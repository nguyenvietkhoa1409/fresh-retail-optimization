# src/inventory/planner_v2.py
"""
ENHANCED INVENTORY PLANNER
Compatible with EnhancedSupplierGenerator and EnhancedProcurementOptimizer

Key Updates:
1. Reads supplier archetypes and metadata
2. Preserves fixed_order_cost information
3. Enhanced supplier-store assignment logic
4. Maintains backward compatibility
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils


class EnhancedInventoryPlanner:
    """
    Inventory planner that works with heterogeneous supplier network
    """
    
    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_DIAGNOSTICS, exist_ok=True)
        self.rng = np.random.default_rng(Cfg.SEED)
    
    def run(self):
        print("\n[Enhanced Inventory Planner] Starting pipeline...")
        
        # 1. Load base demand
        summary = self._load_demand_data()
        
        # 2. Load supply network (with new attributes!)
        suppliers_df, sp_df = self._load_supply_network()
        
        # 3. Enrich with product metadata
        summary = self._enrich_product_info(summary)
        
        # 4. Rescale demand to realistic ranges
        summary = self._rescale_demand_stats(summary)
        
        # 5. Assign suppliers to stores (distance-aware)
        summary = self._assign_suppliers_enhanced(summary, suppliers_df, sp_df)
        
        # 6. Calculate inventory policies
        summary = self._calculate_policies(summary, suppliers_df, sp_df)
        
        # 7. Save artifacts
        self._save_artifacts(summary, suppliers_df, sp_df)
        
        # 8. Diagnostics
        self._run_diagnostics(summary, suppliers_df, sp_df)
        
        print("[Enhanced Inventory Planner] Complete.")
    
    def _load_demand_data(self):
        """Load reconstructed demand from Step 2"""
        print("   -> Loading demand data...")
        
        recon_path = os.path.join(Cfg.OUT_DIR_PART2, "part2_reconstructed.parquet")
        
        if not os.path.exists(recon_path):
            raise FileNotFoundError(f"Missing: {recon_path}. Run Step 2 first.")
        
        df = pd.read_parquet(recon_path)
        
        # Standardize demand column
        if 'y16' in df.columns:
            df['daily_demand'] = df['y16'].apply(
                lambda x: float(np.sum(x)) if isinstance(x, (list, np.ndarray)) else 0.0
            )
        elif 'd_recon' in df.columns:
            df['daily_demand'] = df['d_recon']
        else:
            df['daily_demand'] = df.get('y', 0.0)
        
        # Standardize OOS column
        if 's16' in df.columns:
            df['oos_day'] = df['s16'].apply(
                lambda s: float(np.mean(np.array(s) == 1)) 
                if isinstance(s, (list, np.ndarray)) else 0.0
            )
        else:
            df['oos_day'] = 0.0
        
        # Aggregate by store-product
        summary = df.groupby(['store_id', 'product_id'], observed=True).agg(
            predicted_mean=('daily_demand', 'mean'),
            predicted_std=('daily_demand', lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
            oos_rate=('oos_day', 'mean'),
            n_days=('dt', 'nunique')
        ).reset_index()
        
        # Filter by coverage and volume
        TARGET_STORES = getattr(Cfg, 'GLOBAL_NUM_STORES', 20)
        store_vol = summary.groupby('store_id')['predicted_mean'].sum().sort_values(ascending=False)
        top_stores = store_vol.head(TARGET_STORES).index.tolist()
        
        summary = summary[summary['store_id'].isin(top_stores)].copy()
        summary = summary.sort_values('predicted_mean', ascending=False).head(Cfg.PAIR_LIMIT)
        summary.reset_index(drop=True, inplace=True)
        
        # Type enforcement
        summary['store_id'] = summary['store_id'].astype(str)
        summary['product_id'] = summary['product_id'].astype(int)
        
        print(f"    Loaded {len(summary)} store-product pairs from {len(top_stores)} stores")
        
        return summary
    
    def _load_supply_network(self):
        """
        Load supplier network with enhanced attributes
        
        CRITICAL: Must read from Step 1 artifacts (generator output)
        """
        print("   -> Loading supply network...")
        
        # 1. Load Suppliers (WITH NEW COLUMNS!)
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        
        if not os.path.exists(sup_path):
            raise FileNotFoundError(
                f"Missing: {sup_path}. Run enhanced generator first!"
            )
        
        suppliers_df = pd.read_csv(sup_path)
        suppliers_df['supplier_id'] = suppliers_df['supplier_id'].astype(int)
        
        # Verify new columns exist
        expected_cols = ['archetype', 'distance_tier', 'lead_time_mean_days', 
                        'lead_time_std_days', 'base_price_mult']
        
        missing = [c for c in expected_cols if c not in suppliers_df.columns]
        
        if missing:
            print(f"    ⚠️  WARNING: Missing columns in suppliers.csv: {missing}")
            print(f"    → Regenerate suppliers using EnhancedSupplierGenerator!")
            
            # Add defaults for backward compatibility
            for col in missing:
                if col == 'archetype':
                    suppliers_df['archetype'] = 'generic'
                elif col == 'distance_tier':
                    suppliers_df['distance_tier'] = 'Unknown'
                elif col == 'lead_time_mean_days':
                    suppliers_df['lead_time_mean_days'] = 2.0
                elif col == 'lead_time_std_days':
                    suppliers_df['lead_time_std_days'] = 0.5
                elif col == 'base_price_mult':
                    suppliers_df['base_price_mult'] = 1.0
        
        # Standardize location columns
        if 'sup_lat' in suppliers_df.columns and 'lat' not in suppliers_df.columns:
            suppliers_df['lat'] = suppliers_df['sup_lat']
            suppliers_df['lon'] = suppliers_df['sup_lon']
        
        print(f"    Loaded {len(suppliers_df)} suppliers")
        print(f"    Archetypes: {suppliers_df['archetype'].value_counts().to_dict()}")
        
        # 2. Load Supplier-Product Matrix (WITH FIXED COSTS!)
        sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"Missing: {sp_path}")
        
        sp_df = pd.read_csv(sp_path)
        sp_df['supplier_id'] = sp_df['supplier_id'].astype(int)
        sp_df['product_id'] = sp_df['product_id'].astype(int)
        
        # Verify critical columns
        if 'fixed_order_cost' not in sp_df.columns:
            print("    ⚠️  WARNING: 'fixed_order_cost' missing in supplier_product.csv")
            print("    → Using global default. Sourcing differentiation may be weak!")
            sp_df['fixed_order_cost'] = float(Cfg.FIXED_ORDER_COST)
        
        # Standardize freshness column name
        if 'elapsed_shelf_days' in sp_df.columns and 'freshness_loss_days' not in sp_df.columns:
            sp_df['freshness_loss_days'] = sp_df['elapsed_shelf_days']
        elif 'freshness_loss_days' not in sp_df.columns:
            sp_df['freshness_loss_days'] = 0.0
        
        print(f"    Loaded {len(sp_df)} supplier-product relationships")
        print(f"    Fixed cost range: ${sp_df['fixed_order_cost'].min():.0f} - "
              f"${sp_df['fixed_order_cost'].max():.0f}")
        
        return suppliers_df, sp_df
    
    def _enrich_product_info(self, summary):
        """Add product metadata (category, shelf life, prices)"""
        print("   -> Enriching with product metadata...")
        
        n = len(summary)
        
        # Assign categories (Fresh vs Ambient)
        summary['category_id'] = self.rng.choice([1, 2], size=n, p=[0.5, 0.5])
        
        # Map to simulation product IDs
        summary['sim_product_id'] = summary.apply(
            lambda r: self.rng.choice([101, 102]) if r['category_id'] == 1 
            else self.rng.choice([201, 202]),
            axis=1
        )
        
        # Aggregate duplicates (if same store gets multiple products mapped to same sim_id)
        agg_rules = {
            'predicted_mean': 'sum',
            'predicted_std': lambda x: np.sqrt(np.sum(np.array(x)**2)),
            'oos_rate': 'mean',
            'n_days': 'max',
            'category_id': 'first',
            'sim_product_id': 'first'
        }
        
        summary = summary.groupby(['store_id', 'sim_product_id'], as_index=False).agg(agg_rules)
        summary.rename(columns={'sim_product_id': 'product_id'}, inplace=True)
        
        # Add shelf life
        summary['shelf_life'] = summary['category_id'].map(
            Cfg.SHELF_LIFE_BY_CAT
        ).astype(float)
        
        # Add pricing
        summary['price'] = summary['product_id'].map(
            lambda p: self.rng.uniform(*Cfg.PRICE_RANGE_BY_PRODUCT[p])
        )
        
        # Calculate holding cost
        base_annual_rate = 0.20
        summary['daily_holding_cost_unit'] = (
            summary['price'] * (base_annual_rate / 365.0) * 
            Cfg.HOLDING_COST_MULTIPLIER
        )
        
        # Add demand ranges (for rescaling)
        summary['demand_min'] = summary['product_id'].map(
            lambda p: Cfg.DEMAND_RANGE_BY_PRODUCT[p][0]
        )
        summary['demand_max'] = summary['product_id'].map(
            lambda p: Cfg.DEMAND_RANGE_BY_PRODUCT[p][1]
        )
        
        # Add unit weight
        summary['unit_weight_kg'] = summary['product_id'].map(
            lambda p: Cfg.UNIT_WEIGHT.get(p, 1.0)
        )
        
        return summary
    
    def _rescale_demand_stats(self, summary):
        """Rescale demand to realistic magnitude"""
        print("   -> Rescaling demand statistics...")
        
        # Group statistics by product
        grp_stats = summary.groupby('product_id')['predicted_mean'].agg(
            ['min', 'max']
        ).rename(columns={'min': 'curr_min', 'max': 'curr_max'})
        
        summary = summary.merge(grp_stats, on='product_id', how='left')
        
        # Linear rescaling formula
        numerator = ((summary['predicted_mean'] - summary['curr_min']) * 
                    (summary['demand_max'] - summary['demand_min']))
        denominator = (summary['curr_max'] - summary['curr_min'])
        
        # Handle flat distributions
        mask_flat = denominator < 1e-9
        
        summary.loc[~mask_flat, 'predicted_mean'] = (
            summary['demand_min'] + (numerator / denominator)
        )
        summary.loc[mask_flat, 'predicted_mean'] = (
            (summary['demand_min'] + summary['demand_max']) / 2.0
        )
        
        # Clean up
        summary.drop(columns=['curr_min', 'curr_max'], inplace=True)
        
        # Ensure non-negative
        summary['predicted_std'] = summary['predicted_std'].fillna(0.0).clip(lower=0)
        summary['predicted_mean'] = summary['predicted_mean'].fillna(0.0).clip(lower=0)
        
        # Apply minimum std (CV floor)
        floor_std = Cfg.MIN_STD_FRACTION * summary['predicted_mean']
        summary['predicted_std'] = np.maximum(summary['predicted_std'], floor_std)
        
        return summary
    
    def _assign_suppliers_enhanced(self, summary, suppliers_df, sp_df):
        """
        Enhanced supplier assignment with archetype awareness
        
        Logic:
        - For each store-product, find viable suppliers
        - Consider distance, archetype, and capacity
        - Assign top K candidates (sorted by composite score)
        """
        print("   -> Assigning suppliers to stores (enhanced)...")
        
        # Add store locations
        unique_stores = summary['store_id'].unique()
        store_locs = {}
        
        for sid in unique_stores:
            d = float(self.rng.uniform(*Cfg.STORE_RADIUS_KM))
            b = float(self.rng.uniform(0, 360))
            lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, d, b)
            store_locs[sid] = (lat, lon)
        
        summary['store_lat'] = summary['store_id'].map(lambda s: store_locs[s][0])
        summary['store_lon'] = summary['store_id'].map(lambda s: store_locs[s][1])
        
        # Add time windows (for VRP later)
        def gen_store_tw():
            start_base = Cfg.STORE_OPEN_WINDOW[0]
            start = start_base + self.rng.integers(0, 60)
            duration = self.rng.integers(120, 240)
            return start, start + duration
        
        tw_map = {sid: gen_store_tw() for sid in unique_stores}
        summary['tw_open'] = summary['store_id'].map(lambda s: tw_map[s][0])
        summary['tw_close'] = summary['store_id'].map(lambda s: tw_map[s][1])
        summary['service_time'] = Cfg.SERVICE_TIME_STORE_MINS
        
        # Build supplier-product lookup
        sp_lookup = sp_df.groupby('product_id')['supplier_id'].apply(list).to_dict()
        
        # Supplier metadata lookup
        sup_meta = suppliers_df.set_index('supplier_id').to_dict('index')
        
        # For each store, find viable suppliers
        assigned_suppliers_list = []
        nearest_supplier_list = []
        
        for idx, row in summary.iterrows():
            store_lat, store_lon = row['store_lat'], row['store_lon']
            product_id = row['product_id']
            
            # Get suppliers who can serve this product
            viable_sids = sp_lookup.get(product_id, [])
            
            if not viable_sids:
                assigned_suppliers_list.append([])
                nearest_supplier_list.append(np.nan)
                continue
            
            # Score each supplier (distance + archetype bonus)
            scored = []
            
            for sid in viable_sids:
                if sid not in sup_meta:
                    continue
                
                s = sup_meta[sid]
                
                # Calculate distance
                dist_km = GeoUtils.haversine_km(
                    store_lat, store_lon,
                    s.get('lat', s.get('sup_lat', 0)),
                    s.get('lon', s.get('sup_lon', 0))
                )
                
                # Composite score (lower is better)
                # Penalize distance, but give bonus to certain archetypes
                archetype = s.get('archetype', 'generic')
                
                # Archetype bonuses (adjust competitive balance)
                arch_bonus = {
                    'local_specialty': -10,    # Bonus for local
                    'regional_distributor': 0,  # Neutral
                    'bulk_wholesaler': 5,       # Slight penalty (prefer for bulk only)
                    'farm_direct': 10           # Penalty (prefer for large orders)
                }.get(archetype, 0)
                
                score = dist_km + arch_bonus
                scored.append((sid, score, dist_km))
            
            # Sort by score, take top 10 candidates
            scored.sort(key=lambda x: x[1])
            top_candidates = [s[0] for s in scored[:10]]
            nearest = scored[0][0] if scored else np.nan
            
            assigned_suppliers_list.append(top_candidates)
            nearest_supplier_list.append(nearest)
        
        summary['assigned_suppliers'] = assigned_suppliers_list
        summary['nearest_supplier'] = nearest_supplier_list
        
        print(f"    Assigned average {np.mean([len(a) for a in assigned_suppliers_list]):.1f} "
              f"candidates per store-product")
        
        return summary
    
    def _calculate_policies(self, summary, suppliers_df, sp_df):
        """Calculate inventory policies (ROP, SS, Order Qty)"""
        print("   -> Calculating inventory policies...")
        
        # Service level from OOS rate
        def tau_from_oos(r):
            for lo, hi, tau in Cfg.TAU_BY_OOS:
                if lo <= r < hi:
                    return tau
            return 0.75
        
        summary['tau_i'] = summary['oos_rate'].apply(tau_from_oos)
        summary['z'] = summary['tau_i'].apply(norm.ppf).fillna(1.65)
        
        # Base order quantity (newsvendor)
        summary['order_qty_raw'] = (
            summary['predicted_mean'] + 
            summary['z'] * summary['predicted_std']
        )
        
        # Cap by shelf life constraint
        summary['order_qty_capped'] = np.minimum(
            summary['order_qty_raw'],
            summary['shelf_life'] * summary['predicted_mean']
        ).clip(lower=0.0)
        
        # Convert to units and kg
        summary['order_qty_units'] = np.ceil(summary['order_qty_capped']).astype(int)
        summary['order_qty_kg'] = summary['order_qty_units'] * summary['unit_weight_kg']
        
        # Safety stock
        ss_val = np.maximum(0.0, summary['z'] * summary['predicted_std'])
        summary['safety_stock_units'] = np.ceil(ss_val).astype(int)
        summary['safety_stock_kg'] = summary['safety_stock_units'] * summary['unit_weight_kg']
        
        # ROP (requires lead time from nearest supplier)
        sup_lt_map = suppliers_df.set_index('supplier_id')['lead_time_mean_days'].to_dict()
        sup_lt_std_map = suppliers_df.set_index('supplier_id')['lead_time_std_days'].to_dict()
        
        summary['nearest_lead_mean'] = summary['nearest_supplier'].map(
            lambda s: sup_lt_map.get(s, 2.0) if pd.notna(s) else 2.0
        )
        summary['nearest_lead_std'] = summary['nearest_supplier'].map(
            lambda s: sup_lt_std_map.get(s, 0.5) if pd.notna(s) else 0.5
        )
        
        # ROP formula (accounts for demand + lead time uncertainty)
        L = summary['nearest_lead_mean']
        sigma_L = summary['nearest_lead_std']
        mean_d = summary['predicted_mean']
        std_d = summary['predicted_std']
        
        variance_term = L * (std_d**2) + (mean_d**2) * (sigma_L**2)
        rop_val = (mean_d * L + summary['z'] * np.sqrt(variance_term)).clip(lower=0)
        
        summary['rop_units'] = np.ceil(rop_val).astype(int)
        summary['rop_kg'] = summary['rop_units'] * summary['unit_weight_kg']
        
        return summary
    
    def _save_artifacts(self, summary, suppliers_df, sp_df):
        """Save enriched data for downstream use"""
        print("   -> Saving artifacts...")
        
        # 1. Basic version (for Step 5 optimization)
        basic_cols = [
            'store_id', 'product_id', 'store_lat', 'store_lon',
            'predicted_mean', 'predicted_std', 'unit_weight_kg',
            'oos_rate', 'n_days', 'tw_open', 'tw_close', 'service_time',
            'assigned_suppliers', 'nearest_supplier'
        ]
        
        # Filter to existing columns
        basic_cols_existing = [c for c in basic_cols if c in summary.columns]
        
        out_basic = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement.parquet")
        summary[basic_cols_existing].to_parquet(out_basic, index=False)
        print(f"    Saved: unified_for_procurement.parquet")
        
        # 2. Enhanced version (with full metadata)
        enhanced_cols = basic_cols + [
            'category_id', 'shelf_life', 'tau_i', 'z',
            'order_qty_units', 'order_qty_kg', 'order_qty_capped',
            'safety_stock_units', 'safety_stock_kg',
            'rop_units', 'rop_kg',
            'price', 'daily_holding_cost_unit',
            'nearest_lead_mean', 'nearest_lead_std'
        ]
        
        enhanced_cols_existing = [c for c in enhanced_cols if c in summary.columns]
        
        out_enhanced = os.path.join(Cfg.ARTIFACTS_DIR, 
                                    "unified_for_procurement_enhanced.parquet")
        summary[enhanced_cols_existing].to_parquet(out_enhanced, index=False)
        print(f"    Saved: unified_for_procurement_enhanced.parquet")
        
        # 3. Save harmonized supplier files (optional, for reference)
        # Note: These are just copies/links to original files from generator
        suppliers_out = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv")
        suppliers_df.to_csv(suppliers_out, index=False)
        
        sp_out = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product_harmonized_v2.csv")
        sp_df.to_csv(sp_out, index=False)
        
        print(f"    Saved harmonized supplier references")
    
    def _run_diagnostics(self, summary, suppliers_df, sp_df):
        """Generate diagnostic summary"""
        print("   -> Running diagnostics...")
        
        total_demand_kg = summary['order_qty_kg'].sum()
        
        # Calculate total supply capacity
        if 'supplier_capacity_kg' in sp_df.columns:
            # Aggregate by product
            supply_by_product = sp_df.groupby('product_id')['supplier_capacity_kg'].sum()
            
            # Match with demand
            demand_by_product = summary.groupby('product_id')['order_qty_kg'].sum()
            
            ratios = []
            for pid in demand_by_product.index:
                d = demand_by_product[pid]
                s = supply_by_product.get(pid, 0)
                if d > 0:
                    ratios.append(s / d)
            
            avg_ratio = np.mean(ratios) if ratios else 0.0
        else:
            avg_ratio = 0.0
        
        # Write diagnostics
        diag_path = os.path.join(Cfg.OUT_DIR_DIAGNOSTICS, "diagnostics_summary_v2.txt")
        
        with open(diag_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(" ENHANCED INVENTORY PLANNER - DIAGNOSTICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Demand (kg): {total_demand_kg:,.2f}\n")
            f.write(f"Average Supply/Demand Ratio: {avg_ratio:.2f}x\n\n")
            
            f.write("Supplier Network:\n")
            f.write(f"  Total Suppliers: {len(suppliers_df)}\n")
            
            if 'archetype' in suppliers_df.columns:
                f.write("\n  By Archetype:\n")
                for arch, count in suppliers_df['archetype'].value_counts().items():
                    f.write(f"    {arch}: {count}\n")
            
            f.write(f"\nDemand Summary:\n")
            f.write(f"  Store-Product Pairs: {len(summary)}\n")
            f.write(f"  Unique Stores: {summary['store_id'].nunique()}\n")
            f.write(f"  Unique Products: {summary['product_id'].nunique()}\n")
            
            if 'category_id' in summary.columns:
                f.write("\n  By Category:\n")
                for cat, grp in summary.groupby('category_id'):
                    cat_name = "Fresh" if cat == 1 else "Ambient"
                    f.write(f"    {cat_name}: {len(grp)} pairs, "
                           f"{grp['order_qty_kg'].sum():,.0f} kg\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"    Diagnostics saved to {diag_path}")


# === INTEGRATION HELPER ===
# For backward compatibility, can import as original name
InventoryPlanner = EnhancedInventoryPlanner
