import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class InventoryPlanner:
    """
    Class thực hiện Inventory Planning.
    [FIXED VERSION]
    1. Loads Suppliers/Products directly from Step 1 (Generator) instead of regenerating them.
       -> Solves the ID Mismatch (100005 vs 1) permanently.
    2. Calculates Inventory Policies based on consistent Supply Chain network.
    3. Saves unified parquet with CORRECT assigned_suppliers IDs.
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_DIAGNOSTICS, exist_ok=True)
        self.rng = np.random.default_rng(Cfg.SEED)

    def run(self):
        print("\n[Inventory] Starting Inventory Planning Pipeline...")
        
        # 1. Load Data (Demand & Supply Network)
        summary = self._load_demand_data()
        suppliers_df, sp_df = self._load_supply_network()
        
        # 2. Enrich Product Info (+ Store Time Windows)
        summary = self._enrich_product_info(summary)
        
        # 3. Rescale Demand (Optional normalization)
        summary = self._rescale_demand_stats(summary)
        
        # 4. Map Stores to Suppliers (Distance Matrix Logic)
        summary = self._assign_suppliers(summary, suppliers_df)
        
        # 5. Calculate Policies (Newsvendor, ROP, SS)
        summary, suppliers_df, sp_df = self._calculate_policies_and_scale(summary, suppliers_df, sp_df)
        
        # 6. Save Artifacts
        self._save_artifacts(summary, suppliers_df, sp_df)
        
        # 7. Diagnostics
        self._run_diagnostics(summary, suppliers_df)

    def _load_demand_data(self):
        print("   -> Loading reconstructed demand...")
        recon_path = os.path.join(Cfg.OUT_DIR_PART2, "part2_reconstructed.parquet")
        df = pd.read_parquet(recon_path)
        
        # Standardize demand column
        if 'y16' in df.columns:
            df['daily_demand'] = df['y16'].apply(lambda x: float(np.sum(x)) if isinstance(x, (list,np.ndarray)) else 0.0)
        elif 'd_recon' in df.columns:
            df['daily_demand'] = df['d_recon']
            
        # Standardize OOS column
        if 's16' in df.columns:
            df['oos_day'] = df['s16'].apply(lambda s: float(np.mean(np.array(s) == 1)) if isinstance(s, (list,np.ndarray)) else 0.0)
        else: df['oos_day'] = 0.0
        
        summary = df.groupby(['store_id', 'product_id']).agg(
            predicted_mean=('daily_demand', 'mean'),
            predicted_std=('daily_demand', lambda x: np.std(x, ddof=1) if len(x)>1 else 0.0),
            oos_rate=('oos_day', 'mean'),
            n_days=('dt', 'nunique')
        ).reset_index()
        
        # Filter Logic
        TARGET_STORE_COUNT = getattr(Cfg, 'GLOBAL_NUM_STORES', 20)
        store_vol = summary.groupby('store_id')['predicted_mean'].sum().sort_values(ascending=False)
        top_stores = store_vol.head(TARGET_STORE_COUNT).index.tolist()
        
        summary = summary[summary['store_id'].isin(top_stores)].copy()
        summary = summary.sort_values('predicted_mean', ascending=False).head(Cfg.PAIR_LIMIT)
        summary.reset_index(drop=True, inplace=True)
        
        summary['store_id'] = summary['store_id'].astype(str)
        summary['product_id'] = summary['product_id'].astype(int)
        
        print(f"    [FILTER] Kept {len(top_stores)} Stores, {len(summary)} Pairs.")
        return summary

    def _load_supply_network(self):
        """[CRITICAL FIX] Load generated suppliers instead of creating new ones."""
        print("   -> Loading Supply Network (Step 1 Artifacts)...")
        
        # Load Suppliers
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        if not os.path.exists(sup_path):
            raise FileNotFoundError(f"Missing {sup_path}. Please run Step 1 (Generator) first.")
        suppliers_df = pd.read_csv(sup_path)
        suppliers_df['supplier_id'] = suppliers_df['supplier_id'].astype(int)
        
        # Ensure lat/lon naming consistency
        if 'lat' in suppliers_df.columns:
            suppliers_df.rename(columns={'lat': 'sup_lat', 'lon': 'sup_lon'}, inplace=True)
            
        # Load Supplier-Product
        sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        sp_df = pd.read_csv(sp_path)
        sp_df['supplier_id'] = sp_df['supplier_id'].astype(int)
        sp_df['product_id'] = sp_df['product_id'].astype(int)
        
        # Enrich supplier_product with lead time info from suppliers_df if missing
        # (Assuming simple logic: lead time is mostly geo-based, generated below)
        
        return suppliers_df, sp_df

    def _enrich_product_info(self, summary):
        # ... (Logic giữ nguyên, chỉ đảm bảo consistency)
        n = len(summary)
        summary['category_id'] = self.rng.choice([1, 2], size=n, p=[0.5, 0.5])
        summary['sim_product_id'] = summary.apply(lambda r: self.rng.choice([101, 102]) if r['category_id']==1 else self.rng.choice([201, 202]), axis=1)
        #Đồng bộ IDs
        summary['product_id'] = summary['sim_product_id']
        agg_rules = {
            'predicted_mean': 'sum',
            'predicted_std': lambda x: np.sqrt(np.sum(np.array(x)**2)), # Gộp độ lệch chuẩn
            'oos_rate': 'mean',
            'n_days': 'max',
            'category_id': 'first',
            'sim_product_id': 'first'
        }
        # Thực hiện Groupby
        summary = summary.groupby(['store_id', 'product_id'], as_index=False).agg(agg_rules)

        summary['shelf_life'] = summary['category_id'].map(Cfg.SHELF_LIFE_BY_CAT).astype(float)
        
        summary['price'] = summary['sim_product_id'].map(lambda p: self.rng.uniform(*Cfg.PRICE_RANGE_BY_PRODUCT[p]))
        
        base_annual_rate = 0.20
        summary['daily_holding_cost_unit'] = summary['price'] * (base_annual_rate / 365.0) * Cfg.HOLDING_COST_MULTIPLIER

        summary['demand_min'] = summary['sim_product_id'].map(lambda p: Cfg.DEMAND_RANGE_BY_PRODUCT[p][0])
        summary['demand_max'] = summary['sim_product_id'].map(lambda p: Cfg.DEMAND_RANGE_BY_PRODUCT[p][1])
        summary['unit_weight_kg'] = summary['sim_product_id'].map(lambda p: Cfg.UNIT_WEIGHT.get(p, 1.0))
        
        unique_stores = summary['store_id'].unique()
        
        # Generate Locations & Time Windows for Stores
        store_locs = {sid: GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, float(self.rng.uniform(*Cfg.STORE_RADIUS_KM)), float(self.rng.uniform(0, 360))) for sid in unique_stores}
        summary['store_lat'] = summary['store_id'].map(lambda s: store_locs[s][0])
        summary['store_lon'] = summary['store_id'].map(lambda s: store_locs[s][1])
        
        def gen_store_tw():
            start_base = Cfg.STORE_OPEN_WINDOW[0]
            start = start_base + self.rng.integers(0, 60)
            duration = self.rng.integers(120, 240)
            return start, start + duration

        tw_map = {sid: gen_store_tw() for sid in unique_stores}
        summary['tw_open'] = summary['store_id'].map(lambda s: tw_map[s][0])
        summary['tw_close'] = summary['store_id'].map(lambda s: tw_map[s][1])
        summary['service_time'] = Cfg.SERVICE_TIME_STORE_MINS
        
        return summary

    def _rescale_demand_stats(self, summary):
        # [Keeping your logic, it's fine]
        grp_stats = summary.groupby('sim_product_id')['predicted_mean'].agg(['min', 'max']).rename(columns={'min': 'curr_min', 'max': 'curr_max'})
        summary = summary.merge(grp_stats, on='sim_product_id', how='left')
        
        numerator = (summary['predicted_mean'] - summary['curr_min']) * (summary['demand_max'] - summary['demand_min'])
        denominator = (summary['curr_max'] - summary['curr_min'])
        mask_flat = denominator < 1e-9
        
        summary.loc[~mask_flat, 'predicted_mean'] = summary['demand_min'] + (numerator / denominator)
        summary.loc[mask_flat, 'predicted_mean'] = (summary['demand_min'] + summary['demand_max']) / 2.0
        
        summary.drop(columns=['curr_min', 'curr_max'], inplace=True)
        summary['predicted_std'] = summary['predicted_std'].fillna(0.0)
        summary['predicted_mean'] = summary['predicted_mean'].fillna(0.0)
        
        floor_std = Cfg.MIN_STD_FRACTION * summary['predicted_mean']
        summary['predicted_std'] = np.maximum(summary['predicted_std'], floor_std)
        return summary

    def _assign_suppliers(self, summary, suppliers_df):
        """
        Calculates distances and assigns top-k suppliers.
        Using the LOADED suppliers (ID 1, 2...), not generated ones.
        """
        print("   -> Mapping Stores to Suppliers...")
        
        # Helper to compute distance for each store against all suppliers
        # Optimizing this: Pre-calculate supplier locations
        sup_locs = suppliers_df[['supplier_id', 'sup_lat', 'sup_lon', 'zone_label']].to_dict('records')
        
        # Pre-enrich suppliers_df with necessary columns if missing (e.g. lead time from dist)
        if 'lead_time_mean_days' not in suppliers_df.columns:
             # Basic logic if missing in source
             suppliers_df['lead_time_mean_days'] = 2.0 
             suppliers_df['lead_time_std_days'] = 0.5

        def get_candidates(row):
            # Calculate dist to all suppliers
            # Filter by simple logic (e.g. random for now or all)
            # Since we want "assigned_suppliers" to be valid IDs:
            
            # Simple heuristic: Just pick 5 random suppliers for variety, or nearest.
            # Here we pick nearest 10 to ensure feasibility
            dists = []
            for s in sup_locs:
                d = GeoUtils.haversine_km(row['store_lat'], row['store_lon'], s['sup_lat'], s['sup_lon'])
                dists.append((s['supplier_id'], d))
            
            dists.sort(key=lambda x: x[1])
            return [x[0] for x in dists[:10]]

        # Apply to unique stores only to save time
        unique_stores = summary[['store_id', 'store_lat', 'store_lon']].drop_duplicates()
        unique_stores['assigned_suppliers'] = unique_stores.apply(get_candidates, axis=1)
        
        # Merge back
        summary = summary.merge(unique_stores[['store_id', 'assigned_suppliers']], on='store_id', how='left')
        summary['nearest_supplier'] = summary['assigned_suppliers'].apply(lambda x: x[0] if len(x)>0 else np.nan)
        
        return summary

    def _calculate_policies_and_scale(self, summary, suppliers_df, sp_df):
        print("   -> Calculating Inventory Policies...")
        
        # [Supply Capacity Scaling Logic - Optional]
        # Check global balance
        dem = summary['predicted_mean'].sum() * 1.0 # Rough estimate
        sup = sp_df['supplier_capacity_kg'].sum() if 'supplier_capacity_kg' in sp_df.columns else 0
        
        # If supply info missing in sp_df, merge from suppliers_df (if needed)
        # But assuming Step 1 generated correct sp_df with capacity
        
        # Policy Calc
        def tau_from_oos(r):
            for lo, hi, tau in Cfg.TAU_BY_OOS:
                if lo <= r < hi: return tau
            return 0.75
        
        summary['tau_i'] = summary['oos_rate'].apply(tau_from_oos)
        summary['z'] = summary['tau_i'].apply(norm.ppf).fillna(1.65)
        
        summary['order_qty_raw'] = summary['predicted_mean'] + summary['z'] * summary['predicted_std']
        summary['order_qty_capped'] = np.minimum(
            summary['order_qty_raw'], 
            summary['shelf_life'] * summary['predicted_mean']
        ).clip(lower=0.0)
        
        summary['order_qty_units'] = np.ceil(summary['order_qty_capped'].fillna(0)).astype(int)
        summary['order_qty_kg'] = summary['order_qty_units'] * summary['unit_weight_kg']
        
        ss_val = np.maximum(0.0, summary['z'] * summary['predicted_std'])
        summary['safety_stock_units'] = np.ceil(ss_val.fillna(0)).astype(int)
        summary['safety_stock_kg'] = summary['safety_stock_units'] * summary['unit_weight_kg']
        
        # ROP (Requires Lead Time)
        # Map Nearest Supplier Lead Time
        lt_map = suppliers_df.set_index('supplier_id').to_dict('index')
        
        def get_lt(sid):
            if pd.isna(sid): return 2.0, 0.5
            s = lt_map.get(sid, {})
            # Fallback values if Step 1 didn't generate columns
            return s.get('lead_time_mean_days', 2.0), s.get('lead_time_std_days', 0.5)

        summary['lt_stats'] = summary['nearest_supplier'].apply(get_lt)
        summary['nearest_lead_mean'] = summary['lt_stats'].apply(lambda x: x[0])
        summary['nearest_lead_std'] = summary['lt_stats'].apply(lambda x: x[1])
        summary.drop(columns=['lt_stats'], inplace=True)
        
        L = summary['nearest_lead_mean']
        sigma_L = summary['nearest_lead_std']
        mean = summary['predicted_mean']
        std = summary['predicted_std']
        
        variance_term = L * (std**2) + (mean**2) * (sigma_L**2)
        rop_val = (mean * L + summary['z'] * np.sqrt(variance_term)).clip(lower=0)
        
        summary['rop_units'] = np.ceil(rop_val.fillna(0)).astype(int)
        summary['rop_kg'] = summary['rop_units'] * summary['unit_weight_kg']
        
        return summary, suppliers_df, sp_df

    def _save_artifacts(self, summary, suppliers, sp):
        print("   -> Saving Unified Artifacts...")
        base = ['store_id','product_id','sim_product_id','store_lat','store_lon','predicted_mean',
                'predicted_std','unit_weight_kg','oos_rate','n_days',
                'tw_open', 'tw_close', 'service_time', 
                'assigned_suppliers'] # IMPORTANT: Now contains correct IDs
        
        # Ensure column existence
        for c in base:
            if c not in summary.columns: summary[c] = None
            
        summary[base].to_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement.parquet"), index=False)
        
        enh_cols = base + ['category_id','shelf_life','tau_i','order_qty_units','order_qty_kg','safety_stock_kg',
                           'price','rop_kg','daily_holding_cost_unit', 'safety_stock_units']
        
        summary[enh_cols].to_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet"), index=False)
        
        # We DO NOT save 'suppliers_harmonized_v2.csv' anymore to avoid confusion.
        # Step 5 should read 'suppliers.csv' from Step 1.
        print("   -> Saved unified_for_procurement_enhanced.parquet")

    def _run_diagnostics(self, summary, suppliers_df):
        # Simple diag
        total_d = summary['order_qty_kg'].sum()
        # Ensure we check the LOADED capacity
        if 'capacity_kg' in suppliers_df.columns:
            total_s = suppliers_df['capacity_kg'].sum()
        else:
            total_s = 0.0 # Warning
            
        with open(os.path.join(Cfg.OUT_DIR_DIAGNOSTICS, "diagnostics_summary_v2.txt"), 'w') as f:
            f.write(f"Total Demand (kg): {total_d:.2f}\n")
            f.write(f"Total Supply (kg): {total_s:.2f}\n")
            f.write(f"Ratio: {total_s/total_d if total_d>0 else 0:.3f}\n")