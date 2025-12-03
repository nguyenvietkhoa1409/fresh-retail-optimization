# src/inventory/planner.py
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class InventoryPlanner:
    """
    Class thực hiện Inventory Planning.
    Updated: Removed premature baseline comparison. Focuses strictly on 
    calculating Inventory Policies (Newsvendor, ROP, SS) for the solvers.
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_DIAGNOSTICS, exist_ok=True)
        self.rng = np.random.default_rng(Cfg.SEED)

    def run(self):
        print("\n[Inventory] Starting Inventory Planning Pipeline...")
        
        # 1. Load Data
        summary = self._load_and_aggregate()
        
        # 2. Enrich Product Info
        summary = self._enrich_product_info(summary)
        
        # 3. Rescale Demand & Floor Std
        summary = self._rescale_demand_stats(summary)
        
        # 4. Generate Suppliers
        suppliers_df, sp_df, summary = self._setup_suppliers_and_matrix(summary)
        
        # 5. Calculate Policies (Newsvendor, ROP, SS)
        # This creates the parameters that the Optimization Step will use.
        summary, suppliers_df, sp_df = self._calculate_policies_and_scale(summary, suppliers_df, sp_df)
        
        # 6. Save Artifacts
        self._save_artifacts(summary, suppliers_df, sp_df)
        
        # 7. Diagnostics (Technical checks only, no financial claims yet)
        self._run_diagnostics(summary, suppliers_df)

    def _load_and_aggregate(self):
        print("  -> Loading reconstructed data...")
        recon_path = os.path.join(Cfg.OUT_DIR_PART2, "part2_reconstructed.parquet")
        df = pd.read_parquet(recon_path)
        
        if 'y16' in df.columns:
            df['daily_demand'] = df['y16'].apply(lambda x: float(np.sum(x)) if isinstance(x, (list,np.ndarray)) else 0.0)
        elif 'd_recon' in df.columns:
            df['daily_demand'] = df['d_recon']
            
        if 's16' in df.columns:
            df['oos_day'] = df['s16'].apply(lambda s: float(np.mean(np.array(s) == 1)) if isinstance(s, (list,np.ndarray)) else 0.0)
        else: df['oos_day'] = 0.0
        
        summary = df.groupby(['store_id', 'product_id']).agg(
            predicted_mean=('daily_demand', 'mean'),
            predicted_std=('daily_demand', lambda x: np.std(x, ddof=1) if len(x)>1 else 0.0),
            oos_rate=('oos_day', 'mean'),
            n_days=('dt', 'nunique')
        ).reset_index()
        
        summary = summary[summary['n_days'] >= 60].sort_values('predicted_mean', ascending=False).head(Cfg.PAIR_LIMIT)
        print(f"     Kept {len(summary)} pairs for planning.")
        return summary

    def _enrich_product_info(self, summary):
        n = len(summary)
        summary['category_id'] = self.rng.choice([1, 2], size=n, p=[0.5, 0.5])
        summary['sim_product_id'] = summary['category_id'].apply(lambda c: self.rng.choice([101, 102]) if c==1 else self.rng.choice([201, 202]))
        summary['shelf_life'] = summary['category_id'].map(Cfg.SHELF_LIFE_BY_CAT).astype(float)
        
        summary['price'] = summary['sim_product_id'].apply(lambda pid: float(self.rng.uniform(*Cfg.PRICE_RANGE_BY_PRODUCT[pid])))
        
        # Daily Holding Cost (Used later in Integrated Solver)
        base_annual_rate = 0.20
        summary['daily_holding_cost_unit'] = summary['price'] * (base_annual_rate / 365.0) * Cfg.HOLDING_COST_MULTIPLIER

        summary['demand_min'] = summary['sim_product_id'].apply(lambda pid: Cfg.DEMAND_RANGE_BY_PRODUCT[pid][0])
        summary['demand_max'] = summary['sim_product_id'].apply(lambda pid: Cfg.DEMAND_RANGE_BY_PRODUCT[pid][1])
        summary['unit_weight_kg'] = summary['sim_product_id'].apply(lambda pid: Cfg.UNIT_WEIGHT.get(int(pid), 1.0)).astype(float)
        
        unique_stores = summary['store_id'].unique()
        store_locs = {sid: GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, float(self.rng.uniform(*Cfg.STORE_RADIUS_KM)), float(self.rng.uniform(0, 360))) for sid in unique_stores}
        summary['store_lat'] = summary['store_id'].map(lambda s: store_locs[s][0])
        summary['store_lon'] = summary['store_id'].map(lambda s: store_locs[s][1])
        return summary

    def _rescale_demand_stats(self, summary):
        def rescale_group(g):
            old = g['predicted_mean'].values.astype(float)
            newmin, newmax = float(g['demand_min'].iloc[0]), float(g['demand_max'].iloc[0])
            if (old.max() - old.min()) <= 1e-9: return pd.Series(np.full_like(old, (newmin+newmax)/2.0), index=g.index)
            return pd.Series(newmin + (old - old.min()) * (newmax - newmin) / (old.max() - old.min()), index=g.index)
        
        summary['predicted_mean'] = summary.groupby('sim_product_id', group_keys=False).apply(rescale_group).reset_index(drop=True)
        floor = np.maximum(0.05, Cfg.MIN_STD_FRACTION * np.maximum(1.0, summary['predicted_mean']))
        summary['predicted_std'] = np.maximum(summary['predicted_std'].astype(float), floor)
        return summary

    def _setup_suppliers_and_matrix(self, summary):
        print("  -> Generating enhanced suppliers and matrix...")
        suppliers = []
        for cat in [1, 2]:
            counter = 0
            for count, rmin, rmax in Cfg.SUPPLIER_RINGS:
                for _ in range(count):
                    sid = cat * 100000 + counter; counter += 1
                    lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, float(self.rng.uniform(rmin, rmax)), float(self.rng.uniform(0, 360)))
                    suppliers.append({
                        'supplier_id': int(sid), 'category_id': int(cat), 'sup_lat': lat, 'sup_lon': lon,
                        'reliability': float(self.rng.uniform(0.80, 0.99)),
                        'capacity_kg': float(self.rng.uniform(100.0, 5000.0)),
                        'min_order_qty_units_supplier': self.rng.integers(*Cfg.MOQ_RANGE_UNITS),
                        'lead_time_mean_days': self.rng.integers(1, 4).astype(float),
                        'lead_time_std_days': self.rng.random() * 0.75,
                    })
        suppliers_df = pd.DataFrame(suppliers)
        suppliers_df['on_time_rate'] = (suppliers_df['reliability'] - 0.02).clip(0.5, 0.995)
        
        def assign_top_k(row, k=5):
            cat_sup = suppliers_df[suppliers_df['category_id'] == row['category_id']]
            if cat_sup.empty: return []
            dists = cat_sup.apply(lambda s: GeoUtils.haversine_km(row['store_lat'], row['store_lon'], s['sup_lat'], s['sup_lon']), axis=1)
            return cat_sup.loc[dists.nsmallest(k).index, 'supplier_id'].tolist()
        
        summary['assigned_suppliers'] = summary.apply(assign_top_k, axis=1)
        summary['nearest_supplier'] = summary['assigned_suppliers'].apply(lambda x: x[0] if len(x)>0 else np.nan)
        
        sp_rows = []
        for _, s in suppliers_df.iterrows():
            pids = [101, 102] if s['category_id'] == 1 else [201, 202]
            for pid in pids:
                sp_rows.append({
                    'supplier_id': int(s['supplier_id']), 'product_id': pid,
                    'min_order_qty_units': max(1, int(s['min_order_qty_units_supplier'])),
                    'unit_price': float(np.random.uniform(*Cfg.PRICE_RANGE_BY_PRODUCT[pid]) * self.rng.uniform(0.92, 1.12)),
                    'elapsed_shelf_days': max(0, int(round(s['lead_time_mean_days']))),
                    'lead_time_mean_days': s['lead_time_mean_days'], 'lead_time_std_days': s['lead_time_std_days'],
                    'on_time_rate': s['on_time_rate'], 'supplier_capacity_kg': s['capacity_kg']
                })
        return suppliers_df, pd.DataFrame(sp_rows), summary

    def _calculate_policies_and_scale(self, summary, suppliers_df, sp_df):
        print("  -> Calculating Inventory Policies (ROP, SS)...")
        def tau_from_oos(r):
            for lo, hi, tau in Cfg.TAU_BY_OOS:
                if lo <= r < hi: return tau
            return 0.75
        summary['tau_i'] = summary['oos_rate'].apply(tau_from_oos)
        summary['z'] = summary['tau_i'].apply(norm.ppf)
        
        summary['order_qty_raw'] = summary['predicted_mean'] + summary['z'] * summary['predicted_std']
        summary['order_qty_capped'] = np.minimum(summary['order_qty_raw'], summary['shelf_life'] * summary['predicted_mean']).clip(lower=0.0)
        summary['order_qty_units'] = np.ceil(summary['order_qty_capped']).astype(int)
        summary['order_qty_kg'] = summary['order_qty_units'] * summary['unit_weight_kg']
        summary['safety_stock_units'] = np.ceil(np.maximum(0.0, summary['z'] * summary['predicted_std'])).astype(int)
        summary['safety_stock_kg'] = summary['safety_stock_units'] * summary['unit_weight_kg']

        # Scale Capacity to match Demand (Simulated Environment)
        total_demand = summary['order_qty_kg'].sum()
        current_cap = suppliers_df['capacity_kg'].sum()
        scale = (total_demand * Cfg.TARGET_SUPPLY_DEMAND_RATIO) / max(1.0, current_cap)
        suppliers_df['capacity_kg'] *= scale
        sp_df['supplier_capacity_kg'] = sp_df['supplier_id'].map(suppliers_df.set_index('supplier_id')['capacity_kg'].to_dict())
        
        # ROP
        lt_mean = suppliers_df.set_index('supplier_id')['lead_time_mean_days'].to_dict()
        lt_std = suppliers_df.set_index('supplier_id')['lead_time_std_days'].to_dict()
        summary['nearest_lead_mean'] = summary['nearest_supplier'].map(lt_mean).fillna(2.0)
        summary['nearest_lead_std'] = summary['nearest_supplier'].map(lt_std).fillna(0.5)
        
        mean, std, L, sigma_L = summary['predicted_mean'], summary['predicted_std'], summary['nearest_lead_mean'], summary['nearest_lead_std']
        summary['rop_units'] = np.ceil((mean * L + summary['z'] * np.sqrt(L*std**2 + mean**2*sigma_L**2)).clip(lower=0)).astype(int)
        summary['rop_kg'] = summary['rop_units'] * summary['unit_weight_kg']
        
        return summary, suppliers_df, sp_df

    def _save_artifacts(self, summary, suppliers, sp):
        print("  -> Saving Inventory Artifacts...")
        base = ['store_id','product_id','sim_product_id','store_lat','store_lon','predicted_mean','predicted_std','unit_weight_kg','oos_rate','n_days']
        summary[base].to_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement.parquet"), index=False)
        
        enh_cols = base + ['category_id','shelf_life','tau_i','order_qty_units','order_qty_kg','safety_stock_kg',
                           'assigned_suppliers','price','rop_kg','daily_holding_cost_unit', 'safety_stock_units']
        
        for c in enh_cols: 
            if c not in summary.columns: summary[c] = 0
            
        summary[enh_cols].to_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet"), index=False)
        suppliers.to_csv(os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv"), index=False)
        sp.to_csv(os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product_harmonized_v2.csv"), index=False)

    def _run_diagnostics(self, summary, suppliers_df):
        max_veh_cap = max([v[1] for v in Cfg.VEHICLES_CHECK])
        pct_large = (summary['order_qty_kg'] > max_veh_cap).mean() * 100.0
        total_d = summary['order_qty_kg'].sum()
        total_s = suppliers_df['capacity_kg'].sum()
        
        with open(os.path.join(Cfg.OUT_DIR_DIAGNOSTICS, "diagnostics_summary_v2.txt"), 'w') as f:
            f.write(f"Total Demand (kg): {total_d:.2f}\n")
            f.write(f"Total Supply (kg): {total_s:.2f}\n")
            f.write(f"Ratio: {total_s/total_d:.3f}\n")
            f.write(f"Orders > {max_veh_cap}kg: {pct_large:.2f}%\n")
        print(f"     Diag saved to diagnostics_summary_v2.txt")