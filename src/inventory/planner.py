# src/inventory/planner.py
"""
DATA-DRIVEN INVENTORY PLANNER (SAA Version)
Aligns with Forecasting Hybrid ML/SMA
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class InventoryPlanner:
    """
    DATA-DRIVEN INVENTORY PLANNER (SAA Version)
    
    Core Improvements:
    1. Reads 'future_forecast.parquet' (ML Output) instead of raw history.
    2. Reads 'master_product_catalog.parquet' (Enriched Economics) instead of random gen.
    3. Implements 'Pooled SAA' (Sample Average Approximation):
       - Uses 'forecast_residuals.parquet' to build error distributions.
       - Pools errors by 'risk_group' to handle sparse data/new products.
       - Calculates Safety Stock dynamically based on Target Service Level (Cu/Cu+Co).
    """
    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_DIAGNOSTICS, exist_ok=True)
        self.rng = np.random.default_rng(Cfg.SEED)
        
        # Placeholders
        self.catalog = None
        self.residuals_pool = {} # {risk_group: [scaled_errors]}
    
    def run(self):
        print("\n[Data-Driven Inventory Planner] Starting pipeline...")
        
        # 1. Load Master Catalog
        self.catalog = self._load_catalog()
        
        # 2. Load Forecasts (ML + SMA mixed)
        forecast_df = self._load_forecast_data()
        
        # 3. Load & Process Residuals (Uncertainty)
        self._build_saa_residuals_pool()
        
        # 4. Merge Data Streams
        summary = self._merge_data(forecast_df)
        
        # 5. Load Supply Network
        suppliers_df, sp_df = self._load_supply_network()
        
        # 6. Assign Suppliers
        summary = self._assign_suppliers(summary, suppliers_df, sp_df)
        
        # 7. Calculate Policies (SAA Core)
        summary = self._calculate_policies_saa(summary, suppliers_df)
        
        # 8. Save & Diagnostics
        self._save_artifacts(summary, suppliers_df, sp_df)
        self._run_diagnostics(summary)
        
        print("[Data-Driven Inventory Planner] Complete.")

    def _load_catalog(self):
        cat_path = os.path.join(Cfg.ARTIFACTS_DIR, "master_product_catalog.parquet")
        if not os.path.exists(cat_path):
            raise FileNotFoundError(f"Missing {cat_path}. Run CatalogEnricher first.")
        
        df = pd.read_parquet(cat_path)
        df['product_id'] = df['product_id'].astype(str)
        print(f"  -> Loaded Catalog: {len(df)} SKUs. Risk Groups: {df['risk_group'].unique()}")
        return df

    def _load_forecast_data(self):
        fc_path = os.path.join(Cfg.ARTIFACTS_DIR, "future_forecast.parquet")
        if not os.path.exists(fc_path):
            raise FileNotFoundError(f"Missing {fc_path}. Run Forecasting v5.0 first.")
        
        df = pd.read_parquet(fc_path)
        df['product_id'] = df['product_id'].astype(str)
        df['store_id'] = df['store_id'].astype(str)
        
        # Check if 'method' column exists (from v5.0)
        has_method = 'method' in df.columns
        
        # --- FIX: Sử dụng tuple (Source Column, Function) rõ ràng ---
        agg_dict = {
            'predicted_mean': ('predicted_mean', 'mean'),  # Sửa: Khai báo rõ cột nguồn
            'n_days': ('date', 'nunique')                  # Sửa: Dùng tuple đơn giản thay vì NamedAgg
        }
        
        if has_method:
            agg_dict['forecast_method'] = ('method', 'first')
            
        summary = df.groupby(['store_id', 'product_id'], observed=True).agg(**agg_dict).reset_index()
        
        print(f"  -> Loaded Forecasts: {len(summary)} pairs.")
        if has_method:
            print(f"     Method Breakdown: {summary['forecast_method'].value_counts().to_dict()}")
            
        return summary

    def _build_saa_residuals_pool(self):
        """Build Pooled Error Distributions from Residuals"""
        res_path = os.path.join(Cfg.ARTIFACTS_DIR, "forecast_residuals.parquet")
        if not os.path.exists(res_path):
            print("  -> [Warning] No residuals found. SAA will fallback to Normal approx.")
            return

        df_res = pd.read_parquet(res_path)
        df_res['product_id'] = df_res['product_id'].astype(str)
        
        # Merge with catalog to get Risk Group
        df_res = df_res.merge(self.catalog[['product_id', 'risk_group']], on='product_id', how='left')
        df_res['risk_group'] = df_res['risk_group'].fillna('Normal')
        
        # Calculate Scaled Error (Scale-Free)
        # Handle division by zero/small numbers robustly
        safe_pred = np.maximum(df_res['y_pred'], 0.1)
        df_res['scaled_error'] = df_res['error'] / safe_pred
        
        # Remove extreme outliers (e.g., > 1000% error) that might break SAA
        # df_res = df_res[df_res['scaled_error'].abs() < 10.0] 
        
        self.residuals_pool = {}
        for rg, group in df_res.groupby('risk_group'):
            errors = group['scaled_error'].values
            # Ensure we have enough samples
            if len(errors) > 10:
                self.residuals_pool[rg] = errors
            
        print(f"  -> Built SAA Pools for: {list(self.residuals_pool.keys())}")

    def _merge_data(self, forecast_df):
        merged = forecast_df.merge(self.catalog, on='product_id', how='left')
        
        # Defaults
        merged['price'] = merged['price'].fillna(10.0)
        merged['shelf_life'] = merged['shelf_life'].fillna(3.0)
        merged['unit_weight_kg'] = 1.0
        
        # Target Service Level (Crucial for SAA)
        if 'target_sl_theoretical' not in merged.columns:
             merged['target_sl_theoretical'] = 0.90
        merged['target_sl_theoretical'] = merged['target_sl_theoretical'].fillna(0.90)
        
        return merged

    def _load_supply_network(self):
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        
        suppliers_df = pd.read_csv(sup_path)
        sp_df = pd.read_csv(sp_path)
        
        return suppliers_df, sp_df

    def _assign_suppliers(self, summary, suppliers_df, sp_df):
        """Link stores to potential suppliers based on location/product"""
        print("  -> Assigning suppliers...")
        
        # Type safety
        summary['product_id'] = summary['product_id'].astype(str)
        sp_df['product_id'] = sp_df['product_id'].astype(str)
        sp_df['supplier_id'] = sp_df['supplier_id'].astype(int)
        
        # 1. Generate Store Locations (Deterministic)
        unique_stores = summary['store_id'].unique()
        store_locs = {}
        for sid in unique_stores:
            h = hash(sid)
            rng_local = np.random.default_rng(abs(h))
            d = float(rng_local.uniform(*Cfg.STORE_RADIUS_KM))
            b = float(rng_local.uniform(0, 360))
            lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, d, b)
            store_locs[sid] = (lat, lon)
            
        summary['store_lat'] = summary['store_id'].map(lambda s: store_locs.get(s, (0,0))[0])
        summary['store_lon'] = summary['store_id'].map(lambda s: store_locs.get(s, (0,0))[1])
        
        # 2. Time Windows
        # (Simple random assignment for now)
        tw_map = {}
        rng_tw = np.random.default_rng(42)
        for sid in unique_stores:
            start = Cfg.STORE_OPEN_WINDOW[0] + rng_tw.integers(0, 60)
            tw_map[sid] = (start, start + rng_tw.integers(120, 240))
            
        summary['tw_open'] = summary['store_id'].map(lambda s: tw_map[s][0])
        summary['tw_close'] = summary['store_id'].map(lambda s: tw_map[s][1])
        summary['service_time'] = Cfg.SERVICE_TIME_STORE_MINS
        
        # 3. Supplier Lookup
        sp_lookup = sp_df.groupby('product_id')['supplier_id'].apply(list).to_dict()
        sup_meta = suppliers_df.set_index('supplier_id').to_dict('index')
        
        assigned_list = []
        nearest_list = []
        
        for idx, row in summary.iterrows():
            s_lat, s_lon = row['store_lat'], row['store_lon']
            pid = row['product_id']
            
            candidates = sp_lookup.get(pid, [])
            if not candidates:
                assigned_list.append([])
                nearest_list.append(np.nan)
                continue
                
            # Score suppliers (Distance + Archetype)
            scored = []
            for sid in candidates:
                if sid not in sup_meta: continue
                s_info = sup_meta[sid]
                
                dist = GeoUtils.haversine_km(s_lat, s_lon, 
                                             s_info.get('lat', 0), s_info.get('lon', 0))
                
                # Bonus/Penalty logic
                arch = s_info.get('archetype', '')
                bonus = -10 if arch == 'local_specialty' else (10 if arch == 'farm_direct' else 0)
                
                scored.append((sid, dist + bonus))
            
            scored.sort(key=lambda x: x[1])
            top_ids = [x[0] for x in scored[:10]] # Keep top 10
            
            assigned_list.append(top_ids)
            nearest_list.append(top_ids[0] if top_ids else np.nan)
            
        summary['assigned_suppliers'] = assigned_list
        summary['nearest_supplier'] = nearest_list
        
        return summary

    def _calculate_policies_saa(self, summary, suppliers_df):
        """Calculate Order Qty using SAA"""
        print("  -> Calculating Data-Driven Policies (SAA)...")
        
        # Helper: Get Lead Time
        sup_lt_map = suppliers_df.set_index('supplier_id')['lead_time_mean_days'].to_dict()
        
        results = []
        for idx, row in summary.iterrows():
            mean_demand = row['predicted_mean']
            risk_group = row.get('risk_group', 'Normal')
            target_sl = row.get('target_sl_theoretical', 0.95)
            
            # SAA Logic
            errors = self.residuals_pool.get(risk_group, [])
            if len(errors) == 0:
                # Fallback to Global pool if specific risk group missing
                errors = self.residuals_pool.get('Normal', [])
            
            if len(errors) > 0:
                try:
                    # Quantile of (True - Pred)/Pred
                    # We want Coverage > Demand. 
                    # If error distribution captures under-forecasting (positive error), 
                    # quantile(0.95) gives the buffer needed.
                    safety_factor = np.quantile(errors, target_sl)
                except: safety_factor = 0.5
            else:
                # Fallback Normal approx
                safety_factor = norm.ppf(target_sl) * 0.5 
            
            # Bound safety factor (e.g., don't reduce forecast by > 50%)
            safety_factor = max(safety_factor, -0.5)
            
            # Quantities
            # Note: predicted_mean is DAILY. We need to cover Review Period + Lead Time?
            # For simplicity in this demo, we assume 'predicted_mean' is the Target Daily Rate
            # and we order for 1 cycle (Review Period) + Buffer.
            # Let's assume Review Period = 1 day (Daily ordering).
            
            # Safety Stock (Units)
            safety_stock = mean_demand * safety_factor
            safety_stock = max(0, safety_stock)
            
            # Order Qty (Base Stock Policy)
            # Order = Forecast + SS
            order_qty = mean_demand + safety_stock
            
            # Shelf Life Constraint
            max_sellable = mean_demand * row.get('shelf_life', 3.0)
            order_qty = min(order_qty, max_sellable)
            
            # ROP
            ns = row['nearest_supplier']
            lt = sup_lt_map.get(ns, 2.0) if pd.notna(ns) else 2.0
            rop = (mean_demand * lt) + safety_stock
            
            results.append({
                'safety_stock_units': float(np.ceil(safety_stock)),
                'order_qty_units': float(np.ceil(order_qty)),
                'rop_units': float(np.ceil(rop)),
                'safety_factor_saa': float(safety_factor)
            })
            
        res_df = pd.DataFrame(results, index=summary.index)
        summary = pd.concat([summary, res_df], axis=1)
        summary['order_qty_kg'] = summary['order_qty_units'] * summary.get('unit_weight_kg', 1.0)
        
        return summary

    def _save_artifacts(self, summary, suppliers_df, sp_df):
        print("  -> Saving artifacts...")
        summary['oos_rate'] = 1.0 - summary['target_sl_theoretical']
        
        # Prepare Unified Output
        # Ensure all required columns exist
        cols = [
            'store_id', 'product_id', 'store_lat', 'store_lon',
            'predicted_mean', 'unit_weight_kg', 'oos_rate',
            'category_id', 'shelf_life', 'order_qty_units', 'order_qty_kg',
            'rop_units', 'price', 'assigned_suppliers', 'nearest_supplier',
            'tw_open', 'tw_close', 'service_time'
        ]
        
        if 'forecast_method' in summary.columns:
            cols.append('forecast_method')
            
        for c in cols:
            if c not in summary.columns: summary[c] = 0
            
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        summary[cols].to_parquet(out_path, index=False)
        print(f"    Saved Unified Plan: {out_path}")
        
    def _run_diagnostics(self, summary):
        print("  -> Diagnostics: Policy Stats by Risk Group")
        grp = summary.groupby('risk_group')[['target_sl_theoretical', 'safety_factor_saa', 'order_qty_units']].mean()
        print(grp)
        
        # Check SMA vs ML Logic (if available)
        if 'forecast_method' in summary.columns:
            print("\n  -> Diagnostics: Policy Stats by Method")
            grp_met = summary.groupby('forecast_method')[['predicted_mean', 'safety_factor_saa', 'order_qty_units']].mean()
            print(grp_met)

if __name__ == "__main__":
    planner = InventoryPlanner()
    planner.run()