import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class InventoryPlanner:
    """
    Class thực hiện Inventory Planning.
    Updated: 
    1. Downscaled context generation for Integrated VRP feasibility.
    2. Added Time Window generation for Stores and Suppliers.
    3. Added Capacity Differentiation (Wholesaler vs Farm).
    4. Fixed NaN handling for integer conversions.
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_DIAGNOSTICS, exist_ok=True)
        self.rng = np.random.default_rng(Cfg.SEED)

    def run(self):
        print("\n[Inventory] Starting Inventory Planning Pipeline...")
        
        # 1. Load Data
        summary = self._load_and_aggregate()
        
        # 2. Enrich Product Info (+ Time Windows for Stores)
        summary = self._enrich_product_info(summary)
        
        # 3. Rescale Demand & Floor Std
        summary = self._rescale_demand_stats(summary)
        
        # 4. Generate Suppliers (+ Time Windows + Capacity Differentiation)
        suppliers_df, sp_df, summary = self._setup_suppliers_and_matrix(summary)
        
        # 5. Calculate Policies (Newsvendor, ROP, SS)
        summary, suppliers_df, sp_df = self._calculate_policies_and_scale(summary, suppliers_df, sp_df)
        
        # 6. Save Artifacts
        self._save_artifacts(summary, suppliers_df, sp_df)
        
        # 7. Diagnostics
        self._run_diagnostics(summary, suppliers_df)

    def _load_and_aggregate(self):
        print("   -> Loading reconstructed data...")
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
        
        # --- [MODIFIED & FIXED LOGIC] ---
        # 1. Lọc lấy 40 Store có volume cao nhất
        TARGET_STORE_COUNT = 40  
        store_vol = summary.groupby('store_id')['predicted_mean'].sum().sort_values(ascending=False)
        top_stores = store_vol.head(TARGET_STORE_COUNT).index.tolist()
        
        # 2. Apply Filter
        summary = summary[summary['store_id'].isin(top_stores)].copy()
        
        # 3. Apply Limit Pairs
        summary = summary.sort_values('predicted_mean', ascending=False).head(Cfg.PAIR_LIMIT)
        
        # [QUAN TRỌNG NHẤT] Reset Index để tránh lỗi NaN khi tính toán sau này
        summary.reset_index(drop=True, inplace=True) 
        
        # [TYPE ENFORCEMENT]
        summary['store_id'] = summary['store_id'].astype(str)
        summary['product_id'] = summary['product_id'].astype(int)
        
        print(f"     [FILTER] Kept {len(top_stores)} unique Stores (Target: {TARGET_STORE_COUNT})")
        print(f"     [FILTER] Total Pairs kept: {len(summary)}")
        # --------------------------------
        
        return summary

    def _enrich_product_info(self, summary):
        n = len(summary)
        summary['category_id'] = self.rng.choice([1, 2], size=n, p=[0.5, 0.5])
        summary['sim_product_id'] = summary.apply(lambda r: self.rng.choice([101, 102]) if r['category_id']==1 else self.rng.choice([201, 202]), axis=1)
        summary['shelf_life'] = summary['category_id'].map(Cfg.SHELF_LIFE_BY_CAT).astype(float)
        
        summary['price'] = summary['sim_product_id'].map(lambda p: self.rng.uniform(*Cfg.PRICE_RANGE_BY_PRODUCT[p]))
        
        # Daily Holding Cost
        base_annual_rate = 0.20
        summary['daily_holding_cost_unit'] = summary['price'] * (base_annual_rate / 365.0) * Cfg.HOLDING_COST_MULTIPLIER

        summary['demand_min'] = summary['sim_product_id'].map(lambda p: Cfg.DEMAND_RANGE_BY_PRODUCT[p][0])
        summary['demand_max'] = summary['sim_product_id'].map(lambda p: Cfg.DEMAND_RANGE_BY_PRODUCT[p][1])
        summary['unit_weight_kg'] = summary['sim_product_id'].map(lambda p: Cfg.UNIT_WEIGHT.get(p, 1.0))
        
        unique_stores = summary['store_id'].unique()
        
        # Generate Locations
        store_locs = {sid: GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, float(self.rng.uniform(*Cfg.STORE_RADIUS_KM)), float(self.rng.uniform(0, 360))) for sid in unique_stores}
        summary['store_lat'] = summary['store_id'].map(lambda s: store_locs[s][0])
        summary['store_lon'] = summary['store_id'].map(lambda s: store_locs[s][1])
        
        # [NEW] Generate Time Windows for STORES (Outbound)
        def gen_store_tw():
            start_base = Cfg.STORE_OPEN_WINDOW[0] # 360 (6 AM)
            start_noise = self.rng.integers(0, 60) # + 0-60 mins
            start = start_base + start_noise
            duration = self.rng.integers(120, 240) # 2-4 hours receiving window
            return start, start + duration

        tw_map = {sid: gen_store_tw() for sid in unique_stores}
        summary['tw_open'] = summary['store_id'].map(lambda s: tw_map[s][0])
        summary['tw_close'] = summary['store_id'].map(lambda s: tw_map[s][1])
        summary['service_time'] = Cfg.SERVICE_TIME_STORE_MINS # Fixed service time
        
        return summary

    def _rescale_demand_stats(self, summary):
        # [FIXED LOGIC] Vectorized Rescaling
        grp_stats = summary.groupby('sim_product_id')['predicted_mean'].agg(['min', 'max']).rename(columns={'min': 'curr_min', 'max': 'curr_max'})
        summary = summary.merge(grp_stats, on='sim_product_id', how='left')
        
        numerator = (summary['predicted_mean'] - summary['curr_min']) * (summary['demand_max'] - summary['demand_min'])
        denominator = (summary['curr_max'] - summary['curr_min'])
        
        mask_flat = denominator < 1e-9
        
        summary.loc[~mask_flat, 'predicted_mean'] = summary['demand_min'] + (numerator / denominator)
        summary.loc[mask_flat, 'predicted_mean'] = (summary['demand_min'] + summary['demand_max']) / 2.0
        
        summary.drop(columns=['curr_min', 'curr_max'], inplace=True)
        
        summary['predicted_std'] = summary['predicted_std'].astype(float).fillna(0.0)
        summary['predicted_mean'] = summary['predicted_mean'].astype(float).fillna(0.0)
        
        floor_std = Cfg.MIN_STD_FRACTION * summary['predicted_mean']
        summary['predicted_std'] = np.maximum(summary['predicted_std'], floor_std)
        
        return summary

    def _setup_suppliers_and_matrix(self, summary):
        print("   -> Generating Suppliers with Differential Capacity...")
        suppliers = []
        for cat in [1, 2]: 
            counter = 0
            for count, rmin, rmax, price_factor, fresh_loss, label in Cfg.SUPPLIER_ZONES:
                for _ in range(count):
                    sid = cat * 100000 + counter; counter += 1
                    lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, float(self.rng.uniform(rmin, rmax)), float(self.rng.uniform(0, 360)))
                    
                    # [NEW] Generate Time Windows for SUPPLIERS (Inbound)
                    tw_start = Cfg.SUPPLIER_OPEN_WINDOW[0] + self.rng.integers(0, 60)
                    tw_end = Cfg.SUPPLIER_OPEN_WINDOW[1] - self.rng.integers(0, 60)
                    
                    dist_km = GeoUtils.haversine_km(Cfg.CENTER_LAT, Cfg.CENTER_LON, lat, lon)
                    avg_speed = getattr(Cfg, 'SPEED_KMPH', 35.0)
                    driving_hours_per_day = getattr(Cfg, 'DRIVING_HOURS_PER_DAY', 10.0)
                    transport_hours = dist_km / max(1e-6, avg_speed)
                    transport_days = transport_hours / driving_hours_per_day
                    req_procurement_days = max(1.0, float(np.ceil(transport_days)))

                    
                    # [CRITICAL UPDATE] Capacity Differentiation
                    # Wholesaler (Zone 1): Tiny Capacity (Force overflow to farms for large batches)
                    # Farm (Zone 3): Huge Capacity
                    if "Wholesaler" in label:
                        cap_base = self.rng.uniform(200.0, 500.0) # Very Small
                    elif "Farm" in label:
                        cap_base = self.rng.uniform(10000.0, 20000.0) # Huge
                    else:
                        cap_base = self.rng.uniform(1000.0, 3000.0) # Mid
                    
                    suppliers.append({
                        'supplier_id': int(sid), 'category_id': int(cat), 'sup_lat': lat, 'sup_lon': lon,
                        'zone_label': label, 'distance_km': dist_km,
                        'price_factor': price_factor, 'freshness_loss_days': fresh_loss,
                        'min_procurement_days': max(1.0, req_procurement_days),
                        'capacity_kg': cap_base, # Use differentiated capacity
                        'min_order_qty_units_supplier': self.rng.integers(*Cfg.MOQ_RANGE_UNITS),
                        'lead_time_mean_days': max(1.0, req_procurement_days),
                        'lead_time_std_days': 0.5 if label == 'Farm_Far' else 0.2,
                        'tw_open': tw_start,
                        'tw_close': tw_end,
                        'service_time': Cfg.SERVICE_TIME_SUPPLIER_MINS,
                        'reliability': float(self.rng.uniform(0.80, 0.99))
                    })
        
        suppliers_df = pd.DataFrame(suppliers)
        suppliers_df['on_time_rate'] = (suppliers_df['reliability'] - 0.02).clip(0.5, 0.995)
        
        def assign_top_k(row, k=5):
            cat_sup = suppliers_df[suppliers_df['category_id'] == row['category_id']]
            if cat_sup.empty: return []
            dists = cat_sup.apply(lambda s: GeoUtils.haversine_km(row['store_lat'], row['store_lon'], s['sup_lat'], s['sup_lon']), axis=1)
            # Relax to top 10 so Farms (far away) are considered candidates
            return cat_sup.loc[dists.nsmallest(10).index, 'supplier_id'].tolist()
        
        summary['assigned_suppliers'] = summary.apply(assign_top_k, axis=1)
        summary['nearest_supplier'] = summary['assigned_suppliers'].apply(lambda x: x[0] if len(x)>0 else np.nan)
        
        sp_rows = []
        for _, s in suppliers_df.iterrows():
            pids = [101, 102] if s['category_id'] == 1 else [201, 202]
            for pid in pids:
                base_p = np.random.uniform(*Cfg.PRICE_RANGE_BY_PRODUCT[pid])
                sp_rows.append({
                    'supplier_id': int(s['supplier_id']), 'product_id': pid,
                    'min_order_qty_units': max(1, int(s['min_order_qty_units_supplier'])),
                    'unit_price': float(base_p * s['price_factor']),
                    'elapsed_shelf_days': s['freshness_loss_days'],
                    'lead_time_mean_days': s['lead_time_mean_days'], 'lead_time_std_days': s['lead_time_std_days'],
                    'on_time_rate': s['on_time_rate'], 'supplier_capacity_kg': s['capacity_kg']
                })
        return suppliers_df, pd.DataFrame(sp_rows), summary

    def _calculate_policies_and_scale(self, summary, suppliers_df, sp_df):
        print("   -> Calculating Inventory Policies (ROP, SS)...")
        def tau_from_oos(r):
            for lo, hi, tau in Cfg.TAU_BY_OOS:
                if lo <= r < hi: return tau
            return 0.75
        
        # 1. Tính Z-score
        summary['tau_i'] = summary['oos_rate'].apply(tau_from_oos)
        summary['z'] = summary['tau_i'].apply(norm.ppf)
        
        # 2. Xử lý các giá trị NaN tiềm ẩn ở các cột đầu vào
        summary['predicted_mean'] = summary['predicted_mean'].fillna(0.0)
        summary['predicted_std'] = summary['predicted_std'].fillna(0.0)
        summary['shelf_life'] = summary['shelf_life'].fillna(5.0)
        summary['z'] = summary['z'].fillna(1.65) # Fallback z=1.65 (~95%) nếu lỗi
        
        # 3. Tính Order Quantity (Q)
        summary['order_qty_raw'] = summary['predicted_mean'] + summary['z'] * summary['predicted_std']
        
        summary['order_qty_capped'] = np.minimum(
            summary['order_qty_raw'], 
            summary['shelf_life'] * summary['predicted_mean']
        ).clip(lower=0.0)
        
        # [FIX 1] Fill NaN trước khi ép kiểu int cho Order Qty
        summary['order_qty_units'] = np.ceil(summary['order_qty_capped'].fillna(0)).astype(int)
        summary['order_qty_kg'] = summary['order_qty_units'] * summary['unit_weight_kg']
        
        # 4. Tính Safety Stock (SS)
        ss_val = np.maximum(0.0, summary['z'] * summary['predicted_std'])
        # [FIX 2] Fill NaN trước khi ép kiểu int cho SS (Đây là chỗ gây lỗi cũ của bạn)
        summary['safety_stock_units'] = np.ceil(ss_val.fillna(0)).astype(int)
        summary['safety_stock_kg'] = summary['safety_stock_units'] * summary['unit_weight_kg']

        # 5. Scale Supply Logic
        # Do not uniformly scale up capacity. We want to preserve the "Scarcity" of wholesalers.
        dem = summary['order_qty_kg'].sum(); sup = suppliers_df['capacity_kg'].sum()
        # Scale only if total supply is absolutely less than demand
        if sup < dem * Cfg.TARGET_SUPPLY_DEMAND_RATIO:
             scale = (dem * Cfg.TARGET_SUPPLY_DEMAND_RATIO) / max(sup, 1.0)
             suppliers_df['capacity_kg'] *= scale
             
        sp_df['supplier_capacity_kg'] = sp_df['supplier_id'].map(suppliers_df.set_index('supplier_id')['capacity_kg'].to_dict())
        
        # 6. Tính Reorder Point (ROP)
        lt_mean = suppliers_df.set_index('supplier_id')['lead_time_mean_days'].to_dict()
        lt_std = suppliers_df.set_index('supplier_id')['lead_time_std_days'].to_dict()
        
        summary['nearest_lead_mean'] = summary['nearest_supplier'].map(lt_mean).fillna(2.0)
        summary['nearest_lead_std'] = summary['nearest_supplier'].map(lt_std).fillna(0.5)
        
        mean = summary['predicted_mean']
        std = summary['predicted_std']
        L = summary['nearest_lead_mean']
        sigma_L = summary['nearest_lead_std']
        
        variance_term = L * (std**2) + (mean**2) * (sigma_L**2)
        rop_val = (mean * L + summary['z'] * np.sqrt(variance_term)).clip(lower=0)
        
        summary['rop_units'] = np.ceil(rop_val.fillna(0)).astype(int)
        summary['rop_kg'] = summary['rop_units'] * summary['unit_weight_kg']
        
        return summary, suppliers_df, sp_df

    def _save_artifacts(self, summary, suppliers, sp):
        print("   -> Saving Inventory Artifacts (with TW)...")
        # [MODIFIED] Include Time Window columns in base parquet
        base = ['store_id','product_id','sim_product_id','store_lat','store_lon','predicted_mean',
                'predicted_std','unit_weight_kg','oos_rate','n_days',
                'tw_open', 'tw_close', 'service_time'] # <--- Added these
        
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
            f.write(f"Total Stores: {summary['store_id'].nunique()}\n")
            f.write(f"Total Suppliers: {len(suppliers_df)}\n")
        print(f"     Diag saved to diagnostics_summary_v2.txt (Stores: {summary['store_id'].nunique()}, Suppliers: {len(suppliers_df)})")