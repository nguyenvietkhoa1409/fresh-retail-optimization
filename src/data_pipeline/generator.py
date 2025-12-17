# src/data_pipeline/generator.py
import os
import numpy as np
import pandas as pd
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class SyntheticGenerator:
    """
    Class chịu trách nhiệm sinh dữ liệu giả lập logistic.
    Updated: 
      - Integated Supplier Zones for competitive Capacity/MOQ logic.
      - Fixed Supply/Demand ratio to be realistic (~1.5x).
      - Robust config handling.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(Cfg.SEED)
        os.makedirs(Cfg.ARTIFACTS_DIR, exist_ok=True)

    def generate_all(self, unique_store_ids):
        """Chạy toàn bộ quy trình sinh dữ liệu."""
        print("\n[Generator] Starting synthetic data generation...")
        demand_profile = self._get_demand_summary()
        self._gen_products()
        suppliers_df = self._gen_suppliers()
        self._gen_stores(unique_store_ids)
        self._gen_dist_matrix(suppliers_df)
        #truyền demand_profile để tính Supplier Capaticy tương ứng (1,2-1,5x Demand)
        self._gen_supplier_product_matrix(suppliers_df, demand_profile)
        self._gen_vehicles()
        
        print(f"[Generator] All artifacts saved to {Cfg.ARTIFACTS_DIR}")
    
    def _get_demand_summary(self):
        """
        Đọc file demand output từ Step 2 để định cỡ Capacity cho Supply Chain.
        Logic:
        1. Đọc file parquet từ Step 2.
        2. Tìm cột 'D_recon' (Nhu cầu thực tế đã khôi phục).
        3. Tính tổng nhu cầu toàn hệ thống.
        4. Phân bổ về các Product giả lập (101, 102...).
        """
        # Đường dẫn khớp với output của reconstruction.py
        demand_path = os.path.join(Cfg.OUT_DIR_PART2, "part2_reconstructed.parquet")
        
        # Profile mặc định phòng khi chưa chạy Step 2 (Fallback an toàn)
        default_profile = {101: 5000.0, 102: 4000.0, 201: 8000.0, 202: 6000.0}

        if not os.path.exists(demand_path):
            print(f"[WARN] Demand file not found at {demand_path}. Using default estimates.")
            return default_profile

        print(f" -> Calibrating Supply based on Demand file: {demand_path}")
        
        try:
            df = pd.read_parquet(demand_path)
            
            # --- KIỂM TRA LOGIC CỘT ---
            # reconstruction.py dòng 296 lưu cột là "D_recon"
            target_col = "D_recon"
            
            if target_col not in df.columns:
                print(f"   [CRITICAL WARN] Column '{target_col}' not found! Checking alternatives...")
                if "daily_demand" in df.columns: target_col = "daily_demand"
                elif "y16" in df.columns: 
                    # Rất nguy hiểm, nhưng đành dùng tạm nếu không có recon
                    print("   [WARN] Using raw 'y16' sum (Censored Demand). Supply might be under-sized.")
                    df['D_recon_proxy'] = df['y16'].apply(lambda x: np.sum(x) if isinstance(x, (list, np.ndarray)) else 0)
                    target_col = 'D_recon_proxy'
                else:
                    return default_profile
            
            # --- TÍNH TOÁN TỔNG NHU CẦU ---
            # Đảm bảo không có số âm hoặc NaN
            valid_demand = df[target_col].fillna(0).clip(lower=0)
            
            # Tổng demand toàn hệ thống (đơn vị: giống đơn vị của y16, thường là kg hoặc units)
            total_system_demand = valid_demand.sum()
            
            if total_system_demand == 0:
                print("   [WARN] Total demand is 0. Using defaults.")
                return default_profile

            print(f"   -> Detected System-wide True Demand: {total_system_demand:,.2f}")

            # --- PHÂN BỔ VỀ PRODUCT GIẢ LẬP ---
            # Vì Step 2 dùng dữ liệu thật (Product ID thật), còn Step 1 đang sinh Product ID giả (101, 202...)
            # Ta dùng tỷ lệ heuristic để chia tổng bánh này.
            
            # Tỷ lệ giả định: 
            # - 101, 102 (Fresh): Chiếm 40% tổng lượng
            # - 201, 202 (Ambient): Chiếm 60% tổng lượng (thường bán nhiều hơn)
            profile = {
                101: total_system_demand * 0.20, # 20%
                102: total_system_demand * 0.20, # 20%
                201: total_system_demand * 0.30, # 30%
                202: total_system_demand * 0.30  # 30%
            }
            
            return profile

        except Exception as e:
            print(f"   [ERROR] Failed to read demand: {e}. Using defaults.")
            return default_profile

    def _gen_products(self):
        df = pd.DataFrame(Cfg.PRODUCT_CATEGORIES,
                          columns=["product_id", "category_id", "name", "holding_cost_per_kg_day", "volume_m3_per_kg"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "products.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated products.csv")

    def _gen_suppliers(self):
        suppliers = []
        sid = 1
        
        # 1. Robust Config Check
        if hasattr(Cfg, 'SUPPLIER_ZONES'):
            config_source = Cfg.SUPPLIER_ZONES
        elif hasattr(Cfg, 'SUPPLIER_RINGS'):
            config_source = Cfg.SUPPLIER_RINGS
        else:
            config_source = [(5, 2, 10), (10, 10, 40), (5, 40, 120)] # Fallback

        for item in config_source:
            # Handle tuples of varying lengths (3 or 6) based on Config
            # Item structure: (count, rmin, rmax, price_mult, fresh_loss, label)
            if len(item) >= 3:
                count, rmin, rmax = item[0], item[1], item[2]
                # [FIX] Capture Zone Label for downstream logic
                zone_label = item[5] if len(item) > 5 else "Generic"
            else:
                continue

            for _ in range(count):
                d = float(self.rng.uniform(rmin, rmax))
                b = float(self.rng.uniform(0, 360))
                lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, d, b)
                
                # [FIX] Append zone_label
                suppliers.append((sid, round(lat, 6), round(lon, 6), zone_label))
                sid += 1
        
        # 2. Safety Cap (Only if N_SUPPLIERS exists)
        limit_n = getattr(Cfg, 'N_SUPPLIERS', None)
        if limit_n is not None:
            suppliers = suppliers[:limit_n]
        
        # [FIX] DataFrame now includes 'zone_label'
        df = pd.DataFrame(suppliers, columns=["supplier_id", "lat", "lon", "zone_label"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated suppliers.csv ({len(df)} rows) with Zones")
        return df

    def _gen_stores(self, store_ids):
        # [FIX] Use GLOBAL_NUM_STORES from Config to ensure consistency
        limit_stores = getattr(Cfg, 'GLOBAL_NUM_STORES', 20)
        unique = store_ids[:limit_stores] 
        
        stores = []
        for rid, store_id in enumerate(unique, 1):
            d = float(self.rng.uniform(*Cfg.STORE_RADIUS_KM))
            b = float(self.rng.uniform(0, 360))
            lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, d, b)
            stores.append((rid, store_id, round(lat, 6), round(lon, 6)))
        
        df = pd.DataFrame(stores, columns=["store_rid", "store_id", "lat", "lon"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "stores.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated stores.csv ({len(df)} rows)")

    def _gen_dist_matrix(self, suppliers_df):
        dist_ws = []
        for _, row in suppliers_df.iterrows():
            km = GeoUtils.haversine_km(Cfg.CENTER_LAT, Cfg.CENTER_LON, row["lat"], row["lon"])
            time_min = int(round((km / Cfg.SPEED_KMPH) * 60))
            dist_ws.append((row["supplier_id"], km, time_min))
        
        df = pd.DataFrame(dist_ws, columns=["supplier_id", "dist_km", "time_min"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "dist_ws.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated dist_ws.csv")

    def _gen_supplier_product_matrix(self, suppliers_df, demand_profile):
        sup_ids = suppliers_df["supplier_id"].tolist()
        zone_map = suppliers_df.set_index("supplier_id")["zone_label"].to_dict()
        
        if not sup_ids: return
        self.rng.shuffle(sup_ids)
        k = getattr(Cfg, 'SUPPLIERS_PER_CAT_MAX', 6)
        
        pool_c1 = sorted(sup_ids[:min(k, len(sup_ids))])
        start_c2 = max(0, len(sup_ids)//4)
        pool_c2 = sorted(sup_ids[start_c2 : start_c2 + min(k, len(sup_ids))])
        
        sp_rows = []
        for p in Cfg.PRODUCT_CATEGORIES:
            prod_id = int(p[0])
            cat_id = int(p[1])
            lo, hi = Cfg.PRICE_RANGE_BY_PRODUCT.get(prod_id, (1.0, 10.0))
            elapsed_ranges = getattr(Cfg, 'ELAPSED_RANGE_BY_CAT', {1: (0,2), 2: (0,4)})
            el_lo, el_hi = elapsed_ranges.get(cat_id, (0, 2))
            
            pool = pool_c1 if cat_id == 1 else pool_c2
            if not pool: pool = sup_ids
            
            num_sup = int(self.rng.integers(3, min(6, len(pool)) + 1))
            num_sup = min(num_sup, len(pool))
            chosen = self.rng.choice(pool, size=num_sup, replace=False)
            
            # --- [LOGIC MỚI: Capacity based on Demand] ---
            # 1. Lấy Demand ước tính cho sản phẩm này
            est_total_demand = demand_profile.get(prod_id, 5000.0)
            
            # 2. Tính Target Supply System-wide (1.2x - 1.5x)
            target_system_supply = est_total_demand * self.rng.uniform(1.3, 1.6) # Tăng nhẹ lên 1.3-1.6 cho an toàn
            
            # 3. Capacity trung bình cần thiết cho mỗi Supplier
            avg_cap_needed = target_system_supply / max(1, len(chosen))
            
            for sid in sorted(chosen):
                zone = zone_map.get(sid, "Generic")
                
                # Base Capacity dựa trên nhu cầu thực tế
                base_cap = avg_cap_needed * self.rng.uniform(0.8, 1.2)
                
                # Điều chỉnh theo Zone
                if "Farm" in zone:
                    price_mult, fresh_add = 0.8, 0
                    cap_kg = base_cap * 1.5 # Farm to hơn
                    moq = self.rng.integers(10, 30) # [KEEP LOW MOQ]
                elif "Distributor" in zone:
                    price_mult, fresh_add = 1.0, 1
                    cap_kg = base_cap * 1.0 # Chuẩn
                    moq = self.rng.integers(5, 15) # [KEEP LOW MOQ]
                else: # Wholesaler
                    price_mult, fresh_add = 1.3, 2
                    cap_kg = base_cap * 0.5 # Nhỏ hơn
                    moq = 1
                
                # Đảm bảo mức tối thiểu
                cap_kg = max(cap_kg, 1000.0)

                base_price = self.rng.uniform(lo, hi)
                final_price = float(np.round(base_price * price_mult, 2))
                base_el = self.rng.integers(el_lo, el_hi + 1)
                final_el = min(base_el + fresh_add, 5) 
                
                sp_rows.append((int(sid), prod_id, final_price, final_el, round(cap_kg, 1), int(moq)))
        
        df_sp = pd.DataFrame(sp_rows, columns=["supplier_id", "product_id", "unit_price", "elapsed_shelf_days", "supplier_capacity_kg", "min_order_qty_units"])
        df_sp = df_sp.drop_duplicates(subset=["supplier_id", "product_id"])
        df_sp = df_sp.sort_values(["product_id", "supplier_id"]).reset_index(drop=True)
        
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        df_sp.to_csv(out_path, index=False)
        print(f"  -> Generated supplier_product.csv ({len(df_sp)} rows) - Demand Driven Capacity")

    def _gen_vehicles(self):
        # Robust check for VEHICLES vs VEHICLE_FLEET_DEFINITIONS
        rows = []
        
        # 1. Priority: Check for new detailed Fleet Defs (Strategy B)
        if hasattr(Cfg, 'VEHICLE_FLEET_DEFINITIONS'):
            vehs = Cfg.VEHICLE_FLEET_DEFINITIONS
            for v in vehs:
                rows.append({
                    "type": v.get('type', 'Truck'),
                    "capacity_kg": v.get('capacity', 1000),
                    "var_cost_per_km": v.get('cost_km', 1.0),
                    "fixed_cost": v.get('fixed_cost', 100),
                    "cost_per_hour": 0.0 
                })
                
        # 2. Fallback: Legacy VEHICLES tuple list
        elif hasattr(Cfg, 'VEHICLES'):
            vehs = Cfg.VEHICLES
            for v in vehs:
                if isinstance(v, tuple) and len(v) >= 4:
                    rows.append({
                        "type": v[0],
                        "capacity_kg": v[1],
                        "var_cost_per_km": v[2],
                        "fixed_cost": v[3],
                        "cost_per_hour": v[4] if len(v)>4 else 0.0
                    })
        
        # 3. Ultimate Fallback
        if not rows:
             rows = [{"type": "Default", "capacity_kg": 1000, "var_cost_per_km": 1.0, "fixed_cost": 100.0, "cost_per_hour": 0.0}]
             
        df = pd.DataFrame(rows)
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "vehicles.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated vehicles.csv")