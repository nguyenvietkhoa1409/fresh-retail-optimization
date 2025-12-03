# src/data_pipeline/generator.py
import os
import numpy as np
import pandas as pd
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class SyntheticGenerator:
    """
    Class chịu trách nhiệm sinh dữ liệu giả lập logistic:
    - Products, Suppliers, Stores, Distances
    - Supplier-Product Matrix (giá, lead time, shelf life)
    - Vehicles
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(Cfg.SEED)
        os.makedirs(Cfg.ARTIFACTS_DIR, exist_ok=True)

    def generate_all(self, unique_store_ids):
        """Chạy toàn bộ quy trình sinh dữ liệu."""
        print("\n[Generator] Starting synthetic data generation...")
        
        # 1. Products
        self._gen_products()
        
        # 2. Suppliers
        suppliers_df = self._gen_suppliers()
        
        # 3. Stores (mapped to real IDs)
        self._gen_stores(unique_store_ids)
        
        # 4. Distance Matrix (Warehouse-Supplier)
        self._gen_dist_matrix(suppliers_df)
        
        # 5. Supplier-Product Matrix (Complex Logic)
        self._gen_supplier_product_matrix(suppliers_df)
        
        # 6. Vehicles
        self._gen_vehicles()
        
        print(f"[Generator] All artifacts saved to {Cfg.ARTIFACTS_DIR}")

    def _gen_products(self):
        df = pd.DataFrame(Cfg.PRODUCT_CATEGORIES,
                          columns=["product_id", "category_id", "name", "holding_cost_per_kg_day", "volume_m3_per_kg"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "products.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated products.csv")

    def _gen_suppliers(self):
        suppliers = []
        sid = 1
        # Sinh supplier theo các vòng tròn bán kính khác nhau (Supplier Rings)
        for count, rmin, rmax in Cfg.SUPPLIER_RINGS:
            for _ in range(count):
                d = float(self.rng.uniform(rmin, rmax))
                b = float(self.rng.uniform(0, 360))
                lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, d, b)
                suppliers.append((sid, round(lat, 6), round(lon, 6)))
                sid += 1
        
        # Cắt bớt nếu vượt quá cấu hình tối đa
        suppliers = suppliers[:Cfg.N_SUPPLIERS]
        df = pd.DataFrame(suppliers, columns=["supplier_id", "lat", "lon"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated suppliers.csv ({len(df)} rows)")
        return df

    def _gen_stores(self, store_ids):
        # Lấy tối đa 20 store duy nhất từ dataset thật để simulate logistics cho nhẹ
        unique = store_ids[:20] 
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

    def _gen_supplier_product_matrix(self, suppliers_df):
        """
        Logic phức tạp: Phân chia supplier thành các pool cho từng category,
        gán giá và thời gian shelf-life đã trôi qua (elapsed days).
        """
        sup_ids = suppliers_df["supplier_id"].tolist()
        self.rng.shuffle(sup_ids)
        
        k = Cfg.SUPPLIERS_PER_CAT_MAX
        
        # Tạo 2 pool supplier overlap nhau
        pool_c1 = sorted(sup_ids[:min(k, len(sup_ids))])
        start_c2 = max(0, len(sup_ids)//4)
        pool_c2 = sorted(sup_ids[start_c2 : start_c2 + min(k, len(sup_ids))])
        
        sp_rows = []
        # Duyệt qua từng sản phẩm định nghĩa trong Config
        df_prods = pd.DataFrame(Cfg.PRODUCT_CATEGORIES, 
                                columns=["product_id", "category_id", "name", "h", "v"])
        
        for r in df_prods.itertuples(index=False):
            prod_id = int(r.product_id)
            cat_id = int(r.category_id)
            
            # Lấy range giá và range elapsed days
            lo, hi = Cfg.PRICE_RANGE_BY_PRODUCT.get(prod_id, (1.0, 10.0))
            el_lo, el_hi = Cfg.ELAPSED_RANGE_BY_CAT[cat_id]
            
            # Chọn pool dựa trên category
            pool = pool_c1 if cat_id == 1 else pool_c2
            
            # Chọn ngẫu nhiên số lượng supplier cung cấp sp này (từ 3 đến 6)
            num_sup = int(self.rng.integers(3, min(6, len(pool)) + 1))
            chosen = self.rng.choice(pool, size=num_sup, replace=False)
            
            for sid in sorted(chosen):
                price = float(np.round(self.rng.uniform(lo, hi), 2))
                el = int(self.rng.integers(el_lo, el_hi + 1))
                sp_rows.append((int(sid), prod_id, price, el))
        
        df_sp = pd.DataFrame(sp_rows, columns=["supplier_id", "product_id", "unit_price", "elapsed_shelf_days"])
        df_sp = df_sp.drop_duplicates(subset=["supplier_id", "product_id"])
        df_sp = df_sp.sort_values(["product_id", "supplier_id"]).reset_index(drop=True)
        
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        df_sp.to_csv(out_path, index=False)
        print(f"  -> Generated supplier_product.csv ({len(df_sp)} rows)")

    def _gen_vehicles(self):
        df = pd.DataFrame(Cfg.VEHICLES, columns=["type", "capacity_kg", "var_cost_per_km", "fixed_cost", "cost_per_hour"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "vehicles.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated vehicles.csv")