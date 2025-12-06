# src/data_pipeline/generator.py
import os
import numpy as np
import pandas as pd
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class SyntheticGenerator:
    """
    Class chịu trách nhiệm sinh dữ liệu giả lập logistic.
    Updated: Robust config handling for Vehicles and Suppliers.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(Cfg.SEED)
        os.makedirs(Cfg.ARTIFACTS_DIR, exist_ok=True)

    def generate_all(self, unique_store_ids):
        """Chạy toàn bộ quy trình sinh dữ liệu."""
        print("\n[Generator] Starting synthetic data generation...")
        
        self._gen_products()
        suppliers_df = self._gen_suppliers()
        self._gen_stores(unique_store_ids)
        self._gen_dist_matrix(suppliers_df)
        self._gen_supplier_product_matrix(suppliers_df)
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
        
        # 1. Robust Config Check
        if hasattr(Cfg, 'SUPPLIER_ZONES'):
            config_source = Cfg.SUPPLIER_ZONES
        elif hasattr(Cfg, 'SUPPLIER_RINGS'):
            config_source = Cfg.SUPPLIER_RINGS
        else:
            config_source = [(5, 2, 10), (10, 10, 40), (5, 40, 120)] # Fallback

        for item in config_source:
            # Handle tuples of varying lengths (3 or 6)
            if len(item) >= 3:
                count, rmin, rmax = item[0], item[1], item[2]
            else:
                continue

            for _ in range(count):
                d = float(self.rng.uniform(rmin, rmax))
                b = float(self.rng.uniform(0, 360))
                lat, lon = GeoUtils.dest_from(Cfg.CENTER_LAT, Cfg.CENTER_LON, d, b)
                suppliers.append((sid, round(lat, 6), round(lon, 6)))
                sid += 1
        
        # 2. Safety Cap (Only if N_SUPPLIERS exists)
        limit_n = getattr(Cfg, 'N_SUPPLIERS', None)
        if limit_n is not None:
            suppliers = suppliers[:limit_n]
        
        df = pd.DataFrame(suppliers, columns=["supplier_id", "lat", "lon"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated suppliers.csv ({len(df)} rows)")
        return df

    def _gen_stores(self, store_ids):
        limit_stores = 20 
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

    def _gen_supplier_product_matrix(self, suppliers_df):
        sup_ids = suppliers_df["supplier_id"].tolist()
        if not sup_ids: return
        self.rng.shuffle(sup_ids)
        
        # Use getattr for safety
        k = getattr(Cfg, 'SUPPLIERS_PER_CAT_MAX', 6)
        
        pool_c1 = sorted(sup_ids[:min(k, len(sup_ids))])
        start_c2 = max(0, len(sup_ids)//4)
        pool_c2 = sorted(sup_ids[start_c2 : start_c2 + min(k, len(sup_ids))])
        
        sp_rows = []
        for p in Cfg.PRODUCT_CATEGORIES:
            prod_id = int(p[0])
            cat_id = int(p[1])
            
            lo, hi = Cfg.PRICE_RANGE_BY_PRODUCT.get(prod_id, (1.0, 10.0))
            
            # Fix: Access ELAPSED_RANGE safely
            elapsed_ranges = getattr(Cfg, 'ELAPSED_RANGE_BY_CAT', {1: (0,2), 2: (0,4)})
            el_lo, el_hi = elapsed_ranges.get(cat_id, (0, 2))
            
            pool = pool_c1 if cat_id == 1 else pool_c2
            if not pool: pool = sup_ids
            
            num_sup = int(self.rng.integers(3, min(6, len(pool)) + 1))
            num_sup = min(num_sup, len(pool))
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
        # --- FIX: Robust check for VEHICLES vs VEHICLE_FLEET_DEFINITIONS ---
        rows = []
        
        # 1. Priority: Check for new detailed Fleet Defs (Strategy B)
        if hasattr(Cfg, 'VEHICLE_FLEET_DEFINITIONS'):
            vehs = Cfg.VEHICLE_FLEET_DEFINITIONS
            for v in vehs:
                # Map dict to CSV columns
                rows.append({
                    "type": v.get('type', 'Truck'),
                    "capacity_kg": v.get('capacity', 1000),
                    "var_cost_per_km": v.get('cost_km', 1.0),
                    "fixed_cost": v.get('fixed_cost', 100),
                    "cost_per_hour": 0.0 # Not used in VRP currently
                })
                
        # 2. Fallback: Legacy VEHICLES tuple list
        elif hasattr(Cfg, 'VEHICLES'):
            vehs = Cfg.VEHICLES
            for v in vehs:
                # tuple: (Type, Cap, CostKM, Fixed, CostHour)
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