# src/data_pipeline/generator.py
"""
ENHANCED SUPPLIER GENERATOR (Data-Driven Compatible)
Creates rich trade-off structure for P/U sensitivity using Catalog Data.
"""

import os
import numpy as np
import pandas as pd
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class EnhancedSupplierGenerator:
    """
    Generates heterogeneous supplier network with strategic attributes.
    UPDATES (v4):
    - Reads product list from 'master_product_catalog.parquet' instead of Config.
    - Ensures alignment with Enrichment module.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(Cfg.SEED)
        os.makedirs(Cfg.ARTIFACTS_DIR, exist_ok=True)
        
        # Define supplier archetypes (Keep existing logic)
        self.SUPPLIER_ARCHETYPES = {
            'local_specialty': {
                'distance_km': (5, 30),
                'base_price_mult': 1.4,
                'fixed_cost_mult': 0.5,
                'moq_range': (10, 50),
                'capacity_mult': 0.6,
                'lead_time_days': (0.5, 1.5),
                'lead_time_variability': 0.2,
                'freshness_loss_days': 0,
                'product_specialization': 'narrow'
            },
            'regional_distributor': {
                'distance_km': (30, 100),
                'base_price_mult': 1.0,
                'fixed_cost_mult': 1.0,
                'moq_range': (50, 200),
                'capacity_mult': 1.5,
                'lead_time_days': (1.5, 3.0),
                'lead_time_variability': 0.4,
                'freshness_loss_days': 1,
                'product_specialization': 'medium'
            },
            'bulk_wholesaler': {
                'distance_km': (100, 200),
                'base_price_mult': 0.7,
                'fixed_cost_mult': 2.0,
                'moq_range': (200, 500),
                'capacity_mult': 3.0,
                'lead_time_days': (3.0, 5.0),
                'lead_time_variability': 0.6,
                'freshness_loss_days': 2,
                'product_specialization': 'broad'
            },
            'farm_direct': {
                'distance_km': (150, 400),
                'base_price_mult': 0.5,
                'fixed_cost_mult': 3.0,
                'moq_range': (500, 1000),
                'capacity_mult': 2.5,
                'lead_time_days': (4.0, 7.0),
                'lead_time_variability': 0.8,
                'freshness_loss_days': 0,
                'product_specialization': 'narrow'
            }
        }
        
        self.SUPPLIER_MIX = [
            ('local_specialty', 4),
            ('regional_distributor', 6),
            ('bulk_wholesaler', 4),
            ('farm_direct', 3)
        ]
        
        # Placeholder for product catalog
        self.catalog = None
    
    def generate_all(self, unique_store_ids):
        """Main generation pipeline"""
        print("\n[Enhanced Generator] Creating differentiated supplier network...")
        
        # 1. Load Product Catalog (Critical Fix)
        self.catalog = self._load_catalog()
        if self.catalog is None:
            raise RuntimeError("Catalog not found. Run CatalogEnricher first.")

        # 2. Get demand context
        demand_profile = self._get_demand_summary()
        
        # 3. Generate suppliers
        suppliers_df = self._generate_heterogeneous_suppliers()
        
        # 4. Generate stores
        self._gen_stores(unique_store_ids)
        
        # 5. Generate distance matrix (Depots <-> Suppliers)
        self._gen_dist_matrix(suppliers_df)
        
        # 6. Generate supplier-product matrix
        self._gen_strategic_supplier_product(suppliers_df, demand_profile)
        
        # 7. Generate vehicles
        self._gen_vehicles()
        
        print(f"[Enhanced Generator] Complete. Artifacts in {Cfg.ARTIFACTS_DIR}")

    def _load_catalog(self):
        """Load product info from master catalog"""
        path = os.path.join(Cfg.ARTIFACTS_DIR, "master_product_catalog.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Ensure proper types
            df['product_id'] = df['product_id'].astype(str)
            return df
        return None

    def _generate_heterogeneous_suppliers(self):
        """Creates suppliers with distinct strategic profiles"""
        suppliers = []
        sid = 1
        
        for archetype_name, count in self.SUPPLIER_MIX:
            archetype = self.SUPPLIER_ARCHETYPES[archetype_name]
            
            for _ in range(count):
                dist_km = float(self.rng.uniform(*archetype['distance_km']))
                bearing = float(self.rng.uniform(0, 360))
                lat, lon = GeoUtils.dest_from(
                    Cfg.CENTER_LAT, Cfg.CENTER_LON, dist_km, bearing
                )
                
                lt_mean = float(self.rng.uniform(*archetype['lead_time_days']))
                lt_std = lt_mean * archetype['lead_time_variability']
                
                suppliers.append({
                    'supplier_id': sid,
                    'lat': round(lat, 6),
                    'lon': round(lon, 6),
                    'archetype': archetype_name,
                    'distance_tier': self._classify_distance(dist_km),
                    'lead_time_mean_days': round(lt_mean, 2),
                    'lead_time_std_days': round(lt_std, 2),
                    'base_price_mult': archetype['base_price_mult'],
                    'fixed_cost_mult': archetype['fixed_cost_mult'],
                    'capacity_mult': archetype['capacity_mult'],
                    'freshness_loss_base': archetype['freshness_loss_days'],
                    'product_specialization': archetype['product_specialization']
                })
                sid += 1
        
        df = pd.DataFrame(suppliers)
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        df.to_csv(out_path, index=False)
        return df
    
    def _classify_distance(self, dist_km):
        if dist_km < 50: return "Local"
        elif dist_km < 150: return "Regional"
        else: return "Remote"
    
    def _gen_strategic_supplier_product(self, suppliers_df, demand_profile):
        """Generate SP Matrix using Catalog Info"""
        sp_rows = []
        
        # Use catalog for product info
        # Extract unique products from catalog
        unique_products = self.catalog[['product_id', 'category_id', 'category_name']].drop_duplicates()
        products_list = list(unique_products.itertuples(index=False, name=None)) 
        # structure: (product_id, category_id, category_name)
        
        for _, supplier in suppliers_df.iterrows():
            sid = supplier['supplier_id']
            archetype = supplier['archetype']
            archetype_def = self.SUPPLIER_ARCHETYPES[archetype]
            
            # Determine portfolio based on specialization
            products_to_serve = self._determine_product_portfolio(
                archetype_def['product_specialization'], 
                products_list,
                archetype
            )
            
            for prod_info in products_to_serve:
                prod_id, cat_id, _ = prod_info
                
                # Get base price from catalog if possible, else default
                # Find matching row in catalog
                cat_row = self.catalog[self.catalog['product_id'] == prod_id]
                base_price = float(cat_row['price'].iloc[0]) if not cat_row.empty else 10.0
                
                # Apply supplier multiplier
                unit_price = base_price * supplier['base_price_mult']
                
                # Fixed Cost (Distance Based)
                distance_km = GeoUtils.haversine_km(
                    Cfg.CENTER_LAT, Cfg.CENTER_LON, supplier['lat'], supplier['lon']
                )
                base_fixed = float(Cfg.FIXED_ORDER_COST)
                supplier_fixed_cost = base_fixed * supplier['fixed_cost_mult'] * (1 + distance_km / 200)
                
                # MOQ
                moq_min, moq_max = archetype_def['moq_range']
                moq = int(self.rng.integers(moq_min, moq_max + 1))
                
                # Capacity
                # Heuristic: Allocate share of total demand
                # Assuming simple total demand for now, can improve with actual forecast
                base_capacity = 1000.0 * supplier['capacity_mult'] * self.rng.uniform(0.8, 1.5)
                
                # Freshness Loss
                transport_days = (distance_km / Cfg.SPEED_KMPH) / 24.0
                freshness_loss = supplier['freshness_loss_base'] + transport_days
                
                sp_rows.append({
                    'supplier_id': sid,
                    'product_id': prod_id,
                    'unit_price': round(unit_price, 2),
                    'fixed_order_cost': round(supplier_fixed_cost, 2),
                    'min_order_qty_units': moq,
                    'supplier_capacity_kg': round(base_capacity, 1),
                    'elapsed_shelf_days': round(freshness_loss, 2),
                    'lead_time_mean_days': supplier['lead_time_mean_days'],
                    'lead_time_std_days': supplier['lead_time_std_days']
                })
        
        df_sp = pd.DataFrame(sp_rows)
        df_sp = df_sp.drop_duplicates(subset=['supplier_id', 'product_id'])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        df_sp.to_csv(out_path, index=False)
        print(f"  -> Generated {len(df_sp)} supplier-product relationships")
        return df_sp
    
    def _determine_product_portfolio(self, specialization, products_list, archetype):
        """
        Filter products based on supplier specialization.
        products_list: list of (id, cat_id, name)
        """
        if not products_list: return []
        
        # Separate products by category for logic
        # Assuming category_ids in catalog map to: 1=Veg, 2=Fruit, 3=Meat, 4=Seafood...
        
        if specialization == 'broad':
            return products_list # Serve all
            
        elif specialization == 'narrow':
            # Farm Direct -> Fresh only (Category 1 or specific logic)
            # Local Specialty -> Pick one random category
            if archetype == 'farm_direct':
                # Serve Veg (1) or Fruit (2)
                candidates = [p for p in products_list if p[1] in [1, 2]]
                return candidates if candidates else products_list[:5]
            else:
                # Pick 1 random category
                cats = list(set(p[1] for p in products_list))
                if not cats: return []
                chosen_cat = self.rng.choice(cats)
                return [p for p in products_list if p[1] == chosen_cat]
                
        elif specialization == 'medium':
            # Pick 50% of available products randomly
            n = len(products_list)
            size = max(1, int(n * 0.5))
            chosen_indices = self.rng.choice(n, size=size, replace=False)
            return [products_list[i] for i in chosen_indices]
            
        return []

    def _get_demand_summary(self):
        """Load demand summary (Optional - for capacity sizing)"""
        # Kept simple for now, relying on defaults or basic file check
        return {} # Placeholder

    def _gen_stores(self, store_ids):
        """Generate stores.csv"""
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
        """Generate dist_ws.csv"""
        dist_ws = []
        for _, row in suppliers_df.iterrows():
            km = GeoUtils.haversine_km(Cfg.CENTER_LAT, Cfg.CENTER_LON, 
                                     row["lat"], row["lon"])
            time_min = int(round((km / Cfg.SPEED_KMPH) * 60))
            dist_ws.append((row["supplier_id"], km, time_min))
        
        df = pd.DataFrame(dist_ws, columns=["supplier_id", "dist_km", "time_min"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "dist_ws.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated dist_ws.csv")

    def _gen_vehicles(self):
        """Generate vehicles.csv"""
        rows = []
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
        
        if not rows:
            rows = [{"type": "Default", "capacity_kg": 1000, "var_cost_per_km": 1.0, "fixed_cost": 100.0, "cost_per_hour": 0.0}]
        
        df = pd.DataFrame(rows)
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "vehicles.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated vehicles.csv")

if __name__ == "__main__":
    # Test block
    gen = EnhancedSupplierGenerator()
    # Mock call (requires CatalogEnricher to have run)
    try:
        gen.generate_all([f"s_{i}" for i in range(10)])
    except Exception as e:
        print(f"Test failed: {e}")