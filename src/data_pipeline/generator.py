# src/data_pipeline/generator_v2.py
"""
ENHANCED SUPPLIER GENERATOR
Creates rich trade-off structure for P/U sensitivity

Key Innovations:
1. Supplier Tiers: Volume vs Specialty suppliers
2. Distance-Based Fixed Costs (makes P-sensitivity real)
3. MOQ Scaling with Supplier Size (makes U-sensitivity real)
4. Lead Time Variability (risk-cost trade-off)
5. Capacity Constraints (forces diversification)
"""

import os
import numpy as np
import pandas as pd
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils


class EnhancedSupplierGenerator:
    """
    Generates heterogeneous supplier network with strategic attributes
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(Cfg.SEED)
        os.makedirs(Cfg.ARTIFACTS_DIR, exist_ok=True)
        
        # Define supplier archetypes
        self.SUPPLIER_ARCHETYPES = {
            'local_specialty': {
                'distance_km': (5, 30),
                'base_price_mult': 1.4,
                'fixed_cost_mult': 0.5,  # Low setup cost
                'moq_range': (10, 50),
                'capacity_mult': 0.6,    # Small scale
                'lead_time_days': (0.5, 1.5),
                'lead_time_variability': 0.2,  # Very reliable
                'freshness_loss_days': 0,
                'product_specialization': 'narrow'  # 1-2 categories only
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
                'product_specialization': 'medium'  # 2-3 categories
            },
            'bulk_wholesaler': {
                'distance_km': (100, 200),
                'base_price_mult': 0.7,
                'fixed_cost_mult': 2.0,  # High setup but low per-unit
                'moq_range': (200, 500),
                'capacity_mult': 3.0,    # Large scale
                'lead_time_days': (3.0, 5.0),
                'lead_time_variability': 0.6,
                'freshness_loss_days': 2,
                'product_specialization': 'broad'  # All categories
            },
            'farm_direct': {
                'distance_km': (150, 400),
                'base_price_mult': 0.5,  # Cheapest per-unit
                'fixed_cost_mult': 3.0,  # Very high fixed cost (logistics setup)
                'moq_range': (500, 1000),  # Pallet-based
                'capacity_mult': 2.5,
                'lead_time_days': (4.0, 7.0),
                'lead_time_variability': 0.8,  # Weather-dependent
                'freshness_loss_days': 0,  # Fresh from source
                'product_specialization': 'narrow'  # Single category (Fresh only)
            }
        }
        
        # Supplier mix configuration
        self.SUPPLIER_MIX = [
            ('local_specialty', 4),
            ('regional_distributor', 6),
            ('bulk_wholesaler', 4),
            ('farm_direct', 3)
        ]
    
    def generate_all(self, unique_store_ids):
        """Main generation pipeline"""
        print("\n[Enhanced Generator] Creating differentiated supplier network...")
        
        # Get demand context
        demand_profile = self._get_demand_summary()
        
        # Generate suppliers with archetypes
        suppliers_df = self._generate_heterogeneous_suppliers()
        
        # Generate products
        self._gen_products()
        
        # Generate stores
        self._gen_stores(unique_store_ids)
        
        # Generate distance matrix
        self._gen_dist_matrix(suppliers_df)
        
        # Generate supplier-product matrix WITH STRATEGIC ATTRIBUTES
        self._gen_strategic_supplier_product(suppliers_df, demand_profile)
        
        # Generate vehicles
        self._gen_vehicles()
        
        print(f"[Enhanced Generator] Complete. Artifacts in {Cfg.ARTIFACTS_DIR}")
    
    def _generate_heterogeneous_suppliers(self):
        """
        Creates suppliers with distinct strategic profiles
        """
        suppliers = []
        sid = 1
        
        for archetype_name, count in self.SUPPLIER_MIX:
            archetype = self.SUPPLIER_ARCHETYPES[archetype_name]
            
            for _ in range(count):
                # Generate location
                dist_km = float(self.rng.uniform(*archetype['distance_km']))
                bearing = float(self.rng.uniform(0, 360))
                lat, lon = GeoUtils.dest_from(
                    Cfg.CENTER_LAT, Cfg.CENTER_LON, dist_km, bearing
                )
                
                # Generate lead time (mean + std)
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
        
        print(f"  -> Generated {len(df)} suppliers across {len(self.SUPPLIER_MIX)} archetypes")
        print(f"     Archetype distribution:")
        print(df['archetype'].value_counts().to_string())
        
        return df
    
    def _classify_distance(self, dist_km):
        """Classify supplier by distance tier"""
        if dist_km < 50:
            return "Local"
        elif dist_km < 150:
            return "Regional"
        else:
            return "Remote"
    
    def _gen_strategic_supplier_product(self, suppliers_df, demand_profile):
        """
        Generate supplier-product matrix with P/U sensitive attributes
        
        Key Logic:
        - Distant suppliers: Lower unit price BUT higher fixed costs
        - Large suppliers: Better for high U (can handle large orders)
        - Local suppliers: Better for low P (fast, fresh)
        """
        
        sp_rows = []
        
        # Calculate total system demand for capacity scaling
        total_demand = sum(demand_profile.values())
        
        for _, supplier in suppliers_df.iterrows():
            sid = supplier['supplier_id']
            archetype = supplier['archetype']
            archetype_def = self.SUPPLIER_ARCHETYPES[archetype]
            
            # Determine which products this supplier serves
            products_to_serve = self._determine_product_portfolio(
                archetype_def['product_specialization']
            )
            
            for prod_info in products_to_serve:
                prod_id, cat_id, prod_name = prod_info
                
                # Get base product pricing
                price_range = Cfg.PRICE_RANGE_BY_PRODUCT.get(prod_id, (1.0, 10.0))
                base_price = self.rng.uniform(*price_range)
                
                # Apply supplier's price multiplier
                unit_price = base_price * supplier['base_price_mult']
                
                # --- KEY INNOVATION: Distance-based fixed cost ---
                # This makes P-parameter matter!
                # Far suppliers = high fixed cost, so only economical for high U
                distance_km = GeoUtils.haversine_km(
                    Cfg.CENTER_LAT, Cfg.CENTER_LON, 
                    supplier['lat'], supplier['lon']
                )
                
                # Fixed cost scales with distance tier
                base_fixed = float(Cfg.FIXED_ORDER_COST)
                supplier_fixed_cost = base_fixed * supplier['fixed_cost_mult'] * (1 + distance_km / 200)
                
                # --- KEY INNOVATION: MOQ scales with archetype ---
                # This makes U-parameter matter!
                moq_min, moq_max = archetype_def['moq_range']
                moq = int(self.rng.integers(moq_min, moq_max + 1))
                
                # --- Capacity calibration ---
                product_demand = demand_profile.get(prod_id, 5000.0)
                num_competing = len([s for s in suppliers_df.itertuples() 
                                    if self._can_serve_product(s.archetype, cat_id)])
                
                # Each supplier gets share of demand, scaled by archetype capacity
                base_capacity = (product_demand * 1.5) / max(1, num_competing)
                supplier_capacity_kg = base_capacity * supplier['capacity_mult']
                
                # Add variability
                supplier_capacity_kg *= self.rng.uniform(0.8, 1.3)
                
                # --- Freshness loss calculation ---
                # Combines supplier base loss + transport time
                transport_hours = distance_km / Cfg.SPEED_KMPH
                transport_days = transport_hours / 24.0
                
                freshness_loss = supplier['freshness_loss_base'] + transport_days
                
                # Add to matrix
                sp_rows.append({
                    'supplier_id': sid,
                    'product_id': prod_id,
                    'unit_price': round(unit_price, 2),
                    'fixed_order_cost': round(supplier_fixed_cost, 2),  # NEW!
                    'min_order_qty_units': moq,
                    'supplier_capacity_kg': round(supplier_capacity_kg, 1),
                    'elapsed_shelf_days': round(freshness_loss, 2),
                    'lead_time_mean_days': supplier['lead_time_mean_days'],
                    'lead_time_std_days': supplier['lead_time_std_days']
                })
        
        df_sp = pd.DataFrame(sp_rows)
        df_sp = df_sp.drop_duplicates(subset=['supplier_id', 'product_id'])
        
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        df_sp.to_csv(out_path, index=False)
        
        print(f"  -> Generated {len(df_sp)} supplier-product relationships")
        print(f"     Fixed cost range: ${df_sp['fixed_order_cost'].min():.0f} - ${df_sp['fixed_order_cost'].max():.0f}")
        print(f"     MOQ range: {df_sp['min_order_qty_units'].min()} - {df_sp['min_order_qty_units'].max()} units")
        
        return df_sp
    
    def _determine_product_portfolio(self, specialization):
        """
        Determine which products a supplier can serve based on specialization
        
        Returns: List of (product_id, category_id, name) tuples
        """
        products = Cfg.PRODUCT_CATEGORIES  # (id, cat_id, name, ...)
        
        if specialization == 'narrow':
            # Only serve 1 category (randomly chosen)
            chosen_cat = self.rng.choice([1, 2])
            return [(p[0], p[1], p[2]) for p in products if p[1] == chosen_cat]
        
        elif specialization == 'medium':
            # Serve 2-3 random products
            n_products = self.rng.integers(2, min(4, len(products) + 1))
            chosen = self.rng.choice(len(products), size=n_products, replace=False)
            return [(products[i][0], products[i][1], products[i][2]) for i in chosen]
        
        else:  # broad
            # Serve all products
            return [(p[0], p[1], p[2]) for p in products]
    
    def _can_serve_product(self, archetype_name, category_id):
        """Check if archetype can serve a product category"""
        spec = self.SUPPLIER_ARCHETYPES[archetype_name]['product_specialization']
        
        if spec == 'broad':
            return True
        elif spec == 'narrow':
            # Farm Direct only serves fresh (category 1)
            if archetype_name == 'farm_direct':
                return category_id == 1
            else:
                return True  # Local specialty can do any single category
        else:
            return True
    
    # === Keep existing helper methods ===
    
    def _get_demand_summary(self):
        """Load demand from Step 2 (same as original)"""
        demand_path = os.path.join(Cfg.OUT_DIR_PART2, "part2_reconstructed.parquet")
        default_profile = {101: 5000.0, 102: 4000.0, 201: 8000.0, 202: 6000.0}

        if not os.path.exists(demand_path):
            print(f"[WARN] Demand file not found. Using defaults.")
            return default_profile

        try:
            df = pd.read_parquet(demand_path)
            target_col = "D_recon" if "D_recon" in df.columns else "daily_demand"
            
            if target_col not in df.columns:
                return default_profile
            
            valid_demand = df[target_col].fillna(0).clip(lower=0)
            total_demand = valid_demand.sum()
            
            if total_demand == 0:
                return default_profile
            
            # Allocate by category
            profile = {
                101: total_demand * 0.20,
                102: total_demand * 0.20,
                201: total_demand * 0.30,
                202: total_demand * 0.30
            }
            
            print(f"   -> Calibrated from demand file: Total={total_demand:,.0f}")
            return profile
            
        except Exception as e:
            print(f"   [ERROR] {e}. Using defaults.")
            return default_profile
    
    def _gen_products(self):
        """Same as original"""
        df = pd.DataFrame(Cfg.PRODUCT_CATEGORIES,
                          columns=["product_id", "category_id", "name", 
                                  "holding_cost_per_kg_day", "volume_m3_per_kg"])
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "products.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated products.csv")
    
    def _gen_stores(self, store_ids):
        """Same as original"""
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
        """Same as original"""
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
        """Same as original"""
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
            rows = [{
                "type": "Default", 
                "capacity_kg": 1000, 
                "var_cost_per_km": 1.0, 
                "fixed_cost": 100.0, 
                "cost_per_hour": 0.0
            }]
        
        df = pd.DataFrame(rows)
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "vehicles.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Generated vehicles.csv")


# === Usage in main pipeline ===
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Mock store IDs for testing
    mock_stores = [f"store_{i:04d}" for i in range(50)]
    
    gen = EnhancedSupplierGenerator()
    gen.generate_all(mock_stores)