# src/optimization/procurement.py
import os, ast
import pandas as pd
import numpy as np
import pulp
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class ProcurementOptimizer:
    """
    [DEBUG VERSION] Optimizes sourcing decisions with detailed logging.
    Helps identify why candidates are not being built.
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_PROCUREMENT, exist_ok=True)

    def run_with_constraints(self, max_lead_time_days=None, review_period_days=None, transport_feedback=None):
        self._load_data()
        active_review = review_period_days if review_period_days else Cfg.PROCURE_REVIEW_DAYS
        
        candidates = self._build_candidates(max_lead_time_days, active_review, transport_feedback)
        
        if candidates.empty:
            print(f"    [X] Infeasible: No candidates found.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Infeasible"
            
        return self._solve_milp(candidates)

    def _load_data(self):
        # 1. Load Demand
        self.unified = pd.read_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet"))
        
        # 2. Load Suppliers
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        if not os.path.exists(sup_path):
            sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv")
        self.suppliers = pd.read_csv(sup_path)
        
        # 3. Load Supplier-Product
        sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        if not os.path.exists(sp_path):
            sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product_harmonized_v2.csv")
        self.supplier_product = pd.read_csv(sp_path)
        
        # Type enforcement
        self.unified['store_id'] = self.unified['store_id'].astype(str)
        self.unified['product_id'] = self.unified['product_id'].astype(int)
        self.suppliers['supplier_id'] = self.suppliers['supplier_id'].astype(int)
        self.supplier_product['supplier_id'] = self.supplier_product['supplier_id'].astype(int)
        self.supplier_product['product_id'] = self.supplier_product['product_id'].astype(int)

        # --- DEBUG LOGGING ---
        print("\n    [DEBUG DATA LOAD]")
        print(f"    - Stores loaded: {self.unified['store_id'].nunique()}")
        print(f"    - Suppliers loaded: {self.suppliers['supplier_id'].nunique()}")
        
        # Check Product ID overlap
        demand_prods = set(self.unified['product_id'].unique())
        supply_prods = set(self.supplier_product['product_id'].unique())
        
        print(f"    - Unique Demand Product IDs (First 5): {list(demand_prods)[:5]}")
        print(f"    - Unique Supply Product IDs (First 5): {list(supply_prods)[:5]}")
        
        common = demand_prods.intersection(supply_prods)
        print(f"    - COMMON PRODUCT IDs: {len(common)}")
        
        if len(common) == 0:
            print("    [!!! CRITICAL WARNING] NO PRODUCT ID MATCH BETWEEN DEMAND AND SUPPLY.")
            print("    Please check if 'sim_product_id' mapping is working or if Step 1 and Step 4 are synced.")

    def _build_candidates(self, max_lead_time_days, active_review_period, transport_feedback=None):
        sup_loc = self.suppliers.set_index('supplier_id').to_dict(orient='index')
        spp_grouped = self.supplier_product.groupby('product_id')
        
        rows = []
        if transport_feedback is None: transport_feedback = {}

        count_demand_rows = 0
        count_prod_match = 0
        count_valid_candidate = 0

        for _, r in self.unified.iterrows():
            count_demand_rows += 1
            store = str(r['store_id'])
            
            # Try mapping using sim_product_id if available
            raw_prod = r['product_id']
            if 'sim_product_id' in r and pd.notna(r['sim_product_id']):
                prod = int(r['sim_product_id'])
            else:
                prod = int(raw_prod)
            
            demand_units = float(r.get('predicted_mean', 0)) * active_review_period
            if demand_units <= 0: continue
            
            if prod not in spp_grouped.groups:
                # Debug only first failure
                if count_demand_rows == 1:
                    print(f"    [DEBUG ROW 1] Demand Prod {prod} (Raw {raw_prod}) NOT FOUND in Supply. Skipping.")
                continue
            
            count_prod_match += 1
            possible_suppliers = spp_grouped.get_group(prod)
            
            for _, sp in possible_suppliers.iterrows():
                sid = int(sp['supplier_id'])
                if sid not in sup_loc: continue
                
                s_lat = sup_loc[sid].get('lat', sup_loc[sid].get('sup_lat'))
                s_lon = sup_loc[sid].get('lon', sup_loc[sid].get('sup_lon'))
                dist_km = GeoUtils.haversine_km(r['store_lat'], r['store_lon'], s_lat, s_lon)
                
                lead_time = float(sp.get('lead_time_mean_days', 1.0))
                elapsed = float(sp.get('elapsed_shelf_days', 0.0))
                
                avg_speed = getattr(Cfg, 'SPEED_KMPH', 35.0)
                driving_hours_per_day = getattr(Cfg, 'DRIVING_HOURS_PER_DAY', 10.0)
                transport_hours = dist_km / max(1e-6, avg_speed)
                transport_days = transport_hours / driving_hours_per_day
                
                procure_cycle_time = max(lead_time, transport_days)
                freshness_loss_at_arrival = elapsed + procure_cycle_time
                
                # Checks
                if max_lead_time_days and procure_cycle_time > max_lead_time_days: continue
                if (procure_cycle_time + elapsed) >= float(r.get('shelf_life', 5.0)): continue
                
                # Cost
                if sid in transport_feedback:
                    transport_cost = transport_feedback[sid] * r['unit_weight_kg']
                else:
                    transport_cost = Cfg.TRANSPORT_COST_PER_KG_KM * dist_km * r['unit_weight_kg']
                
                total_unit_cost = sp['unit_price'] + transport_cost
                
                rows.append({
                    'store_id': store, 'product_id': prod, 'supplier_id': sid,
                    'total_cost_per_unit': float(total_unit_cost),
                    'min_order_qty_units': int(max(1, sp.get('min_order_qty_units', 1))),
                    'demand_units': float(demand_units), 'unit_weight_kg': float(r['unit_weight_kg']),
                    'supplier_capacity_kg': float(sp.get('supplier_capacity_kg', np.inf)),
                    'unit_price': float(sp['unit_price']),
                    'daily_holding_cost_unit': float(r.get('daily_holding_cost_unit', 0.0)),
                    'lead_time_mean_days': lead_time,
                    'freshness_loss_days': float(freshness_loss_at_arrival)
                })
                count_valid_candidate += 1
        
        # [DEBUG SUMMARY]
        print(f"    [DEBUG CANDIDATES] Checked {count_demand_rows} rows. Found Supply for {count_prod_match} rows. Generated {count_valid_candidate} candidates.")
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset=['store_id', 'product_id', 'supplier_id'])
        return df

    def _solve_milp(self, cand_df):
        cand_df['key'] = list(zip(cand_df['store_id'], cand_df['product_id'], cand_df['supplier_id']))
        cand_map = cand_df.set_index('key').to_dict(orient='index')
        keys = cand_df['key'].tolist()
        
        prob = pulp.LpProblem("Procurement_MIP", pulp.LpMinimize)
        
        q = pulp.LpVariable.dicts("q", keys, lowBound=0, cat='Integer' if Cfg.ROUND_Q_TO_INT else 'Continuous')
        y = pulp.LpVariable.dicts("y", keys, lowBound=0, upBound=1, cat='Binary')
        
        bigM = {}
        for k in keys:
            dem_units = int(max(1, cand_map[k]['demand_units']))
            cap_units = int((cand_map[k]['supplier_capacity_kg'] / max(1e-6, cand_map[k]['unit_weight_kg'])))
            bigM[k] = max(dem_units, cap_units) + 1000
        
        objective_terms = []
        freshness_penalty_rate = float(getattr(Cfg, 'FRESHNESS_PENALTY_PER_DAY', 0.05))

        for k in keys:
            base_cost = cand_map[k]['total_cost_per_unit'] * q[k]
            fixed_cost = Cfg.FIXED_ORDER_COST * y[k]
            f_days = cand_map[k].get('freshness_loss_days', 0.0)
            u_price = cand_map[k]['unit_price']
            freshness_cost = freshness_penalty_rate * u_price * f_days * q[k]
            objective_terms.append(base_cost + fixed_cost + freshness_cost)

        prob += pulp.lpSum(objective_terms)      
        
        # Constraints
        for (s, p), grp in cand_df.groupby(['store_id', 'product_id']):
            prob += pulp.lpSum([q[(s, p, sid)] for sid in grp['supplier_id']]) >= grp['demand_units'].iloc[0] * Cfg.SERVICE_LEVEL
        
        for sid, grp in cand_df.groupby('supplier_id'):
            relevant_keys = [tuple(x) for x in grp['key'].tolist()]
            prob += pulp.lpSum([q[k] * cand_map[k]['unit_weight_kg'] for k in relevant_keys]) <= grp['supplier_capacity_kg'].iloc[0]
        
        for k in keys:
            prob += q[k] <= bigM[k] * y[k]
            prob += q[k] >= cand_map[k]['min_order_qty_units'] * y[k]
        
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=int(Cfg.MAX_SOLVE_TIME_S), gapRel=float(Cfg.GAP_REL))
        prob.solve(solver)
        status = pulp.LpStatus.get(prob.status, "Unknown")
        
        sol_rows = []
        for k in keys:
            q_val = q[k].value()
            y_val = y[k].value()
            if q_val is None: continue
            if q_val > 0:
                qty = int(q_val) if Cfg.ROUND_Q_TO_INT else q_val
                r = cand_map[k]
                fixed_cost_val = (Cfg.FIXED_ORDER_COST if (y_val and y_val>0.5) else 0.0)
                sol_rows.append({
                    'store_id': k[0], 'product_id': k[1], 'supplier_id': k[2],
                    'order_qty_units': qty,
                    'order_weight_kg': qty * r['unit_weight_kg'],
                    'unit_price': r['unit_price'],
                    'fixed_order_cost_reported': fixed_cost_val, 
                    'fixed_order_cost': fixed_cost_val, 
                    'daily_holding_cost_unit': r['daily_holding_cost_unit'],
                    'candidate_unit_cost': r.get('total_cost_per_unit', 0.0),
                    'freshness_loss_days': r.get('freshness_loss_days', 0.0)
                })

        sol_df = pd.DataFrame(sol_rows)
        if sol_df.empty:
            return sol_df, pd.DataFrame(), pd.DataFrame(), "Infeasible"
        
        agg_df = sol_df.groupby(['supplier_id', 'store_id'])['order_weight_kg'].sum().reset_index().rename(columns={'order_weight_kg': 'total_kg'})
        return sol_df, agg_df, pd.DataFrame(), status