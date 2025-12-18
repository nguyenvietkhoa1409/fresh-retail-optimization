# src/optimization/procurement_v2.py
"""
UPDATED PROCUREMENT SOLVER
Now reads supplier-specific fixed costs from data
"""

import os
import pandas as pd
import numpy as np
import pulp
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils


class EnhancedProcurementOptimizer:
    """
    Procurement optimizer with supplier-specific fixed costs
    """
    
    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_PROCUREMENT, exist_ok=True)
    
    def run_with_constraints(self, max_lead_time_days=None, review_period_days=None, 
                            transport_feedback=None):
        """Main entry point"""
        self._load_data()
        active_review = review_period_days if review_period_days else Cfg.PROCURE_REVIEW_DAYS
        
        candidates = self._build_candidates(max_lead_time_days, active_review, transport_feedback)
        
        if candidates.empty:
            print(f"    [X] Infeasible: No candidates found.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Infeasible"
        
        return self._solve_milp(candidates)
    
    def _load_data(self):
        """Load all required data"""
        # 1. Demand
        self.unified = pd.read_parquet(
            os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        )
        
        # 2. Suppliers
        self.suppliers = pd.read_csv(os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv"))
        
        # 3. Supplier-Product (NOW WITH FIXED COSTS!)
        self.supplier_product = pd.read_csv(
            os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        )
        
        # Type enforcement
        self.unified['store_id'] = self.unified['store_id'].astype(str)
        self.unified['product_id'] = self.unified['product_id'].astype(int)
        self.suppliers['supplier_id'] = self.suppliers['supplier_id'].astype(int)
        self.supplier_product['supplier_id'] = self.supplier_product['supplier_id'].astype(int)
        self.supplier_product['product_id'] = self.supplier_product['product_id'].astype(int)
        
        print(f"\n    [Data Load] Stores: {self.unified['store_id'].nunique()}, "
              f"Suppliers: {self.suppliers['supplier_id'].nunique()}, "
              f"SP relationships: {len(self.supplier_product)}")
    
    def _build_candidates(self, max_lead_time_days, active_review_period, transport_feedback=None):
        """
        Build candidate sourcing options with ALL costs
        """
        
        sup_loc = self.suppliers.set_index('supplier_id').to_dict(orient='index')
        spp_grouped = self.supplier_product.groupby('product_id')
        
        rows = []
        if transport_feedback is None: 
            transport_feedback = {}
        
        for _, demand_row in self.unified.iterrows():
            store = str(demand_row['store_id'])
            
            # Get product ID (handle sim_product_id if exists)
            if 'sim_product_id' in demand_row and pd.notna(demand_row['sim_product_id']):
                prod = int(demand_row['sim_product_id'])
            else:
                prod = int(demand_row['product_id'])
            
            demand_units = float(demand_row.get('predicted_mean', 0)) * active_review_period
            if demand_units <= 0: 
                continue
            
            # Find suppliers for this product
            if prod not in spp_grouped.groups:
                continue
            
            possible_suppliers = spp_grouped.get_group(prod)
            
            for _, sp in possible_suppliers.iterrows():
                sid = int(sp['supplier_id'])
                
                if sid not in sup_loc:
                    continue
                
                # Get supplier location
                s_lat = sup_loc[sid].get('lat', sup_loc[sid].get('sup_lat'))
                s_lon = sup_loc[sid].get('lon', sup_loc[sid].get('sup_lon'))
                
                dist_km = GeoUtils.haversine_km(
                    demand_row['store_lat'], demand_row['store_lon'], 
                    s_lat, s_lon
                )
                
                # Lead time calculation
                lead_time = float(sp.get('lead_time_mean_days', 1.0))
                
                # Freshness calculation
                elapsed = float(sp.get('elapsed_shelf_days', 0.0))
                
                # Transport time
                transport_hours = dist_km / max(1e-6, Cfg.SPEED_KMPH)
                transport_days = transport_hours / 24.0
                
                procure_cycle_time = max(lead_time, transport_days)
                freshness_loss_at_arrival = elapsed + procure_cycle_time
                
                # === CONSTRAINTS CHECK ===
                
                # 1. Lead time constraint (P parameter)
                if max_lead_time_days and procure_cycle_time > max_lead_time_days:
                    continue
                
                # 2. Shelf life feasibility
                if (procure_cycle_time + elapsed) >= float(demand_row.get('shelf_life', 5.0)):
                    continue
                
                # === COST CALCULATION ===
                
                # Unit cost (price + transport)
                if sid in transport_feedback:
                    transport_cost = transport_feedback[sid] * demand_row['unit_weight_kg']
                else:
                    transport_cost = (Cfg.TRANSPORT_COST_PER_KG_KM * 
                                    dist_km * demand_row['unit_weight_kg'])
                
                total_unit_cost = sp['unit_price'] + transport_cost
                
                # === KEY CHANGE: Read fixed cost from data ===
                supplier_fixed_cost = float(sp.get('fixed_order_cost', Cfg.FIXED_ORDER_COST))
                
                # Store candidate
                rows.append({
                    'store_id': store,
                    'product_id': prod,
                    'supplier_id': sid,
                    'total_cost_per_unit': float(total_unit_cost),
                    'fixed_order_cost': supplier_fixed_cost,  # Supplier-specific!
                    'min_order_qty_units': int(max(1, sp.get('min_order_qty_units', 1))),
                    'demand_units': float(demand_units),
                    'unit_weight_kg': float(demand_row['unit_weight_kg']),
                    'supplier_capacity_kg': float(sp.get('supplier_capacity_kg', np.inf)),
                    'unit_price': float(sp['unit_price']),
                    'daily_holding_cost_unit': float(demand_row.get('daily_holding_cost_unit', 0.0)),
                    'lead_time_mean_days': lead_time,
                    'freshness_loss_days': float(freshness_loss_at_arrival),
                    'distance_km': float(dist_km)
                })
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df = df.drop_duplicates(subset=['store_id', 'product_id', 'supplier_id'])
            
            # Debug output
            print(f"    [Candidates] Generated {len(df)} options")
            print(f"      Fixed cost range: ${df['fixed_order_cost'].min():.0f} - "
                  f"${df['fixed_order_cost'].max():.0f}")
            print(f"      Distance range: {df['distance_km'].min():.1f} - "
                  f"{df['distance_km'].max():.1f} km")
        
        return df
    
    def _solve_milp(self, cand_df):
        """
        Solve procurement MILP with supplier-specific fixed costs
        """
        
        cand_df['key'] = list(zip(cand_df['store_id'], 
                                 cand_df['product_id'], 
                                 cand_df['supplier_id']))
        cand_map = cand_df.set_index('key').to_dict(orient='index')
        keys = cand_df['key'].tolist()
        
        # Create problem
        prob = pulp.LpProblem("Procurement_Enhanced", pulp.LpMinimize)
        
        # Decision variables
        q = pulp.LpVariable.dicts("q", keys, lowBound=0, 
                                 cat='Integer' if Cfg.ROUND_Q_TO_INT else 'Continuous')
        y = pulp.LpVariable.dicts("y", keys, lowBound=0, upBound=1, cat='Binary')
        
        # Big-M for linearization
        bigM = {}
        for k in keys:
            dem = int(max(1, cand_map[k]['demand_units']))
            cap = int(cand_map[k]['supplier_capacity_kg'] / 
                     max(1e-6, cand_map[k]['unit_weight_kg']))
            bigM[k] = max(dem, cap) + 1000
        
        # === OBJECTIVE FUNCTION ===
        objective_terms = []
        freshness_penalty_rate = float(getattr(Cfg, 'FRESHNESS_PENALTY_PER_DAY', 0.05))
        
        for k in keys:
            candidate = cand_map[k]
            
            # Variable cost (unit price + transport)
            var_cost = candidate['total_cost_per_unit'] * q[k]
            
            # Fixed cost (SUPPLIER-SPECIFIC!)
            fixed_cost = candidate['fixed_order_cost'] * y[k]
            
            # Freshness penalty
            f_days = candidate.get('freshness_loss_days', 0.0)
            u_price = candidate['unit_price']
            fresh_penalty = freshness_penalty_rate * u_price * f_days * q[k]
            
            objective_terms.append(var_cost + fixed_cost + fresh_penalty)
        
        prob += pulp.lpSum(objective_terms)
        
        # === CONSTRAINTS ===
        
        # 1. Demand satisfaction (with service level)
        for (s, p), grp in cand_df.groupby(['store_id', 'product_id']):
            relevant_keys = [tuple(x) for x in grp['key'].tolist()]
            min_demand = grp['demand_units'].iloc[0] * Cfg.SERVICE_LEVEL
            
            prob += (pulp.lpSum([q[k] for k in relevant_keys]) >= min_demand,
                    f"Demand_{s}_{p}")
        
        # 2. Supplier capacity
        for sid, grp in cand_df.groupby('supplier_id'):
            relevant_keys = [tuple(x) for x in grp['key'].tolist()]
            max_cap = grp['supplier_capacity_kg'].iloc[0]
            
            prob += (pulp.lpSum([q[k] * cand_map[k]['unit_weight_kg'] 
                                for k in relevant_keys]) <= max_cap,
                    f"Capacity_{sid}")
        
        # 3. Fixed cost activation & MOQ
        for k in keys:
            prob += (q[k] <= bigM[k] * y[k], f"FixedActivate_{k}")
            prob += (q[k] >= cand_map[k]['min_order_qty_units'] * y[k], 
                    f"MOQ_{k}")
        
        # === SOLVE ===
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=int(Cfg.MAX_SOLVE_TIME_S),
            gapRel=float(Cfg.GAP_REL)
        )
        
        prob.solve(solver)
        status = pulp.LpStatus.get(prob.status, "Unknown")
        
        print(f"    [Solver] Status: {status}, Objective: ${pulp.value(prob.objective):,.2f}")
        
        # === EXTRACT SOLUTION ===
        sol_rows = []
        
        for k in keys:
            q_val = q[k].value()
            y_val = y[k].value()
            
            if q_val is None or q_val < 0.01:
                continue
            
            qty = int(q_val) if Cfg.ROUND_Q_TO_INT else q_val
            r = cand_map[k]
            
            fixed_cost_incurred = (r['fixed_order_cost'] 
                                  if (y_val and y_val > 0.5) else 0.0)
            
            sol_rows.append({
                'store_id': k[0],
                'product_id': k[1],
                'supplier_id': k[2],
                'order_qty_units': qty,
                'order_weight_kg': qty * r['unit_weight_kg'],
                'unit_price': r['unit_price'],
                'fixed_order_cost': fixed_cost_incurred,
                'daily_holding_cost_unit': r['daily_holding_cost_unit'],
                'candidate_unit_cost': r['total_cost_per_unit'],
                'freshness_loss_days': r['freshness_loss_days'],
                'distance_km': r['distance_km']
            })
        
        sol_df = pd.DataFrame(sol_rows)
        
        if sol_df.empty:
            return sol_df, pd.DataFrame(), pd.DataFrame(), "Infeasible"
        
        # Aggregation for logistics
        agg_df = (sol_df.groupby(['supplier_id', 'store_id'])['order_weight_kg']
                 .sum().reset_index()
                 .rename(columns={'order_weight_kg': 'total_kg'}))
        
        return sol_df, agg_df, pd.DataFrame(), status
    
ProcurementOptimizer = EnhancedProcurementOptimizer