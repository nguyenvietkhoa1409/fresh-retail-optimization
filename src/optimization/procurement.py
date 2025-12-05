# src/optimization/procurement.py
import os, ast
import pandas as pd
import numpy as np
import pulp
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class ProcurementOptimizer:
    """
    Optimizes sourcing decisions (Which Supplier -> Which Store).
    Updated:
      - Unified transport days formula using Cfg.SPEED_KMPH and Cfg.DRIVING_HOURS_PER_DAY
      - Remove freshness_cost from procurement unit cost (freshness penalty handled centrally)
      - MIP formulation: integer q (units) + binary y (order placed)
      - MOQ constraints and fixed order cost included in objective
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_PROCUREMENT, exist_ok=True)

    def run_with_constraints(self, max_lead_time_days=None, review_period_days=None):
        self._load_data()
        active_review = review_period_days if review_period_days else Cfg.PROCURE_REVIEW_DAYS
        candidates = self._build_candidates(max_lead_time_days, active_review)
        
        if candidates.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Infeasible"
            
        return self._solve_milp(candidates)

    def _load_data(self):
        self.unified = pd.read_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet"))
        self.suppliers = pd.read_csv(os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv"))
        self.supplier_product = pd.read_csv(os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product_harmonized_v2.csv"))
        
        # Type enforcement
        self.unified['store_id'] = self.unified['store_id'].astype(str)
        self.unified['product_id'] = self.unified['product_id'].astype(int)
        
        def parse_list(x):
            if isinstance(x, str):
                try: return ast.literal_eval(x)
                except: return []
            return list(x) if isinstance(x, (list, np.ndarray)) else []
        self.unified['assigned_suppliers'] = self.unified['assigned_suppliers'].apply(parse_list)

    def _build_candidates(self, max_lead_time_days, active_review_period):
        sup_loc = self.suppliers.set_index('supplier_id').to_dict(orient='index')
        spp = self.supplier_product.copy()
        rows = []
        
        # We remove procurement-side freshness cost; freshness penalty will be computed centrally.
        for _, r in self.unified.iterrows():
            store = str(r['store_id'])
            # sim_product_id used as product mapping earlier
            prod = int(r['sim_product_id'] if pd.notna(r.get('sim_product_id')) else r['product_id'])
            demand_units = float(r.get('predicted_mean', 0)) * active_review_period
            if demand_units <= 0: continue
            
            # Use assigned supplier candidates (top few)
            assigned_ids = list(dict.fromkeys(r.get('assigned_suppliers', [])[:10]))  # keep order, unique
            for sid in assigned_ids:
                sid = int(sid)
                if sid not in sup_loc: continue
                sp_row = spp[(spp['supplier_id']==sid) & (spp['product_id']==prod)]
                if sp_row.empty: continue
                sp = sp_row.iloc[0]
                
                # Lead time from supplier data
                lead_time = float(sp.get('lead_time_mean_days', 1.0))
                elapsed = float(sp.get('elapsed_shelf_days', 0.0))
                # distance store <-> supplier
                dist_km = GeoUtils.haversine_km(r['store_lat'], r['store_lon'], sup_loc[sid]['sup_lat'], sup_loc[sid]['sup_lon'])
                
                # Unified transport_days formula:
                avg_speed = getattr(Cfg, 'SPEED_KMPH', 35.0)
                driving_hours_per_day = getattr(Cfg, 'DRIVING_HOURS_PER_DAY', 10.0)
                transport_hours = dist_km / max(1e-6, avg_speed)
                transport_days = transport_hours / driving_hours_per_day
                
                procure_cycle_time = lead_time + transport_days
                
                # Feasibility checks
                if max_lead_time_days and procure_cycle_time > max_lead_time_days:
                    continue
                if (procure_cycle_time + elapsed) >= float(r.get('shelf_life', 5.0)):
                    # would arrive expired or at shelf-life -> infeasible
                    continue
                
                # Transport cost per unit = cost_per_kg_km * dist_km * unit_weight_kg
                transport_cost = Cfg.TRANSPORT_COST_PER_KG_KM * dist_km * r['unit_weight_kg']
                
                # Total unit procurement cost = supplier unit price + transport (no freshness here)
                total_unit_cost = sp['unit_price'] + transport_cost
                
                rows.append({
                    'store_id': store, 'product_id': prod, 'supplier_id': sid,
                    'total_cost_per_unit': float(total_unit_cost),
                    'min_order_qty_units': int(max(1, sp.get('min_order_qty_units', 1))),
                    'demand_units': float(demand_units), 'unit_weight_kg': float(r['unit_weight_kg']),
                    'supplier_capacity_kg': float(sp.get('supplier_capacity_kg', np.inf)),
                    'unit_price': float(sp['unit_price']),
                    'daily_holding_cost_unit': float(r.get('daily_holding_cost_unit', 0.0)),
                    'lead_time_mean_days': lead_time
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset=['store_id', 'product_id', 'supplier_id'])
        return df

    def _solve_milp(self, cand_df):
        # Keys are tuples (store, product, supplier)
        cand_df['key'] = list(zip(cand_df['store_id'], cand_df['product_id'], cand_df['supplier_id']))
        cand_map = cand_df.set_index('key').to_dict(orient='index')
        keys = cand_df['key'].tolist()
        
        prob = pulp.LpProblem("Procurement_MIP", pulp.LpMinimize)
        
        # Decision variables
        # Quantities in units (Integer if ROUND_Q_TO_INT True)
        q = pulp.LpVariable.dicts("q", keys, lowBound=0,
                                  cat='Integer' if Cfg.ROUND_Q_TO_INT else 'Continuous')
        # Binary y_k = 1 if we place any order with candidate k (to capture fixed order cost)
        y = pulp.LpVariable.dicts("y", keys, lowBound=0, upBound=1, cat='Binary')
        
        # Build Big-M for each candidate (keep conservative but not too large)
        bigM = {}
        for k in keys:
            dem_units = int(max(1, cand_map[k]['demand_units']))
            # capacity in units = capacity_kg / unit_weight_kg
            cap_units = int((cand_map[k]['supplier_capacity_kg'] / max(1e-6, cand_map[k]['unit_weight_kg'])))
            bigM[k] = max(dem_units, cap_units) + 1000  # cushion
        
        # Objective: unit costs * qty + fixed order cost * y
        prob += pulp.lpSum([cand_map[k]['total_cost_per_unit'] * q[k] + (Cfg.FIXED_ORDER_COST * y[k]) for k in keys])
        
        # Constraints
        # 1) Demand coverage per (store, product) at service level
        for (s, p), grp in cand_df.groupby(['store_id', 'product_id']):
            prob += pulp.lpSum([q[(s, p, sid)] for sid in grp['supplier_id']]) >= grp['demand_units'].iloc[0] * Cfg.SERVICE_LEVEL
        
        # 2) Supplier capacity (kg)
        for sid, grp in cand_df.groupby('supplier_id'):
            # sum q_k * unit_weight <= capacity_kg
            relevant_keys = [tuple(x) for x in grp['key'].tolist()]
            prob += pulp.lpSum([q[k] * cand_map[k]['unit_weight_kg'] for k in relevant_keys]) <= grp['supplier_capacity_kg'].iloc[0]
        
        # 3) Link q and y (Big-M) & MOQ
        for k in keys:
            prob += q[k] <= bigM[k] * y[k]
            prob += q[k] >= cand_map[k]['min_order_qty_units'] * y[k]
        
        # Solve using CBC (default)
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=int(Cfg.MAX_SOLVE_TIME_S), gapRel=float(Cfg.GAP_REL))
        res = prob.solve(solver)
        status = pulp.LpStatus.get(prob.status, "Unknown")
        
        sol_rows = []
        for k in keys:
            q_val = q[k].value()
            y_val = y[k].value()
            if q_val is None: continue
            if q_val and q_val > 0:
                qty = int(q_val) if Cfg.ROUND_Q_TO_INT else q_val
                r = cand_map[k]
                fixed_cost_val = (Cfg.FIXED_ORDER_COST if (y_val and y_val>0.5) else 0.0)
                sol_rows.append({
                    'store_id': k[0], 'product_id': k[1], 'supplier_id': k[2],
                    'order_qty_units': qty,
                    'order_weight_kg': qty * r['unit_weight_kg'],
                    'unit_price': r['unit_price'],
                    'fixed_order_cost_reported': fixed_cost_val,   # new
                    'fixed_order_cost': fixed_cost_val,            # keep old name too
                    'daily_holding_cost_unit': r['daily_holding_cost_unit'],
                    'candidate_unit_cost': r.get('total_cost_per_unit', r.get('unit_price', 0.0))
                })

        
        sol_df = pd.DataFrame(sol_rows)
        if sol_df.empty:
            return sol_df, pd.DataFrame(), pd.DataFrame(), "Infeasible"
        
        agg_df = sol_df.groupby(['supplier_id', 'store_id'])['order_weight_kg'].sum().reset_index().rename(columns={'order_weight_kg': 'total_kg'})
        return sol_df, agg_df, pd.DataFrame(), status
