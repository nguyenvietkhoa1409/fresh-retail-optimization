# src/optimization/procurement.py
import os
import time
import math
import ast
import pandas as pd
import numpy as np
import pulp
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class ProcurementOptimizer:
    """
    Class thực hiện tối ưu hóa mua hàng (Stage 1) - UPDATED STRATEGY A (BATCHING).
    Input: Unified Demand, Suppliers Info, Supplier-Product Matrix.
    Output: Kế hoạch đặt hàng tối ưu (Order Quantity per Store-Product-Supplier).
    Method: MILP with Joint Replenishment (Supplier-Store Link Cost) & Freshness Penalty.
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_PROCUREMENT, exist_ok=True)
        self.unified = None; self.suppliers = None; self.supplier_product = None

    def run(self):
        print("\n[Procurement] Starting Pipeline (Strategy A: Joint Replenishment)...")
        # 1. Load Data
        self._load_data()
        
        # 2. Build Candidates (Default Run)
        candidates = self._build_candidates(max_lead_time_days=None, active_review_period=Cfg.PROCURE_REVIEW_DAYS)
        if candidates.empty: 
            print("[Procurement] No feasible candidates found. Aborting.")
            return None
            
        # 3. Solve MILP
        sol_df, agg_df, util_df, status = self._solve_milp(candidates)
        
        # 4. Save Results
        if sol_df is not None and not sol_df.empty:
            self._save_results(sol_df, agg_df, util_df)
        
        print(f"[Procurement] Solver Status: {status}")
        return sol_df

    # --- NEW METHOD FOR INTEGRATED SOLVER ---
    def run_with_constraints(self, max_lead_time_days=None, review_period_days=None):
        """Used by Integrated Solver to inject P_lim and U_limit constraints"""
        self._load_data()
        
        # Override Config Review Period if provided
        active_review_period = review_period_days if review_period_days else Cfg.PROCURE_REVIEW_DAYS
        
        candidates = self._build_candidates(max_lead_time_days, active_review_period)
        if candidates.empty: 
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Infeasible"
            
        sol_df, agg_df, util_df, status = self._solve_milp(candidates)
        return sol_df, agg_df, util_df, status
    # ----------------------------------------

    def _load_data(self):
        # print("  -> Loading artifacts...")
        uni_path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv")
        sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product_harmonized_v2.csv")
        
        if not os.path.exists(uni_path): raise FileNotFoundError(uni_path)
        
        self.unified = pd.read_parquet(uni_path)
        self.suppliers = pd.read_csv(sup_path)
        self.supplier_product = pd.read_csv(sp_path)
        
        self.unified['store_id'] = self.unified['store_id'].astype(str)
        
        # Ensure unit_weight_kg exists
        if 'unit_weight_kg' not in self.unified.columns or self.unified['unit_weight_kg'].isna().any():
            self.unified['unit_weight_kg'] = self.unified['sim_product_id'].map(
                lambda x: Cfg.UNIT_WEIGHT.get(int(x), 1.0) if pd.notna(x) else 1.0
            ).astype(float)
            
        def parse_list(x):
            if isinstance(x, (list, np.ndarray)): return list(x)
            if isinstance(x, str):
                try: return ast.literal_eval(x)
                except: pass
            return []
        self.unified['assigned_suppliers'] = self.unified['assigned_suppliers'].apply(parse_list)

    def _build_candidates(self, max_lead_time_days, active_review_period):
        # print("  -> Building candidates...")
        
        # [FIX] Removed 'elapsed_shelf_days' from here (it is in supplier_product)
        sup_loc = self.suppliers.set_index('supplier_id')[
            ['sup_lat','sup_lon','capacity_kg','lead_time_mean_days']
        ].to_dict(orient='index')
        
        spp = self.supplier_product.copy()
        rows = []
        
        # FRESHNESS DECAY PARAMETER (Scientific Addition)
        # Loss of value per day of shelf-life consumed (e.g., 5%)
        FRESHNESS_DECAY_RATE = 0.05 
        
        for _, r in self.unified.iterrows():
            store = str(r['store_id']); prod = int(r['sim_product_id']) if pd.notna(r.get('sim_product_id')) else int(r['product_id'])
            store_lat = float(r.get('store_lat',0)); store_lon = float(r.get('store_lon',0)); unit_w = float(r.get('unit_weight_kg',1))
            shelf_life = float(r.get('shelf_life', np.nan))
            
            # [FIX] Pass holding cost to output
            daily_holding = float(r.get('daily_holding_cost_unit', 0.0))
            
            # Update Demand based on dynamic U_limit (Review Period)
            daily_mean = float(r.get('predicted_mean', 0.0))
            demand_units = daily_mean * active_review_period
            
            if demand_units <= 0: continue 

            assigned = r.get('assigned_suppliers', [])
            # Top 5 suppliers
            for sid in assigned[:5]: 
                try: sid = int(sid)
                except: continue
                if sid not in sup_loc: continue
                
                sp_row = spp[(spp['supplier_id']==sid) & (spp['product_id']==prod)]
                if sp_row.empty: continue 
                
                sp = sp_row.iloc[0]
                unit_price = float(sp.get('unit_price',0)); min_q = int(sp.get('min_order_qty_units',1))
                cap_kg = float(sp.get('supplier_capacity_kg',0))
                
                lead_time = float(sp.get('lead_time_mean_days', 2.0))
                # [FIX] Get elapsed from Product Matrix
                elapsed = int(sp.get('elapsed_shelf_days', 0))
                
                dist_km = GeoUtils.haversine_km(store_lat, store_lon, sup_loc[sid]['sup_lat'], sup_loc[sid]['sup_lon'])
                # Estimate transport time (assuming 50km/h and 8h driving day) -> Days
                transport_days = (dist_km / 50.0) / 8.0 
                total_procure_time = lead_time + transport_days + elapsed

                # --- CONSTRAINT 1: SHELF LIFE FEASIBILITY ---
                if pd.notna(shelf_life) and (total_procure_time > shelf_life): continue

                # --- CONSTRAINT 2: P_LIMIT (TIME BUDGET) ---
                if max_lead_time_days is not None:
                    if total_procure_time > max_lead_time_days:
                        continue 
                
                transport_cost = Cfg.TRANSPORT_COST_PER_KG_KM * dist_km * unit_w
                
                # --- FRESHNESS COST ---
                freshness_cost_unit = unit_price * FRESHNESS_DECAY_RATE * total_procure_time
                
                cap_units = cap_kg / unit_w if unit_w > 0 else 0
                bigM = int(math.ceil(min(cap_units, demand_units))) if cap_units > 0 else int(demand_units)
                
                rows.append({
                    'store_id': store, 'product_id': prod, 'supplier_id': sid,
                    'unit_weight_kg': unit_w, 'unit_price': unit_price,
                    'transport_cost_unit': transport_cost,
                    'freshness_cost_unit': freshness_cost_unit, # Store for analytics
                    # Objective = Price + Transport + Freshness Penalty
                    'total_cost_per_unit': unit_price + transport_cost + freshness_cost_unit,
                    'min_order_qty_units': min_q, 'bigM_units': max(1, bigM),
                    'demand_units': demand_units, 'supplier_capacity_kg': cap_kg,
                    'daily_holding_cost_unit': daily_holding # Pass through
                })
        
        cand_df = pd.DataFrame(rows)
        if not cand_df.empty: cand_df = cand_df.drop_duplicates(subset=['store_id','product_id','supplier_id'])
        return cand_df

    def _solve_milp(self, cand_df):
        # print("  -> Solving MILP...")
        prob = pulp.LpProblem("Procurement_Opt_StrategyA", pulp.LpMinimize)
        cand_df['key'] = cand_df.apply(lambda r: (str(r['store_id']), int(r['product_id']), int(r['supplier_id'])), axis=1)
        keys = cand_df['key'].tolist(); cand_map = cand_df.set_index('key').to_dict(orient='index')
        link_keys = list(set((k[0], k[2]) for k in keys))
        
        q_vars = pulp.LpVariable.dicts("q", keys, lowBound=0, cat='Continuous')
        y_vars = pulp.LpVariable.dicts("y", keys, cat='Binary')
        z_vars = pulp.LpVariable.dicts("z", link_keys, cat='Binary') 
        
        pairs = sorted(list(set((k[0], k[1]) for k in keys)))
        short_vars = {}
        if Cfg.ALLOW_SHORTAGE: short_vars = pulp.LpVariable.dicts("short", pairs, lowBound=0, cat='Continuous')

        obj_terms = []
        # Variable Cost (Price + Trans + Freshness)
        for k in keys: obj_terms.append(cand_map[k]['total_cost_per_unit'] * q_vars[k])
        # Fixed Cost (Per Link)
        for lk in link_keys: obj_terms.append(Cfg.FIXED_ORDER_COST * z_vars[lk])
        # Shortage Penalty
        if Cfg.ALLOW_SHORTAGE:
            for p in pairs: obj_terms.append(Cfg.SHORTAGE_COST * short_vars[p])
        prob += pulp.lpSum(obj_terms)

        # 1. Demand Coverage
        demand_map = { (str(r['store_id']), int(r['product_id'])): r['demand_units'] for _, r in cand_df.iterrows() }
        pair_to_keys = {}
        for k in keys: 
            p = (k[0], k[1])
            if p not in pair_to_keys: pair_to_keys[p] = []
            pair_to_keys[p].append(k)
            
        for p in pairs:
            demand = demand_map.get(p, 0)
            lhs = pulp.lpSum([q_vars[k] for k in pair_to_keys.get(p, [])])
            if Cfg.ALLOW_SHORTAGE: lhs += short_vars[p]
            prob += (lhs >= demand * Cfg.SERVICE_LEVEL)

        # 2. MOQ & Linking
        for k in keys:
            r = cand_map[k]
            prob += (q_vars[k] >= r['min_order_qty_units'] * y_vars[k])
            prob += (q_vars[k] <= r['bigM_units'] * y_vars[k])
            prob += (y_vars[k] <= z_vars[(k[0], k[2])]) # Product implies Link active

        # 3. Supplier Capacity
        sup_to_keys = {}
        for k in keys:
            s = k[2]
            if s not in sup_to_keys: sup_to_keys[s] = []
            sup_to_keys[s].append(k)
        cap_map = { int(r['supplier_id']): r['supplier_capacity_kg'] for _, r in cand_df.iterrows() }
        for sid, k_list in sup_to_keys.items():
            cap = cap_map.get(sid, 0)
            prob += (pulp.lpSum([q_vars[k] * cand_map[k]['unit_weight_kg'] for k in k_list]) <= cap)

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=Cfg.MAX_SOLVE_TIME_S, gapRel=Cfg.GAP_REL)
        prob.solve(solver)
        
        sol_rows = []
        link_counts = {}
        for k in keys:
            if q_vars[k].value() and q_vars[k].value() > 1e-5:
                lk = (k[0], k[2])
                link_counts[lk] = link_counts.get(lk, 0) + 1
        
        for k in keys:
            qv = q_vars[k].value()
            if qv and qv > 1e-5:
                if Cfg.ROUND_Q_TO_INT: qv = round(qv)
                r = cand_map[k]; lk = (k[0], k[2])
                dist_fixed = Cfg.FIXED_ORDER_COST / link_counts.get(lk, 1)
                
                sol_rows.append({
                    'store_id': k[0], 'product_id': k[1], 'supplier_id': k[2],
                    'order_qty_units': qv, 'order_weight_kg': qv * r['unit_weight_kg'],
                    'unit_price': r['unit_price'], 'transport_unit_cost': r['transport_cost_unit'],
                    'fixed_order_cost': dist_fixed,
                    # Real financial cost (excluding freshness penalty) for Report
                    'order_cost': (r['unit_price'] + r['transport_cost_unit']) * qv + dist_fixed,
                    'freshness_cost': r['freshness_cost_unit'] * qv,
                    'daily_holding_cost_unit': r.get('daily_holding_cost_unit', 0.0)
                })
                
        sol_df = pd.DataFrame(sol_rows)
        agg_df = pd.DataFrame(); util_df = pd.DataFrame()
        if not sol_df.empty:
            agg_df = sol_df.groupby(['supplier_id','store_id']).agg(
                total_units=('order_qty_units','sum'), total_kg=('order_weight_kg','sum'),
                total_cost=('order_cost','sum')
            ).reset_index()
            used = sol_df.groupby('supplier_id')['order_weight_kg'].sum().reset_index()
            util_df = used.rename(columns={'order_weight_kg':'used_kg'})
            util_df['capacity_kg'] = util_df['supplier_id'].map(cap_map)
            
        return sol_df, agg_df, util_df, pulp.LpStatus[prob.status]

    def _save_results(self, sol_df, agg_df, util_df):
        sol_df.to_csv(os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_solution.csv"), index=False)
        agg_df.to_csv(os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv"), index=False)
        util_df.to_csv(os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_supplier_utilization.csv"), index=False)