# src/optimization/integrated_solver.py
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import ProjectConfig as Cfg
from src.optimization.procurement import ProcurementOptimizer
from src.optimization.logistics import LogisticsManager
from src.optimization.cost_evaluator import CostEvaluator

class IntegratedSolver:
    """
    Implements the Adaptive Sequential Decision-Making Approach.
    Iterates through Strategic Scenarios (Matrix) to find the Optimal Business Model.
    
    [UPDATED] Now supports Iterative Feedback Loop (Procurement <-> Logistics)
    to solve the Local Optima problem.
    """
    
    def __init__(self, override_output_dir=None):
        self.results = []
        # Dynamic Output Directory
        if override_output_dir:
            self.analysis_dir = override_output_dir
            self.logistics_dir = os.path.join(override_output_dir, "logistics_logs")
        else:
            self.analysis_dir = Cfg.OUT_DIR_ANALYSIS # Default to Analysis folder
            self.logistics_dir = None # Use default inside LogisticsManager
            
        os.makedirs(self.analysis_dir, exist_ok=True)
        self._load_references()

    def _load_references(self):
        print("[Integrated] Loading reference data...")
        try:
            # 1. Load Inventory Ref
            inv = pd.read_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet"))
            inv['store_id'] = inv['store_id'].astype(str).str.strip()
            
            if 'sim_product_id' in inv.columns:
                inv['product_id'] = inv['sim_product_id'].astype(int)
            else:
                inv['product_id'] = inv['product_id'].astype(int)
                
            self.inventory_ref = inv[['store_id', 'product_id', 'safety_stock_kg']].drop_duplicates()
            
            # 2. Load Supplier Ref (FIXED LOGIC)
            sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
            sp = pd.read_csv(sp_path)
            sp['product_id'] = sp['product_id'].astype(int)
            sp['supplier_id'] = sp['supplier_id'].astype(int)
            
            # [FIX 1] Rename column safely
            if 'elapsed_shelf_days' in sp.columns:
                sp.rename(columns={'elapsed_shelf_days': 'freshness_loss_days'}, inplace=True)
            
            # [FIX 2] Create default if missing
            if 'freshness_loss_days' not in sp.columns:
                sp['freshness_loss_days'] = 0.0
                print('[Warn] freshness_loss_days column missing in supplier_product.csv, defaulting to 0.0')
            
            self.sp_ref = sp[['supplier_id', 'product_id', 'freshness_loss_days']].drop_duplicates()
                
        except Exception as e:
            print(f"[Critical Error] Ref load failed: {e}")
            self.inventory_ref = pd.DataFrame()
            self.sp_ref = pd.DataFrame()

    def run(self):
        print("\n=======================================================")
        print("   INTEGRATED STRATEGIC SIMULATION LOOP (ITERATIVE)")
        print("=======================================================")
        
        scenarios = Cfg.STRATEGIC_SCENARIOS
        proc_opt = ProcurementOptimizer()
        
        # [CONFIG] Number of iterations for Feedback Loop
        MAX_ITERATIONS = 3 
        DAMPING_FACTOR = 0.5 # Alpha for smoothing updates
        
        for sc in scenarios:
            p, u = sc['p'], sc['u']
            name = sc['name']
            desc = sc.get('desc', '')
            
            print(f"\n>>> TESTING STRATEGY: {name} (P={p}, U={u})")
            print(f"    Context: {desc}")
            
            transport_feedback = {} # Reset feedback for new scenario
            final_sol_df = pd.DataFrame()
            
            # --- ITERATIVE OPTIMIZATION LOOP ---
            for i in range(MAX_ITERATIONS):
                print(f"    [Iteration {i+1}/{MAX_ITERATIONS}] Running Procurement...")
                
                # --- STEP A: PROCUREMENT (with Feedback) ---
                sol_df, agg_df, _, status = proc_opt.run_with_constraints(
                    max_lead_time_days=p, 
                    review_period_days=u,
                    transport_feedback=transport_feedback # Pass learned costs
                )
                
                if sol_df.empty or status != "Optimal":
                    print(f"    [X] Infeasible at Iteration {i+1}.")
                    break
                
                # Save temp summary for Logistics
                agg_df.to_csv(os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv"), index=False)
                
                # --- STEP B: LOGISTICS ---
                log_mgr = LogisticsManager(override_output_dir=self.logistics_dir)
                routes = log_mgr.run() 
                
                if not routes:
                    print("    [!] No routes generated. Stopping iterations.")
                    break

                # --- STEP C: CALCULATE FEEDBACK ---
                # Calculate actual cost per kg for Inbound routes
                new_feedback = self._calculate_transport_feedback(routes)
                
                # Update with Damping
                transport_feedback = self._update_feedback(transport_feedback, new_feedback, DAMPING_FACTOR)
                
                # Log convergence info
                if i > 0:
                    print(f"    -> Feedback updated for {len(new_feedback)} suppliers.")
                
                # Keep the last successful result
                final_sol_df = sol_df
                
                # If this is the last iteration, proceed to Final Metrics
                if i == MAX_ITERATIONS - 1:
                    print("    -> Convergence reached (Max Iterations). Computing final metrics.")
            
            # --- FINAL EVALUATION (Based on last iteration) ---
            if final_sol_df.empty:
                self.results.append({
                    'Strategy': name, 'P_lim': p, 'U_limit': u, 
                    'Total_Daily_Cost': float('inf')
                })
                continue

            self._evaluate_final_result(final_sol_df, routes, p, u, name, desc)

        self._save_results()

    def _calculate_transport_feedback(self, routes):
        """
        Analyses VRP routes to calculate actual Transport Cost per Kg for each Supplier.
        Only considers 'Inbound' routes (Supplier -> Warehouse).
        """
        sup_costs = {} # {sid: [unit_costs]}
        
        for r in routes:
            # Only feedback Inbound costs to procurement
            if r['role'] != 'Inbound' or r['total_load_kg'] <= 0:
                continue
            
            # Heuristic: Unit cost for this route = Total Cost / Total Load
            # This rewards suppliers on efficient (dense/short) routes
            route_unit_cost = r['cost'] / r['total_load_kg']
            
            # Assign this cost to all suppliers on this route
            for step in r['steps']:
                if step['type'] == 'SUPPLIER':
                    try:
                        sid = int(step['id'])
                        if sid not in sup_costs: sup_costs[sid] = []
                        sup_costs[sid].append(route_unit_cost)
                    except: pass
        
        # Average if a supplier is visited multiple times (split loads)
        feedback = {}
        for sid, costs in sup_costs.items():
            feedback[sid] = sum(costs) / len(costs)
            
        return feedback

    def _update_feedback(self, old_fb, new_fb, alpha):
        """Soft update: New = alpha * New + (1-alpha) * Old"""
        all_keys = set(old_fb.keys()) | set(new_fb.keys())
        updated = {}
        for k in all_keys:
            v_old = old_fb.get(k, None)
            v_new = new_fb.get(k, None)
            
            if v_old is None: updated[k] = v_new
            elif v_new is None: updated[k] = v_old
            else:
                updated[k] = alpha * v_new + (1 - alpha) * v_old
        return updated

    def _evaluate_final_result(self, sol_df, routes, p, u, name, desc):
        # Type Cleaning & Merge
        try:
            # 1. Standardize IDs
            sol_df['store_id'] = sol_df['store_id'].astype(str).str.strip()
            sol_df['product_id'] = sol_df['product_id'].astype(int)
            sol_df['supplier_id'] = sol_df['supplier_id'].astype(int)
            
            # 2. Merge Inventory Info
            if not self.inventory_ref.empty:
                sol_df = sol_df.merge(self.inventory_ref, on=['store_id', 'product_id'], how='left')
                sol_df['safety_stock_kg'] = sol_df['safety_stock_kg'].fillna(0.0)
            else: sol_df['safety_stock_kg'] = 0.0
            
            # 3. Merge Supplier Info (Freshness) - CRITICAL FIX
            if not self.sp_ref.empty:
                # Ensure Reference Types Match
                self.sp_ref['supplier_id'] = self.sp_ref['supplier_id'].astype(int)
                self.sp_ref['product_id'] = self.sp_ref['product_id'].astype(int)
                
                # DROP existing column if present to avoid suffixes (_x, _y)
                if 'freshness_loss_days' in sol_df.columns:
                    sol_df = sol_df.drop(columns=['freshness_loss_days'])
                
                # MERGE
                sol_df = sol_df.merge(self.sp_ref, on=['supplier_id', 'product_id'], how='left')
            
            # [CRITICAL FIX] Ensure column exists BEFORE access
            if 'freshness_loss_days' not in sol_df.columns:
                print("   [WARN] 'freshness_loss_days' missing after merge. Defaulting to 0.0")
                sol_df['freshness_loss_days'] = 0.0
            else:
                sol_df['freshness_loss_days'] = sol_df['freshness_loss_days'].fillna(0.0)

            # Merge Product Name for Readability (Demonstration)
            try:
                prods = pd.read_csv(os.path.join(Cfg.ARTIFACTS_DIR, "products.csv"))
                if 'name' in prods.columns:
                    prod_map = prods.set_index('product_id')['name'].to_dict()
                    sol_df['product_name'] = sol_df['product_id'].map(prod_map)
            except: pass
        
        except Exception as e: 
            print(f"[Warn] Eval merge issue: {e}")
            if 'freshness_loss_days' not in sol_df.columns: sol_df['freshness_loss_days'] = 0.0
        
        # Calculate Costs
        total_cost = sum(r['cost'] for r in routes) if routes else 0.0
        
        # Extract VRP Metrics
        in_routes = [r for r in routes if r['role'] == 'Inbound']
        last_arr = max([r['steps'][-1]['arrival_time_min'] for r in in_routes]) if in_routes else 240
        out_start = last_arr + Cfg.SERVICE_TIME_CROSSDOCK_MINS
        is_next_day = out_start > 540
        crossdock_time = f"{int(out_start//60)}:{int(out_start%60):02d}"
        
        # Financials
        real_product_cost = (sol_df['order_qty_units'] * sol_df['unit_price']).sum()
        
        # Fixed cost handling
        sol_df['fixed_order_cost'] = sol_df.get('fixed_order_cost', sol_df.get('fixed_order_cost_reported', 0.0))
        real_fixed_cost = sol_df['fixed_order_cost'].sum()
        if real_fixed_cost == 0: # Fallback if column missing
             n_orders = len(sol_df)
             real_fixed_cost = float(n_orders) * float(Cfg.FIXED_ORDER_COST)

        raw_proc_total = real_product_cost + real_fixed_cost
        
        # Inventory Value
        total_kg = sol_df['order_weight_kg'].sum()
        avg_price_kg = (real_product_cost / total_kg) if total_kg > 0 else 0
        raw_safety_value = sol_df['safety_stock_kg'].sum() * avg_price_kg
        
        # Freshness
        total_qty = sol_df['order_qty_units'].sum()
        
        # [SAFE ACCESS] Now guaranteed to be safe
        w_loss = (sol_df['order_qty_units'] * sol_df['freshness_loss_days']).sum()
        avg_loss = (w_loss / total_qty) if total_qty > 0 else 0
        
        extra_hold = 1.0 if is_next_day else 0.0
        total_fresh_loss = avg_loss + extra_hold
        avg_price_per_unit = (real_product_cost / total_qty) if total_qty > 0 else 0.0
        
        print(f"    DEBUG_COSTS: total_qty={total_qty:.0f} | avg_freshness_loss={total_fresh_loss:.3f} days (Source Avg: {avg_loss:.3f})")

        # Metrics
        metrics = CostEvaluator.calculate_integrated_daily_cost(
            procurement_cost_total=raw_proc_total,
            distribution_cost_total=total_cost,
            avg_inventory_value=raw_safety_value,
            P_lim=p, U_lim=u,
            freshness_loss_days_avg=total_fresh_loss,
            extra_holding_days=extra_hold,
            total_order_units=float(total_qty),
            avg_price_per_unit=float(avg_price_per_unit)
        )
        
        print(f"    [FINAL METRICS] Proc=${raw_proc_total:,.0f} | Log=${total_cost:,.0f} | Daily=${metrics['Total_Daily_Cost']:,.2f}")
        
        res = metrics.copy()
        res.update({
            'Strategy': name,
            'Description': desc,
            'Crossdock_Time': crossdock_time, 
            'Is_Next_Day': is_next_day
        })
        self.results.append(res)
        
        # Save Final Detailed Plan for THIS Strategy
        detailed_filename = f"procurement_plan_{name}_P{p}_U{u}.csv"
        cols_order = ['store_id', 'product_id', 'product_name', 'supplier_id', 
                      'order_qty_units', 'unit_price', 'freshness_loss_days', 
                      'order_weight_kg', 'candidate_unit_cost']
        final_cols = [c for c in cols_order if c in sol_df.columns]
        
        sol_df[final_cols].to_csv(os.path.join(self.analysis_dir, detailed_filename), index=False)
        print(f"    -> Detailed Plan saved: {detailed_filename}")
        
        # Save routes for this scenario
        current_log_dir = self.logistics_dir if self.logistics_dir else Cfg.OUT_DIR_LOGISTICS
        pd.DataFrame(routes).to_csv(os.path.join(current_log_dir, f"routes_{name}.csv"), index=False)

    def _save_results(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        df = df[df['Total_Daily_Cost'] != float('inf')]
        
        if df.empty:
            print("[Integrated] No feasible solutions found.")
            return

        out_path = os.path.join(self.analysis_dir, "integrated_optimization_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\n[Integrated] Results saved to {out_path}")
        
        # Save Best Scenario Info
        best = df.loc[df['Total_Daily_Cost'].idxmin()]
        best_p, best_u = int(best['P_lim']), int(best['U_lim'])
        print(f"\n*** WINNING STRATEGY: {best['Strategy']} (P={best_p}, U={best_u}) ***")
        print(f"*** Reason: {best['Description']} ***")
        
        with open(os.path.join(Cfg.OUT_DIR_ANALYSIS, "best_scenario_config.json"), 'w') as f:
            import json
            json.dump({"P": best_p, "U": best_u}, f)
        
        # Plotting
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Strategy', y='Total_Daily_Cost', hue='Strategy', palette='viridis', legend=False)
            plt.title('Strategic Comparison (Iterative Optimization)', fontsize=14)
            plt.ylabel('Total Daily Cost ($)')
            plt.xlabel('Strategy')
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(self.analysis_dir, "06_strategic_comparison_bar.png"))
            plt.close()
        except: pass