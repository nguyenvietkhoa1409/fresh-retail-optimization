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
    """
    
    def __init__(self, override_output_dir=None):
        self.results = []
        # [MODIFIED] Dynamic Output Directory
        if override_output_dir:
            self.analysis_dir = override_output_dir
            # Với sensitivity, ta có thể không cần lưu route map chi tiết, 
            # hoặc lưu vào subfolder logistic bên trong sensitivity
            self.logistics_dir = os.path.join(override_output_dir, "logistics_logs")
        else:
            self.analysis_dir = Cfg.OUT_DIR_ANALYSIS
            self.logistics_dir = None # Use default inside LogisticsManager
            
        os.makedirs(self.analysis_dir, exist_ok=True)
        self._load_references()

    def _load_references(self):
        print("[Integrated] Loading reference data...")
        try:
            inv = pd.read_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet"))
            inv['store_id'] = inv['store_id'].astype(str).str.strip()
            
            if 'sim_product_id' in inv.columns:
                inv['product_id'] = inv['sim_product_id'].astype(int)
            else:
                inv['product_id'] = inv['product_id'].astype(int)
                
            self.inventory_ref = inv[['store_id', 'product_id', 'safety_stock_kg']].drop_duplicates()
            
            sp = pd.read_csv(os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product_harmonized_v2.csv"))
            sp['product_id'] = sp['product_id'].astype(int)
            sp['supplier_id'] = sp['supplier_id'].astype(int)
            
            if 'elapsed_shelf_days' in sp.columns:
                self.sp_ref = sp[['supplier_id', 'product_id', 'elapsed_shelf_days']].rename(columns={'elapsed_shelf_days': 'freshness_loss_days'})
            else:
                self.sp_ref = sp[['supplier_id', 'product_id', 'freshness_loss_days']]
            self.sp_ref = self.sp_ref.drop_duplicates()
                
        except Exception as e:
            print(f"[Critical Error] Ref load failed: {e}")
            self.inventory_ref = pd.DataFrame()
            self.sp_ref = pd.DataFrame()

    def run(self):
        print("\n=======================================================")
        print("   INTEGRATED STRATEGIC SIMULATION LOOP")
        print("=======================================================")
        
        # [MODIFIED] Use Strategic Scenarios from Config
        scenarios = Cfg.STRATEGIC_SCENARIOS
        
        proc_opt = ProcurementOptimizer()
        log_mgr = LogisticsManager()
        
        for sc in scenarios:
            p, u = sc['p'], sc['u']
            name = sc['name']
            desc = sc.get('desc', '')
            
            print(f"\n>>> TESTING STRATEGY: {name} (P={p}, U={u})")
            print(f"    Context: {desc}")
            
            # --- STEP A: PROCUREMENT ---
            sol_df, agg_df, _, status = proc_opt.run_with_constraints(max_lead_time_days=p, review_period_days=u)
            
            if sol_df.empty or status != "Optimal":
                print(f"    [X] Infeasible: Cannot source within {p} days.")
                self.results.append({
                    'Strategy': name, 'P_lim': p, 'U_limit': u, 
                    'Total_Daily_Cost': float('inf')
                })
                continue
            
            # Type Cleaning & Merge
            try:
                sol_df['store_id'] = sol_df['store_id'].astype(str).str.strip()
                sol_df['product_id'] = sol_df['product_id'].astype(int)
                sol_df['supplier_id'] = sol_df['supplier_id'].astype(int)
                
                if not self.inventory_ref.empty:
                    sol_df = sol_df.merge(self.inventory_ref, on=['store_id', 'product_id'], how='left')
                    sol_df['safety_stock_kg'] = sol_df['safety_stock_kg'].fillna(0.0)
                else: sol_df['safety_stock_kg'] = 0.0
                
                if not self.sp_ref.empty:
                    sol_df = sol_df.merge(self.sp_ref, on=['supplier_id', 'product_id'], how='left')
                    sol_df['freshness_loss_days'] = sol_df['freshness_loss_days'].fillna(0.0)
                else: sol_df['freshness_loss_days'] = 0.0
            except: pass
            
            agg_df.to_csv(os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv"), index=False)
            
            # --- STEP B: LOGISTICS ---
            log_mgr = LogisticsManager(override_output_dir=self.logistics_dir)
            log_mgr.run() 
            current_log_dir = self.logistics_dir if self.logistics_dir else Cfg.OUT_DIR_LOGISTICS
            # Save Route File
            src_routes = os.path.join(current_log_dir, "vrp_routes_solution.csv")
            dst_routes = os.path.join(current_log_dir, f"routes_P{p}_U{u}.csv")
            if os.path.exists(src_routes):
                shutil.copy(src_routes, dst_routes)
            
            # --- STEP C: METRICS ---
            vrp_sum = pd.read_csv(os.path.join(Cfg.OUT_DIR_LOGISTICS, "vrp_summary.csv"))
            dist_cost = vrp_sum['Cost_USD'].iloc[0]
            is_next_day = bool(vrp_sum['Is_Next_Day'].iloc[0]) if 'Is_Next_Day' in vrp_sum.columns else False
            crossdock_time = vrp_sum['Crossdock_Ready_Time'].iloc[0] if 'Crossdock_Ready_Time' in vrp_sum.columns else "N/A"
            
            # Product cost = sum unit price * units
            real_product_cost = (sol_df['order_qty_units'] * sol_df['unit_price']).sum()

            # Robust fixed cost handling:
            # - prefer 'fixed_order_cost' (older name)
            # - else use 'fixed_order_cost_reported' (new procurement patch)
            # - else fallback: count number of orders and multiply by Cfg.FIXED_ORDER_COST
            if 'fixed_order_cost' in sol_df.columns:
                real_fixed_cost = sol_df['fixed_order_cost'].sum()
            elif 'fixed_order_cost_reported' in sol_df.columns:
                real_fixed_cost = sol_df['fixed_order_cost_reported'].sum()
            else:
                # fallback: assume each returned row is an order placed
                n_orders = len(sol_df)
                real_fixed_cost = float(n_orders) * float(Cfg.FIXED_ORDER_COST)

            # Ensure a consistent column exists for downstream code
            sol_df['fixed_order_cost'] = sol_df.get('fixed_order_cost', sol_df.get('fixed_order_cost_reported', 0.0))

            raw_proc_total = real_product_cost + real_fixed_cost

            
            total_kg = sol_df['order_weight_kg'].sum()
            avg_price_kg = (real_product_cost / total_kg) if total_kg > 0 else 0
            raw_safety_value = sol_df['safety_stock_kg'].sum() * avg_price_kg
            
            total_qty = sol_df['order_qty_units'].sum()
            w_loss = (sol_df['order_qty_units'] * sol_df['freshness_loss_days']).sum()
            avg_loss = (w_loss / total_qty) if total_qty > 0 else 0
            
            extra_hold = 1.0 if is_next_day else 0.0
            total_fresh_loss = avg_loss + extra_hold
            
            # compute avg price per unit (fallback if total_qty==0)
            avg_price_per_unit = (real_product_cost / total_qty) if total_qty > 0 else 0.0
            #Debugging step: 
            print("DEBUG_COSTS: total_qty =", total_qty,
            "total_order_units =", float(total_qty),
            "avg_price_per_unit =", round(avg_price_per_unit,3),
            "freshness_loss_days_avg =", round(total_fresh_loss,3),
            "Cfg.FRESHNESS_PENALTY_PER_DAY =", getattr(Cfg, 'FRESHNESS_PENALTY_PER_DAY'))

            # --- STEP D: EVALUATION ---
            metrics = CostEvaluator.calculate_integrated_daily_cost(
                procurement_cost_total=raw_proc_total,
                distribution_cost_total=dist_cost,
                avg_inventory_value=raw_safety_value,
                P_lim=p, U_lim=u,
                freshness_loss_days_avg=total_fresh_loss,
                extra_holding_days=extra_hold,
                total_order_units=float(total_qty),
                avg_price_per_unit=float(avg_price_per_unit)
            )
            
            print(f"    [METRICS] Proc=${raw_proc_total:,.0f} | Log=${dist_cost:,.0f} | HoldVal=${raw_safety_value:,.0f}")
            print(f"    >>> DAILY COST: ${metrics['Total_Daily_Cost']:,.2f} (NextDay={is_next_day})")
            
            res = metrics.copy()
            res.update({
                'Strategy': name, # Save strategy name
                'Description': desc,
                'Crossdock_Time': crossdock_time, 
                'Is_Next_Day': is_next_day
            })
            self.results.append(res)
            
        self._save_results()

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
        
        # Update Chart to use Strategy Names on X-axis if possible
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Strategy', y='Total_Daily_Cost', palette='viridis')
        plt.title('Strategic Comparison: Daily Operational Cost', fontsize=14)
        plt.ylabel('Total Daily Cost ($)')
        plt.xlabel('Strategy')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(self.analysis_dir, "06_strategic_comparison_bar.png"))
        plt.close()