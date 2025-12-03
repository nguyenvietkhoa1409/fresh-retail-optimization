# src/optimization/integrated_solver.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import ProjectConfig as Cfg
from src.optimization.procurement import ProcurementOptimizer
from src.optimization.logistics import LogisticsManager

class IntegratedSolver:
    """
    Implements the Adaptive Sequential Decision-Making Approach.
    Iterates through 'Time Budget' scenarios to minimize Total Daily Cost.
    Updated: Calculates "Real Financial Cost" using exact unit holding costs.
    """
    
    def __init__(self):
        self.results = []
        os.makedirs(Cfg.OUT_DIR_ANALYSIS, exist_ok=True)

    def run(self):
        print("\n=======================================================")
        print("   INTEGRATED TIME-BUDGET OPTIMIZATION LOOP")
        print("=======================================================")
        
        # Define Trade-off Scenarios
        # P_lim: Procurement Time (Days) -> Allows reaching further suppliers
        # U_limit: Utilization Time (Days) -> Determines Sales Cycle Length
        scenarios = [
            {'p': 2, 'u': 5}, # Local sourcing, Long sales cycle
            {'p': 3, 'u': 4},
            {'p': 4, 'u': 3},
            {'p': 5, 'u': 2}, # Distant sourcing, Short sales cycle
        ]
        
        proc_opt = ProcurementOptimizer()
        log_mgr = LogisticsManager()
        
        for sc in scenarios:
            p_lim = sc['p']
            u_lim = sc['u']
            print(f"\n>>> TESTING SCENARIO: P_lim={p_lim} days | U_limit={u_lim} days")
            
            # --- STEP A: PROCUREMENT OPTIMIZATION ---
            sol_df, agg_df, _, status = proc_opt.run_with_constraints(max_lead_time_days=p_lim, review_period_days=u_lim)
            
            if sol_df.empty or status != "Optimal":
                print(f"    [X] Infeasible: Cannot source products within {p_lim} days.")
                self.results.append({'P_lim': p_lim, 'U_limit': u_lim, 'Total_Daily_Cost': float('inf')})
                continue
            
            # Save temp file for Logistics to consume
            agg_path = os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv")
            agg_df.to_csv(agg_path, index=False)
            
            # --- STEP B: LOGISTICS VRP OPTIMIZATION ---
            log_mgr.run() 
            
            # Read VRP Cost results
            vrp_sum_path = os.path.join(Cfg.OUT_DIR_LOGISTICS, "vrp_summary.csv")
            if os.path.exists(vrp_sum_path):
                vrp_df = pd.read_csv(vrp_sum_path)
                total_logistics_cost = vrp_df['Cost_USD'].sum()
            else:
                total_logistics_cost = 0.0
                
            # --- STEP C: FINANCIAL COST CALCULATION ---
            
            # 1. Real Procurement Spend (COGS + Fixed Order Fees)
            # We exclude transport estimates from MILP because VRP gives the real transport cost.
            real_product_cost = (sol_df['order_qty_units'] * sol_df['unit_price']).sum()
            real_fixed_cost = sol_df['fixed_order_cost'].sum()
            total_procure_cash = real_product_cost + real_fixed_cost
            
            # 2. Holding Cost (Opportunity Cost of Inventory)
            # Formula: Avg Inventory * Daily Cost * Cycle Length
            # Avg Inventory ~= Order Qty / 2 (Cycle Stock)
            if 'daily_holding_cost_unit' in sol_df.columns:
                # Exact Calculation per SKU
                line_holding_cost = (sol_df['order_qty_units'] / 2) * sol_df['daily_holding_cost_unit'] * u_lim
                holding_cost_per_cycle = line_holding_cost.sum()
            else:
                # Fallback Approximation (Safety Net)
                print("    [Warn] Exact holding cost missing, using approximation.")
                total_kg = sol_df['order_weight_kg'].sum()
                # Approx $0.1/kg/day * Multiplier
                holding_rate = 0.1 * Cfg.HOLDING_COST_MULTIPLIER
                holding_cost_per_cycle = (total_kg / 2) * holding_rate * u_lim
            
            # 3. Total Daily Cost (Financial View)
            # Sum everything and divide by the Cycle Length (U_limit)
            daily_financial_cost = (total_procure_cash + total_logistics_cost + holding_cost_per_cycle) / u_lim
            
            print(f"    [FINANCE] Product=${total_procure_cash:,.0f} | Logistics=${total_logistics_cost:,.0f} | Hold=${holding_cost_per_cycle:,.0f}")
            print(f"    >>> REAL DAILY COST: ${daily_financial_cost:,.2f}")
            
            self.results.append({
                'P_lim': p_lim,
                'U_limit': u_lim,
                'Product_Cost_Cycle': total_procure_cash,
                'Logistics_Cost_Cycle': total_logistics_cost,
                'Holding_Cost_Cycle': holding_cost_per_cycle,
                'Total_Daily_Cost': daily_financial_cost
            })
            
        self._save_results()

    def _save_results(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        df = df[df['Total_Daily_Cost'] != float('inf')]
        
        # Save results CSV
        out_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "integrated_optimization_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\n[Integrated] Results saved to {out_path}")
        
        # Generate Chart
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='P_lim', y='Total_Daily_Cost', marker='o', linewidth=2.5, color='darkgreen')
        
        if not df.empty:
            # Highlight Optimal Point
            best = df.loc[df['Total_Daily_Cost'].idxmin()]
            plt.plot(best['P_lim'], best['Total_Daily_Cost'], 'ro', markersize=12, label=f"Optimal (P={int(best['P_lim'])})")
        
        plt.title('Integrated Optimization: Real Daily Financial Cost', fontsize=14)
        plt.xlabel('Procurement Time Limit (Days)', fontsize=12)
        plt.ylabel('Total Daily Cost ($)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plot_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "06_integrated_optimization_curve.png")
        plt.savefig(plot_path)
        plt.close()