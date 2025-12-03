# src/analysis/sensitivity.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import ProjectConfig as Cfg
from src.optimization.procurement import ProcurementOptimizer

class SensitivityAnalyzer:
    """
    Thực hiện One-Way Sensitivity Analysis.
    Updated: Plots 'Fulfillment Cost' instead of 'Total Cost' to show logistics sensitivity clearly.
    """
    def __init__(self):
        self.results = []
        os.makedirs(Cfg.OUT_DIR_ANALYSIS, exist_ok=True)
        # Backup defaults
        self.defaults = {
            "TRANSPORT": Cfg.TRANSPORT_COST_PER_KG_KM,
            "FIXED": Cfg.FIXED_ORDER_COST,
            "SHORTAGE": Cfg.SHORTAGE_COST
        }

    def run(self):
        print("\n=== STARTING DECOUPLED SENSITIVITY ANALYSIS ===")
        
        multipliers = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        
        print("-> Testing Transport Cost...")
        for m in multipliers:
            Cfg.TRANSPORT_COST_PER_KG_KM = self.defaults["TRANSPORT"] * m
            self._run_single_scenario("Transport Cost", m)
        Cfg.TRANSPORT_COST_PER_KG_KM = self.defaults["TRANSPORT"]

        print("-> Testing Fixed Cost...")
        for m in multipliers:
            Cfg.FIXED_ORDER_COST = self.defaults["FIXED"] * m
            self._run_single_scenario("Fixed Order Cost", m)
        Cfg.FIXED_ORDER_COST = self.defaults["FIXED"]

        print("-> Testing Shortage Penalty...")
        for m in multipliers:
            Cfg.SHORTAGE_COST = self.defaults["SHORTAGE"] * m
            self._run_single_scenario("Shortage Penalty", m)
        Cfg.SHORTAGE_COST = self.defaults["SHORTAGE"]

        self._plot_results()
        print("\n=== COMPLETE ===")

    def _run_single_scenario(self, param_name, multiplier):
        Cfg.VERBOSE_SOLVER = False 
        optimizer = ProcurementOptimizer()
        sol_df, _, _, _ = optimizer.run_with_constraints()
        
        if sol_df.empty:
            fulfill_cost = float('inf')
        else:
            # --- DECOUPLING LOGIC ---
            # Total Cost = ProductCost + Transport + Fixed + Shortage + FreshnessPenalty
            # We want: Fulfillment Cost = Total - ProductCost
            
            # Re-calculate Product Cost manually to be safe
            product_cost = (sol_df['order_qty_units'] * sol_df['unit_price']).sum()
            total_order_cost = sol_df['order_cost'].sum() # This includes Price + Transport + Fixed
            
            # Note: order_cost in ProcurementOptimizer output = (Price + Trans + Freshness) * Q + Fixed
            # So subtracting ProductCost leaves: Transport + Freshness + Fixed
            fulfill_cost = total_order_cost - product_cost
            
        print(f"   [{param_name}] x{multiplier}: Fulfillment Cost = ${fulfill_cost:,.0f}")
        
        self.results.append({
            "Parameter": param_name,
            "Multiplier": multiplier,
            "Fulfillment_Cost": fulfill_cost
        })

    def _plot_results(self):
        df = pd.DataFrame(self.results)
        df = df[df['Fulfillment_Cost'] != float('inf')]
        
        df.to_csv(os.path.join(Cfg.OUT_DIR_ANALYSIS, "sensitivity_results_decoupled.csv"), index=False)
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Normalized Plot (Relative to Baseline 1.0)
        # Allows comparing Transport vs Fixed slope directly
        
        sns.lineplot(
            data=df, 
            x="Multiplier", 
            y="Fulfillment_Cost", 
            hue="Parameter", 
            style="Parameter", 
            markers=True, 
            linewidth=2.5
        )
        
        plt.title("Sensitivity Analysis: Impact on Logistics & Fulfillment Cost (Decoupled)", fontsize=14)
        plt.xlabel("Parameter Multiplier (1.0 = Baseline)", fontsize=12)
        plt.ylabel("Fulfillment Cost (Transport + Fixed + Freshness) [$]", fontsize=12)
        plt.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
        
        out_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "07_sensitivity_analysis_decoupled.png")
        plt.savefig(out_path)
        print(f"-> Chart saved to {out_path}")

if __name__ == "__main__":
    analyzer = SensitivityAnalyzer()
    analyzer.run()