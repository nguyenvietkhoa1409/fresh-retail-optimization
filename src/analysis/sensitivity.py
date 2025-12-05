import os
import sys
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import ProjectConfig as Cfg
from src.optimization.integrated_solver import IntegratedSolver

class SensitivityAnalyzer:
    """
    Performs Strategic Robustness Analysis.
    Generates a Tornado Chart to rank parameter impact.
    """
    def __init__(self, fast_mode: bool = True):
        self.results = []
        self.fast_mode = fast_mode 
        
        self.base_sensitivity_dir = os.path.join(Cfg.BASE_DIR, "data", "artifacts", "sensitivity")
        os.makedirs(self.base_sensitivity_dir, exist_ok=True)

        # Backup defaults
        self._defaults = {
            "TRANSPORT_COST_PER_KG_KM": float(Cfg.TRANSPORT_COST_PER_KG_KM),
            "FIXED_ORDER_COST": float(Cfg.FIXED_ORDER_COST),
            "DAILY_HOLDING_RATE_PCT": float(Cfg.DAILY_HOLDING_RATE_PCT),
            "PAIR_LIMIT": int(getattr(Cfg, 'PAIR_LIMIT', 200)),
            "TARGET_STORE_COUNT": int(40),
            "VRP_SEARCH_TIME_LIMIT_SEC": int(getattr(Cfg, 'VRP_SEARCH_TIME_LIMIT_SEC', 60)),
            "MAX_SOLVE_TIME_S": int(getattr(Cfg, 'MAX_SOLVE_TIME_S', 900)),
            "GAP_REL": float(getattr(Cfg, 'GAP_REL', 0.02)),
            "VEHICLE_FLEET_DEFINITIONS": copy.deepcopy(getattr(Cfg, 'VEHICLE_FLEET_DEFINITIONS', [])),
        }

    def _preflight(self):
        missing = []
        required = [
            os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet"),
            os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv"),
            os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product_harmonized_v2.csv")
        ]
        for p in required:
            if not os.path.exists(p): missing.append(p)
        if missing:
            print("[PREFLIGHT] Missing artifacts. Run InventoryPlanner first.")
            return False
        return True

    def _apply_fast_mode(self):
        print("[FAST MODE] Reducing solver budget for sensitivity check.")
        Cfg.PAIR_LIMIT = min(100, getattr(Cfg, 'PAIR_LIMIT', 200))
        Cfg.VRP_SEARCH_TIME_LIMIT_SEC = max(5, int(self._defaults['VRP_SEARCH_TIME_LIMIT_SEC'] / 6))
        Cfg.MAX_SOLVE_TIME_S = max(10, int(self._defaults['MAX_SOLVE_TIME_S'] / 10))
        Cfg.GAP_REL = max(0.05, float(self._defaults['GAP_REL']) * 4)

    def _restore_defaults(self):
        print("[RESTORE] Restoring defaults.")
        Cfg.TRANSPORT_COST_PER_KG_KM = self._defaults['TRANSPORT_COST_PER_KG_KM']
        Cfg.FIXED_ORDER_COST = self._defaults['FIXED_ORDER_COST']
        Cfg.DAILY_HOLDING_RATE_PCT = self._defaults['DAILY_HOLDING_RATE_PCT']
        Cfg.PAIR_LIMIT = self._defaults['PAIR_LIMIT']
        Cfg.VRP_SEARCH_TIME_LIMIT_SEC = self._defaults['VRP_SEARCH_TIME_LIMIT_SEC']
        Cfg.MAX_SOLVE_TIME_S = self._defaults['MAX_SOLVE_TIME_S']
        Cfg.GAP_REL = self._defaults['GAP_REL']
        Cfg.VEHICLE_FLEET_DEFINITIONS = copy.deepcopy(self._defaults['VEHICLE_FLEET_DEFINITIONS'])

    def run(self):
        print("\n=======================================================")
        print("   STRATEGIC SENSITIVITY ANALYSIS (TORNADO CHART)")
        print("=======================================================")

        if not self._preflight(): return
        if self.fast_mode: self._apply_fast_mode()

        # Define ranges to stress-test the model
        # These should represent "Low" vs "High" scenarios
        tests = [
            ("Logistics Cost", "TRANSPORT_COST_PER_KG_KM", [0.005, 0.01, 0.015, 0.03, 0.05]), # 0.015 is Baseline
            ("Holding Cost", "DAILY_HOLDING_RATE_PCT", [0.1, 0.3, 0.5, 1.0, 2.0]),            # 0.5 is Baseline
            ("Fixed Overhead", "FIXED_ORDER_COST", [1.0, 5.0, 20.0, 50.0])                      # 5.0 is Baseline
        ]

        start_all = time.perf_counter()
        
        # We need to establish the Baseline Cost first (using defaults)
        print(">>> Establishing Baseline...")
        solver = IntegratedSolver(override_output_dir=self.base_sensitivity_dir)
        solver.run()
        df_base = pd.DataFrame(solver.results)
        if not df_base.empty:
            self.baseline_cost = df_base['Total_Daily_Cost'].min()
            print(f"    Baseline Cost: ${self.baseline_cost:,.2f}")
        else:
            print("[Error] Baseline run failed.")
            return

        try:
            for name, param_attr, values in tests:
                print(f"\n>>> Analyzing Sensitivity: {name}...")
                base_val = getattr(Cfg, param_attr)
                
                for v in values:
                    # Skip if it's the baseline value (we already know the cost)
                    # Float comparison tolerance
                    if abs(v - base_val) < 1e-6: 
                        cost = self.baseline_cost
                        strat = df_base.loc[df_base['Total_Daily_Cost'].idxmin(), 'Strategy']
                        runtime_s = 0
                    else:
                        print(f"    Setting {param_attr} = {v}")
                        setattr(Cfg, param_attr, float(v))

                        t0 = time.perf_counter()
                        solver = IntegratedSolver(override_output_dir=self.base_sensitivity_dir)
                        solver.run()
                        runtime_s = time.perf_counter() - t0

                        df_res = pd.DataFrame(solver.results)
                        if df_res.empty: continue
                        
                        best = df_res.loc[df_res['Total_Daily_Cost'].idxmin()]
                        cost = float(best['Total_Daily_Cost'])
                        strat = best['Strategy']

                    self.results.append({
                        "Dimension": name,
                        "Value": float(v),
                        "Winning_Strategy": strat,
                        "Min_Total_Cost": cost,
                        "Delta": cost - self.baseline_cost
                    })
                
                setattr(Cfg, param_attr, float(base_val)) # Reset

        finally:
            self._restore_defaults()
            print(f"\n[Sensitivity] Completed in {time.perf_counter() - start_all:.1f}s")

        # Visualization
        self._plot_tornado_chart()
        self._plot_trend_lines()

    def _plot_tornado_chart(self):
        """
        Generates a Tornado Chart showing the range of impact for each dimension.
        """
        df = pd.DataFrame(self.results)
        out_csv = os.path.join(Cfg.OUT_DIR_ANALYSIS, "sensitivity_data.csv")
        df.to_csv(out_csv, index=False)
        
        if df.empty: return

        # Calculate Range (Swing) for each Dimension
        summary = df.groupby("Dimension")['Min_Total_Cost'].agg(['min', 'max'])
        summary['swing'] = summary['max'] - summary['min']
        summary = summary.sort_values('swing', ascending=True) # Sort for horizontal bar
        
        # Baseline line
        base = self.baseline_cost

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bars
        # Plot from Min to Baseline (Left side) and Baseline to Max (Right side)
        # Or simpler: Plot a bar from Min to Max, centered visually? 
        # Standard Tornado: Bars centered on Baseline 0.
        
        y_pos = np.arange(len(summary))
        
        # Values relative to baseline
        left_bar = summary['min'] - base
        right_bar = summary['max'] - base
        
        # Plot 'Low' impact (Left)
        rects1 = ax.barh(y_pos, left_bar, align='center', color='#3498db', alpha=0.8, label='Cost Reduction Potential')
        # Plot 'High' impact (Right)
        rects2 = ax.barh(y_pos, right_bar, align='center', color='#e74c3c', alpha=0.8, label='Cost Risk Exposure')
        
        # Decoration
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary.index, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel('Impact on Daily Cost ($) relative to Baseline')
        ax.set_title('Sensitivity Tornado Chart: What drives costs?', fontweight='bold')
        ax.legend(loc='lower right')
        
        # Add value labels
        def autolabel(rects, is_left):
            for rect in rects:
                width = rect.get_width()
                label_x = width if not is_left else width - (width * 0.1)
                val = f"${int(width)}" if abs(width) > 10 else ""
                if val:
                    ha = 'left' if not is_left else 'right'
                    ax.annotate(val, xy=(width, rect.get_y() + rect.get_height() / 2),
                                xytext=(3 if not is_left else -3, 0), textcoords="offset points",
                                ha=ha, va='center', fontsize=9)
        
        autolabel(rects1, True)
        autolabel(rects2, False)

        plt.tight_layout()
        path = os.path.join(self.base_sensitivity_dir, "08_Tornado_Chart.png")
        plt.savefig(path, dpi=300)
        print(f"-> Tornado chart saved: {path}")

    def _plot_trend_lines(self):
        """Secondary chart: Line plots to see non-linearity."""
        df = pd.DataFrame(self.results)
        g = sns.FacetGrid(df, col="Dimension", sharex=False, sharey=False, height=4)
        g.map(sns.lineplot, "Value", "Min_Total_Cost", marker="o")
        path = os.path.join(self.base_sensitivity_dir, "08_Sensitivity_Trends.png")
        plt.savefig(path)
