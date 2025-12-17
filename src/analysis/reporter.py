# src/analysis/reporter.py
import os
import ast
import glob
import math
import warnings
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import sys
from matplotlib.gridspec import GridSpec

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import ProjectConfig as Cfg

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except Exception:
    HAS_TABULATE = False

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class PipelineReporter:
    """
    Enhanced Reporter with 'Zoom-In' visualizations for better insight extraction.
    """

    def __init__(self,
                 out_dir: Optional[str] = None,
                 sensitivity_csv: Optional[str] = None,
                 save_svg: bool = True):
        self.OUT_DIR = out_dir or Cfg.OUT_DIR_ANALYSIS
        os.makedirs(self.OUT_DIR, exist_ok=True)
        # Logistics maps usually in distinct folder
        self.OUT_DIR_LOGISTICS = getattr(Cfg, "OUT_DIR_LOGISTICS", os.path.join(os.path.dirname(self.OUT_DIR), "vrp_route_maps"))
        # Procurement dir (optional, mostly for temp files now)
        self.PROCUREMENT_DIR = getattr(Cfg, "OUT_DIR_PROCUREMENT", os.path.join(os.path.dirname(self.OUT_DIR), "procurement"))
        
        self.sensitivity_csv = sensitivity_csv
        self.save_svg = save_svg
        self._set_style()

    def _set_style(self):
        sns.set_style("whitegrid")
        self.palette = sns.color_palette("deep") 
        
        self.colors = {
            'Procurement': '#1f77b4', # Blue
            'Logistics': '#ff7f0e',   # Orange
            'Holding': '#2ca02c',     # Green
            'Freshness': '#d62728',   # Red
            'Total': '#9467bd'        # Purple
        }

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'figure.dpi': 150,
        })

    def _savefig(self, fig, name: str, dpi=300):
        png_path = os.path.join(self.OUT_DIR, f"{name}.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        if self.save_svg:
            svg_path = os.path.join(self.OUT_DIR, f"{name}.svg")
            fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {png_path}")

    def run(self):
        print("[Reporter] Generating visuals...")
        
        self._analyze_reconstruction()
        self._analyze_forecast()
        
        # Optimization Results
        opt_path = os.path.join(self.OUT_DIR, "integrated_optimization_results.csv")
        if os.path.exists(opt_path):
            df_res = pd.read_csv(opt_path)
            
            # 1. Trade-off Analysis
            self._plot_cost_tradeoff_split(df_res) 
            
            # 2. Heatmaps
            self._plot_cost_heatmaps(df_res)
            
            # 3. Logistics
            self._plot_logistics_gantt_chart(df_res)
            
            # 4. Sourcing Analysis [FIXED PATH]
            self._plot_sourcing_comparison(df_res) 
            
            # 5. Summary Table
            self._generate_summary_table(df_res)
        else:
            print("[Warn] integrated_optimization_results.csv not found.")

        print("[Reporter] Done.")

    # ------------------------------------------------------
    # 1. Split Panel Trade-off Chart
    # ------------------------------------------------------
    def _plot_cost_tradeoff_split(self, df: pd.DataFrame):
        if df.empty: return
        df = df.copy()
        
        if 'Strategy' in df.columns:
            df['label'] = df['Strategy']
        else:
            df['label'] = [f"P={p}, U={u}" for p, u in zip(df['P_lim'], df['U_lim'])]

        cols = {
            'Procurement': 'Daily_Procurement_Cost',
            'Logistics': 'Daily_Distribution_Cost',
            'Holding': 'Daily_Holding_Cost',
            'Freshness': 'Daily_Freshness_Penalty',
            'Total': 'Total_Daily_Cost'
        }
        
        # Ensure columns exist
        for k, v in cols.items():
            if v not in df.columns: df[v] = 0.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
        plt.subplots_adjust(hspace=0.1)
        
        x = np.arange(len(df))
        width = 0.6
        
        # --- TOP PANEL ---
        ax1.plot(x, df[cols['Total']], marker='o', markersize=10, color='black', linewidth=2, label='Total Daily Cost', zorder=10)
        ax1.bar(x, df[cols['Procurement']], width=width, color=self.colors['Procurement'], alpha=0.6, label='Procurement Cost')
        
        for i, v in enumerate(df[cols['Total']]):
            ax1.text(i, v + (v*0.01), f"${v/1000:.1f}k", ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax1.set_ylabel("Major Costs ($)")
        ax1.set_title("Strategic Cost Analysis: Total vs. Operational Trade-offs", pad=20)
        ax1.legend(loc='upper right')
        ax1.grid(axis='x')
        
        # --- BOTTOM PANEL ---
        w = 0.25
        ax2.bar(x - w, df[cols['Logistics']], width=w, label='Logistics', color=self.colors['Logistics'])
        ax2.bar(x, df[cols['Holding']], width=w, label='Holding', color=self.colors['Holding'])
        ax2.bar(x + w, df[cols['Freshness']], width=w, label='Freshness Penalty', color=self.colors['Freshness'])
        
        ax2.set_ylabel("Operational Trade-offs ($)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['label'], fontweight='bold')
        ax2.legend(loc='upper left', title="Operational Components")
        
        # Highlight Winner
        if not df[cols['Total']].empty:
            best_idx = df[cols['Total']].idxmin()
            ax1.axvspan(best_idx - 0.4, best_idx + 0.4, color='gold', alpha=0.1, zorder=0)
            ax2.axvspan(best_idx - 0.4, best_idx + 0.4, color='gold', alpha=0.1, zorder=0)
            ax2.text(best_idx, ax2.get_ylim()[1]*0.95, "WINNER", ha='center', color='goldenrod', fontweight='bold')

        self._savefig(fig, "04_Strategic_Tradeoff_Split")

    # ------------------------------------------------------
    # 2. UPGRADED: Sourcing Comparison (Stacked Bar) - FIXED
    # ------------------------------------------------------
    def _plot_sourcing_comparison(self, df_res: pd.DataFrame):
        """
        Compare where each strategy buys from.
        [FIXED] Looks in self.OUT_DIR (analysis folder) for plan files.
        """
        if df_res.empty: return
        
        strategies = []
        zone_data = {} 
        
        # Load supplier info
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        if not os.path.exists(sup_path): 
            print(f"[Warn] Suppliers file missing: {sup_path}")
            return
            
        df_sup = pd.read_csv(sup_path)
        if 'zone_label' not in df_sup.columns: df_sup['zone_label'] = 'Unknown'
        df_sup['supplier_id'] = df_sup['supplier_id'].astype(int)
        sup_map = df_sup.set_index('supplier_id')['zone_label'].to_dict()

        found_any = False
        print("[Reporter] Analyzing Sourcing Mix...")

        for _, row in df_res.iterrows():
            strat = row.get('Strategy', f"P{row['P_lim']}U{row['U_lim']}")
            strategies.append(strat)
            p, u = int(row['P_lim']), int(row['U_lim'])
            
            # [FIXED PATH] Look in OUT_DIR (Analysis folder) where IntegratedSolver saves
            filename = f"procurement_plan_{strat}_P{p}_U{u}.csv"
            plan_path = os.path.join(self.OUT_DIR, filename)
            
            # Fallback to Procurement Dir just in case
            if not os.path.exists(plan_path):
                plan_path = os.path.join(self.PROCUREMENT_DIR, filename)
            
            # Fallback filename pattern
            if not os.path.exists(plan_path):
                plan_path = os.path.join(self.OUT_DIR, f"procurement_plan_P{p}_U{u}.csv")

            if os.path.exists(plan_path):
                try:
                    df_plan = pd.read_csv(plan_path)
                    if df_plan.empty: continue
                    
                    df_plan['supplier_id'] = df_plan['supplier_id'].astype(int)
                    df_plan['zone'] = df_plan['supplier_id'].map(sup_map).fillna('Unknown')
                    
                    # Aggregate Volume
                    vol_by_zone = df_plan.groupby('zone')['order_weight_kg'].sum().to_dict()
                    zone_data[strat] = vol_by_zone
                    found_any = True
                except Exception: pass
            else:
                zone_data[strat] = {}

        if not found_any or not zone_data: 
            print("   [Warn] No valid plan files found in Analysis directory.")
            return

        # Plot
        df_plot = pd.DataFrame(zone_data).T.fillna(0)
        if df_plot.empty or df_plot.shape[1] == 0: return

        # Normalize to 100%
        row_sums = df_plot.sum(axis=1)
        df_plot_pct = df_plot.div(row_sums.replace(0, 1), axis=0) * 100
        
        if df_plot_pct.select_dtypes(include=np.number).empty: return

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_plot_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', width=0.7)
            
            ax.set_ylabel("Volume Share (%)")
            ax.set_title("Sourcing Strategy Mix: Supplier Zones")
            ax.legend(title="Supplier Zone", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            
            for c in ax.containers:
                labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]
                ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', weight='bold')

            self._savefig(fig, "07_Sourcing_Comparison")
        except Exception as e:
            print(f"   [Error] Plotting failed: {e}")
            plt.close(fig)

    # ------------------------------------------------------
    # 3. Heatmaps
    # ------------------------------------------------------
    def _plot_cost_heatmaps(self, df: pd.DataFrame):
        if df.empty: return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total Cost
        piv_total = df.pivot_table(index='P_lim', columns='U_lim', values='Total_Daily_Cost', aggfunc='mean')
        sns.heatmap(piv_total, annot=True, fmt=".0f", cmap="viridis_r", ax=ax1, cbar_kws={'label': 'Cost ($)'})
        ax1.set_title("Total Daily Cost ($)")
        
        # Freshness Penalty
        if 'Daily_Freshness_Penalty' in df.columns:
            piv_fresh = df.pivot_table(index='P_lim', columns='U_lim', values='Daily_Freshness_Penalty', aggfunc='mean')
            sns.heatmap(piv_fresh, annot=True, fmt=".0f", cmap="Reds", ax=ax2, cbar_kws={'label': 'Penalty ($)'})
            ax2.set_title("Freshness Penalty ($)")
        
        self._savefig(fig, "06_Strategic_Heatmaps")

    # ------------------------------------------------------
    # 4. Logistics Timeline
    # ------------------------------------------------------
    def _plot_logistics_gantt_chart(self, df_res: pd.DataFrame):
        if df_res.empty: return
        best = df_res.loc[df_res['Total_Daily_Cost'].idxmin()]
        strategy_name = best.get('Strategy', 'Unknown')
        
        # [FIX PATH] Logistics logs are typically in OUT_DIR_LOGISTICS
        route_path = os.path.join(self.OUT_DIR_LOGISTICS, f"routes_{strategy_name}.csv")
        
        if not os.path.exists(route_path): return

        df_routes = pd.read_csv(route_path)
        if 'steps' not in df_routes.columns: return
        df_routes['steps'] = df_routes['steps'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
        
        df_routes['start_time'] = df_routes['steps'].apply(lambda x: x[0].get('arrival_time_min', 0) if x else 0)
        df_routes = df_routes.sort_values('start_time')

        fig, ax = plt.subplots(figsize=(12, max(5, len(df_routes)*0.4)))
        
        y_ticks, y_labels = [], []
        for i, (_, row) in enumerate(df_routes.iterrows()):
            steps = row['steps']
            if not steps: continue
            
            y_pos = i
            y_ticks.append(y_pos)
            veh_type = row.get('vehicle_type', 'Truck')
            load_pct = (row.get('total_load_kg', 0) / 1000) * 100 
            y_labels.append(f"{veh_type} #{i+1} ({load_pct:.0f}% Load)")
            
            start = steps[0].get('arrival_time_min', 0)
            end = steps[-1].get('arrival_time_min', start)
            
            col = self.colors['Procurement'] if str(row.get('role')).lower() == 'inbound' else self.colors['Logistics']
            
            ax.barh(y_pos, (end - start)/60, left=start/60, height=0.5, color=col, alpha=0.8, edgecolor='black')
            
            for s in steps:
                t = s.get('arrival_time_min')
                if t: 
                    marker = 's' if s.get('type') == 'SUPPLIER' else 'o'
                    ax.scatter(t/60, y_pos, color='white', edgecolor='k', marker=marker, s=30, zorder=5)

        ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time of Day (Hours)")
        ax.set_title(f"Logistics Execution Plan — {strategy_name}")
        
        wh_open, wh_close = getattr(Cfg, "WAREHOUSE_WINDOW", (360, 1020))
        ax.axvspan(wh_open/60, wh_close/60, color='gray', alpha=0.1, label='Warehouse Open')
        
        self._savefig(fig, "05_Logistics_Timeline")

    # ------------------------------------------------------
    # 5. Base Analytics
    # ------------------------------------------------------
    def _analyze_reconstruction(self):
        path = os.path.join(Cfg.OUT_DIR_PART2, "reconstruction_accuracy_by_product.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(10, 6))

        if 'Mean_Sales' not in df.columns: return
        sizes = (df['Recon_RMSE'] if 'Recon_RMSE' in df.columns else df['Mean_Sales']).fillna(1).values
        sizes = np.interp(sizes, (sizes.min(), sizes.max()), (30, 400))

        sc = ax.scatter(df['Mean_Sales'], df['Recon_MAPE'], s=sizes,
                        c=df['Recon_MAPE'], cmap="viridis_r", alpha=0.75, edgecolors='k', linewidth=0.3)

        ax.set_xscale('log')
        ax.set_xlabel("Mean Daily Sales (log)")
        ax.set_ylabel("Reconstruction Error (MAPE %)")
        ax.set_title("Data Quality — Demand Reconstruction")
        fig.colorbar(sc, ax=ax, label="MAPE (%)")
        
        # Add Quadrants
        ax.axhline(df['Recon_MAPE'].median(), color='k', linestyle='--', alpha=0.3)
        ax.axvline(df['Mean_Sales'].median(), color='k', linestyle='--', alpha=0.3)
        
        self._savefig(fig, "01_Data_Reconstruction_Quality")

    def _analyze_forecast(self):
        path = os.path.join(Cfg.OUT_DIR_FORECAST, "per_horizon_metrics.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df['horizon'], df['WAPE'], marker='o', label='WAPE (%)', color=self.colors['Procurement'])
        ax1.set_xlabel("Forecast Horizon (days)")
        ax1.set_ylabel("WAPE (%)", color=self.colors['Procurement'])
        
        ax2 = ax1.twinx()
        ax2.plot(df['horizon'], df['RMSE'], marker='s', linestyle='--', label='RMSE', color=self.colors['Logistics'])
        ax2.set_ylabel("RMSE (units)", color=self.colors['Logistics'])
        
        ax1.set_title("Forecast Performance vs Horizon")
        self._savefig(fig, "02_Forecast_Performance")

    def _generate_summary_table(self, df: pd.DataFrame):
        cols_map = {
            'Strategy': 'Strategy',
            'P_lim': 'P', 'U_lim': 'U',
            'Daily_Procurement_Cost': 'Procure($)',
            'Daily_Distribution_Cost': 'Logistics($)',
            'Daily_Freshness_Penalty': 'Freshness($)',
            'Total_Daily_Cost': 'TOTAL($)',
            'Is_Next_Day': 'NextDay'
        }
        avail = [c for c in cols_map if c in df.columns]
        disp = df[avail].copy().rename(columns=cols_map)
        
        for c in disp.columns:
            if '($)' in c:
                disp[c] = disp[c].apply(lambda x: f"${x:,.0f}")
        
        txt = tabulate(disp, headers='keys', tablefmt='github', showindex=False) if HAS_TABULATE else disp.to_string(index=False)
        
        out = os.path.join(self.OUT_DIR, "00_Executive_Summary.txt")
        with open(out, 'w', encoding='utf-8') as f:
            f.write(txt)
        print("Saved executive summary.")

if __name__ == "__main__":
    reporter = PipelineReporter()
    reporter.run()