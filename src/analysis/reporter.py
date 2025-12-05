# reporter_refactor.py
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
import matplotlib.patches as mpatches
import seaborn as sns
import sys
from matplotlib.gridspec import GridSpec

# Add project root (if script run from subfolder)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import ProjectConfig as Cfg

# Optional pretty printing
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except Exception:
    HAS_TABULATE = False

# Avoid seaborn/matplotlib warnings for edgecolor
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class PipelineReporter:
    """
    Nâng cấp reporter: consistent visuals, improved charts, sensitivity integration, MC plots.
    Usage: reporter = PipelineReporter(); reporter.run()
    """

    def __init__(self,
                 out_dir: Optional[str] = None,
                 sensitivity_csv: Optional[str] = None,
                 save_svg: bool = True):
        self.OUT_DIR = out_dir or Cfg.OUT_DIR_ANALYSIS
        os.makedirs(self.OUT_DIR, exist_ok=True)
        self.OUT_DIR_LOGISTICS = getattr(Cfg, "OUT_DIR_LOGISTICS", os.path.join(self.OUT_DIR, "logistics"))
        os.makedirs(self.OUT_DIR_LOGISTICS, exist_ok=True)
        self.sensitivity_csv = sensitivity_csv  # optional override
        self.save_svg = save_svg

        self._set_style()

    # -----------------------
    # Styling & helpers
    # -----------------------
    def _set_style(self):
        """Set a consistent, publication-ready style."""
        sns.set_style("whitegrid")
        self.palette = sns.color_palette("colorblind")
        # accent colors
        self.color_proc = self.palette[0]  # procurement
        self.color_log = self.palette[1]   # logistics
        self.color_hold = self.palette[2]  # holding
        self.color_fresh = "#e74c3c"       # freshness (red accent)
        self.color_best = "#ffd700"        # gold for best

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 16,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'grid.linestyle': '--',
            'lines.linewidth': 2,
            'figure.dpi': 150,
        })

    def _savefig(self, fig, name: str, dpi=300):
        """Save PNG and optionally SVG for publication."""
        png_path = os.path.join(self.OUT_DIR, f"{name}.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        if self.save_svg:
            svg_path = os.path.join(self.OUT_DIR, f"{name}.svg")
            fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {png_path}")

    # -----------------------
    # Main runner
    # -----------------------
    def run(self):
        print("[Reporter] Generating visuals...")
        self._analyze_reconstruction()
        self._analyze_forecast()
        self._analyze_inventory()

        opt_path = os.path.join(self.OUT_DIR, "integrated_optimization_results.csv")
        if os.path.exists(opt_path):
            df_res = pd.read_csv(opt_path)
            self._plot_cost_tradeoff_detailed(df_res)
            self._plot_cost_heatmaps(df_res)
            self._plot_logistics_gantt_chart(df_res)
            self._generate_summary_table(df_res)
        else:
            print("[Warn] integrated_optimization_results.csv not found in OUT_DIR. Skipping optimization figs.")

        # Sensitivity (try to load CSV automatically)
        sens_path = self.sensitivity_csv or os.path.join(self.OUT_DIR, "sensitivity_strategic_robustness.csv")
        if os.path.exists(sens_path):
            try:
                df_sens = pd.read_csv(sens_path)
                # Normalize column names (strip BOM / weird spaces)
                df_sens.columns = [c.strip() for c in df_sens.columns]
                self._plot_tornado_chart(df_sens)
                self._plot_sensitivity_trends(df_sens)
            except Exception as e:
                print(f"[Warn] cannot load sensitivity CSV: {e}")
        else:
            print(f"[Info] sensitivity file not found at {sens_path}. Skipping sensitivity figs.")

        # Monte Carlo winner distribution (if exists)
        mc_path = os.path.join(self.OUT_DIR, "mc_results.csv")
        if os.path.exists(mc_path):
            self._plot_montecarlo_summary(mc_path)

        print("[Reporter] Done. Outputs in:", self.OUT_DIR)

    # -----------------------
    # Part 1: Data & Forecasting
    # -----------------------
    def _analyze_reconstruction(self):
        path = os.path.join(Cfg.OUT_DIR_PART2, "reconstruction_accuracy_by_product.csv")
        if not os.path.exists(path):
            print("[Info] reconstruction file missing:", path); return
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(10, 6))

        # safe columns
        if 'Mean_Sales' not in df.columns or 'Recon_MAPE' not in df.columns:
            print("[Warn] reconstruction file missing required columns"); return

        # bubble size scaled (use RMSE if present)
        sizes = (df['Recon_RMSE'] if 'Recon_RMSE' in df.columns else df['Mean_Sales']).fillna(1).values
        # avoid degenerate
        if sizes.max() == sizes.min():
            sizes = np.full_like(sizes, 50)
        else:
            sizes = np.interp(sizes, (sizes.min(), sizes.max()), (30, 400))

        sc = ax.scatter(df['Mean_Sales'], df['Recon_MAPE'], s=sizes,
                        c=df['Recon_MAPE'], cmap="viridis_r", alpha=0.75, edgecolors='k', linewidth=0.3)

        ax.set_xscale('log')
        ax.set_xlabel("Mean Daily Sales (log scale)")
        ax.set_ylabel("Reconstruction Error (MAPE %)")
        ax.set_title("Data Quality — Demand Reconstruction", pad=8)

        # regression fit to show trend
        try:
            mask = np.isfinite(df['Mean_Sales']) & np.isfinite(df['Recon_MAPE']) & (df['Mean_Sales'] > 0)
            if mask.sum() >= 3:
                coef = np.polyfit(np.log(df.loc[mask, 'Mean_Sales']), df.loc[mask, 'Recon_MAPE'], 1)
                x_ = np.logspace(np.log10(df.loc[mask, 'Mean_Sales'].min()), np.log10(df.loc[mask, 'Mean_Sales'].max()), 50)
                ax.plot(x_, np.polyval(coef, np.log(x_)), color='black', linestyle='--', lw=1.5, label='Trend')
                ax.legend()
        except Exception:
            pass

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("MAPE (%)")
        # annotate top-k outliers
        if 'product_id' in df.columns or 'sku' in df.columns:
            df['score'] = df['Recon_MAPE'] * np.log1p(df['Mean_Sales'])
            for _, r in df.nlargest(6, 'score').iterrows():
                ax.annotate(str(r.get('product_id', r.get('sku', ''))),
                            (r['Mean_Sales'], r['Recon_MAPE']), xytext=(5, 5), textcoords='offset points', fontsize=9)

        self._savefig(fig, "01_Data_Reconstruction_Quality")

    def _analyze_forecast(self):
        path = os.path.join(Cfg.OUT_DIR_FORECAST, "per_horizon_metrics.csv")
        if not os.path.exists(path):
            print("[Info] forecast metrics missing:", path); return
        df = pd.read_csv(path)
        if 'horizon' not in df.columns or 'WAPE' not in df.columns or 'RMSE' not in df.columns:
            print("[Warn] per_horizon_metrics missing columns"); return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df['horizon'], df['WAPE'], marker='o', label='WAPE (%)', color=self.color_proc)
        ax1.set_xlabel("Forecast Horizon (days)")
        ax1.set_ylabel("WAPE (%)", color=self.color_proc)
        ax1.tick_params(axis='y', labelcolor=self.color_proc)

        ax2 = ax1.twinx()
        ax2.plot(df['horizon'], df['RMSE'], marker='s', linestyle='--', label='RMSE', color=self.color_log)
        ax2.set_ylabel("RMSE (units)", color=self.color_log)
        ax2.tick_params(axis='y', labelcolor=self.color_log)

        # Optional: plot WAPE baseline if present
        if 'WAPE_CI_low' in df.columns and 'WAPE_CI_high' in df.columns:
            ax1.fill_between(df['horizon'], df['WAPE_CI_low'], df['WAPE_CI_high'], color=self.color_proc, alpha=0.15)

        # legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        ax1.set_title("Forecast Performance vs Horizon")
        self._savefig(fig, "02_Forecast_Performance")

    def _analyze_inventory(self):
        path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        if not os.path.exists(path):
            print("[Info] inventory artifact missing:", path); return
        df = pd.read_parquet(path)

        # pick columns
        if 'predicted_mean' not in df.columns or 'safety_stock_kg' not in df.columns:
            print("[Warn] unified_for_procurement_enhanced.parquet missing columns"); return

        # Create GridSpec with reserved column for colorbar and a small inset for histogram
        fig = plt.figure(figsize=(11, 6))
        gs = GridSpec(1, 10, figure=fig)
        ax = fig.add_subplot(gs[:, :8])        # main scatter
        cbar_ax = fig.add_subplot(gs[:, 8:9])  # colorbar column
        inset_ax = fig.add_subplot(gs[0, 9])   # small inset for histogram (far right)

        # scatter
        cmap = plt.get_cmap('rocket_r')
        norms = df.get('predicted_std', df['predicted_mean'])
        # handle nan
        norms = norms.fillna(norms.median()) if hasattr(norms, "fillna") else norms
        sc = ax.scatter(df['predicted_mean'], df['safety_stock_kg'],
                        c=norms, cmap=cmap, alpha=0.82, edgecolors='k', linewidth=0.25, s=50)

        ax.set_xscale('symlog')  # robust to zero
        ax.set_xlabel('Predicted Daily Demand (units)', fontsize=12)
        ax.set_ylabel('Safety Stock (kg)', fontsize=12)
        ax.set_title('Inventory Policy — Demand vs Safety Stock', fontsize=16)

        # colorbar put in cbar_ax (vertical)
        cb = fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
        cb.set_label('Predicted Std', fontsize=10)

        # inset histogram (small) for predicted_mean distribution
        inset_ax.hist(df['predicted_mean'].clip(lower=0.1), bins=30)
        inset_ax.set_title('Demand dist', fontsize=9)
        inset_ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        out = os.path.join(self.OUT_DIR, "03_Inventory_Safety_Logic.png")
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved:", out)

    # -----------------------
    # Part 2: Optimization visuals
    # -----------------------
    def _plot_cost_tradeoff_detailed(self, df: pd.DataFrame):
        """Stacked component bars + total line + highlight best."""
        if df.empty: return
        df = df.copy()
        df['scenario_label'] = [f"P={int(p)}\nU={int(u)}" for p, u in zip(df['P_lim'], df['U_lim'])]

        # ensure columns exist
        comp_cols = {
            'Daily_Procurement_Cost': self.color_proc,
            'Daily_Distribution_Cost': self.color_log,
            'Daily_Holding_Cost': self.color_hold,
            'Daily_Freshness_Penalty': self.color_fresh
        }
        present = [c for c in comp_cols.keys() if c in df.columns]

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(df))
        bottom = np.zeros(len(df))
        for c in present:
            vals = df[c].fillna(0).values
            ax.bar(x, vals, bottom=bottom, label=c.replace('Daily_', '').replace('_', ' '),
                   color=comp_cols[c], edgecolor='white')
            bottom += vals

        # total line
        if 'Total_Daily_Cost' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(x, df['Total_Daily_Cost'], color='k', marker='o', lw=2.5, label='Total Daily Cost')
            # annotate totals
            for xi, v in zip(x, df['Total_Daily_Cost']):
                ax2.text(xi, v * 1.02, f"${v:,.0f}", ha='center', va='bottom', fontsize=9)

            # highlight best
            best_idx = int(df['Total_Daily_Cost'].idxmin())
            ax2.scatter([best_idx], [df['Total_Daily_Cost'].iloc[best_idx]], s=220,
                        marker='*', color=self.color_best, edgecolors='black', zorder=5)

            ax2.set_ylabel("Total Daily Cost ($)", color='k')
            ax2.tick_params(axis='y', labelcolor='k')

        ax.set_xticks(x)
        ax.set_xticklabels(df['scenario_label'], rotation=0)
        ax.set_ylabel("Component Costs ($)")
        ax.set_title("Strategic Cost Trade-off")

        # smart legend: show friendly names
        handles, labels = ax.get_legend_handles_labels()
        if 'Total_Daily_Cost' in df.columns:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2; labels += labels2
        ax.legend(handles, [l.replace('Daily_', '') for l in labels], loc='upper center', bbox_to_anchor=(0.5, 1.12),
                  ncol=min(4, len(labels)), frameon=False)

        self._savefig(fig, "04_Integrated_Tradeoff")

    def _plot_cost_heatmaps(self, df: pd.DataFrame):
        """Pivot P_lim x U_lim and show Total cost + component breakdown heatmaps."""
        if df.empty: return
        # pivot total cost
        if 'Total_Daily_Cost' in df.columns:
            pivot = df.pivot_table(index='P_lim', columns='U_lim', values='Total_Daily_Cost', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
            ax.set_title("Total Daily Cost (P x U)")
            ax.set_xlabel("U (Sales cycle)")
            ax.set_ylabel("P (Proc. days)")
            self._savefig(fig, "06_Cost_Heatmap_Total")

        # also heatmap of component intensity (normalized)
        comps = [c for c in ['Daily_Procurement_Cost', 'Daily_Distribution_Cost', 'Daily_Holding_Cost', 'Daily_Freshness_Penalty'] if c in df.columns]
        if comps:
            norm_df = df.copy()
            # normalize per row to % of total
            norm_df['row_total'] = norm_df[comps].sum(axis=1).replace(0, np.nan)
            for c in comps:
                norm_df[c + "_pct"] = norm_df[c] / norm_df['row_total']
            # For clarity, save CSV of percentages
            norm_df.to_csv(os.path.join(self.OUT_DIR, "component_percentages_by_scenario.csv"), index=False)

    def _plot_logistics_gantt_chart(self, df_res: pd.DataFrame):
        """Gantt-style timeline for the selected best scenario. Robust parsing for steps field."""
        if df_res.empty: return
        best = df_res.loc[df_res['Total_Daily_Cost'].idxmin()]
        p, u = int(best['P_lim']), int(best['U_lim'])
        # find route file matching best scenario, fallback to any routes_ file
        fname = f"routes_P{p}_U{u}.csv"
        route_path = os.path.join(self.OUT_DIR_LOGISTICS, fname)
        if not os.path.exists(route_path):
            files = glob.glob(os.path.join(self.OUT_DIR_LOGISTICS, "routes_P*_U*.csv"))
            if files:
                route_path = sorted(files)[0]
            else:
                print("[Info] No route files found"); return

        df_routes = pd.read_csv(route_path)
        # ensure steps parsed
        if 'steps' in df_routes.columns:
            def safe_parse(s):
                if pd.isna(s): return []
                if isinstance(s, (list, dict)): return s
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return []
            df_routes['steps'] = df_routes['steps'].apply(safe_parse)
        else:
            print("[Warn] routes file missing 'steps' column")
            return

        fig, ax = plt.subplots(figsize=(14, max(6, len(df_routes) * 0.5)))
        # warehouse window highlight
        wh_open, wh_close = getattr(Cfg, "WAREHOUSE_WINDOW", (360, 1020))  # minutes
        ax.axvspan(wh_open / 60.0, wh_close / 60.0, color='gray', alpha=0.08, label='Warehouse Window')

        y_ticks = []
        y_labels = []
        for i, row in df_routes.iterrows():
            steps = row['steps']
            role = row.get('role', 'Unknown')
            vid = row.get('vehicle_id', i)
            y_pos = i
            y_ticks.append(y_pos); y_labels.append(f"{role} #{vid}")
            col = self.color_log if str(role).lower() != 'inbound' else self.color_proc
            if not steps:
                continue
            start = steps[0].get('arrival_time_min', 0)
            end = steps[-1].get('arrival_time_min', start)
            ax.barh(y_pos, (end - start) / 60.0, left=start / 60.0, height=0.4, color=col, edgecolor='k', alpha=0.9)
            for s in steps:
                t = s.get('arrival_time_min', None)
                if t is not None:
                    ax.scatter(t / 60.0, y_pos, color='white', edgecolor='k', s=30, zorder=5)

        # show crossdock ready time if inbound exist
        inbound = df_routes[df_routes.get('role', '') == 'Inbound'] if 'role' in df_routes.columns else df_routes
        if not inbound.empty:
            try:
                last_in = max(r['steps'][-1]['arrival_time_min'] for _, r in inbound.iterrows() if r['steps'])
                ready = last_in + getattr(Cfg, "SERVICE_TIME_CROSSDOCK_MINS", 30)
                ax.axvline(ready / 60.0, color='purple', linestyle='--', lw=2, label='Outbound Ready')
            except Exception:
                pass

        ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time of day (hours)")
        ax.set_title(f"Logistics Timeline — Scenario P={p}, U={u}")
        max_h = max((r['steps'][-1].get('arrival_time_min', 0) if r['steps'] else 0) for _, r in df_routes.iterrows()) / 60.0
        ax.set_xlim(0, max(30, max_h + 2))
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.legend(loc='upper right')
        self._savefig(fig, "05_Logistics_Timeline")

    def _generate_summary_table(self, df: pd.DataFrame):
        # friendly mapping and write executive text
        cols_map = {
            'P_lim': 'Proc. Days', 'U_lim': 'Sales Cycle',
            'Daily_Procurement_Cost': 'Procurement ($)',
            'Daily_Distribution_Cost': 'Logistics ($)',
            'Daily_Holding_Cost': 'Holding ($)',
            'Daily_Freshness_Penalty': 'Freshness Pen. ($)',
            'Total_Daily_Cost': 'TOTAL ($)',
            'Is_Next_Day': 'Next Day?'
        }
        avail = [c for c in cols_map if c in df.columns]
        disp = df[avail].copy().rename(columns=cols_map)
        if 'TOTAL ($)' in disp.columns:
            disp['Rank'] = disp['TOTAL ($)'].rank(method='min').astype(int)
            cols = ['Rank'] + [c for c in disp.columns if c != 'Rank']
            disp = disp[cols].sort_values('Rank')

        # format currency
        for c in disp.columns:
            if '($)' in c:
                disp[c] = disp[c].apply(lambda x: f"${x:,.2f}")

        txt = tabulate(disp, headers='keys', tablefmt='github', showindex=False) if HAS_TABULATE else disp.to_string(index=False)
        out = os.path.join(self.OUT_DIR, "00_Executive_Summary.txt")
        with open(out, 'w', encoding='utf-8') as f:
            f.write("FRESH RETAIL SIMULATION RESULTS\n")
            f.write("===============================\n\n")
            f.write(txt)
            f.write("\n\n")
            if not df.empty and 'Total_Daily_Cost' in df.columns:
                best = df.loc[df['Total_Daily_Cost'].idxmin()]
                f.write(f"* OPTIMAL STRATEGY: P={int(best['P_lim'])}, U={int(best['U_lim'])}.\n")
        print("Saved executive summary:", out)

    # -----------------------
    # Sensitivity visuals
    # -----------------------
    def _plot_tornado_chart(self, df_sens: pd.DataFrame):
        """
        Improved tornado: draw left-bar = baseline - min (if >0), right-bar = max - baseline (if >0).
        Bars centered visually on baseline so baseline always meaningful.
        """
        if df_sens is None or df_sens.empty:
            print("[Info] sensitivity dataframe empty"); return

        if 'Dimension' not in df_sens.columns:
            print("[Warn] sensitivity file missing 'Dimension'"); return

        # Detect a numeric cost column to use
        cost_col = None
        candidates = ['Min_Total_Cost', 'Total', 'Total_Daily_Cost', 'MinCost', 'Min Cost']
        for c in candidates:
            if c in df_sens.columns:
                cost_col = c
                break
        if cost_col is None:
            # fallback: pick last numeric column excluding 'Value'
            numeric_cols = [c for c in df_sens.columns if np.issubdtype(df_sens[c].dtype, np.number) and c not in ('Value',)]
            if numeric_cols:
                cost_col = numeric_cols[-1]
            else:
                print("[Warn] cannot find numeric cost column in sensitivity CSV"); return

        # compute per-dimension min/max/swing
        summary = df_sens.groupby('Dimension')[cost_col].agg(['min', 'max']).rename(columns={'min':'min','max':'max'})
        summary['swing'] = summary['max'] - summary['min']
        summary = summary.sort_values('swing', ascending=False)  # most influential on top

        if summary.empty:
            print("[Info] no summary to plot"); return

        # baseline: prefer integrated results file's best cost; fallback to median(min)
        baseline = None
        try:
            opt_path = os.path.join(self.OUT_DIR, "integrated_optimization_results.csv")
            if os.path.exists(opt_path):
                df_opt = pd.read_csv(opt_path)
                if 'Total_Daily_Cost' in df_opt.columns:
                    baseline = df_opt['Total_Daily_Cost'].min()
        except Exception:
            baseline = None
        if baseline is None:
            baseline = float(summary['min'].median())

        # prepare plotting arrays
        dims = summary.index.tolist()
        y = np.arange(len(dims))
        min_vals = summary['min'].values
        max_vals = summary['max'].values
        left_widths = np.maximum(0.0, baseline - min_vals)   # length to left
        left_starts = baseline - left_widths                 # equals min_vals when baseline>=min
        right_widths = np.maximum(0.0, max_vals - baseline)  # length to right
        right_starts = np.full_like(right_widths, baseline, dtype=float)

        fig, ax = plt.subplots(figsize=(10, max(4, len(dims)*0.6)))

        # left bars (reduction potential) and right bars (risk)
        left_bars = ax.barh(y, left_widths, left=left_starts, color=self.color_proc, alpha=0.9, label='Cost Reduction Potential')
        right_bars = ax.barh(y, right_widths, left=right_starts, color=self.color_fresh, alpha=0.9, label='Cost Risk Exposure')

        # baseline vertical line
        ax.axvline(baseline, color='k', linestyle='--', lw=1.2, label='Baseline')

        # labels and ticks
        ax.set_yticks(y); ax.set_yticklabels(dims, fontsize=10)
        ax.set_xlabel('Daily Cost ($)')
        ax.set_title('Sensitivity Tornado (impact relative to baseline)', fontsize=14)

        # format x ticks as dollars
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

        # annotate min and max at ends
        for yi, (mn, mx, lw, rw) in enumerate(zip(min_vals, max_vals, left_widths, right_widths)):
            # annotate min at leftmost edge (if left_width > 0)
            if lw > 0:
                ax.text(mn, yi, f"${int(mn):,}", va='center', ha='right', fontsize=9, color='white' if lw > (0.08*(summary['max'].max()-summary['min'].min()+1e-9)) else 'black', weight='bold')
            # annotate max at rightmost edge (if right_width > 0)
            if rw > 0:
                ax.text(mx, yi, f"${int(mx):,}", va='center', ha='left', fontsize=9, color='white' if rw > (0.08*(summary['max'].max()-summary['min'].min()+1e-9)) else 'black', weight='bold')

        # show legend on top-right
        ax.legend(loc='upper right')
        plt.tight_layout()
        self._savefig(fig, "08_Tornado_Chart_improved")

    def _plot_sensitivity_trends(self, df_sens: pd.DataFrame):
        """Guaranteed no-overlap version: large top margin + layout rect."""
        if df_sens.empty:
            return
        if "Dimension" not in df_sens.columns:
            print("[Warn] missing Dimension column"); return

        val_col = "Value" if "Value" in df_sens.columns else None
        cost_col = "Min_Total_Cost" if "Min_Total_Cost" in df_sens.columns else (
            "Total" if "Total" in df_sens.columns else None
        )

        if val_col is None or cost_col is None:
            print("[Warn] missing Value or Cost column"); return

        df = df_sens.copy()
        df["Value_num"] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["Value_num", cost_col])

        dims = sorted(df["Dimension"].unique())
        n = len(dims)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.8), squeeze=False)

        for i, dim in enumerate(dims):
            ax = axes[i // cols, i % cols]
            sub = df[df["Dimension"] == dim].sort_values("Value_num")

            ax.plot(sub["Value_num"], sub[cost_col],
                    marker="o", linewidth=2, color=self.color_proc)

            ax.set_title(dim, fontsize=10, pad=6)
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("Cost ($)")
            ax.grid(True, linestyle="--", alpha=0.25)

            # Nice x-axis formatting
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, pos: f"{x:g}")
            )

        # Remove unused axes
        for j in range(len(dims), rows * cols):
            fig.delaxes(axes[j // cols, j % cols])

        # === The magic: reserve huge space for suptitle ===
        fig.suptitle("Sensitivity Trends by Dimension",
                    fontsize=16,
                    y=1.05,           # way above normal
                    weight="bold")

        # Make room for suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.subplots_adjust(top=0.88, hspace=0.40)

        out = os.path.join(self.OUT_DIR, "09_Sensitivity_Trends.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)


    
    # -----------------------
    # Monte Carlo summary
    # -----------------------
    def _plot_montecarlo_summary(self, mc_csv_path: str):
        """
        Expected mc CSV with columns: sim_id, scenario_label (or P_lim/U_lim), Total_Daily_Cost
        """
        df = pd.read_csv(mc_csv_path)
        if df.empty or 'Total_Daily_Cost' not in df.columns:
            print("[Warn] MC file invalid"); return

        # winner counts per scenario
        winner = df.loc[df.groupby('sim_id')['Total_Daily_Cost'].idxmin()]
        if 'scenario_label' not in winner.columns:
            # create scenario_label from P_lim/U_lim if present
            if 'P_lim' in winner.columns and 'U_lim' in winner.columns:
                winner['scenario_label'] = winner.apply(lambda r: f"P={int(r['P_lim'])}_U={int(r['U_lim'])}", axis=1)
            else:
                winner['scenario_label'] = 'scenario'

        win_counts = winner['scenario_label'].value_counts().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # histogram of total costs (all sims)
        sns.histplot(df, x='Total_Daily_Cost', hue='scenario_label', element='step', stat='density', common_norm=False, ax=axes[0])
        axes[0].set_title("Monte Carlo: Total Cost Distribution by Scenario")
        axes[0].xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

        # bar: winner freq
        win_counts.plot(kind='bar', ax=axes[1], color=self.palette)
        axes[1].set_title("Monte Carlo: Win counts per scenario")
        axes[1].set_ylabel("Number of simulations won")

        self._savefig(fig, "10_MonteCarlo_Summary")


# end of file
