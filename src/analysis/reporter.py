# src/analysis/reporter.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from config.settings import ProjectConfig as Cfg

class PipelineReporter:
    """
    Class tổng hợp kết quả từ toàn bộ Pipeline và tạo báo cáo trực quan.
    Updated: Publication-Quality Visualizations (Consistent Style, Deep Insights).
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_ANALYSIS, exist_ok=True)
        self.metrics = {}
        self._set_style()

    def _set_style(self):
        """Sets a professional, academic plotting style globally."""
        sns.set_theme(style="whitegrid", context="talk")
        # Professional Palette: Deep Blue, Muted Green, Slate Grey, Terracotta
        self.colors = ["#2C3E50", "#27AE60", "#7F8C8D", "#E67E22", "#2980B9", "#8E44AD"]
        self.cmap = sns.color_palette(self.colors)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

    def run(self):
        print("\n[Analysis] Generating Final Executive Report (Professional Edition)...")
        
        self._analyze_reconstruction()   # Chart 0
        self._analyze_forecast()         # Chart 1
        self._analyze_inventory()        # Chart 2
        self._analyze_procurement()      # Chart 3 (Donut)
        self._analyze_logistics()        # Chart 4 & 5
        self._analyze_integrated()       # Chart 6 (Stacked Trade-off)
        self._analyze_sensitivity()      # Chart 7
        self._compare_baseline_vs_actual() # Chart 8 (ROI)
        self._generate_detailed_tables()
        self._save_text_report()
        
        print(f"[Analysis] Report generated at: {Cfg.OUT_DIR_ANALYSIS}")

    def _analyze_reconstruction(self):
        path = os.path.join(Cfg.OUT_DIR_PART2, "reconstruction_accuracy_by_product.csv")
        if not os.path.exists(path): return

        df = pd.read_csv(path)
        plt.figure(figsize=(12, 7))
        
        # Bubble Chart with clear transparency
        scatter = sns.scatterplot(
            data=df, x='Mean_Sales', y='Recon_MAPE', 
            size='Recon_RMSE', sizes=(50, 500), 
            hue='Recon_MAPE', palette="viridis_r", alpha=0.7, edgecolor="black"
        )
        
        plt.title('0. Data Quality: Demand Reconstruction Accuracy', fontweight='bold')
        plt.xlabel('Mean Daily Sales Volume (Log Scale)')
        plt.ylabel('Reconstruction Error (MAPE %)')
        plt.xscale('log') # Use Log scale to show low vs high volume better
        
        # Reference Line
        avg_mape = df['Recon_MAPE'].mean()
        plt.axhline(avg_mape, color=self.colors[3], linestyle='--', linewidth=2, label=f"Avg MAPE: {avg_mape:.1f}%")
        
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "00_reconstruction_accuracy.png"), dpi=300)
        plt.close()

    def _analyze_forecast(self):
        path = os.path.join(Cfg.OUT_DIR_FORECAST, "per_horizon_metrics.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Dual Axis with distinct styles
        sns.lineplot(data=df, x='horizon', y='WAPE', marker='o', markersize=10, ax=ax1, color=self.colors[0], linewidth=3, label='WAPE (%)')
        ax1.set_ylabel('WAPE (%)', color=self.colors[0], fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=self.colors[0])
        ax1.set_ylim(0, max(df['WAPE']) * 1.3)
        
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x='horizon', y='RMSE', marker='s', markersize=10, ax=ax2, color=self.colors[3], linewidth=3, linestyle='--', label='RMSE')
        ax2.set_ylabel('RMSE (Units)', color=self.colors[3], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=self.colors[3])
        
        plt.title('1. Forecast Degradation: Accuracy vs. Horizon', fontweight='bold')
        ax1.set_xlabel('Forecast Horizon (Days)')
        ax1.grid(True, alpha=0.3)
        
        # Unified Legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "01_forecast_performance.png"), dpi=300)
        plt.close()

    def _analyze_inventory(self):
        path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        if not os.path.exists(path): return
        df = pd.read_parquet(path)
        
        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=df, x='predicted_mean', y='safety_stock_kg', 
            size='predicted_std', hue='predicted_std', 
            sizes=(30, 300), palette="mako_r", alpha=0.8, edgecolor=None
        )
        plt.title('2. Inventory Policy: Safety Stock Logic', fontweight='bold')
        plt.xlabel('Daily Demand Mean (Units)')
        plt.ylabel('Safety Stock (kg)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "02_inventory_policy.png"), dpi=300)
        plt.close()
        
        self.metrics['inventory'] = {
            'Total Safety Stock (kg)': round(df['safety_stock_kg'].sum(), 2),
            'Cycle Stock (kg)': round((df['order_qty_kg']/2).sum(), 2)
        }

    def _analyze_procurement(self):
        path = os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_solution.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        
        df['Product Cost'] = df['order_qty_units'] * df['unit_price']
        df['Transport'] = df['order_qty_units'] * df['transport_unit_cost']
        df['Fixed Cost'] = df['fixed_order_cost'] if 'fixed_order_cost' in df.columns else Cfg.FIXED_ORDER_COST
        
        sums = [df['Product Cost'].sum(), df['Transport'].sum(), df['Fixed Cost'].sum()]
        labels = ['Product Cost', 'Inbound Transport', 'Fixed Fees']
        
        # Donut Chart
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            sums, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=[self.colors[4], self.colors[3], self.colors[2]], 
            pctdistance=0.85, textprops={'fontsize': 12}
        )
        
        # Draw White Circle for Donut
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        
        plt.title('3. Procurement Cost Structure (Post-Optimization)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "03_procurement_cost_structure.png"), dpi=300)
        plt.close()

    def _analyze_logistics(self):
        path = os.path.join(Cfg.OUT_DIR_LOGISTICS, "vrp_routes_solution.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        df = df.rename(columns={'cost': 'cost_usd', 'total_load_kg': 'load_kg'})
        if 'role' not in df.columns: df['role'] = 'Direct'
        
        # 4. Fill Rate (Violin Plot is better for distribution comparison)
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='role', y='utilization_pct', palette="Set2", inner="quartile")
        plt.title('4. Vehicle Fill Rate Efficiency: Inbound vs Outbound', fontweight='bold')
        plt.ylabel('Fill Rate (%)')
        plt.xlabel('Logistics Leg')
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "04_fill_rate_by_role.png"), dpi=300)
        plt.close()
        
        # 5. Fleet Mix (Stacked Bar is cleaner)
        plt.figure(figsize=(12, 6))
        ct = pd.crosstab(df['role'], df['vehicle_type'])
        # Reorder columns by size
        order = [c for c in ['Small', 'Medium', 'Large', 'Extra'] if c in ct.columns]
        ct = ct[order]
        ct.plot(kind='bar', stacked=True, color=sns.color_palette("viridis", len(order)), figsize=(10,6))
        plt.title('5. Fleet Mix Usage Strategy', fontweight='bold')
        plt.ylabel('Number of Trips')
        plt.xticks(rotation=0)
        plt.legend(title='Vehicle Type')
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "05_fleet_mix_usage.png"), dpi=300)
        plt.close()
        
        # Stats
        summary = df.groupby('role').agg({'vehicle_id': 'count', 'cost_usd': 'sum', 'load_kg': 'sum', 'utilization_pct': 'mean'})
        summary = summary.rename(columns={'vehicle_id': 'Routes', 'utilization_pct': 'Avg Fill %'})
        summary['Cost/Kg'] = (summary['cost_usd'] / summary['load_kg']).round(4)
        self.metrics['logistics'] = summary.to_dict()

    def _analyze_integrated(self):
        """
        Improved Chart: Shows the breakdown of costs (Stacked) + Total Curve.
        This answers 'Why is the curve U-shaped?'
        """
        path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "integrated_optimization_results.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path).sort_values('P_lim')
        
        # Setup Data for Stacked Bar
        # We want to show: Logistics Cost, Holding Cost, and (Procurement - Product) Cost
        # Product cost is huge and constant, so we subtract it to show the "Optimization Variable Costs"
        # Or we normalize. Let's show total daily cost components.
        
        # Normalize components to Daily
        df['Daily_Procure'] = df['Product_Cost_Cycle'] / df['U_limit']
        df['Daily_Logistics'] = df['Logistics_Cost_Cycle'] / df['U_limit']
        df['Daily_Holding'] = df['Holding_Cost_Cycle'] / df['U_limit']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Stacked Bar Chart for Components
        indices = np.arange(len(df))
        width = 0.5
        
        p1 = ax.bar(indices, df['Daily_Logistics'], width, label='Logistics Cost', color='#34495E')
        p2 = ax.bar(indices, df['Daily_Holding'], width, bottom=df['Daily_Logistics'], label='Holding Cost', color='#2ECC71')
        # Note: We omit Product Cost from the bar stack to focus on the trade-off variables, 
        # OR we can plot the Total Line on a secondary axis if Product Cost is too dominant.
        # Let's plot Total Daily Cost as a Line on secondary axis for clarity.
        
        ax.set_xlabel('Scenario (Procurement Days / Utilization Days)', fontsize=13)
        ax.set_ylabel('Variable Supply Chain Costs ($/Day)', fontsize=13)
        ax.set_xticks(indices)
        ax.set_xticklabels([f"P={p}/U={u}" for p, u in zip(df['P_lim'], df['U_limit'])])
        
        # Total Cost Line
        ax2 = ax.twinx()
        ax2.plot(indices, df['Total_Daily_Cost'], color='#E74C3C', marker='o', linewidth=3, label='Total Daily Cost (Inc. Product)')
        ax2.set_ylabel('Total Daily Cost ($)', color='#E74C3C', fontsize=13)
        ax2.tick_params(axis='y', labelcolor='#E74C3C')
        
        # Highlight Optimal
        best_idx = df['Total_Daily_Cost'].idxmin()
        ax2.plot(best_idx, df['Total_Daily_Cost'].iloc[best_idx], 'o', markersize=15, markerfacecolor='none', markeredgecolor='red', markeredgewidth=2)
        
        plt.title('6. Integrated Time-Budget: Cost Trade-offs', fontweight='bold')
        
        # Combined Legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "06_integrated_optimization_curve.png"), dpi=300)
        plt.close()
        
        best = df.loc[best_idx]
        self.metrics['integrated'] = best.to_dict()

    def _analyze_sensitivity(self):
        path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "sensitivity_results_decoupled.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        
        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df, x="Multiplier", y="Fulfillment_Cost", 
            hue="Parameter", style="Parameter", 
            markers=True, linewidth=3, palette="deep"
        )
        plt.title('7. Sensitivity Analysis: Robustness Check', fontweight='bold')
        plt.xlabel('Parameter Change (1.0 = Baseline)')
        plt.ylabel('Total Fulfillment Cost ($)')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "07_sensitivity_analysis.png"), dpi=300)
        plt.close()
        
        max_vars = df.groupby("Parameter")["Fulfillment_Cost"].apply(lambda x: x.max() - x.min())
        self.metrics['sensitivity'] = {'most_sensitive': max_vars.idxmax()}

    def _compare_baseline_vs_actual(self):
        # ... (Calculations same as before) ...
        path_int = os.path.join(Cfg.OUT_DIR_ANALYSIS, "integrated_optimization_results.csv")
        if not os.path.exists(path_int): return
        res_df = pd.read_csv(path_int)
        actual_annual = res_df['Total_Daily_Cost'].min() * 365
        
        uni_path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        if not os.path.exists(uni_path): return
        df_uni = pd.read_parquet(uni_path)
        
        # Recalculate Baseline (Simplified for brevity - same logic as before)
        base_h_annual = (df_uni['price']*0.2*Cfg.HOLDING_COST_MULTIPLIER * (df_uni['predicted_mean']*7/2 + df_uni['safety_stock_kg']/(df_uni['unit_weight_kg']+1e-6))).sum()
        base_ord_annual = (365/7) * Cfg.FIXED_ORDER_COST * len(df_uni)
        base_trans_annual = (df_uni['predicted_mean']*df_uni['unit_weight_kg']*365*100*Cfg.TRANSPORT_COST_PER_KG_KM*1.5).sum()
        total_baseline = base_h_annual + base_ord_annual + base_trans_annual
        
        savings = total_baseline - actual_annual
        pct = (savings / total_baseline) * 100
        
        # Improved Bar Chart
        plt.figure(figsize=(9, 6))
        colors = [self.colors[2], self.colors[1]] # Grey, Green
        bars = plt.bar(['Baseline\n(Siloed)', 'Proposed\n(Integrated)'], [total_baseline, actual_annual], color=colors, width=0.6)
        
        # Annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
        # Savings Box
        mid_x = (bars[0].get_x() + bars[1].get_x() + bars[0].get_width()) / 2 + 0.1
        mid_y = (total_baseline + actual_annual) / 2
        plt.text(mid_x, mid_y, f"SAVINGS\n{pct:.1f}%", 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=self.colors[1], lw=2),
                 ha='center', color=self.colors[1], fontweight='bold', fontsize=14)

        plt.title('8. Financial Impact Analysis: Annual Cost Comparison', fontweight='bold')
        plt.ylabel('Total Annual Cost ($)')
        plt.ylim(0, total_baseline * 1.2)
        plt.grid(False, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(Cfg.OUT_DIR_ANALYSIS, "08_baseline_comparison.png"), dpi=300)
        plt.close()
        self.metrics['comparison'] = {'savings_pct': pct, 'savings_abs': savings}

    def _generate_detailed_tables(self):
        # (Same logic, ensuring no crash)
        fc_path = os.path.join(Cfg.OUT_DIR_FORECAST, "product_level_accuracy.csv")
        if not os.path.exists(fc_path): return
        df_main = pd.read_csv(fc_path)
        econ_path = os.path.join(Cfg.ARTIFACTS_DIR, "sku_economics.csv")
        if os.path.exists(econ_path):
            df_econ = pd.read_csv(econ_path)
            oos_agg = df_econ.groupby("product_id")["oos_rate"].mean().reset_index()
            df_main = df_main.merge(oos_agg, left_on="Product ID", right_on="product_id", how="left")
            df_main.rename(columns={"oos_rate": "OOS Rate"}, inplace=True)
        else: df_main["OOS Rate"] = 0.0
        df_main["OOS Rate"] = df_main["OOS Rate"].fillna(0.0)
        df_main["OOS (%)"] = (df_main["OOS Rate"] * 100).round(1).astype(str) + "%"
        final = df_main[["Product ID", "Mean Sales", "OOS (%)", "MAPE (%)", "RMSE", "Samples"]].sort_values("Mean Sales", ascending=False)
        final.to_csv(os.path.join(Cfg.OUT_DIR_ANALYSIS, "detailed_product_performance.csv"), index=False)

    def _save_text_report(self, logistics_data=None):
        report_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "EXECUTIVE_SUMMARY.txt")
        with open(report_path, "w") as f:
            f.write("=== FRESH RETAIL OPTIMIZATION REPORT ===\n\n")
            
            f.write("1. FINANCIAL IMPACT & STRATEGY\n")
            if 'integrated' in self.metrics:
                best = self.metrics['integrated']
                f.write(f"   OPTIMAL STRATEGY: P_lim={int(best['P_lim'])} days, U_limit={int(best['U_limit'])} days.\n")
                f.write(f"   Minimum Daily Cost: ${best['Total_Daily_Cost']:,.2f}\n")
            else: f.write("   (Integrated Optimization step was not run)\n")
            f.write("\n")
            
            f.write("2. ROBUSTNESS & SENSITIVITY\n")
            if 'sensitivity' in self.metrics:
                f.write(f"   Most Sensitive Parameter: {self.metrics['sensitivity']['most_sensitive']}\n")
                f.write("   (See Chart 07 for detailed stress-test results)\n")
            f.write("\n")
            
            f.write("3. LOGISTICS PERFORMANCE\n")
            if 'logistics' in self.metrics:
                data = self.metrics['logistics']
                try:
                    for role in data['Routes'].keys():
                        f.write(f"   --- {role} ---\n")
                        f.write(f"   Routes: {data['Routes'][role]}\n")
                        f.write(f"   Avg Fill Rate: {data['Avg Fill %'][role]:.2f}%\n")
                        f.write(f"   Efficiency: ${data['Cost/Kg ($)'][role]}/kg\n")
                except: f.write("   (Data format error)\n")