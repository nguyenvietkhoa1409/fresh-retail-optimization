# src/analysis/reporter_v2.py
"""
ENHANCED PIPELINE REPORTER
Publication-quality visualizations for heterogeneous supplier analysis

Key Improvements:
1. Reads archetype (not zone_label) for sourcing mix
2. New visualizations: P-U sensitivity, trade-off frontier
3. Better color schemes and annotations
4. Executive dashboard generation
"""

import os
import ast
import glob
import warnings
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import ProjectConfig as Cfg

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class PipelineReporter:
    """
    Enhanced reporter with publication-quality visualizations
    """
    
    def __init__(self, out_dir: Optional[str] = None, save_svg: bool = True):
        self.OUT_DIR = out_dir or Cfg.OUT_DIR_ANALYSIS
        os.makedirs(self.OUT_DIR, exist_ok=True)
        
        self.OUT_DIR_LOGISTICS = getattr(Cfg, "OUT_DIR_LOGISTICS", 
                                         os.path.join(os.path.dirname(self.OUT_DIR), "vrp_route_maps"))
        
        self.save_svg = save_svg
        
        # Professional color palette
        self.colors = {
            'Procurement': '#2E86AB',      # Deep Blue
            'Logistics': '#A23B72',        # Magenta
            'Holding': '#F18F01',          # Amber
            'Freshness': '#C73E1D',        # Burnt Orange
            'Total': '#6A4C93',            # Purple
            'Optimal': '#06A77D'           # Teal
        }
        
        # Archetype colors (consistent across charts)
        self.archetype_colors = {
            'local_specialty': '#E63946',      # Red
            'regional_distributor': '#F77F00', # Orange
            'bulk_wholesaler': '#06A77D',      # Green
            'farm_direct': '#4361EE',          # Blue
            'Unknown': '#CCCCCC'               # Gray
        }
        
        self._set_style()
    
    def _set_style(self):
        """Set publication-quality matplotlib style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_context("paper", font_scale=1.3)
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.titleweight': 'bold',
            'figure.titlesize': 16,
            'figure.titleweight': 'bold',
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2.5,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'figure.dpi': 150,
        })
    
    def _savefig(self, fig, name: str, dpi=300):
        """Save figure with optional SVG"""
        png_path = os.path.join(self.OUT_DIR, f"{name}.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        
        if self.save_svg:
            svg_path = os.path.join(self.OUT_DIR, f"{name}.svg")
            fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
        
        plt.close(fig)
        print(f"  ✓ Saved: {name}")
    
    def run(self):
        """Generate all reports"""
        print("\n" + "="*70)
        print("  ENHANCED PIPELINE REPORTER")
        print("="*70)
        
        # Load optimization results
        opt_path = os.path.join(self.OUT_DIR, "integrated_optimization_results.csv")
        
        if not os.path.exists(opt_path):
            print(f"\n❌ Missing: {opt_path}")
            print("   Run IntegratedSolver first!")
            return
        
        df_res = pd.read_csv(opt_path)
        
        print(f"\n[Data] Loaded {len(df_res)} strategic scenarios")
        
        # Generate visualizations
        print("\n[Phase 1] Core Visualizations...")
        self._plot_strategic_waterfall(df_res)
        self._plot_pu_sensitivity_heatmap(df_res)
        self._plot_tradeoff_frontier(df_res)
        
        print("\n[Phase 2] Sourcing Analysis...")
        self._plot_sourcing_by_archetype(df_res)
        self._plot_supplier_utilization(df_res)
        
        print("\n[Phase 3] Operational Analysis...")
        self._plot_distance_cost_analysis(df_res)
        self._plot_fixed_cost_impact(df_res)
        
        print("\n[Phase 4] Executive Dashboard...")
        self._generate_executive_dashboard(df_res)
        
        print("\n[Phase 5] Supplementary...")
        self._analyze_reconstruction()
        self._analyze_forecast()
        self._generate_summary_table(df_res)
        
        print("\n" + "="*70)
        print("  ✅ ALL REPORTS GENERATED")
        print(f"  📁 Location: {self.OUT_DIR}")
        print("="*70 + "\n")
    
    # ========================================================================
    # CORE VISUALIZATIONS
    # ========================================================================
    
    def _plot_strategic_waterfall(self, df: pd.DataFrame):
        """
        Waterfall chart showing cost evolution across strategies
        """
        if df.empty:
            return
        
        # Sort by total cost
        df = df.sort_values('Total_Daily_Cost').reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        strategies = df['Strategy'].values
        costs = df['Total_Daily_Cost'].values
        
        # Calculate deltas
        baseline = costs[0]
        deltas = [0] + list(costs[1:] - costs[:-1])
        cumulative = np.cumsum(deltas) + baseline
        
        # Color by increase/decrease
        colors_bars = ['#06A77D' if d == 0 else ('#C73E1D' if d > 0 else '#2E86AB') 
                      for d in deltas]
        
        # Plot bars
        bars = ax.bar(range(len(strategies)), deltas, bottom=cumulative - deltas,
                     color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add connectors
        for i in range(len(strategies) - 1):
            ax.plot([i + 0.4, i + 0.6], [cumulative[i], cumulative[i]], 
                   'k--', alpha=0.5, linewidth=1)
        
        # Annotations
        for i, (s, c) in enumerate(zip(strategies, cumulative)):
            ax.text(i, c + 500, f'${c/1000:.1f}k', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Highlight optimal
        optimal_idx = costs.argmin()
        ax.patches[optimal_idx].set_edgecolor('#06A77D')
        ax.patches[optimal_idx].set_linewidth(4)
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=0, ha='center', fontweight='bold')
        ax.set_ylabel("Total Daily Cost ($)")
        ax.set_title("Strategic Cost Waterfall: Evolution from Baseline", pad=20)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#06A77D', label='Baseline / Optimal'),
            mpatches.Patch(color='#C73E1D', label='Cost Increase'),
            mpatches.Patch(color='#2E86AB', label='Cost Decrease')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        self._savefig(fig, "01_Strategic_Cost_Waterfall")
    
    def _plot_pu_sensitivity_heatmap(self, df: pd.DataFrame):
        """
        Interactive P-U sensitivity heatmap with annotations
        """
        if 'P_lim' not in df.columns or 'U_lim' not in df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Total Cost Heatmap
        pivot_cost = df.pivot_table(index='P_lim', columns='U_lim', 
                                    values='Total_Daily_Cost', aggfunc='mean')
        
        sns.heatmap(pivot_cost, annot=True, fmt='.0f', cmap='RdYlGn_r',
                   ax=axes[0], cbar_kws={'label': 'Total Daily Cost ($)'},
                   linewidths=2, linecolor='white', annot_kws={'fontsize': 11, 'weight': 'bold'})
        
        axes[0].set_title("Total Daily Cost", fontweight='bold', pad=15)
        axes[0].set_xlabel("U (Review Period, days)", fontweight='bold')
        axes[0].set_ylabel("P (Max Lead Time, days)", fontweight='bold')
        
        # Mark optimal
        min_val = pivot_cost.min().min()
        for i in range(pivot_cost.shape[0]):
            for j in range(pivot_cost.shape[1]):
                if abs(pivot_cost.iloc[i, j] - min_val) < 1:
                    axes[0].add_patch(plt.Rectangle((j, i), 1, 1, 
                                     fill=False, edgecolor='blue', lw=4))
        
        # 2. Freshness Penalty Heatmap
        if 'Daily_Freshness_Penalty' in df.columns:
            pivot_fresh = df.pivot_table(index='P_lim', columns='U_lim',
                                        values='Daily_Freshness_Penalty', aggfunc='mean')
            
            sns.heatmap(pivot_fresh, annot=True, fmt='.0f', cmap='Oranges',
                       ax=axes[1], cbar_kws={'label': 'Freshness Penalty ($)'},
                       linewidths=2, linecolor='white', annot_kws={'fontsize': 11, 'weight': 'bold'})
            
            axes[1].set_title("Freshness Penalty Impact", fontweight='bold', pad=15)
            axes[1].set_xlabel("U (Review Period, days)", fontweight='bold')
            axes[1].set_ylabel("P (Max Lead Time, days)", fontweight='bold')
        
        plt.tight_layout()
        self._savefig(fig, "02_PU_Sensitivity_Heatmap")
    
    def _plot_tradeoff_frontier(self, df: pd.DataFrame):
        """
        Pareto frontier: Cost vs Freshness trade-off
        """
        if df.empty or 'Daily_Freshness_Penalty' not in df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x = df['Daily_Freshness_Penalty']
        y = df['Total_Daily_Cost']
        strategies = df['Strategy']
        
        # Scatter with labels
        scatter = ax.scatter(x, y, s=300, c=y, cmap='RdYlGn_r', 
                           alpha=0.7, edgecolor='black', linewidth=2)
        
        # Annotate each point
        for i, (xi, yi, s) in enumerate(zip(x, y, strategies)):
            ax.annotate(s, (xi, yi), 
                       xytext=(10, -5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        # Highlight optimal
        optimal_idx = y.idxmin()
        ax.scatter(x.iloc[optimal_idx], y.iloc[optimal_idx], 
                  s=500, marker='*', color='gold', edgecolor='black', 
                  linewidth=2, zorder=10, label='Optimal')
        
        # Pareto frontier (if applicable)
        # Sort by freshness
        sorted_idx = x.argsort()
        x_sorted = x.iloc[sorted_idx].values
        y_sorted = y.iloc[sorted_idx].values
        
        # Find Pareto-efficient points
        pareto_idx = [0]
        current_min_cost = y_sorted[0]
        
        for i in range(1, len(y_sorted)):
            if y_sorted[i] < current_min_cost:
                pareto_idx.append(i)
                current_min_cost = y_sorted[i]
        
        if len(pareto_idx) > 1:
            ax.plot(x_sorted[pareto_idx], y_sorted[pareto_idx], 
                   'b--', linewidth=2, alpha=0.5, label='Pareto Frontier')
        
        ax.set_xlabel("Freshness Penalty ($/day)", fontweight='bold')
        ax.set_ylabel("Total Daily Cost ($/day)", fontweight='bold')
        ax.set_title("Cost-Freshness Trade-off Frontier", fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Total Cost ($)", fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._savefig(fig, "03_Tradeoff_Frontier")
    
    # ========================================================================
    # SOURCING ANALYSIS (FIXED!)
    # ========================================================================
    
    def _plot_sourcing_by_archetype(self, df_res: pd.DataFrame):
        """
        FIXED: Sourcing mix by supplier archetype (not zone_label)
        """
        if df_res.empty:
            return
        
        print("  [Sourcing Analysis] Loading procurement plans...")
        
        # Load supplier archetype mapping
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        
        if not os.path.exists(sup_path):
            print(f"    ⚠️  Missing: {sup_path}")
            return
        
        df_sup = pd.read_csv(sup_path)
        
        # Check for archetype column (new schema)
        if 'archetype' not in df_sup.columns:
            print("    ⚠️  'archetype' column missing. Using fallback...")
            # Fallback to distance_tier if available
            if 'distance_tier' in df_sup.columns:
                archetype_col = 'distance_tier'
            else:
                print("    ❌ No archetype data. Skipping sourcing analysis.")
                return
        else:
            archetype_col = 'archetype'
        
        df_sup['supplier_id'] = df_sup['supplier_id'].astype(int)
        sup_map = df_sup.set_index('supplier_id')[archetype_col].to_dict()
        
        # Collect sourcing data per strategy
        strategies = []
        archetype_data = {}
        
        for _, row in df_res.iterrows():
            strat = row.get('Strategy', f"P{row['P_lim']}U{row['U_lim']}")
            p, u = int(row['P_lim']), int(row['U_lim'])
            
            strategies.append(strat)
            
            # Try multiple filename patterns
            possible_files = [
                f"procurement_plan_{strat}_P{p}_U{u}.csv",
                f"procurement_plan_P{p}_U{u}.csv"
            ]
            
            plan_path = None
            for fname in possible_files:
                test_path = os.path.join(self.OUT_DIR, fname)
                if os.path.exists(test_path):
                    plan_path = test_path
                    break
            
            if not plan_path:
                print(f"    ⚠️  Plan not found for {strat}")
                archetype_data[strat] = {}
                continue
            
            try:
                df_plan = pd.read_csv(plan_path)
                
                if df_plan.empty:
                    archetype_data[strat] = {}
                    continue
                
                df_plan['supplier_id'] = df_plan['supplier_id'].astype(int)
                df_plan['archetype'] = df_plan['supplier_id'].map(sup_map).fillna('Unknown')
                
                # Aggregate volume by archetype
                vol_by_arch = df_plan.groupby('archetype')['order_weight_kg'].sum().to_dict()
                archetype_data[strat] = vol_by_arch
                
            except Exception as e:
                print(f"    ⚠️  Error loading {strat}: {e}")
                archetype_data[strat] = {}
        
        if not archetype_data:
            print("    ❌ No valid sourcing data found")
            return
        
        # Create DataFrame
        df_plot = pd.DataFrame(archetype_data).T.fillna(0)
        
        if df_plot.empty or df_plot.shape[1] == 0:
            print("    ⚠️  No archetype data to plot")
            return
        
        # Normalize to percentages
        row_sums = df_plot.sum(axis=1)
        df_plot_pct = df_plot.div(row_sums.replace(0, 1), axis=0) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Use archetype-specific colors
        colors_to_use = [self.archetype_colors.get(col, '#CCCCCC') 
                        for col in df_plot_pct.columns]
        
        df_plot_pct.plot(kind='bar', stacked=True, ax=ax, 
                        color=colors_to_use, width=0.75,
                        edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel("Volume Share (%)", fontweight='bold', fontsize=12)
        ax.set_xlabel("Strategy", fontweight='bold', fontsize=12)
        ax.set_title("Sourcing Mix by Supplier Archetype", 
                    fontweight='bold', fontsize=14, pad=20)
        
        ax.legend(title="Supplier Archetype", 
                 bbox_to_anchor=(1.05, 1), loc='upper left',
                 fontsize=10, title_fontsize=11, frameon=True)
        
        plt.xticks(rotation=0, ha='center', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' 
                     for v in container]
            ax.bar_label(container, labels=labels, label_type='center',
                        fontsize=9, color='white', weight='bold')
        
        plt.tight_layout()
        self._savefig(fig, "04_Sourcing_Mix_by_Archetype")
    
    def _plot_supplier_utilization(self, df_res: pd.DataFrame):
        """
        Supplier utilization matrix: which suppliers used in which strategy
        """
        if df_res.empty:
            return
        
        print("  [Supplier Utilization] Analyzing supplier selection patterns...")
        
        # Load suppliers
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        if not os.path.exists(sup_path):
            return
        
        df_sup = pd.read_csv(sup_path)
        
        # Build utilization matrix
        utilization = []
        
        for _, row in df_res.iterrows():
            strat = row.get('Strategy', f"P{row['P_lim']}U{row['U_lim']}")
            p, u = int(row['P_lim']), int(row['U_lim'])
            
            plan_path = os.path.join(self.OUT_DIR, f"procurement_plan_{strat}_P{p}_U{u}.csv")
            
            if not os.path.exists(plan_path):
                continue
            
            try:
                df_plan = pd.read_csv(plan_path)
                suppliers_used = df_plan['supplier_id'].unique()
                
                for sid in df_sup['supplier_id']:
                    utilization.append({
                        'Strategy': strat,
                        'Supplier': f"S{sid}",
                        'Used': 1 if sid in suppliers_used else 0
                    })
            except:
                continue
        
        if not utilization:
            return
        
        df_util = pd.DataFrame(utilization)
        pivot = df_util.pivot_table(index='Supplier', columns='Strategy', 
                                    values='Used', fill_value=0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.3)))
        
        sns.heatmap(pivot, cmap='YlGnBu', cbar_kws={'label': 'Utilized'},
                   ax=ax, linewidths=1, linecolor='gray',
                   annot=False, fmt='d')
        
        ax.set_title("Supplier Utilization Matrix", fontweight='bold', pad=15)
        ax.set_xlabel("Strategy", fontweight='bold')
        ax.set_ylabel("Supplier ID", fontweight='bold')
        
        plt.tight_layout()
        self._savefig(fig, "05_Supplier_Utilization_Matrix")
    
    # ========================================================================
    # OPERATIONAL ANALYSIS
    # ========================================================================
    
    def _plot_distance_cost_analysis(self, df_res: pd.DataFrame):
        """
        Validate distance-cost relationship across strategies
        """
        print("  [Distance Analysis] Computing sourcing distances...")
        
        # Load suppliers with locations
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        if not os.path.exists(sup_path):
            return
        
        df_sup = pd.read_csv(sup_path)
        
        # Compute average sourcing distance per strategy
        strategy_distances = []
        
        for _, row in df_res.iterrows():
            strat = row.get('Strategy', f"P{row['P_lim']}U{row['U_lim']}")
            p, u = int(row['P_lim']), int(row['U_lim'])
            
            plan_path = os.path.join(self.OUT_DIR, f"procurement_plan_{strat}_P{p}_U{u}.csv")
            
            if not os.path.exists(plan_path):
                continue
            
            try:
                df_plan = pd.read_csv(plan_path)
                
                # Merge with supplier distances
                if 'distance_km' in df_plan.columns:
                    avg_dist = np.average(df_plan['distance_km'], 
                                         weights=df_plan['order_weight_kg'])
                else:
                    # Calculate from lat/lon if not in plan
                    avg_dist = 100.0  # Placeholder
                
                strategy_distances.append({
                    'Strategy': strat,
                    'P_lim': p,
                    'Avg_Distance_km': avg_dist,
                    'Procurement_Cost': row['Daily_Procurement_Cost'],
                    'Logistics_Cost': row['Daily_Distribution_Cost']
                })
            except:
                continue
        
        if not strategy_distances:
            return
        
        df_dist = pd.DataFrame(strategy_distances)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Distance vs P parameter
        ax1 = axes[0]
        for p_val in sorted(df_dist['P_lim'].unique()):
            subset = df_dist[df_dist['P_lim'] == p_val]
            ax1.scatter(subset['P_lim'], subset['Avg_Distance_km'], 
                       s=200, label=f'P={p_val}', alpha=0.7)
        
        ax1.set_xlabel("P (Max Lead Time, days)", fontweight='bold')
        ax1.set_ylabel("Average Sourcing Distance (km)", fontweight='bold')
        ax1.set_title("Lead Time Constraint Impact", fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        
        # 2. Distance vs Cost
        ax2 = axes[1]
        scatter = ax2.scatter(df_dist['Avg_Distance_km'], 
                             df_dist['Logistics_Cost'],
                             s=200, c=df_dist['P_lim'], cmap='viridis',
                             alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for _, row in df_dist.iterrows():
            ax2.annotate(row['Strategy'], 
                        (row['Avg_Distance_km'], row['Logistics_Cost']),
                        fontsize=9, xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel("Average Sourcing Distance (km)", fontweight='bold')
        ax2.set_ylabel("Logistics Cost ($/day)", fontweight='bold')
        ax2.set_title("Distance-Cost Relationship", fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("P (Lead Time)", fontweight='bold')
        
        plt.tight_layout()
        self._savefig(fig, "06_Distance_Cost_Analysis")
    
    def _plot_fixed_cost_impact(self, df_res: pd.DataFrame):
        """
        Show how U parameter affects fixed cost burden
        """
        if df_res.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Group by U parameter
        for u_val in sorted(df_res['U_lim'].unique()):
            subset = df_res[df_res['U_lim'] == u_val]
            
            ax.scatter(subset['U_lim'], subset['Daily_Procurement_Cost'],
                      s=300, label=f'U={u_val}', alpha=0.7,
                      edgecolor='black', linewidth=1.5)
            
            # Annotate with strategy name
            for _, row in subset.iterrows():
                ax.annotate(row['Strategy'], 
                           (row['U_lim'], row['Daily_Procurement_Cost']),
                           fontsize=9, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel("U (Review Period, days)", fontweight='bold')
        ax.set_ylabel("Procurement Cost ($/day)", fontweight='bold')
        ax.set_title("Fixed Cost Amortization: Impact of Order Frequency", 
                    fontweight='bold', pad=20)
        
        # Add trend line
        if len(df_res) > 2:
            z = np.polyfit(df_res['U_lim'], df_res['Daily_Procurement_Cost'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_res['U_lim'].min(), df_res['U_lim'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, 
                   label='Trend')
        
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._savefig(fig, "07_Fixed_Cost_Impact")
    
    # ========================================================================
    # EXECUTIVE DASHBOARD
    # ========================================================================
    
    def _generate_executive_dashboard(self, df: pd.DataFrame):
        """
        Single-page executive summary with key metrics
        """
        if df.empty:
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
        
        # === TOP ROW: Key Metrics ===
        
        # 1. Cost Comparison Bar
        ax1 = fig.add_subplot(gs[0, :2])
        
        strategies = df['Strategy'].values
        costs = df['Total_Daily_Cost'].values
        
        bars = ax1.barh(range(len(strategies)), costs, 
                       color=self.colors['Total'], alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        
        # Highlight optimal
        optimal_idx = costs.argmin()
        bars[optimal_idx].set_color(self.colors['Optimal'])
        bars[optimal_idx].set_alpha(0.9)
        
        ax1.set_yticks(range(len(strategies)))
        ax1.set_yticklabels(strategies, fontweight='bold')
        ax1.set_xlabel("Total Daily Cost ($)", fontweight='bold')
        ax1.set_title("Strategic Cost Ranking", fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Key Metrics Table
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        best = df.loc[df['Total_Daily_Cost'].idxmin()]
        worst = df.loc[df['Total_Daily_Cost'].idxmax()]
        
        savings = worst['Total_Daily_Cost'] - best['Total_Daily_Cost']
        savings_pct = (savings / worst['Total_Daily_Cost']) * 100
        
        metrics_text = f"""
EXECUTIVE SUMMARY

Optimal Strategy: {best['Strategy']}
• P (Lead Time): {best['P_lim']} days
• U (Review): {best['U_lim']} days

Cost Performance:
• Best: ${best['Total_Daily_Cost']:,.0f}/day
• Worst: ${worst['Total_Daily_Cost']:,.0f}/day
• Savings: ${savings:,.0f}/day ({savings_pct:.1f}%)

Annual Impact:
• Cost Savings: ${savings * 365:,.0f}/year
        """
        
        ax2.text(0.1, 0.5, metrics_text, fontsize=10, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # === MIDDLE ROW: Cost Breakdown ===
        
        # 3. Stacked Bar Chart
        ax3 = fig.add_subplot(gs[1, :])
        
        components = ['Daily_Procurement_Cost', 'Daily_Distribution_Cost',
                     'Daily_Holding_Cost', 'Daily_Freshness_Penalty']
        component_labels = ['Procurement', 'Logistics', 'Holding', 'Freshness']
        
        bottom = np.zeros(len(strategies))
        
        for i, (comp, label) in enumerate(zip(components, component_labels)):
            if comp in df.columns:
                vals = df[comp].values
                ax3.bar(range(len(strategies)), vals, bottom=bottom,
                       label=label, color=list(self.colors.values())[i],
                       alpha=0.8, edgecolor='white', linewidth=1)
                bottom += vals
        
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels(strategies, rotation=0, ha='center', fontweight='bold')
        ax3.set_ylabel("Cost ($/day)", fontweight='bold')
        ax3.set_title("Cost Component Breakdown by Strategy", fontweight='bold')
        ax3.legend(loc='upper left', ncol=4, fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
        # === BOTTOM ROW: Insights ===
        
        # 4. P-U Heatmap (mini)
        ax4 = fig.add_subplot(gs[2, 0])
        
        if 'P_lim' in df.columns and 'U_lim' in df.columns:
            pivot = df.pivot_table(index='P_lim', columns='U_lim',
                                  values='Total_Daily_Cost', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                       ax=ax4, cbar=False, linewidths=1,
                       annot_kws={'fontsize': 9})
            ax4.set_title("P-U Sensitivity", fontweight='bold', fontsize=11)
            ax4.set_xlabel("U", fontweight='bold', fontsize=10)
            ax4.set_ylabel("P", fontweight='bold', fontsize=10)
        
        # 5. Cost Variance
        ax5 = fig.add_subplot(gs[2, 1])
        
        cost_std = df['Total_Daily_Cost'].std()
        cost_mean = df['Total_Daily_Cost'].mean()
        cv = (cost_std / cost_mean) * 100
        
        ax5.pie([savings, worst['Total_Daily_Cost'] - savings],
               labels=['Potential Savings', 'Base Cost'],
               colors=['#06A77D', '#CCCCCC'],
               autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 11, 'weight': 'bold'})
        ax5.set_title("Savings Opportunity", fontweight='bold', fontsize=11)
        
        # 6. Recommendations
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        recommendation = f"""
RECOMMENDATION

Implement: {best['Strategy']}

Rationale:
• Lowest total cost
• P={best['P_lim']}: {self._interpret_p(best['P_lim'])}
• U={best['U_lim']}: {self._interpret_u(best['U_lim'])}

Next Steps:
1. Validate with pilot stores
2. Negotiate supplier contracts
3. Monitor KPIs: cost, freshness
        """
        
        ax6.text(0.1, 0.5, recommendation, fontsize=9,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Overall title
        fig.suptitle("EXECUTIVE DASHBOARD: Supply Chain Optimization Results",
                    fontsize=16, fontweight='bold', y=0.98)
        
        self._savefig(fig, "00_Executive_Dashboard")
    
    def _interpret_p(self, p):
        """Interpret P value"""
        if p <= 2:
            return "Local sourcing priority"
        elif p <= 3:
            return "Regional mix"
        else:
            return "Access to distant suppliers"
    
    def _interpret_u(self, u):
        """Interpret U value"""
        if u <= 2:
            return "Frequent ordering"
        elif u <= 3:
            return "Balanced frequency"
        else:
            return "Consolidated orders"
    
    # ========================================================================
    # SUPPLEMENTARY
    # ========================================================================
    
    def _analyze_reconstruction(self):
        """Demand reconstruction quality chart"""
        path = os.path.join(Cfg.OUT_DIR_PART2, "reconstruction_accuracy_by_product.csv")
        if not os.path.exists(path):
            return
        
        df = pd.read_csv(path)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'Mean_Sales' not in df.columns:
            return
        
        sizes = df.get('Recon_RMSE', df['Mean_Sales']).fillna(1).values
        sizes = np.interp(sizes, (sizes.min(), sizes.max()), (50, 500))
        
        scatter = ax.scatter(df['Mean_Sales'], df['Recon_MAPE'],
                           s=sizes, c=df['Recon_MAPE'], cmap='viridis_r',
                           alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xscale('log')
        ax.set_xlabel("Mean Daily Sales (log scale)", fontweight='bold')
        ax.set_ylabel("Reconstruction Error (MAPE %)", fontweight='bold')
        ax.set_title("Data Quality: Demand Reconstruction Performance",
                    fontweight='bold', pad=15)
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("MAPE (%)", fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self._savefig(fig, "08_Data_Reconstruction_Quality")
    
    def _analyze_forecast(self):
        """Forecast performance chart"""
        path = os.path.join(Cfg.OUT_DIR_FORECAST, "per_horizon_metrics.csv")
        if not os.path.exists(path):
            return
        
        df = pd.read_csv(path)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = self.colors['Procurement']
        ax1.plot(df['horizon'], df['WAPE'], marker='o', color=color1,
                linewidth=2.5, markersize=8, label='WAPE (%)')
        ax1.set_xlabel("Forecast Horizon (days)", fontweight='bold')
        ax1.set_ylabel("WAPE (%)", color=color1, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = self.colors['Logistics']
        ax2.plot(df['horizon'], df['RMSE'], marker='s', linestyle='--',
                color=color2, linewidth=2.5, markersize=8, label='RMSE')
        ax2.set_ylabel("RMSE (units)", color=color2, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.set_title("Forecast Performance Degradation vs Horizon",
                     fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        self._savefig(fig, "09_Forecast_Performance")
    
    def _generate_summary_table(self, df: pd.DataFrame):
        """Generate text summary table"""
        cols_order = ['Strategy', 'P_lim', 'U_lim',
                     'Daily_Procurement_Cost', 'Daily_Distribution_Cost',
                     'Daily_Holding_Cost', 'Daily_Freshness_Penalty',
                     'Total_Daily_Cost']
        
        cols_available = [c for c in cols_order if c in df.columns]
        df_display = df[cols_available].copy()
        
        # Format currency columns
        for col in df_display.columns:
            if 'Cost' in col or 'Penalty' in col:
                df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}")
        
        # Save as text
        out_path = os.path.join(self.OUT_DIR, "10_Summary_Table.txt")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("  STRATEGIC OPTIMIZATION RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(df_display.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
        
        print(f"  ✓ Saved: Summary Table")

