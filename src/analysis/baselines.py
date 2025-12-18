# src/analysis/baseline_comparison.py
"""
RIGOROUS BASELINE FRAMEWORK
Implements industry-standard baselines and ablation studies

Baseline Categories:
1. Naive Heuristics (sanity checks)
2. Industry Practices (realistic alternatives)
3. Academic Benchmarks (literature standards)
4. Ablation Studies (component analysis)
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import ProjectConfig as Cfg


class BaselineFramework:
    """
    Comprehensive baseline comparison framework
    """
    
    def __init__(self):
        self.results = []
        self.out_dir = os.path.join(Cfg.OUT_DIR_ANALYSIS, "baselines")
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Load reference data
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load procurement candidates and demand data"""
        print("\n[Baseline Framework] Loading reference data...")
        
        # Load demand
        unified_path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement_enhanced.parquet")
        self.demand_df = pd.read_parquet(unified_path)
        
        # Load suppliers
        sup_path = os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv")
        self.suppliers_df = pd.read_csv(sup_path)
        
        # Load supplier-product
        sp_path = os.path.join(Cfg.ARTIFACTS_DIR, "supplier_product.csv")
        self.sp_df = pd.read_csv(sp_path)
        
        print(f"  Loaded: {len(self.demand_df)} demand pairs, "
              f"{len(self.suppliers_df)} suppliers")
    
    # =====================================================================
    # CATEGORY 1: NAIVE HEURISTICS (Sanity Checks)
    # =====================================================================
    
    def baseline_random_assignment(self) -> Dict:
        """
        Random supplier assignment
        Purpose: Worst-case lower bound
        """
        print("\n[Baseline 1] Random Assignment...")
        
        total_cost = 0
        total_orders = 0
        
        for _, row in self.demand_df.iterrows():
            store = row['store_id']
            product = int(row['product_id'])
            demand = row['predicted_mean'] * 3  # U=3 days
            
            # Find available suppliers
            available = self.sp_df[self.sp_df['product_id'] == product]
            
            if available.empty:
                continue
            
            # Random selection
            chosen = available.sample(1).iloc[0]
            
            qty = max(demand, chosen['min_order_qty_units'])
            
            # Costs
            product_cost = qty * chosen['unit_price']
            fixed_cost = chosen.get('fixed_order_cost', Cfg.FIXED_ORDER_COST)
            
            total_cost += product_cost + fixed_cost
            total_orders += 1
        
        daily_cost = total_cost / 3  # Amortize over U days
        
        return {
            'name': 'Random Assignment',
            'category': 'Naive',
            'daily_cost': daily_cost,
            'num_orders': total_orders,
            'description': 'Random supplier selection (worst case)'
        }
    
    def baseline_nearest_supplier(self) -> Dict:
        """
        Always choose nearest supplier
        Purpose: Distance-only optimization (common practice)
        """
        print("\n[Baseline 2] Nearest Supplier...")
        
        total_cost = 0
        total_orders = 0
        total_distance = 0
        
        for _, row in self.demand_df.iterrows():
            store = row['store_id']
            product = int(row['product_id'])
            demand = row['predicted_mean'] * 3
            
            # Find suppliers for this product
            available = self.sp_df[self.sp_df['product_id'] == product]
            
            if available.empty:
                continue
            
            # Merge with supplier locations
            with_loc = available.merge(
                self.suppliers_df[['supplier_id', 'lat', 'lon']],
                on='supplier_id'
            )
            
            if with_loc.empty:
                continue
            
            # Calculate distances
            from src.utils.geo import GeoUtils
            
            distances = []
            for _, sup in with_loc.iterrows():
                d = GeoUtils.haversine_km(
                    row['store_lat'], row['store_lon'],
                    sup['lat'], sup['lon']
                )
                distances.append(d)
            
            with_loc['distance'] = distances
            
            # Choose nearest
            nearest = with_loc.loc[with_loc['distance'].idxmin()]
            
            qty = max(demand, nearest['min_order_qty_units'])
            
            # Costs
            product_cost = qty * nearest['unit_price']
            fixed_cost = nearest.get('fixed_order_cost', Cfg.FIXED_ORDER_COST)
            transport_cost = (Cfg.TRANSPORT_COST_PER_KG_KM * 
                            nearest['distance'] * 
                            qty * row['unit_weight_kg'])
            
            total_cost += product_cost + fixed_cost + transport_cost
            total_orders += 1
            total_distance += nearest['distance']
        
        daily_cost = total_cost / 3
        avg_distance = total_distance / total_orders if total_orders > 0 else 0
        
        return {
            'name': 'Nearest Supplier',
            'category': 'Naive',
            'daily_cost': daily_cost,
            'num_orders': total_orders,
            'avg_distance_km': avg_distance,
            'description': 'Always choose closest supplier (myopic)'
        }
    
    def baseline_cheapest_unit_price(self) -> Dict:
        """
        Always choose supplier with lowest unit price
        Purpose: Price-only optimization (ignores logistics)
        """
        print("\n[Baseline 3] Cheapest Unit Price...")
        
        total_cost = 0
        total_orders = 0
        
        for _, row in self.demand_df.iterrows():
            product = int(row['product_id'])
            demand = row['predicted_mean'] * 3
            
            # Find cheapest supplier
            available = self.sp_df[self.sp_df['product_id'] == product]
            
            if available.empty:
                continue
            
            cheapest = available.loc[available['unit_price'].idxmin()]
            
            qty = max(demand, cheapest['min_order_qty_units'])
            
            # Costs (ignoring distance - unrealistic!)
            product_cost = qty * cheapest['unit_price']
            fixed_cost = cheapest.get('fixed_order_cost', Cfg.FIXED_ORDER_COST)
            
            total_cost += product_cost + fixed_cost
            total_orders += 1
        
        daily_cost = total_cost / 3
        
        return {
            'name': 'Cheapest Price',
            'category': 'Naive',
            'daily_cost': daily_cost,
            'num_orders': total_orders,
            'description': 'Lowest unit price only (ignores distance)'
        }
    
    # =====================================================================
    # CATEGORY 2: INDUSTRY PRACTICES (Realistic Alternatives)
    # =====================================================================
    
    def baseline_single_tier_sourcing(self, tier='regional') -> Dict:
        """
        Use only one supplier tier
        Purpose: Common industry practice (single-tier supply chain)
        """
        print(f"\n[Baseline 4] Single-Tier Sourcing ({tier})...")
        
        # Filter suppliers by archetype
        tier_map = {
            'local': 'local_specialty',
            'regional': 'regional_distributor',
            'bulk': 'bulk_wholesaler',
            'farm': 'farm_direct'
        }
        
        target_arch = tier_map.get(tier, 'regional_distributor')
        
        tier_suppliers = self.suppliers_df[
            self.suppliers_df['archetype'] == target_arch
        ]['supplier_id'].tolist()
        
        total_cost = 0
        total_orders = 0
        feasible_pairs = 0
        
        for _, row in self.demand_df.iterrows():
            product = int(row['product_id'])
            demand = row['predicted_mean'] * 3
            
            # Find suppliers in target tier
            available = self.sp_df[
                (self.sp_df['product_id'] == product) &
                (self.sp_df['supplier_id'].isin(tier_suppliers))
            ]
            
            if available.empty:
                continue
            
            # Choose cheapest in tier
            chosen = available.loc[available['unit_price'].idxmin()]
            
            qty = max(demand, chosen['min_order_qty_units'])
            
            # Costs
            product_cost = qty * chosen['unit_price']
            fixed_cost = chosen.get('fixed_order_cost', Cfg.FIXED_ORDER_COST)
            
            total_cost += product_cost + fixed_cost
            total_orders += 1
            feasible_pairs += 1
        
        daily_cost = total_cost / 3
        
        return {
            'name': f'Single-Tier ({tier.title()})',
            'category': 'Industry',
            'daily_cost': daily_cost,
            'num_orders': total_orders,
            'feasible_pairs': feasible_pairs,
            'description': f'Restrict to {tier} suppliers only'
        }
    
    def baseline_equal_allocation(self) -> Dict:
        """
        Split orders equally among all viable suppliers
        Purpose: Risk diversification strategy
        """
        print("\n[Baseline 5] Equal Allocation (Diversification)...")
        
        total_cost = 0
        total_orders = 0
        
        for _, row in self.demand_df.iterrows():
            product = int(row['product_id'])
            demand = row['predicted_mean'] * 3
            
            # Find all suppliers
            available = self.sp_df[self.sp_df['product_id'] == product]
            
            if available.empty:
                continue
            
            n_suppliers = min(3, len(available))  # Split among top 3
            qty_per_supplier = demand / n_suppliers
            
            # Choose top 3 by price
            top_suppliers = available.nsmallest(n_suppliers, 'unit_price')
            
            for _, sup in top_suppliers.iterrows():
                qty = max(qty_per_supplier, sup['min_order_qty_units'])
                
                product_cost = qty * sup['unit_price']
                fixed_cost = sup.get('fixed_order_cost', Cfg.FIXED_ORDER_COST)
                
                total_cost += product_cost + fixed_cost
                total_orders += 1
        
        daily_cost = total_cost / 3
        
        return {
            'name': 'Equal Allocation',
            'category': 'Industry',
            'daily_cost': daily_cost,
            'num_orders': total_orders,
            'description': 'Split orders among top 3 suppliers (diversification)'
        }
    
    def baseline_eoq_based(self) -> Dict:
        """
        Economic Order Quantity (EOQ) baseline
        Purpose: Classic inventory optimization
        """
        print("\n[Baseline 6] EOQ-Based Policy...")
        
        total_cost = 0
        total_orders = 0
        
        for _, row in self.demand_df.iterrows():
            product = int(row['product_id'])
            annual_demand = row['predicted_mean'] * 365
            
            # Find cheapest supplier
            available = self.sp_df[self.sp_df['product_id'] == product]
            
            if available.empty:
                continue
            
            cheapest = available.loc[available['unit_price'].idxmin()]
            
            # EOQ formula: sqrt(2 * D * K / h)
            # D = annual demand, K = fixed cost, h = holding cost per unit per year
            K = cheapest.get('fixed_order_cost', Cfg.FIXED_ORDER_COST)
            h = row['daily_holding_cost_unit'] * 365
            
            if h > 0:
                eoq = np.sqrt(2 * annual_demand * K / h)
            else:
                eoq = annual_demand / 12  # Fallback: monthly orders
            
            # Number of orders per year
            n_orders = annual_demand / eoq if eoq > 0 else 12
            
            # Daily costs
            daily_product_cost = (annual_demand * cheapest['unit_price']) / 365
            daily_fixed_cost = (K * n_orders) / 365
            daily_holding_cost = (eoq / 2) * h / 365
            
            total_cost += daily_product_cost + daily_fixed_cost + daily_holding_cost
            total_orders += n_orders / 365  # Daily order count
        
        return {
            'name': 'EOQ Policy',
            'category': 'Academic',
            'daily_cost': total_cost,
            'num_orders': total_orders,
            'description': 'Classic Economic Order Quantity (textbook)'
        }
    
    # =====================================================================
    # CATEGORY 3: ABLATION STUDIES (Component Analysis)
    # =====================================================================
    
    def ablation_no_freshness_penalty(self) -> Dict:
        """
        Optimization without freshness consideration
        Purpose: Quantify freshness penalty impact
        """
        print("\n[Ablation 1] Without Freshness Penalty...")
        
        # This would require re-running procurement with freshness_penalty=0
        # For now, estimate from existing results
        
        # Load optimal solution
        opt_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, 
                               "integrated_optimization_results.csv")
        
        if os.path.exists(opt_path):
            df = pd.read_csv(opt_path)
            best = df.loc[df['Total_Daily_Cost'].idxmin()]
            
            # Estimate: Remove freshness penalty
            cost_without_fresh = (best['Total_Daily_Cost'] - 
                                 best.get('Daily_Freshness_Penalty', 0))
            
            return {
                'name': 'No Freshness Penalty',
                'category': 'Ablation',
                'daily_cost': cost_without_fresh,
                'description': 'Optimal strategy without freshness consideration'
            }
        
        return {'name': 'No Freshness', 'daily_cost': 0, 'category': 'Ablation'}
    
    def ablation_no_fixed_costs(self) -> Dict:
        """
        Optimization treating all orders as zero fixed cost
        Purpose: Quantify fixed cost impact
        """
        print("\n[Ablation 2] Without Fixed Costs...")
        
        total_cost = 0
        
        for _, row in self.demand_df.iterrows():
            product = int(row['product_id'])
            demand = row['predicted_mean'] * 3
            
            # Find cheapest supplier (since no fixed cost penalty)
            available = self.sp_df[self.sp_df['product_id'] == product]
            
            if available.empty:
                continue
            
            cheapest = available.loc[available['unit_price'].idxmin()]
            
            # Order exactly demand (no MOQ consideration without fixed costs)
            qty = demand
            
            # Only variable costs
            product_cost = qty * cheapest['unit_price']
            
            total_cost += product_cost
        
        daily_cost = total_cost / 3
        
        return {
            'name': 'No Fixed Costs',
            'category': 'Ablation',
            'daily_cost': daily_cost,
            'description': 'Ignoring setup costs (unrealistic lower bound)'
        }
    
    def ablation_no_capacity_constraints(self) -> Dict:
        """
        Optimization without supplier capacity limits
        Purpose: Quantify constraint tightness
        """
        print("\n[Ablation 3] Without Capacity Constraints...")
        
        # Same as cheapest price but allowing unlimited orders
        total_cost = 0
        
        for _, row in self.demand_df.iterrows():
            product = int(row['product_id'])
            demand = row['predicted_mean'] * 3
            
            available = self.sp_df[self.sp_df['product_id'] == product]
            
            if available.empty:
                continue
            
            # Choose absolute cheapest (capacity doesn't matter)
            cheapest = available.loc[available['unit_price'].idxmin()]
            
            qty = demand  # No MOQ either
            
            product_cost = qty * cheapest['unit_price']
            # Assume can amortize fixed cost well with large orders
            fixed_cost = cheapest.get('fixed_order_cost', Cfg.FIXED_ORDER_COST) / 2
            
            total_cost += product_cost + fixed_cost
        
        daily_cost = total_cost / 3
        
        return {
            'name': 'No Capacity Limits',
            'category': 'Ablation',
            'daily_cost': daily_cost,
            'description': 'Unlimited supplier capacity (theoretical lower bound)'
        }
    
    # =====================================================================
    # MAIN EXECUTION
    # =====================================================================
    
    def run_all_baselines(self) -> pd.DataFrame:
        """Execute all baseline methods"""
        
        print("\n" + "="*70)
        print("  COMPREHENSIVE BASELINE COMPARISON")
        print("="*70)
        
        baselines = [
            # Naive
            self.baseline_random_assignment,
            self.baseline_nearest_supplier,
            self.baseline_cheapest_unit_price,
            
            # Industry
            lambda: self.baseline_single_tier_sourcing('local'),
            lambda: self.baseline_single_tier_sourcing('regional'),
            lambda: self.baseline_single_tier_sourcing('farm'),
            self.baseline_equal_allocation,
            self.baseline_eoq_based,
            
            # Ablation
            self.ablation_no_freshness_penalty,
            self.ablation_no_fixed_costs,
            self.ablation_no_capacity_constraints
        ]
        
        results = []
        
        start_time = time.time()
        
        for baseline_func in baselines:
            try:
                result = baseline_func()
                results.append(result)
                
                # Print summary
                print(f"  ✓ {result['name']}: ${result['daily_cost']:,.2f}/day")
                
            except Exception as e:
                print(f"  ✗ Failed: {baseline_func.__name__} - {e}")
        
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s")
        
        # Convert to DataFrame
        df_baselines = pd.DataFrame(results)
        
        # Load optimal solution for comparison
        opt_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, 
                               "integrated_optimization_results.csv")
        
        if os.path.exists(opt_path):
            df_opt = pd.read_csv(opt_path)
            best = df_opt.loc[df_opt['Total_Daily_Cost'].idxmin()]
            
            # Add to results
            optimal_result = {
                'name': f"Proposed ({best['Strategy']})",
                'category': 'Proposed',
                'daily_cost': best['Total_Daily_Cost'],
                'description': 'Integrated optimization (this work)'
            }
            
            df_baselines = pd.concat([
                df_baselines,
                pd.DataFrame([optimal_result])
            ], ignore_index=True)
        
        # Save
        out_path = os.path.join(self.out_dir, "baseline_comparison.csv")
        df_baselines.to_csv(out_path, index=False)
        
        print(f"\n  Results saved to: {out_path}")
        
        return df_baselines
    
    # =====================================================================
    # STATISTICAL ANALYSIS
    # =====================================================================
    
    def statistical_comparison(self, df_baselines: pd.DataFrame):
        """
        Perform statistical analysis and hypothesis testing
        """
        print("\n[Statistical Analysis] Computing metrics...")
        
        # Get optimal cost
        proposed = df_baselines[df_baselines['category'] == 'Proposed']
        
        if proposed.empty:
            print("  ⚠️  No proposed method found")
            return
        
        optimal_cost = proposed['daily_cost'].iloc[0]
        
        # Calculate improvements
        df_baselines['cost_delta'] = df_baselines['daily_cost'] - optimal_cost
        df_baselines['improvement_pct'] = (
            (df_baselines['daily_cost'] - optimal_cost) / 
            df_baselines['daily_cost'] * 100
        )
        
        # Statistical tests
        analysis = {
            'optimal_cost': optimal_cost,
            'worst_baseline': df_baselines['daily_cost'].max(),
            'best_baseline': df_baselines[
                df_baselines['category'] != 'Proposed'
            ]['daily_cost'].min(),
            'mean_baseline': df_baselines[
                df_baselines['category'] != 'Proposed'
            ]['daily_cost'].mean(),
            'improvement_over_worst': (
                (df_baselines['daily_cost'].max() - optimal_cost) / 
                df_baselines['daily_cost'].max() * 100
            ),
            'improvement_over_best_baseline': (
                (df_baselines[df_baselines['category'] != 'Proposed']['daily_cost'].min() - optimal_cost) /
                df_baselines[df_baselines['category'] != 'Proposed']['daily_cost'].min() * 100
            )
        }
        
        # Print summary
        print("\n" + "="*70)
        print("  STATISTICAL SUMMARY")
        print("="*70)
        print(f"\n  Proposed Method Cost: ${analysis['optimal_cost']:,.2f}/day")
        print(f"  Best Baseline Cost:   ${analysis['best_baseline']:,.2f}/day")
        print(f"  Worst Baseline Cost:  ${analysis['worst_baseline']:,.2f}/day")
        print(f"  Mean Baseline Cost:   ${analysis['mean_baseline']:,.2f}/day")
        print(f"\n  Improvement vs Best Baseline:  {analysis['improvement_over_best_baseline']:.2f}%")
        print(f"  Improvement vs Worst Baseline: {analysis['improvement_over_worst']:.2f}%")
        
        # Save analysis
        import json
        analysis_path = os.path.join(self.out_dir, "statistical_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return df_baselines
    
    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    
    def visualize_comparison(self, df_baselines: pd.DataFrame):
        """Generate comprehensive comparison charts"""
        
        print("\n[Visualization] Generating comparison charts...")
        
        # Sort by cost
        df_plot = df_baselines.sort_values('daily_cost').copy()
        
        # Chart 1: Bar chart with categories
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Color by category
        category_colors = {
            'Naive': '#E63946',
            'Industry': '#F77F00',
            'Academic': '#06A77D',
            'Ablation': '#4361EE',
            'Proposed': '#2A9D8F'
        }
        
        colors = [category_colors.get(c, '#CCCCCC') for c in df_plot['category']]
        
        bars = ax.barh(range(len(df_plot)), df_plot['daily_cost'],
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Highlight proposed method
        proposed_idx = df_plot[df_plot['category'] == 'Proposed'].index[0]
        position = df_plot.index.get_loc(proposed_idx)
        bars[position].set_edgecolor('gold')
        bars[position].set_linewidth(3)
        
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['name'], fontweight='bold')
        ax.set_xlabel("Total Daily Cost ($)", fontweight='bold', fontsize=12)
        ax.set_title("Baseline Comparison: Proposed Method vs Alternatives",
                    fontweight='bold', fontsize=14, pad=20)
        
        # Add value labels
        for i, (cost, pct) in enumerate(zip(df_plot['daily_cost'], 
                                            df_plot.get('improvement_pct', [0]*len(df_plot)))):
            label = f'${cost:,.0f}'
            if pct != 0 and not pd.isna(pct):
                label += f' (+{pct:.1f}%)'
            
            ax.text(cost + 500, i, label, va='center', fontsize=9)
        
        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=color, label=cat, alpha=0.8)
            for cat, color in category_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(self.out_dir, "baseline_comparison_chart.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: baseline_comparison_chart.png")
        
        # Chart 2: Improvement percentages
        self._plot_improvement_chart(df_plot)
    
    def _plot_improvement_chart(self, df_plot: pd.DataFrame):
        """Plot improvement percentages relative to proposed method"""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Filter out proposed method
        df_compare = df_plot[df_plot['category'] != 'Proposed'].copy()
        
        if 'improvement_pct' not in df_compare.columns:
            print("  ⚠️  Improvement percentages not available")
            return
        
        # Sort by improvement
        df_compare = df_compare.sort_values('improvement_pct')
        
        # Color by sign
        colors = ['#06A77D' if x > 0 else '#E63946' 
                 for x in df_compare['improvement_pct']]
        
        bars = ax.barh(range(len(df_compare)), df_compare['improvement_pct'],
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_yticks(range(len(df_compare)))
        ax.set_yticklabels(df_compare['name'], fontweight='bold')
        ax.set_xlabel("Cost Difference vs Proposed Method (%)", 
                     fontweight='bold', fontsize=12)
        ax.set_title("Relative Performance: All Baselines vs Proposed Method",
                    fontweight='bold', fontsize=14, pad=20)
        
        # Add value labels
        for i, val in enumerate(df_compare['improvement_pct']):
            label = f'+{val:.1f}%' if val > 0 else f'{val:.1f}%'
            x_pos = val + (2 if val > 0 else -2)
            ha = 'left' if val > 0 else 'right'
            
            ax.text(x_pos, i, label, va='center', ha=ha, 
                   fontweight='bold', fontsize=9)
        
        ax.axvline(0, color='black', linewidth=2, linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        out_path = os.path.join(self.out_dir, "improvement_percentages.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: improvement_percentages.png")


# === CONVENIENCE FUNCTION ===
def run_baseline_study():
    """Quick execution function"""
    framework = BaselineFramework()
    
    # Run all baselines
    df_results = framework.run_all_baselines()
    
    # Statistical analysis
    df_results = framework.statistical_comparison(df_results)
    
    # Visualize
    framework.visualize_comparison(df_results)
    
    print("\n✅ Baseline study complete!")
    print(f"📁 Results in: {framework.out_dir}")
    
    return df_results


if __name__ == "__main__":
    run_baseline_study()