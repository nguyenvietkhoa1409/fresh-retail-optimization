# src/analysis/pareto_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.settings import ProjectConfig as Cfg

class ParetoAnalyzer:
    """
    PARETO & REVENUE CONTRIBUTION ANALYZER
    Goal: Prove that the small subset of ML items drives a significant 
    portion of Total Volume/Revenue (The "Vital Few").
    """
    
    def __init__(self):
        self.catalog = None
        self.forecast = None
        
    def run(self):
        print("\n[Analysis] Starting Pareto Analysis (ML Importance)...")
        
        # 1. Load Data
        self._load_data()
        
        if self.forecast is None or self.catalog is None:
            print("  -> [Error] Missing artifacts. Run pipeline first.")
            return

        # 2. Prepare Aggregated Data
        df = self._aggregate_data()
        
        # 3. Calculate Stats
        stats = self._calculate_contribution(df)
        
        # 4. Generate Pareto Plot
        self._plot_pareto(df)
        
        # 5. Print Executive Summary
        self._print_summary(stats)
        
    def _load_data(self):
        # Load Forecast (Demand & Method)
        fc_path = os.path.join(Cfg.ARTIFACTS_DIR, "future_forecast.parquet")
        if os.path.exists(fc_path):
            self.forecast = pd.read_parquet(fc_path)
            self.forecast['product_id'] = self.forecast['product_id'].astype(str)
        
        # Load Catalog (Price)
        cat_path = os.path.join(Cfg.ARTIFACTS_DIR, "master_product_catalog.parquet")
        if os.path.exists(cat_path):
            self.catalog = pd.read_parquet(cat_path)
            self.catalog['product_id'] = self.catalog['product_id'].astype(str)

    def _aggregate_data(self):
        print("  -> Aggregating Demand & Revenue...")
        
        # Merge Price into Forecast
        # We use 'left' merge on forecast
        df = self.forecast.merge(self.catalog[['product_id', 'price']], on='product_id', how='left')
        
        # Fill missing price with mean or default
        avg_price = df['price'].mean()
        df['price'] = df['price'].fillna(avg_price if not pd.isna(avg_price) else 10.0)
        
        # Calculate Estimated Revenue for the forecast horizon
        df['revenue'] = df['predicted_mean'] * df['price']
        
        # Group by Store-Product Pair (The unit of analysis)
        # We verify if 'method' is consistent per pair (it should be)
        agg = df.groupby(['store_id', 'product_id']).agg({
            'predicted_mean': 'sum', # Total Volume over horizon
            'revenue': 'sum',        # Total Revenue over horizon
            'method': 'first'        # ml or sma
        }).reset_index()
        
        return agg

    def _calculate_contribution(self, df):
        total_items = len(df)
        total_vol = df['predicted_mean'].sum()
        total_rev = df['revenue'].sum()
        
        stats = {}
        
        for method in ['ml', 'sma']:
            sub = df[df['method'] == method]
            n_items = len(sub)
            vol = sub['predicted_mean'].sum()
            rev = sub['revenue'].sum()
            
            stats[method] = {
                'items_count': n_items,
                'items_pct': (n_items / total_items) * 100,
                'volume_sum': vol,
                'volume_pct': (vol / total_vol) * 100,
                'revenue_sum': rev,
                'revenue_pct': (rev / total_rev) * 100
            }
            
        return stats

    def _plot_pareto(self, df):
        # Sort by Revenue Descending
        df_sorted = df.sort_values('revenue', ascending=False).reset_index(drop=True)
        
        # Cumulative calculations
        df_sorted['cum_revenue'] = df_sorted['revenue'].cumsum()
        df_sorted['cum_rev_pct'] = 100 * df_sorted['cum_revenue'] / df_sorted['revenue'].sum()
        df_sorted['cum_items_pct'] = 100 * (df_sorted.index + 1) / len(df_sorted)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted['cum_items_pct'], df_sorted['cum_rev_pct'], label='Lorenz Curve (Revenue)', linewidth=2)
        plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Equality Line')
        
        # Highlight ML Cutoff point (approximate)
        # We find where the ML items are concentrated (usually at the top)
        # Since we sorted by revenue, let's see where the top X% (ML count) falls
        n_ml = len(df[df['method']=='ml'])
        pct_ml = (n_ml / len(df)) * 100
        
        # Get Revenue % at that item %
        # Find index closest to pct_ml
        cutoff_idx = int(n_ml) - 1
        if cutoff_idx >= 0 and cutoff_idx < len(df_sorted):
            rev_at_cutoff = df_sorted.iloc[cutoff_idx]['cum_rev_pct']
            
            plt.scatter(pct_ml, rev_at_cutoff, color='red', s=100, zorder=5)
            plt.annotate(f'ML Group\n({pct_ml:.1f}% Items -> {rev_at_cutoff:.1f}% Revenue)', 
                         (pct_ml, rev_at_cutoff), xytext=(pct_ml+5, rev_at_cutoff-10),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        plt.title('Pareto Analysis: Forecast Method Contribution')
        plt.xlabel('Cumulative % of Store-Product Pairs')
        plt.ylabel('Cumulative % of Total Revenue')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        out_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "pareto_revenue_chart.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"  -> Chart saved to {out_path}")
        plt.close()

    def _print_summary(self, stats):
        print("\n" + "="*60)
        print("   STRATEGIC RESOURCE ALLOCATION (PARETO ANALYSIS)")
        print("="*60)
        
        header = f"{'Method':<10} | {'Items':<10} {'(%)':<8} | {'Volume':<10} {'(%)':<8} | {'Revenue':<10} {'(%)':<8}"
        print(header)
        print("-" * 70)
        
        for m in ['ml', 'sma']:
            s = stats[m]
            row = f"{m.upper():<10} | {s['items_count']:<10} {s['items_pct']:<6.1f}% | {s['volume_sum']:<10.0f} {s['volume_pct']:<6.1f}% | ${s['revenue_sum']:<9.0f} {s['revenue_pct']:<6.1f}%"
            print(row)
            
        print("-" * 70)
        
        ml_impact = stats['ml']['revenue_pct'] / stats['ml']['items_pct']
        print(f" * IMPACT FACTOR: ML items are {ml_impact:.1f}x more valuable than average.")
        print(f" * CONCLUSION: Applying ML to the top {stats['ml']['items_pct']:.1f}% of items")
        print(f"               secures {stats['ml']['revenue_pct']:.1f}% of the Total Revenue.")
        print("="*60 + "\n")

if __name__ == "__main__":
    analyzer = ParetoAnalyzer()
    analyzer.run()