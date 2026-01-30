# src/inventory/evaluator.py
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import ProjectConfig as Cfg

class PolicyEvaluator:
    """
    POLICY EVALUATION & BENCHMARKING MODULE (Ex-Post Simulation)
    
    Goal:
    Prove that Data-Driven SAA reduces Total Cost (Waste + Lost Sales) compared 
    to Traditional methods (Naive Forecast & Parametric Safety Stock).
    
    Optimization Update:
    - Vectorized Quantile Calculation: Drastically speeds up SAA simulation by 
      calculating safety factors only for unique Target_SL values per group.
    """

    def __init__(self):
        self.catalog = None
        self.saa_pool = {}
        
    def run(self):
        print("\n[Policy Evaluator] Starting Ex-Post Benchmarking...")
        
        # 1. Load Data
        self.catalog = self._load_catalog()
        residuals_df = self._load_residuals()
        
        if residuals_df.empty:
            print("  -> [Error] No residuals found. Cannot run evaluation.")
            return

        # 2. Prepare Simulation Data
        sim_df = self._prepare_simulation_data(residuals_df)
        
        # 3. Calculate Economics (Cu, Co, Target SL)
        sim_df = self._calculate_economics(sim_df)
        
        # 4. Run Simulation Loop
        results = self._simulate_policies(sim_df)
        
        # 5. Generate Scorecard
        self._print_scorecard(results)
        self._save_results(results)
        
        print("[Policy Evaluator] Benchmarking Complete.")

    def _load_catalog(self):
        cat_path = os.path.join(Cfg.ARTIFACTS_DIR, "master_product_catalog.parquet")
        if os.path.exists(cat_path):
            df = pd.read_parquet(cat_path)
            df['product_id'] = df['product_id'].astype(str)
            return df
        else:
            raise FileNotFoundError("Master Catalog needed for Cost parameters.")

    def _load_residuals(self):
        res_path = os.path.join(Cfg.ARTIFACTS_DIR, "forecast_residuals.parquet")
        if os.path.exists(res_path):
            df = pd.read_parquet(res_path)
            df['product_id'] = df['product_id'].astype(str)
            df['store_id'] = df['store_id'].astype(str)
            return df
        return pd.DataFrame()

    def _prepare_simulation_data(self, residuals_df):
        """
        Aggregate Test Set Data for Simulation.
        """
        print("  -> Preparing simulation scenarios from Test Set...")
        
        # [CRITICAL FIX]: Calculate Scaled Error FIRST before groupby
        # Scaled Error = Error / Forecast (Avoid div by zero)
        residuals_df['scaled_error'] = residuals_df['error'] / np.maximum(residuals_df['y_pred'], 0.1)
        
        # Merge Risk Group
        residuals_df = residuals_df.merge(self.catalog[['product_id', 'risk_group']], on='product_id', how='left')
        residuals_df['risk_group'] = residuals_df['risk_group'].fillna('Normal')
        
        # 1. Calculate Error Stats per Product (for Parametric Baseline)
        prod_stats = residuals_df.groupby('product_id').agg(
            rmse=('error', lambda x: np.sqrt(np.mean(x**2)))
        ).reset_index()
        
        # 2. Build SAA Error Pools (for SAA Policy)
        self.saa_pool = {}
        for rg, group in residuals_df.groupby('risk_group'):
            # Drop NaNs or Infs just in case
            valid_errors = group['scaled_error'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_errors) > 0:
                self.saa_pool[rg] = valid_errors.values

        # 3. Create Simulation DataFrame
        # We simulate each "forecast instance" as a decision point
        sim_df = residuals_df[['store_id', 'product_id', 'risk_group', 'y_pred', 'y_true', 'scaled_error']].copy()
        
        # Merge RMSE for Parametric Policy calculation
        sim_df = sim_df.merge(prod_stats[['product_id', 'rmse']], on='product_id', how='left')
        
        return sim_df

    def _calculate_economics(self, df):
        """Add Cost Parameters"""
        # Merge Price/Cost info
        econ = self.catalog[['product_id', 'price', 'shelf_life']].copy()
        
        df = df.merge(econ, on='product_id', how='left')
        df['price'] = df['price'].fillna(10.0)
        
        # Assumptions:
        # Margin = 30% -> Cu (Profit Opportunity)
        # Cost = 70% -> Co (Waste Cost)
        margin_pct = 0.30
        
        df['Cu'] = df['price'] * margin_pct
        df['Co'] = df['price'] * (1 - margin_pct)
        
        # Optimal Service Level (Newsvendor Formula)
        # SL* = Cu / (Cu + Co)
        df['Target_SL'] = df['Cu'] / (df['Cu'] + df['Co'])
        
        # Fill NaN Target_SL and round to reduce unique values for optimization
        df['Target_SL'] = df['Target_SL'].fillna(0.5).round(4)
        
        return df

    def _simulate_policies(self, df):
        print("  -> Simulating 3 Policies (Naive vs Normal vs SAA)...")
        
        results = []
        
        # --- 1. Naive Policy (Order = Forecast) ---
        df['Q_naive'] = df['y_pred']
        
        # --- 2. Parametric Normal (Order = Forecast + Z * RMSE) ---
        # Z score for Target SL
        df['Z'] = norm.ppf(df['Target_SL'])
        df['Z'] = df['Z'].clip(-3, 3) # Bound Z to avoid extreme outliers
        df['Q_normal'] = df['y_pred'] + (df['Z'] * df['rmse'])
        
        # --- 3. Data-Driven SAA (Order = Forecast * (1 + Quantile_Error)) ---
        df['Q_saa'] = df['y_pred'] # Initialize
        
        unique_groups = df['risk_group'].unique()
        
        # [PERFORMANCE OPTIMIZATION]
        # Instead of calculating quantile for every row (50k+ times),
        # we find unique Target_SL values per group (~100 times) and map them.
        
        for rg in unique_groups:
            mask = df['risk_group'] == rg
            pool = self.saa_pool.get(rg, np.array([0.0]))
            
            if len(pool) == 0: continue
            
            # Get subset
            subset = df[mask]
            
            # Find unique Service Level targets in this subset
            unique_sl_values = subset['Target_SL'].unique()
            
            # Calculate quantile ONLY for unique values
            sl_to_factor_map = {}
            for sl in unique_sl_values:
                try:
                    # np.quantile sorts the pool array, which is heavy. 
                    # Doing this 50 times is much faster than 10,000 times.
                    val = np.quantile(pool, sl)
                except:
                    val = 0.0
                sl_to_factor_map[sl] = val
            
            # Map back to the dataframe (Vectorized lookup)
            safety_factors = subset['Target_SL'].map(sl_to_factor_map)
            
            # Cap safety factor to avoid huge overstock on outliers
            safety_factors = safety_factors.clip(lower=-0.5, upper=2.0)
            
            # Calculate Q_saa
            df.loc[mask, 'Q_saa'] = subset['y_pred'] * (1 + safety_factors)

        # --- CALCULATE COSTS (VECTORIZED) ---
        policies = ['naive', 'normal', 'saa']
        
        sim_summary = []
        
        for p in policies:
            q_col = f'Q_{p}'
            # Constraints: Non-negative and integer units
            qty = np.ceil(df[q_col].clip(lower=0))
            
            # Performance Metrics
            sold = np.minimum(qty, df['y_true'])
            lost = np.maximum(0, df['y_true'] - qty)
            waste = np.maximum(0, qty - df['y_true'])
            
            cost_lost = lost * df['Cu']
            cost_waste = waste * df['Co']
            total_cost = cost_lost + cost_waste
            
            sim_summary.append({
                'Policy': p.upper(),
                'Total_Cost': total_cost.sum(),
                'Total_Waste_Units': waste.sum(),
                'Total_Lost_Units': lost.sum(),
                'Service_Level': sold.sum() / df['y_true'].sum() if df['y_true'].sum() > 0 else 0
            })
            
        return pd.DataFrame(sim_summary)

    def _print_scorecard(self, res_df):
        # Set Naive as baseline
        try:
            base = res_df[res_df['Policy'] == 'NAIVE'].iloc[0]
            base_cost = base['Total_Cost']
            base_waste = base['Total_Waste_Units']
        except IndexError:
            return

        res_df['Cost_Savings_%'] = (base_cost - res_df['Total_Cost']) / base_cost * 100
        res_df['Waste_Reduct_%'] = (base_waste - res_df['Total_Waste_Units']) / base_waste * 100
        
        print("\n" + "="*65)
        print("   INVENTORY POLICY SCORECARD (BENCHMARK)")
        print("="*65)
        
        # Formatting
        display_df = res_df.copy()
        for col in ['Total_Cost']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
        for col in ['Cost_Savings_%', 'Waste_Reduct_%']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
        display_df['Service_Level'] = (display_df["Service_Level"] * 100).apply(lambda x: f"{x:.1f}%")
        print(display_df[['Policy', 'Total_Cost', 'Cost_Savings_%', 'Waste_Reduct_%', 'Service_Level']].to_string(index=False))
        print("-" * 65)
        print(" * Naive: Order = Forecast (Zero Safety Stock)")
        print(" * Normal: Parametric (Assumes Gaussian Error)")
        print(" * SAA: Data-Driven (Uses Empirical Error Distribution)")
        print("="*65 + "\n")

    def _save_results(self, res_df):
        out_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, "policy_benchmark_results.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        res_df.to_csv(out_path, index=False)
        print(f"  -> Detailed benchmark saved to {out_path}")

if __name__ == "__main__":
    evaluator = PolicyEvaluator()
    evaluator.run()