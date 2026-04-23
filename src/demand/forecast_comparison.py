import os
import json
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

try:
    from pmdarima import auto_arima
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from prophet import Prophet
    import lightgbm as lgb
except ImportError as e:
    print(f"Missing dependency for baseline comparison. Please install: {e}")

from config.settings import ProjectConfig as Cfg
from src.demand.forecasting import DemandForecaster

class ForecastComparisonEngine:
    """
    Evaluates multiple baseline forecasting models on the exact same hold-out test
    set used by DemandForecaster._build_test_batch().
    Designed to be interrupt-friendly and resumable.
    """
    def __init__(self, out_dir=None):
        self.out_dir = out_dir if out_dir is not None else Cfg.FC_COMPARISON_OUT_DIR
        os.makedirs(self.out_dir, exist_ok=True)
        self.raw_results_path = os.path.join(self.out_dir, "comparison_results_raw.csv")
        self.metrics_path = os.path.join(self.out_dir, "forecast_comparison_metrics.json")
        self.summary_path = os.path.join(self.out_dir, "forecast_comparison_summary.csv")
        
        # Instantiate forecaster to reuse its data loading logic
        self.forecaster = DemandForecaster()

    def run(self):
        print("\n" + "="*60)
        print("   FORECAST MODEL COMPARISON ENGINE")
        print("="*60)
        
        # 1. Load Data
        self.forecaster._load_catalog()
        df = self.forecaster._load_data()
        
        # 2. Get ML eligible items
        valid_items, _, _ = self.forecaster._segment_data(df)
        df_ml = df[df.set_index(['store_id', 'product_id']).index.isin(valid_items)].copy()
        
        # 3. Extract test batch with full history
        print("  -> Building test batches with full history...")
        test_samples = self._build_full_history_test_batch(df_ml)
        
        max_series = getattr(Cfg, 'FC_MAX_SERIES_FOR_SARIMA', 1000)
        if len(test_samples) > max_series:
            # Sort by total historical volume
            test_samples = sorted(test_samples, key=lambda x: np.sum(x['y_history']), reverse=True)[:max_series]
            
        print(f"  -> Total series to evaluate: {len(test_samples)}")
        
        # 4. Load processed keys
        processed_keys = set()
        if os.path.exists(self.raw_results_path):
            existing_df = pd.read_csv(self.raw_results_path)
            for _, row in existing_df.iterrows():
                processed_keys.add(f"{row['store_id']}_{row['product_id']}_{row['method']}")
            print(f"  -> Found {len(existing_df)} records already processed. Resuming...")
            
        # 5. Evaluate Statistical Baselines
        models = [
            ('SARIMA', self._run_sarima),
            ('ETS', self._run_ets),
            ('Prophet', self._run_prophet),
            ('SNaive', self._run_snaive),
            ('SMA-7 (Test Eval)', self._run_sma)
        ]

        with open(self.raw_results_path, 'a' if processed_keys else 'w') as f:
            if not processed_keys:
                f.write("store_id,product_id,method,horizon,date,y_true,y_pred\n")
            
            for method_name, method_func in models:
                # Check if this model is already fully done
                done_count = sum(1 for s in test_samples if f"{s['store_id']}_{s['product_id']}_{method_name}" in processed_keys)
                if done_count == len(test_samples):
                    print(f"  -> [{method_name}] already completed. Skipping.")
                    continue
                    
                for sample in tqdm(test_samples, desc=f"Evaluating {method_name}"):
                    key = f"{sample['store_id']}_{sample['product_id']}_{method_name}"
                    if key in processed_keys:
                        continue
                    
                    try:
                        method_func(sample, f)
                    except Exception:
                        pass
                
        # 6. Run Ablation (LightGBM without Lags)
        print("\n  -> Running LightGBM Ablation (No Lag Features)...")
        self._run_lgb_ablation(df_ml)
        
        # 7. Aggregate & Plot
        self._aggregate_and_report()

    def _build_full_history_test_batch(self, df):
        df_grouped = df.groupby(['store_id', 'product_id'])
        samples = []
        for (store, prod), group in df_grouped:
            g = group.sort_values('dt')
            vals = g['y'].values
            dates = g['dt'].values
            n = len(vals)
            needed = Cfg.SEQ_LEN + Cfg.HORIZON
            if n < needed: continue
            
            truth_indices = slice(n - Cfg.HORIZON, n)
            history_indices = slice(0, n - Cfg.HORIZON)
            
            y_truth = vals[truth_indices]
            dates_truth = dates[truth_indices]
            
            y_history = vals[history_indices]
            dates_history = dates[history_indices]
            
            samples.append({
                'store_id': store,
                'product_id': prod,
                'y_history': y_history,
                'dates_history': dates_history,
                'y_truth': y_truth,
                'dates_truth': dates_truth
            })
        return samples

    def _run_sarima(self, sample, file_handle):
        sarima = auto_arima(sample['y_history'], seasonal=Cfg.FC_SARIMA_SEASONAL, m=Cfg.FC_SARIMA_M, 
                            max_p=Cfg.FC_SARIMA_MAX_P, max_q=Cfg.FC_SARIMA_MAX_Q, 
                            max_P=1, max_Q=1, 
                            trace=False, error_action='ignore', suppress_warnings=True)
        y_pred = sarima.predict(n_periods=Cfg.HORIZON)
        self._write_results(file_handle, sample['store_id'], sample['product_id'], 'SARIMA', sample['dates_truth'], sample['y_truth'], y_pred)

    def _run_ets(self, sample, file_handle):
        ets = ExponentialSmoothing(sample['y_history'], seasonal_periods=7, trend='add', seasonal='add', initialization_method="estimated").fit()
        y_pred = ets.forecast(Cfg.HORIZON)
        self._write_results(file_handle, sample['store_id'], sample['product_id'], 'ETS', sample['dates_truth'], sample['y_truth'], y_pred)

    def _run_prophet(self, sample, file_handle):
        df_prophet = pd.DataFrame({'ds': sample['dates_history'], 'y': sample['y_history']})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False, 
                    changepoint_prior_scale=Cfg.FC_PROPHET_CHANGEPOINT_SCALE)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=Cfg.HORIZON)
        forecast = m.predict(future)
        y_pred = forecast['yhat'].values[-Cfg.HORIZON:]
        self._write_results(file_handle, sample['store_id'], sample['product_id'], 'Prophet', sample['dates_truth'], sample['y_truth'], y_pred)

    def _run_snaive(self, sample, file_handle):
        y_hist = sample['y_history']
        if len(y_hist) >= 7:
            y_pred = [y_hist[-7 + (h-1) % 7] for h in range(1, Cfg.HORIZON + 1)]
            self._write_results(file_handle, sample['store_id'], sample['product_id'], 'SNaive', sample['dates_truth'], sample['y_truth'], y_pred)

    def _run_sma(self, sample, file_handle):
        y_hist = sample['y_history']
        if len(y_hist) >= 7:
            sma_val = np.mean(y_hist[-7:])
            y_pred = [sma_val] * Cfg.HORIZON
            self._write_results(file_handle, sample['store_id'], sample['product_id'], 'SMA-7 (Test Eval)', sample['dates_truth'], sample['y_truth'], y_pred)

    def _write_results(self, file_handle, store, prod, method, dates, y_true, y_pred):
        for h, (d, yt, yp) in enumerate(zip(dates, y_true, y_pred), start=1):
            yp_clipped = max(0.0, yp)
            d_str = pd.to_datetime(d).strftime('%Y-%m-%d')
            file_handle.write(f"{store},{prod},{method},{h},{d_str},{yt},{yp_clipped}\n")
        file_handle.flush()

    def _run_lgb_ablation(self, df):
        # Prevent running ablation repeatedly if it's already in the raw results
        if os.path.exists(self.raw_results_path):
            existing = pd.read_csv(self.raw_results_path)
            if 'LGB-NoLag' in existing['method'].unique():
                print("  -> LightGBM Ablation already computed. Skipping.")
                return

        X, Yh, meta = self.forecaster._build_direct_h_windows(
            df, Cfg.SEQ_LEN, Cfg.HORIZON, Cfg.MAX_PAIRS, Cfg.MIN_DAYS_PAIR, Cfg.MAX_SAMPLES
        )
        if len(X) == 0: return
        dates = pd.to_datetime(meta[:, 2])
        order = np.argsort(dates)
        X_ord, Yh_ord, meta_ord = X[order], Yh[order], meta[order]
        n = len(X_ord)
        i1, i2 = int(n * 0.80), int(n * 0.90)
        
        # Assemble feature dfs and aggressively drop ALL lags
        Xtr_df, _, _ = self.forecaster._assemble_features(X_ord[:i1], meta_ord[:i1], df)
        lag_cols = [c for c in Xtr_df.columns if c.startswith('lag_')]
        Xtr_df = Xtr_df.drop(columns=lag_cols)
        
        Xval_df, _, _ = self.forecaster._assemble_features(X_ord[i1:i2], meta_ord[i1:i2], df)
        Xval_df = Xval_df.drop(columns=lag_cols)
        
        models = {}
        bias_log_h = {}
        for h in range(1, Cfg.HORIZON + 1):
            ytr = np.log1p(Yh_ord[:i1, h-1])
            yval = np.log1p(Yh_ord[i1:i2, h-1])
            
            cat_feats = ['store_code', 'prod_code', 'wday', 'month', 'promo_bin', 'is_event']
            dtrain = lgb.Dataset(Xtr_df, label=ytr, categorical_feature=cat_feats, free_raw_data=False)
            dval = lgb.Dataset(Xval_df, label=yval, reference=dtrain, free_raw_data=False)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            
            params = Cfg.LGB_PARAMS.copy()
            params['verbosity'] = -1
            
            model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain, dval], callbacks=callbacks)
            models[h] = model
            
            try:
                preds = model.predict(Xval_df)
                bias = float(np.mean(preds - yval))
            except: bias = 0.0
            bias_log_h[h] = bias

        # Eval on Test Set
        X_test, Y_test, meta_test = self.forecaster._build_test_batch(df)
        if len(X_test) == 0: return
        
        results_list = []
        for h in range(1, Cfg.HORIZON + 1):
            meta_h = meta_test.copy()
            base_dates = pd.to_datetime(meta_test[:, 2])
            shifted_dates = base_dates + pd.Timedelta(days=h-1)
            meta_h[:, 2] = shifted_dates.strftime('%Y-%m-%d')
            
            X_df_h, _, _ = self.forecaster._assemble_features(X_test, meta_h, df)
            X_df_h = X_df_h.drop(columns=lag_cols)
            
            y_log = models[h].predict(X_df_h)
            bias = bias_log_h.get(h, 0.0)
            y_pred = np.expm1(y_log - bias)
            
            y_true = Y_test[:, h-1]
            y_pred = np.clip(y_pred, 0.0, 1000.0)
            
            results_list.append(pd.DataFrame({
                'store_id': meta_test[:, 0], 'product_id': meta_test[:, 1],
                'method': 'LGB-NoLag', 'horizon': h, 'date': shifted_dates,
                'y_true': y_true, 'y_pred': y_pred
            }))
            
        all_res = pd.concat(results_list, ignore_index=True)
        # Order columns to match CSV
        all_res = all_res[['store_id', 'product_id', 'method', 'horizon', 'date', 'y_true', 'y_pred']]
        all_res.to_csv(self.raw_results_path, mode='a', header=False, index=False)

    def _aggregate_and_report(self):
        print("  -> Aggregating and computing metrics...")
        df_res = pd.read_csv(self.raw_results_path)
        
        # Extract main LightGBM results from forecast residuals
        ml_path = os.path.join(Cfg.ARTIFACTS_DIR, "forecast_residuals.parquet")
        if os.path.exists(ml_path):
            full_res = pd.read_parquet(ml_path)
            ml_test_res = full_res[full_res['method'] == 'ml'].copy()
            ml_test_res = ml_test_res[['store_id', 'product_id', 'method', 'horizon', 'date', 'y_true', 'y_pred']]
            ml_test_res['method'] = 'LightGBM (Proposed)'
            
            df_res['method'] = df_res['method'].replace({'LGB-NoLag': 'LightGBM (No Lags)'})
            df_all = pd.concat([df_res, ml_test_res], ignore_index=True)
        else:
            print("  [Warn] forecast_residuals.parquet not found. Computing metrics without Proposed ML.")
            df_all = df_res

        metrics = []
        for method, sub in df_all.groupby('method'):
            y_true = sub['y_true'].values
            y_pred = sub['y_pred'].values
            
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            denom = np.sum(np.abs(y_true))
            wape = (np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
            
            metrics.append({
                'Model': method,
                'WAPE (%)': round(wape, 2),
                'RMSE': round(rmse, 3),
                'MAE': round(mae, 3),
                'Samples': len(sub)
            })
            
        df_metrics = pd.DataFrame(metrics).sort_values('WAPE (%)')
        
        df_metrics.to_csv(self.summary_path, index=False)
        with open(self.metrics_path, 'w') as f:
            json.dump(df_metrics.to_dict(orient='records'), f, indent=2)
            
        print("\n" + "="*50)
        print(" FORECAST COMPARISON RESULTS (ML-Eligible subset)")
        print("="*50)
        print(df_metrics.to_string(index=False))
        print("="*50)
        
        # Plotting
        self._plot_results(df_metrics, df_all)
        print(f"\n  -> Artifacts saved to: {self.out_dir}")

    def _plot_results(self, df_metrics, df_all):
        # 1. Bar Chart WAPE
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_metrics, x='WAPE (%)', y='Model', palette='viridis')
        plt.title('WAPE Comparison across Models (Lower is Better)')
        for i, v in enumerate(df_metrics['WAPE (%)']):
            plt.text(v + 0.5, i, f"{v:.1f}%", va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "comparison_wape_bar.png"), dpi=300)
        plt.close()
        
        # 2. Time Series Overlay for Top 1 Item (by volume)
        try:
            top_item = df_all.groupby(['store_id', 'product_id'])['y_true'].sum().idxmax()
            df_sub = df_all[(df_all['store_id'] == top_item[0]) & (df_all['product_id'] == top_item[1])]
            
            if not df_sub.empty:
                plt.figure(figsize=(12, 6))
                df_true = df_sub[df_sub['method'] == df_sub['method'].iloc[0]].sort_values('date')
                plt.plot(df_true['date'], df_true['y_true'], label='Actual Demand', color='black', linewidth=2.5, marker='o')
                
                # Plot proposed first for visibility
                if 'LightGBM (Proposed)' in df_sub['method'].unique():
                    df_m = df_sub[df_sub['method'] == 'LightGBM (Proposed)'].sort_values('date')
                    plt.plot(df_m['date'], df_m['y_pred'], label='LightGBM (Proposed)', color='blue', linewidth=2, linestyle='-')
                
                other_methods = [m for m in df_sub['method'].unique() if m != 'LightGBM (Proposed)']
                for method in other_methods:
                    df_m = df_sub[df_sub['method'] == method].sort_values('date')
                    plt.plot(df_m['date'], df_m['y_pred'], label=method, linestyle='--', alpha=0.7)
                    
                plt.title(f'Forecast Overlay (Test Period) - High Volume Item\nStore: {top_item[0]} | Product: {top_item[1]}')
                plt.xlabel('Date')
                plt.ylabel('Demand')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "comparison_time_series_overlay.png"), dpi=300)
                plt.close()
        except Exception as e:
            print(f"  [Warn] Failed to plot overlay: {e}")
