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
    Evaluates multiple baseline forecasting models using walk-forward cross-validation
    on the exact same hold-out windows used by DemandForecaster.

    FAIRNESS FIXES applied vs. original version:
      Bug #2: _aggregate_and_report now aligns LightGBM results via inner-join on
              the exact (store_id, product_id) pairs evaluated by baselines.
      Bug #7: Walk-forward evaluation with FC_N_WF_FOLDS folds; each fold is a
              separate HORIZON-length window stepped back through recent history.
              All models (baselines + LightGBM) are measured on identical windows.
    """

    def __init__(self, out_dir=None, exclude_models=None, include_models=None):
        self.out_dir = out_dir if out_dir is not None else Cfg.FC_COMPARISON_OUT_DIR
        self.exclude_models = [m.lower() for m in (exclude_models or [])]
        self.include_models = [m.lower() for m in (include_models or ["all"])]
        os.makedirs(self.out_dir, exist_ok=True)
        self.raw_results_path = os.path.join(self.out_dir, "comparison_results_raw.csv")
        self.metrics_path     = os.path.join(self.out_dir, "forecast_comparison_metrics.json")
        self.summary_path     = os.path.join(self.out_dir, "forecast_comparison_summary.csv")
        self.n_folds = getattr(Cfg, 'FC_N_WF_FOLDS', 1)
        self.forecaster = DemandForecaster()

    # =========================================================================
    # ENTRY POINT
    # =========================================================================
    def run(self):
        print("\n" + "=" * 60)
        print("   FORECAST MODEL COMPARISON ENGINE  (Fair Walk-Forward)")
        print(f"   Walk-forward folds: {self.n_folds} x {Cfg.HORIZON} days")
        print("=" * 60)

        # 1. Load Data
        self.forecaster._load_catalog()
        df = self.forecaster._load_data()

        # 2. Restrict to ML-eligible items (same population as LightGBM)
        valid_items, _, _ = self.forecaster._segment_data(df)
        df_ml = df[df.set_index(['store_id', 'product_id']).index.isin(valid_items)].copy()

        # 3. Build walk-forward test samples for baselines
        print(f"  -> Building {self.n_folds}-fold walk-forward test batches...")
        test_samples = self._build_wf_test_batch(df_ml, self.n_folds)

        # 4. Limit to top-N unique series by historical volume (Bug #2 helper:
        #    same series selection as LightGBM which sorts by volume in _build_direct_h_windows)
        max_series = getattr(Cfg, 'FC_MAX_SERIES_FOR_SARIMA', 1000)
        unique_pairs = {}
        for s in test_samples:
            key = (s['store_id'], s['product_id'])
            unique_pairs[key] = unique_pairs.get(key, 0) + np.sum(s['y_history'])
        top_pairs = set(
            sorted(unique_pairs, key=unique_pairs.get, reverse=True)[:max_series]
        )
        test_samples = [s for s in test_samples if (s['store_id'], s['product_id']) in top_pairs]
        print(f"  -> Unique series: {len(top_pairs)}, Total fold-samples: {len(test_samples)}")

        # 5. Resume checkpointing — key now includes fold to avoid stale hits
        processed_keys = set()

        def _clean_id(val):
            return str(val).split('.')[0]

        if os.path.exists(self.raw_results_path):
            existing_df = pd.read_csv(self.raw_results_path)
            fold_col = existing_df['fold'] if 'fold' in existing_df.columns else 1
            for _, row in existing_df.iterrows():
                fold_val = row.get('fold', 1)
                k = f"{_clean_id(row['store_id'])}_{_clean_id(row['product_id'])}_{row['method']}_f{fold_val}"
                processed_keys.add(k)
            print(f"  -> Found {len(existing_df)} records already processed. Resuming...")

        # 6. Evaluate Statistical Baselines
        all_models = [
            ('SARIMA',           self._run_sarima),
            ('ETS',              self._run_ets),
            ('Prophet',          self._run_prophet),
            ('SNaive',           self._run_snaive),
            ('SMA-7 (Baseline)', self._run_sma),
        ]
        
        models = []
        for name, func in all_models:
            n_lower = name.lower()
            if "all" not in self.include_models and not any(m in n_lower for m in self.include_models):
                continue
            if any(m in n_lower for m in self.exclude_models):
                continue
            models.append((name, func))

        if not models:
            print("  -> No baseline models selected for evaluation.")

        header = "store_id,product_id,method,horizon,date,y_true,y_pred,fold\n"
        with open(self.raw_results_path, 'a' if processed_keys else 'w') as f:
            if not processed_keys:
                f.write(header)

            for method_name, method_func in models:
                total = len(test_samples)
                done  = sum(
                    1 for s in test_samples
                    if f"{_clean_id(s['store_id'])}_{_clean_id(s['product_id'])}_{method_name}_f{s['fold']}"
                    in processed_keys
                )
                if done == total:
                    print(f"  -> [{method_name}] already completed. Skipping.")
                    continue

                for sample in tqdm(test_samples, desc=f"Evaluating {method_name}"):
                    key = (f"{_clean_id(sample['store_id'])}_{_clean_id(sample['product_id'])}"
                           f"_{method_name}_f{sample['fold']}")
                    if key in processed_keys:
                        continue
                    try:
                        method_func(sample, f)
                    except Exception:
                        pass

        # 7. Ablation: LightGBM without Lag features
        run_lgb = True
        if "all" not in self.include_models and not any("lgb" in m for m in self.include_models):
            run_lgb = False
        if any("lgb" in m for m in self.exclude_models):
            run_lgb = False
            
        if run_lgb:
            print("\n  -> Running LightGBM Ablation (No Lag Features)...")
            self._run_lgb_ablation(df_ml)

        # 8. Aggregate & Plot
        self._aggregate_and_report()

    # =========================================================================
    # WALK-FORWARD TEST BATCH BUILDER  (Bug #7)
    # =========================================================================
    def _build_wf_test_batch(self, df, n_folds):
        """
        For each (store, product) series build n_folds evaluation samples.
        Fold k (0-indexed) tests on days [end-k*H : end-(k-1)*H] where end = len(series).
        History for fold k is everything before the test window.
        """
        df_grouped = df.groupby(['store_id', 'product_id'])
        samples = []
        needed_len = Cfg.SEQ_LEN + n_folds * Cfg.HORIZON

        for (store, prod), group in df_grouped:
            g     = group.sort_values('dt')
            vals  = g['y'].values
            dates = g['dt'].values
            n     = len(vals)
            if n < needed_len:
                continue

            for fold_idx in range(n_folds):
                offset    = fold_idx * Cfg.HORIZON
                end_idx   = n - offset

                truth_slice   = slice(end_idx - Cfg.HORIZON, end_idx)
                history_slice = slice(0, end_idx - Cfg.HORIZON)

                samples.append({
                    'store_id':      store,
                    'product_id':    prod,
                    'fold':          fold_idx + 1,          # 1-indexed
                    'y_history':     vals[history_slice],
                    'dates_history': dates[history_slice],
                    'y_truth':       vals[truth_slice],
                    'dates_truth':   dates[truth_slice],
                })
        return samples

    # =========================================================================
    # BASELINE MODEL RUNNERS
    # =========================================================================
    def _run_sarima(self, sample, fh):
        m = auto_arima(
            sample['y_history'],
            seasonal=Cfg.FC_SARIMA_SEASONAL, m=Cfg.FC_SARIMA_M,
            max_p=Cfg.FC_SARIMA_MAX_P, max_q=Cfg.FC_SARIMA_MAX_Q,
            max_P=1, max_Q=1,
            trace=False, error_action='ignore', suppress_warnings=True,
        )
        y_pred = m.predict(n_periods=Cfg.HORIZON)
        self._write_results(fh, sample, 'SARIMA', y_pred)

    def _run_ets(self, sample, fh):
        m = ExponentialSmoothing(
            sample['y_history'],
            seasonal_periods=7, trend='add', seasonal='add',
            initialization_method="estimated",
        ).fit()
        self._write_results(fh, sample, 'ETS', m.forecast(Cfg.HORIZON))

    def _run_prophet(self, sample, fh):
        df_p = pd.DataFrame({'ds': sample['dates_history'], 'y': sample['y_history']})
        m = Prophet(
            weekly_seasonality=True, yearly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=Cfg.FC_PROPHET_CHANGEPOINT_SCALE,
        )
        m.fit(df_p)
        future   = m.make_future_dataframe(periods=Cfg.HORIZON)
        forecast = m.predict(future)
        self._write_results(fh, sample, 'Prophet', forecast['yhat'].values[-Cfg.HORIZON:])

    def _run_snaive(self, sample, fh):
        y_hist = sample['y_history']
        if len(y_hist) >= 7:
            y_pred = [y_hist[-7 + (h - 1) % 7] for h in range(1, Cfg.HORIZON + 1)]
            self._write_results(fh, sample, 'SNaive', y_pred)

    def _run_sma(self, sample, fh):
        """SMA-7: mean of last 7 history days (no clipping to stay symmetric)."""
        y_hist = sample['y_history']
        if len(y_hist) >= 7:
            sma_val = np.mean(y_hist[-7:])
            self._write_results(fh, sample, 'SMA-7 (Baseline)', [sma_val] * Cfg.HORIZON)

    # =========================================================================
    # CSV WRITER  — includes fold column
    # =========================================================================
    def _write_results(self, fh, sample, method, y_pred):
        fold = sample.get('fold', 1)
        for h, (d, yt, yp) in enumerate(
            zip(sample['dates_truth'], sample['y_truth'], y_pred), start=1
        ):
            yp_floor = max(0.0, yp)   # floor at 0; no upper cap (Bug #4 symmetric fix)
            d_str = pd.to_datetime(d).strftime('%Y-%m-%d')
            fh.write(f"{sample['store_id']},{sample['product_id']},"
                     f"{method},{h},{d_str},{yt},{yp_floor},{fold}\n")
        fh.flush()

    # =========================================================================
    # ABLATION: LightGBM without Lag features
    # =========================================================================
    def _run_lgb_ablation(self, df):
        if os.path.exists(self.raw_results_path):
            existing = pd.read_csv(self.raw_results_path)
            if 'LGB-NoLag' in existing['method'].unique():
                print("  -> LightGBM Ablation already computed. Skipping.")
                return

        req_cols = ['promo_bin', 'is_event', 'discount', 'avg_temperature']
        for c in req_cols:
            if c not in df.columns:
                df[c] = 0.0 if c in ['discount', 'avg_temperature'] else 0

        X, Yh, meta = self.forecaster._build_direct_h_windows(
            df, Cfg.SEQ_LEN, Cfg.HORIZON, Cfg.MAX_PAIRS, Cfg.MIN_DAYS_PAIR, Cfg.MAX_SAMPLES
        )
        if len(X) == 0:
            return

        order = np.argsort(pd.to_datetime(meta[:, 2]))
        X, Yh, meta = X[order], Yh[order], meta[order]
        n = len(X)
        i1, i2 = int(n * 0.80), int(n * 0.90)

        Xtr_df, _, _ = self.forecaster._assemble_features(X[:i1], meta[:i1], df)
        lag_cols = [c for c in Xtr_df.columns if c.startswith('lag_')]
        Xtr_df = Xtr_df.drop(columns=lag_cols)
        Xval_df, _, _ = self.forecaster._assemble_features(X[i1:i2], meta[i1:i2], df)
        Xval_df = Xval_df.drop(columns=lag_cols)

        models, bias_log_h = {}, {}
        for h in range(1, Cfg.HORIZON + 1):
            ytr  = np.log1p(Yh[:i1, h - 1])
            yval = np.log1p(Yh[i1:i2, h - 1])
            params = {**Cfg.LGB_PARAMS, 'verbosity': -1}
            cat_feats = ['store_code', 'prod_code', 'wday', 'month', 'promo_bin', 'is_event']
            dtrain = lgb.Dataset(Xtr_df, label=ytr,  categorical_feature=cat_feats, free_raw_data=False)
            dval   = lgb.Dataset(Xval_df, label=yval, reference=dtrain, free_raw_data=False)
            model  = lgb.train(
                params, dtrain, num_boost_round=1000,
                valid_sets=[dtrain, dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            models[h] = model
            try:
                bias_log_h[h] = float(np.mean(model.predict(Xval_df) - yval))
            except Exception:
                bias_log_h[h] = 0.0

        # Walk-forward eval for ablation (same n_folds)
        results_list = []
        for fold_idx in range(self.n_folds):
            offset = fold_idx * Cfg.HORIZON
            X_test, Y_test, meta_test = self.forecaster._build_test_batch(df, offset=offset)
            if len(X_test) == 0:
                continue
            for h in range(1, Cfg.HORIZON + 1):
                meta_h = meta_test.copy()
                base_dates    = pd.to_datetime(meta_test[:, 2])
                shifted_dates = base_dates + pd.Timedelta(days=h)  # Bug #6 fix
                meta_h[:, 2]  = shifted_dates.strftime('%Y-%m-%d')

                X_df_h, _, _ = self.forecaster._assemble_features(X_test, meta_h, df)
                X_df_h = X_df_h.drop(columns=lag_cols, errors='ignore')

                y_log  = models[h].predict(X_df_h)
                y_pred = np.expm1(y_log - bias_log_h.get(h, 0.0))
                y_pred = np.clip(y_pred, 0.0, None)   # Bug #4: only floor

                results_list.append(pd.DataFrame({
                    'store_id':   meta_test[:, 0],
                    'product_id': meta_test[:, 1],
                    'method':     'LGB-NoLag',
                    'horizon':    h,
                    'date':       shifted_dates,
                    'y_true':     Y_test[:, h - 1],
                    'y_pred':     y_pred,
                    'fold':       fold_idx + 1,
                }))

        if results_list:
            all_res = pd.concat(results_list, ignore_index=True)
            all_res = all_res[['store_id', 'product_id', 'method', 'horizon',
                               'date', 'y_true', 'y_pred', 'fold']]
            all_res.to_csv(self.raw_results_path, mode='a', header=False, index=False)

    # =========================================================================
    # AGGREGATION & REPORTING  — Bug #2 fix: inner-join population alignment
    # =========================================================================
    def _aggregate_and_report(self):
        print("  -> Aggregating and computing metrics...")
        df_res = pd.read_csv(self.raw_results_path)
        if 'fold' not in df_res.columns:
            df_res['fold'] = 1   # backward compat with old checkpoint files

        df_res['method'] = df_res['method'].replace({'LGB-NoLag': 'LightGBM (No Lags)'})

        # ── Bug #2 fix: align LightGBM residuals to the exact same pairs ────
        ml_path = os.path.join(Cfg.ARTIFACTS_DIR, "forecast_residuals.parquet")
        if os.path.exists(ml_path):
            full_res = pd.read_parquet(ml_path)
            ml_test_res = full_res[full_res['method'] == 'ml'].copy()
            ml_test_res = ml_test_res[['store_id', 'product_id', 'method',
                                       'horizon', 'date', 'y_true', 'y_pred']]
            ml_test_res['method'] = 'LightGBM (Proposed)'
            if 'fold' not in ml_test_res.columns:
                ml_test_res['fold'] = 1

            # Inner-join: keep only series present in baseline evaluation
            baseline_pairs = (
                df_res[['store_id', 'product_id']]
                .drop_duplicates()
                .astype(str)
            )
            ml_test_res['store_id']   = ml_test_res['store_id'].astype(str)
            ml_test_res['product_id'] = ml_test_res['product_id'].astype(str)
            ml_test_res = pd.merge(
                ml_test_res, baseline_pairs,
                on=['store_id', 'product_id'], how='inner'
            )
            n_before = len(full_res[full_res['method'] == 'ml'])
            print(f"  -> LightGBM aligned: {n_before} → {len(ml_test_res)} rows "
                  f"(inner-join on {len(baseline_pairs)} baseline pairs)")
            df_all = pd.concat([df_res, ml_test_res], ignore_index=True)
        else:
            print("  [Warn] forecast_residuals.parquet not found. Skipping Proposed LightGBM.")
            df_all = df_res

        # ── Metrics ─────────────────────────────────────────────────────────
        metrics = []
        for method, sub in df_all.groupby('method'):
            y_true = sub['y_true'].values
            y_pred = sub['y_pred'].values
            mae    = np.mean(np.abs(y_true - y_pred))
            rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
            denom  = np.sum(np.abs(y_true))
            wape   = (np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
            n_folds_seen = sub['fold'].nunique() if 'fold' in sub.columns else 1
            metrics.append({
                'Model':     method,
                'WAPE (%)':  round(wape, 2),
                'RMSE':      round(rmse, 3),
                'MAE':       round(mae, 3),
                'Samples':   len(sub),
                'WF Folds':  n_folds_seen,
            })

        df_metrics = pd.DataFrame(metrics).sort_values('WAPE (%)')
        df_metrics.to_csv(self.summary_path, index=False)
        with open(self.metrics_path, 'w') as f:
            json.dump(df_metrics.to_dict(orient='records'), f, indent=2)

        print("\n" + "=" * 60)
        print(f" FAIR FORECAST COMPARISON  ({self.n_folds}-Fold Walk-Forward)")
        print("=" * 60)
        print(df_metrics.to_string(index=False))
        print("=" * 60)

        self._plot_results(df_metrics, df_all)
        print(f"\n  -> Artifacts saved to: {self.out_dir}")

    # =========================================================================
    # PLOTTING
    # =========================================================================
    def _plot_results(self, df_metrics, df_all):
        # 1. WAPE bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_metrics, x='WAPE (%)', y='Model', palette='viridis')
        plt.title(f'WAPE Comparison — {self.n_folds}-Fold Walk-Forward (Lower is Better)')
        for i, v in enumerate(df_metrics['WAPE (%)']):
            plt.text(v + 0.3, i, f"{v:.1f}%", va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "comparison_wape_bar.png"), dpi=300)
        plt.close()

        # 2. Time-series overlay for highest-volume item (fold 1 = most recent)
        try:
            fold1 = df_all[df_all.get('fold', pd.Series(1, index=df_all.index)) == 1] \
                if 'fold' in df_all.columns else df_all
            top_item = fold1.groupby(['store_id', 'product_id'])['y_true'].sum().idxmax()
            df_sub   = fold1[(fold1['store_id']   == top_item[0]) &
                              (fold1['product_id'] == top_item[1])]

            if not df_sub.empty:
                plt.figure(figsize=(12, 6))
                ref_method = df_sub['method'].iloc[0]
                df_true = df_sub[df_sub['method'] == ref_method].sort_values('date')
                plt.plot(df_true['date'], df_true['y_true'],
                         label='Actual', color='black', linewidth=2.5, marker='o')

                if 'LightGBM (Proposed)' in df_sub['method'].unique():
                    df_m = df_sub[df_sub['method'] == 'LightGBM (Proposed)'].sort_values('date')
                    plt.plot(df_m['date'], df_m['y_pred'],
                             label='LightGBM (Proposed)', color='royalblue',
                             linewidth=2, linestyle='-')

                for method in [m for m in df_sub['method'].unique()
                                if m != 'LightGBM (Proposed)']:
                    df_m = df_sub[df_sub['method'] == method].sort_values('date')
                    plt.plot(df_m['date'], df_m['y_pred'],
                             label=method, linestyle='--', alpha=0.75)

                plt.title(
                    f'Forecast Overlay — Most Recent Fold\n'
                    f'Store: {top_item[0]} | Product: {top_item[1]}'
                )
                plt.xlabel('Date'); plt.ylabel('Demand')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "comparison_time_series_overlay.png"),
                            dpi=300)
                plt.close()
        except Exception as e:
            print(f"  [Warn] Failed to plot overlay: {e}")
