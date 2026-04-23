# src/demand/forecasting.py
"""
DEMAND FORECASTING ENGINE (v5.0 - Hybrid Segmentation)
------------------------------------------------------
Scientific Approach:
1. Data Qualification: Split products into 'Forecastable' (ML) and 'Intermittent' (SMA).
   - ML Criteria: Mean Sales >= 1.0 AND Density >= 0.6 (configurable).
2. Hybrid Pipeline:
   - ML Branch: LightGBM Direct Multi-horizon (Train/Valid/Test Split).
   - SMA Branch: Moving Average (Robust baseline for noisy data).
3. Universal Coverage: Outputs cover 100% of SKUs for Inventory Planning.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from config.settings import ProjectConfig as Cfg

class DemandForecaster:
    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_FORECAST, exist_ok=True)
        self.models = {}         
        self.bias_log_h = {}     
        self.store_map = {}      
        self.prod_map = {}       
        self.global_cap = 1000.0 
        self.product_catalog = None
        
        # --- SEGMENTATION THRESHOLDS ---
        # Can be moved to Cfg later
        self.THRES_MEAN_SALES = Cfg.THRES_MEAN_SALES   # Min avg daily sales
        self.THRES_DATA_DENSITY = Cfg.THRES_DATA_DENSITY # Min % of days with sales > 0

    def run(self):
        print("\n[Forecasting] Starting Hybrid Forecasting Pipeline (ML + SMA)...")
        
        # 0. Load Catalog & Data
        self._load_catalog()
        df = self._load_data()
        
        # 1. Segmentation (The Scientific Filter)
        print("  -> Performing Data Qualification & Segmentation...")
        valid_items, fallback_items, seg_stats = self._segment_data(df)
        
        print(f"     [Segmentation Stats]")
        print(f"     - ML Eligible (Forecastable): {len(valid_items)} pairs")
        print(f"     - SMA Fallback (Intermittent): {len(fallback_items)} pairs")
        print(f"     - ML Coverage: {len(valid_items) / (len(valid_items)+len(fallback_items)):.1%}")

        # 2. Run Pipelines
        # A. ML Branch
        df_ml = df[df.set_index(['store_id', 'product_id']).index.isin(valid_items)].copy()
        ml_future, ml_residuals, ml_metrics = self._run_ml_pipeline(df_ml)
        
        # B. SMA Branch
        df_sma = df[df.set_index(['store_id', 'product_id']).index.isin(fallback_items)].copy()
        sma_future, sma_residuals = self._run_sma_pipeline(df_sma)
        
        # 3. Combine Results (Unification)
        print("  -> Combining results from ML and SMA branches...")
        
        # Future Forecasts
        full_future = pd.concat([ml_future, sma_future], ignore_index=True)
        out_future = os.path.join(Cfg.ARTIFACTS_DIR, "future_forecast.parquet")
        full_future.to_parquet(out_future)
        print(f"     Saved Combined Forecasts: {out_future} ({len(full_future)} rows)")
        
        # Residuals (For SAA Planner)
        full_residuals = pd.concat([ml_residuals, sma_residuals], ignore_index=True)
        out_res = os.path.join(Cfg.ARTIFACTS_DIR, "forecast_residuals.parquet")
        full_residuals.to_parquet(out_res)
        print(f"     Saved Combined Residuals: {out_res} ({len(full_residuals)} rows)")
        
        # 4. Reporting (Focus on ML Performance)
        # We only report ML metrics because SMA on noisy data is naturally high-error
        metrics_path = os.path.join(Cfg.OUT_DIR_FORECAST, "ml_performance_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(ml_metrics, f, indent=2)
            
        print(f"\n  -> Pipeline Complete. ML Metrics saved to {metrics_path}")
        return ml_metrics

    # -------------------------------------------------------------------------
    # 1. SEGMENTATION LOGIC
    # -------------------------------------------------------------------------
    def _segment_data(self, df):
        """
        Classify Store-Product pairs into 'Forecastable' (ML) vs 'Intermittent' (SMA).
        Criteria: Mean Sales > X AND Data Density > Y
        """
        # Calculate stats per pair
        stats = df.groupby(['store_id', 'product_id'])['y'].agg(
            mean_sales='mean',
            count='count',
            non_zeros=lambda x: (x > 0).sum()
        ).reset_index()
        
        stats['density'] = stats['non_zeros'] / stats['count']
        
        # Apply Filters
        mask_ml = (stats['mean_sales'] >= self.THRES_MEAN_SALES) & \
                  (stats['density'] >= self.THRES_DATA_DENSITY)
        
        valid_pairs = list(zip(stats[mask_ml]['store_id'], stats[mask_ml]['product_id']))
        fallback_pairs = list(zip(stats[~mask_ml]['store_id'], stats[~mask_ml]['product_id']))
        
        return valid_pairs, fallback_pairs, stats

    # -------------------------------------------------------------------------
    # 2A. ML PIPELINE (LightGBM - Batch Optimized)
    # -------------------------------------------------------------------------
    def _run_ml_pipeline(self, df):
        if df.empty:
            print("     [Warn] No items qualified for ML. Skipping ML branch.")
            return pd.DataFrame(), pd.DataFrame(), {}
            
        print("  -> [ML Branch] Building training windows...")
        X, Yh, meta = self._build_direct_h_windows(
            df, Cfg.SEQ_LEN, Cfg.HORIZON, Cfg.MAX_PAIRS, Cfg.MIN_DAYS_PAIR, Cfg.MAX_SAMPLES
        )
        
        if len(X) == 0: return pd.DataFrame(), pd.DataFrame(), {}

        # Strict Split
        dates = pd.to_datetime(meta[:, 2])
        order = np.argsort(dates)
        X_ord, Yh_ord, meta_ord = X[order], Yh[order], meta[order]
        
        n = len(X_ord)
        i1, i2 = int(n * 0.80), int(n * 0.90)
        
        X_tr, y_tr_1step = X_ord[:i1], np.log1p(Yh_ord[:i1, 0])
        X_val, y_val_1step = X_ord[i1:i2], np.log1p(Yh_ord[i1:i2, 0])
        meta_tr, meta_val = meta_ord[:i1], meta_ord[i1:i2]

        Xtr_df, self.store_map, self.prod_map = self._assemble_features(X_tr, meta_tr, df)
        Xval_df, _, _ = self._assemble_features(X_val, meta_val, df)
        
        # Train
        print("     [ML Branch] Training Models...")
        base_model = self._train_lgb_single(Xtr_df, y_tr_1step, Xval_df, y_val_1step)
        self._calibrate(base_model, Xval_df, y_val_1step, Yh_ord[:i2])
        
        self._train_direct_horizons(X_ord, Yh_ord, meta_ord, df, i1, i2)
        
        # Evaluate (Generate Residuals for ML items)
        print("     [ML Branch] Evaluating & Generating Residuals...")
        residuals_df, metrics = self._evaluate_ml_batch(df)
        
        # Future Forecast
        print("     [ML Branch] Forecasting Future...")
        future_df = self._generate_future_ml_batch(df)
        
        return future_df, residuals_df, metrics

    def _evaluate_ml_batch(self, df):
        """
        Evaluate ML models across N walk-forward folds.
        FAIRNESS FIXES:
          - Bug #7: Walk-forward CV with FC_N_WF_FOLDS folds instead of single window.
          - Bug #6: Date shift corrected to `days=h` (was `days=h-1`).
          - Bug #4: Upper-bound clipping removed; only floor at 0 to match baselines.
        """
        n_folds = getattr(Cfg, 'FC_N_WF_FOLDS', 1)
        print(f"     [ML Branch] Walk-Forward Evaluation over {n_folds} fold(s)...")

        all_results = []
        for fold_idx in range(n_folds):
            # fold_idx=0 → most recent window (original test)
            # fold_idx=k → window ending k*HORIZON days earlier
            offset = fold_idx * Cfg.HORIZON
            X_test, Y_test, meta_test = self._build_test_batch(df, offset=offset)
            if len(X_test) == 0:
                continue

            for h in range(1, Cfg.HORIZON + 1):
                meta_h = meta_test.copy()
                base_dates = pd.to_datetime(meta_test[:, 2])
                # Bug #6 fix: prediction for day h after anchor, so +h (not +h-1)
                shifted_dates = base_dates + pd.Timedelta(days=h)
                meta_h[:, 2] = shifted_dates.strftime('%Y-%m-%d')

                X_df_h, _, _ = self._assemble_features(X_test, meta_h, df)

                if h in self.models:
                    y_log = self.models[h].predict(X_df_h)
                    bias = self.bias_log_h.get(h, 0.0)
                    y_pred = np.expm1(y_log - bias)
                else:
                    y_pred = np.zeros(len(X_test))

                # Bug #4 fix: only floor at 0; no upper cap so evaluation is
                # symmetric with baselines (SARIMA/ETS/SMA have no cap).
                y_pred = np.clip(y_pred, 0.0, None)

                y_true = Y_test[:, h - 1]
                error  = y_true - y_pred

                all_results.append(pd.DataFrame({
                    'store_id':   meta_test[:, 0],
                    'product_id': meta_test[:, 1],
                    'horizon': h,
                    'date':    shifted_dates,
                    'y_true':  y_true,
                    'y_pred':  y_pred,
                    'error':   error,
                    'method':  'ml',
                    'fold':    fold_idx + 1,
                }))

        if not all_results:
            return pd.DataFrame(), {}

        all_res = pd.concat(all_results, ignore_index=True)
        metrics  = self._calc_grouped_metrics(all_res)
        self._save_product_metrics(all_res, "ml_product_accuracy.csv")
        return all_res, metrics

    def _generate_future_ml_batch(self, df):
        """Generate T+1..T+H forecasts for ML items"""
        # (Reusing previous logic but simplified for brevity)
        df_grouped = df.groupby(['store_id', 'product_id'])
        X_list, meta_list = [], []
        
        for (store, prod), group in df_grouped:
            g = group.sort_values('dt')
            vals = g['y'].values; dates = g['dt'].values
            if len(vals) < Cfg.SEQ_LEN: continue
            X_list.append(vals[-Cfg.SEQ_LEN:])
            meta_list.append((store, prod, str(pd.to_datetime(dates[-1]).date())))
            
        if not X_list: return pd.DataFrame()
        
        Batch_X = np.stack(X_list)
        Batch_Meta = np.array(meta_list, dtype=object)
        future_results = []
        
        for h in range(1, Cfg.HORIZON + 1):
            meta_h = Batch_Meta.copy()
            base_dates = pd.to_datetime(Batch_Meta[:, 2])
            target_dates = base_dates + pd.Timedelta(days=h)
            meta_h[:, 2] = target_dates.strftime('%Y-%m-%d')
            
            X_df_h, _, _ = self._assemble_features(Batch_X, meta_h, df)
            
            if h in self.models:
                y_log = self.models[h].predict(X_df_h)
                bias = self.bias_log_h.get(h, 0.0)
                y_pred = np.expm1(y_log - bias)
            else: y_pred = np.zeros(len(Batch_X))
                
            recent_mean = np.mean(Batch_X[:, -7:], axis=1)
            max_allowed = np.minimum(self.global_cap * 1.2, np.maximum(1e-6, recent_mean) * 6.0)
            y_pred = np.clip(y_pred, 0.0, max_allowed)
            
            future_results.append(pd.DataFrame({
                'store_id': Batch_Meta[:, 0], 'product_id': Batch_Meta[:, 1],
                'date': target_dates, 'predicted_mean': y_pred, 'method': 'ml'
            }))
            
        return pd.concat(future_results, ignore_index=True)

    # -------------------------------------------------------------------------
    # 2B. SMA PIPELINE (Simple Moving Average - Robust Fallback)
    # -------------------------------------------------------------------------
    def _run_sma_pipeline(self, df):
        """
        Runs SMA-7 forecasting for intermittent/noisy items.
        1. Future Forecast = Mean of last 7 days.
        2. Residuals = Errors of SMA-7 on historical window (last 30 days).
        """
        if df.empty: return pd.DataFrame(), pd.DataFrame()
        print(f"  -> [SMA Branch] Running Moving Average for {df['product_id'].nunique()} items...")
        
        df_grouped = df.groupby(['store_id', 'product_id'])
        future_rows = []
        residual_rows = []
        
        WINDOW = 7
        RES_HISTORY = 30 # Look back 30 days to generate residual samples
        
        for (store, prod), group in df_grouped:
            g = group.sort_values('dt')
            vals = g['y'].values
            dates = g['dt'].values
            
            if len(vals) == 0: continue
            
            # --- 1. Future Forecast ---
            # Robust: Take last available days up to 7
            last_window = vals[-WINDOW:]
            forecast_val = float(np.mean(last_window)) if len(last_window) > 0 else 0.0
            last_date = dates[-1]
            
            for h in range(1, Cfg.HORIZON + 1):
                target_date = last_date + pd.Timedelta(days=h)
                future_rows.append({
                    'store_id': store, 'product_id': prod,
                    'date': target_date, 'predicted_mean': forecast_val,
                    'method': 'sma'
                })
            
            # --- 2. Generate Historical Residuals (Proxy) ---
            # To allow SAA to calculate safety stock, we need error samples.
            # We simulate "What if we used SMA-7 in the past month?"
            if len(vals) > WINDOW:
                # Iterate back RES_HISTORY days
                # Start index such that we have at least WINDOW points before it
                start_idx = max(WINDOW, len(vals) - RES_HISTORY)
                
                for t in range(start_idx, len(vals)):
                    # SMA prediction for time t using t-7...t-1
                    past_window = vals[t-WINDOW : t]
                    pred = np.mean(past_window)
                    actual = vals[t]
                    
                    # Store residual
                    # Horizon 1 is enough proxy for flat SMA
                    residual_rows.append({
                        'store_id': store, 'product_id': prod,
                        'horizon': 1, # Dummy horizon
                        'date': dates[t],
                        'y_true': actual, 'y_pred': pred,
                        'error': actual - pred,
                        'method': 'sma'
                    })
                    
        return pd.DataFrame(future_rows), pd.DataFrame(residual_rows)

    # --- SHARED HELPERS ---
    def _save_product_metrics(self, res_df, filename):
        """Helper to save product-level accuracy CSV"""
        prod_metrics = []
        for pid, sub in res_df.groupby('product_id'):
            y_true, y_pred = sub['y_true'].values, sub['y_pred'].values
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            mean_sales = np.mean(y_true)
            mask = y_true > 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
            
            prod_metrics.append({
                'Product ID': pid,
                'Mean Sales': round(mean_sales, 2),
                'RMSE': round(rmse, 2),
                'MAPE (%)': round(mape, 1) if not np.isnan(mape) else 'N/A',
                'Samples': len(sub)
            })
        pd.DataFrame(prod_metrics).sort_values('Mean Sales', ascending=False)\
            .to_csv(os.path.join(Cfg.OUT_DIR_FORECAST, filename), index=False)

    def _calc_grouped_metrics(self, df):
        if df.empty: return {'WAPE': np.nan, 'RMSE': np.nan, 'MAE': np.nan}
        y_true, y_pred = df['y_true'].values, df['y_pred'].values
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        denom = np.sum(np.abs(y_true))
        wape = (np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
        return {"WAPE": float(wape), "RMSE": float(rmse), "MAE": float(mae), "n_samples": len(df)}

    # ... (Keep _load_catalog, _load_data, _build_direct_h_windows, _assemble_features, 
    #      _train_lgb_single, _calibrate, _train_direct_horizons, _build_test_batch 
    #      exactly as they were in v4.4 - No changes needed there) ...
    
    # [NOTE]: Copy-paste the helper methods (_load_catalog, _load_data, etc.) 
    # from previous version v4.4 here to complete the class.
    # I am omitting them here for brevity but they are required.
    
    def _load_catalog(self):
        cat_path = os.path.join(Cfg.ARTIFACTS_DIR, "master_product_catalog.parquet")
        if os.path.exists(cat_path):
            self.product_catalog = pd.read_parquet(cat_path)
            self.product_catalog['product_id'] = self.product_catalog['product_id'].astype(str)
            self.product_catalog = self.product_catalog.set_index('product_id')
            print("  -> Loaded Master Product Catalog.")
        else:
            print("  -> [Warning] Catalog not found.")

    def _load_data(self):
        # ... (Identical to v4.4) ...
        recon_path = os.path.join(Cfg.OUT_DIR_PART2, "part2_reconstructed.parquet")
        print(f"  -> Loading {recon_path}...")
        df = pd.read_parquet(recon_path)
        df.columns = [c.lower() for c in df.columns]
        tgt = 'd_recon' if 'd_recon' in df.columns else ('y' if 'y' in df.columns else 'y16')
        if tgt == 'y16':
            df['d_recon'] = df['y16'].apply(lambda a: float(np.nansum(np.asarray(a, dtype=float))))
            tgt = 'd_recon'
        date_col = next((c for c in ['dt','date','timestamp'] if c in df.columns), None)
        df[date_col] = pd.to_datetime(df[date_col]).dt.floor('d')
        df = df.rename(columns={date_col:'dt', tgt:'y'})
        prep_path = os.path.join(Cfg.ARTIFACTS_DIR, "preprocessed.parquet")
        if os.path.exists(prep_path):
            prep = pd.read_parquet(prep_path)
            prep['dt'] = pd.to_datetime(prep['dt']).dt.floor('d')
            cols = ['store_id', 'product_id', 'dt', 'promo_bin', 'is_event', 'discount', 'avg_temperature']
            cols = [c for c in cols if c in prep.columns]
            df = df.merge(prep[cols], on=['store_id', 'product_id', 'dt'], how='left')
        for c in ['promo_bin', 'is_event']: 
            if c in df: df[c] = df[c].fillna(0).astype(int)
        for c in ['discount', 'avg_temperature']:
            if c in df: df[c] = df[c].fillna(0.0).astype(float)
        if 'store_id' not in df: df['store_id'] = 'unk'
        df = df.groupby(['store_id', 'product_id', 'dt'], observed=True, as_index=False)['y'].sum()
        if os.path.exists(prep_path):
            df = df.merge(prep[['store_id', 'product_id', 'dt', 'promo_bin', 'is_event', 'discount', 'avg_temperature']], 
                          on=['store_id', 'product_id', 'dt'], how='left')
        return df

    def _build_direct_h_windows(self, df, seq_len, horizon, max_pairs, min_days, max_total):
        # ... (Identical to v4.4) ...
        agg = df.groupby(['store_id','product_id'], observed=True)['y'].agg(['count','sum']).reset_index()
        agg = agg[agg['count'] >= min_days].sort_values('sum', ascending=False)
        pairs = list(zip(agg['store_id'].values[:max_pairs], agg['product_id'].values[:max_pairs]))
        X_list, Yh_list, meta = [], [], []
        for store, prod in pairs:
            s = df[(df['store_id']==store) & (df['product_id']==prod)].sort_values('dt')
            s = s.set_index('dt')['y'].asfreq('D', fill_value=0.0)
            vals = s.values.astype(float)
            n = len(vals)
            if n < seq_len + horizon: continue
            for i in range(n - seq_len - horizon + 1):
                x = vals[i : i+seq_len]
                y_vec = vals[i+seq_len : i+seq_len+horizon]
                X_list.append(x); Yh_list.append(y_vec); meta.append((store, prod, str(s.index[i+seq_len])))
                if len(Yh_list) >= max_total: break
            if len(Yh_list) >= max_total: break
        if len(X_list) == 0: return np.array([]), np.array([]), np.array([])
        return np.stack(X_list), np.stack(Yh_list), np.array(meta, dtype=object)

    def _assemble_features(self, X, meta, df_all_features):
        # ... (Identical to v4.4) ...
        if len(X) == 0: return pd.DataFrame(), {}, {}
        n, seq_len = X.shape
        feat = {}
        for i in range(seq_len): feat[f'lag_{i+1}'] = X[:, seq_len-1-i]
        feat['lag_mean_7'] = np.mean(X[:, -7:], axis=1)
        feat['lag_std_7'] = np.std(X[:, -7:], axis=1)
        meta_df = pd.DataFrame(meta, columns=['store_id', 'product_id', 'dt_str'])
        meta_df['dt'] = pd.to_datetime(meta_df['dt_str'])
        feat['wday'] = meta_df['dt'].dt.weekday.values.astype(int)
        feat['month'] = meta_df['dt'].dt.month.values.astype(int)
        merged = meta_df.merge(df_all_features[['store_id', 'product_id', 'dt', 'promo_bin', 'is_event', 'discount', 'avg_temperature']],
                               on=['store_id', 'product_id', 'dt'], how='left')
        feat['promo_bin'] = merged['promo_bin'].fillna(0).values.astype(int)
        feat['is_event'] = merged['is_event'].fillna(0).values.astype(int)
        feat['discount'] = merged['discount'].fillna(0.0).values.astype(float)
        feat['avg_temp'] = merged['avg_temperature'].fillna(0.0).values.astype(float)
        if not self.store_map: 
            stores = np.unique(meta[:, 0]); prods = np.unique(meta[:, 1])
            self.store_map = {s: i for i, s in enumerate(stores)}
            self.prod_map = {p: i for i, p in enumerate(prods)}
        feat['store_code'] = np.array([self.store_map.get(s, 0) for s in meta[:, 0]], dtype=int)
        feat['prod_code'] = np.array([self.prod_map.get(p, 0) for p in meta[:, 1]], dtype=int)
        return pd.DataFrame(feat), self.store_map, self.prod_map

    def _train_lgb_single(self, X_train, y_train, X_val, y_val):
        # ... (Identical to v4.4) ...
        cat_feats = ['store_code', 'prod_code', 'wday', 'month', 'promo_bin', 'is_event']
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        model = lgb.train(Cfg.LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dtrain, dval], callbacks=callbacks)
        return model

    def _calibrate(self, base_model, X_val, y_val_log, Yh_raw):
        # ... (Identical to v4.4) ...
        try:
            preds_log = base_model.predict(X_val)
            bias = float(np.mean(preds_log - y_val_log))
            self.bias_log_h[1] = bias
        except: self.bias_log_h[1] = 0.0
        try:
            flat_targets = Yh_raw.flatten()
            self.global_cap = float(np.percentile(flat_targets, 99.5))
        except: self.global_cap = 1000.0

    def _train_direct_horizons(self, X, Yh, meta, df, idx_train_end, idx_valid_end):
        # ... (Identical to v4.4) ...
        train_idx = range(0, idx_train_end)
        val_idx = range(idx_train_end, idx_valid_end)
        print(f"     [Split] Train samples: {len(train_idx)}, Valid samples: {len(val_idx)}")
        Xtr_df, _, _ = self._assemble_features(X[train_idx], meta[train_idx], df)
        Xval_df, _, _ = self._assemble_features(X[val_idx], meta[val_idx], df)
        for h in range(1, Cfg.HORIZON + 1):
            y_h_log = np.log1p(Yh[:, h-1])
            ytr, yval = y_h_log[train_idx], y_h_log[val_idx]
            model_h = self._train_lgb_single(Xtr_df, ytr, Xval_df, yval)
            self.models[h] = model_h
            try:
                preds = model_h.predict(Xval_df)
                bias = float(np.mean(preds - yval))
            except: bias = 0.0
            self.bias_log_h[h] = bias
            model_h.save_model(os.path.join(Cfg.OUT_DIR_FORECAST, f"lgb_direct_h{h}.txt"))

    def _build_test_batch(self, df, offset=0):
        """
        Build a single-window test batch.
        `offset` (days) shifts the window earlier for walk-forward evaluation:
          offset=0  → last HORIZON days (most-recent, original behaviour)
          offset=H  → one fold earlier, etc.
        Bug #7 support: called once per fold by _evaluate_ml_batch.
        """
        df_grouped = df.groupby(['store_id', 'product_id'])
        X_list, Y_list, meta_list = [], [], []
        for (store, prod), group in df_grouped:
            g = group.sort_values('dt')
            vals = g['y'].values; dates = g['dt'].values
            n = len(vals)
            needed = Cfg.SEQ_LEN + Cfg.HORIZON + offset
            if n < needed: continue
            end_idx     = n - offset
            truth_indices = slice(end_idx - Cfg.HORIZON, end_idx)
            input_indices = slice(end_idx - Cfg.HORIZON - Cfg.SEQ_LEN, end_idx - Cfg.HORIZON)
            y_truth    = vals[truth_indices]
            x_input    = vals[input_indices]
            anchor_date = dates[end_idx - Cfg.HORIZON - 1]
            X_list.append(x_input); Y_list.append(y_truth)
            meta_list.append((store, prod, str(pd.to_datetime(anchor_date).date())))
        if not X_list: return np.array([]), np.array([]), np.array([])
        return np.stack(X_list), np.stack(Y_list), np.array(meta_list, dtype=object)