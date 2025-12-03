# src/demand/forecasting.py
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from config.settings import ProjectConfig as Cfg
from src.utils.common import DataUtils

class DemandForecaster:
    """
    Class thực hiện Demand Forecasting sử dụng LightGBM.
    Chiến lược: Direct Multi-horizon (Train riêng 1 model cho mỗi horizon t+1...t+7).
    """

    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_FORECAST, exist_ok=True)
        self.models = {}         # Stores models per horizon {1: model_h1, 2: model_h2...}
        self.bias_log_h = {}     # Stores bias correction per horizon
        self.store_map = {}      # Encoding map for Store IDs
        self.prod_map = {}       # Encoding map for Product IDs
        self.global_cap = 1000.0 # Will be updated during calibration
        # Mapping Product ID -> Meta (Name, Category) for Reporting
        self.product_meta = {}
        for p in Cfg.PRODUCT_CATEGORIES:
            # p structure: (id, category_id, name, ...)
            self.product_meta[int(p[0])] = {
                'name': p[2],
                'category_id': int(p[1])
            }
    def run(self):
        print("\n[Forecasting] Starting Forecasting Pipeline...")
        
        # 1. Load Data
        df = self._load_data()
        
        # 2. Build Training Windows
        print("  -> Building training windows...")
        X, Yh, meta = self._build_direct_h_windows(
            df, 
            seq_len=Cfg.SEQ_LEN, 
            horizon=Cfg.HORIZON, 
            max_pairs=Cfg.MAX_PAIRS, 
            min_days=Cfg.MIN_DAYS_PAIR, 
            max_total=Cfg.MAX_SAMPLES
        )
        
        if len(X) == 0:
            raise RuntimeError("No training windows created. Check data filters.")

        # 3. Train Base Model
        dates = pd.to_datetime(meta[:, 2])
        order = np.argsort(dates)
        X_ord, Yh_ord, meta_ord = X[order], Yh[order], meta[order]
        
        n = len(X_ord)
        i1, i2 = int(n * 0.8), int(n * 0.9)
        
        X_tr, y_tr_1step, meta_tr = X_ord[:i1], np.log1p(Yh_ord[:i1, 0]), meta_ord[:i1]
        X_val, y_val_1step, meta_val = X_ord[i1:i2], np.log1p(Yh_ord[i1:i2, 0]), meta_ord[i1:i2]

        Xtr_df, self.store_map, self.prod_map = self._assemble_features(X_tr, meta_tr, df)
        Xval_df, _, _ = self._assemble_features(X_val, meta_val, df)
        
        # Train Base
        print("  -> Training Base Model (Pooled)...")
        base_model = self._train_lgb_single(Xtr_df, y_tr_1step, Xval_df, y_val_1step)
        base_model.save_model(os.path.join(Cfg.OUT_DIR_FORECAST, "lgb_pooled.txt"))
        
        # Calibrate
        self._calibrate(base_model, Xval_df, y_val_1step, Yh_ord[:i2])
        
        # 4. Train Direct-Horizon Models
        print(f"  -> Training Direct-Horizon Models (H=1..{Cfg.HORIZON})...")
        self._train_direct_horizons(X_ord, Yh_ord, meta_ord, df, i1)
        
        # 5. Deep Evaluation (Per-Product/Category)
        print("  -> Running Deep Evaluation (Per-Product & Per-Category)...")
        metrics = self._evaluate(df)
        
        # Save Summary Metrics
        metrics_path = os.path.join(Cfg.OUT_DIR_FORECAST, "phase3_fixed_eval.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"  -> Forecasting completed. Detailed metrics saved to {Cfg.OUT_DIR_FORECAST}")
        return metrics

    # --- Data Loading & Prep ---
    def _load_data(self):
        # Load Reconstruction Target
        recon_path = os.path.join(Cfg.OUT_DIR_PART2, "part2_reconstructed.parquet")
        print(f"  -> Loading {recon_path}...")
        df = pd.read_parquet(recon_path)
        df.columns = [c.lower() for c in df.columns]
        
        # Resolve Target Column
        tgt = 'd_recon' if 'd_recon' in df.columns else ('y' if 'y' in df.columns else 'y16')
        if tgt == 'y16':
            df['d_recon'] = df['y16'].apply(lambda a: float(np.nansum(np.asarray(a, dtype=float))))
            tgt = 'd_recon'
            
        date_col = next((c for c in ['dt','date','timestamp'] if c in df.columns), None)
        df[date_col] = pd.to_datetime(df[date_col]).dt.floor('d')
        df = df.rename(columns={date_col:'dt', tgt:'y'})
        
        # Load Features from Part 1
        prep_path = os.path.join(Cfg.ARTIFACTS_DIR, "preprocessed.parquet")
        if os.path.exists(prep_path):
            prep = pd.read_parquet(prep_path)
            prep['dt'] = pd.to_datetime(prep['dt']).dt.floor('d')
            # Merge features
            cols_to_merge = ['store_id', 'product_id', 'dt', 'promo_bin', 'is_event', 'discount', 'avg_temperature']
            cols_to_merge = [c for c in cols_to_merge if c in prep.columns]
            df = df.merge(prep[cols_to_merge], on=['store_id', 'product_id', 'dt'], how='left')
            
        # Fill NaNs
        for c in ['promo_bin', 'is_event']: 
            if c in df: df[c] = df[c].fillna(0).astype(int)
        for c in ['discount', 'avg_temperature']:
            if c in df: df[c] = df[c].fillna(0.0).astype(float)
            
        # Group by day
        if 'store_id' not in df: df['store_id'] = 'unk'
        df = df.groupby(['store_id', 'product_id', 'dt'], observed=True, as_index=False)['y'].sum()
        
        # Merge back static cols if lost during groupby
        # (Simplified: assume we just need the keys and y, dynamic feats come from merge again or logic)
        # Re-merge dynamic features because groupby might have dropped them
        if os.path.exists(prep_path):
            df = df.merge(prep[['store_id', 'product_id', 'dt', 'promo_bin', 'is_event', 'discount', 'avg_temperature']], 
                          on=['store_id', 'product_id', 'dt'], how='left')
            
        return df

    def _build_direct_h_windows(self, df, seq_len, horizon, max_pairs, min_days, max_total):
        # Select Top Pairs
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
            
            # Sliding Window
            # (Loop optimization: plain python loop is slow but robust for sliding window with meta)
            for i in range(n - seq_len - horizon + 1):
                x = vals[i : i+seq_len]
                y_vec = vals[i+seq_len : i+seq_len+horizon]
                
                X_list.append(x)
                Yh_list.append(y_vec)
                meta.append((store, prod, str(s.index[i+seq_len])))
                
                if len(Yh_list) >= max_total: break
            if len(Yh_list) >= max_total: break
            
        if len(X_list) == 0: return np.array([]), np.array([]), np.array([])
        return np.stack(X_list), np.stack(Yh_list), np.array(meta, dtype=object)

    def _assemble_features(self, X, meta, df_all_features):
        if len(X) == 0: return pd.DataFrame(), {}, {}
        
        n, seq_len = X.shape
        feat = {}
        
        # Lag Features
        for i in range(seq_len):
            feat[f'lag_{i+1}'] = X[:, seq_len-1-i]
        
        feat['lag_mean_7'] = np.mean(X[:, -7:], axis=1)
        feat['lag_std_7'] = np.std(X[:, -7:], axis=1)
        
        # External Features from Meta & DF
        meta_df = pd.DataFrame(meta, columns=['store_id', 'product_id', 'dt_str'])
        meta_df['dt'] = pd.to_datetime(meta_df['dt_str'])
        
        feat['wday'] = meta_df['dt'].dt.weekday.values.astype(int)
        feat['month'] = meta_df['dt'].dt.month.values.astype(int)
        
        # Merge external features efficiently
        # Note: df_all_features passed here must have the columns
        merged = meta_df.merge(df_all_features[['store_id', 'product_id', 'dt', 'promo_bin', 'is_event', 'discount', 'avg_temperature']],
                               on=['store_id', 'product_id', 'dt'], how='left')
        
        feat['promo_bin'] = merged['promo_bin'].fillna(0).values.astype(int)
        feat['is_event'] = merged['is_event'].fillna(0).values.astype(int)
        feat['discount'] = merged['discount'].fillna(0.0).values.astype(float)
        feat['avg_temp'] = merged['avg_temperature'].fillna(0.0).values.astype(float)
        
        # Categorical Encoding
        if not self.store_map: # Create map if first time
            stores = np.unique(meta[:, 0])
            prods = np.unique(meta[:, 1])
            self.store_map = {s: i for i, s in enumerate(stores)}
            self.prod_map = {p: i for i, p in enumerate(prods)}
            
        # Map values (handle unseen with .get(x, 0))
        feat['store_code'] = np.array([self.store_map.get(s, 0) for s in meta[:, 0]], dtype=int)
        feat['prod_code'] = np.array([self.prod_map.get(p, 0) for p in meta[:, 1]], dtype=int)
        
        return pd.DataFrame(feat), self.store_map, self.prod_map

    # --- Training Logic ---
    def _train_lgb_single(self, X_train, y_train, X_val, y_val):
        cat_feats = ['store_code', 'prod_code', 'wday', 'month', 'promo_bin', 'is_event']
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        model = lgb.train(
            Cfg.LGB_PARAMS, 
            dtrain, 
            num_boost_round=2000, 
            valid_sets=[dtrain, dval], 
            callbacks=callbacks
        )
        return model

    def _calibrate(self, base_model, X_val, y_val_log, Yh_raw):
        # 1. Bias Correction
        try:
            preds_log = base_model.predict(X_val)
            bias = float(np.mean(preds_log - y_val_log))
            self.bias_log_h[1] = bias # Base bias for h=1 (default)
            print(f"  -> Calibration Bias (Log Scale): {bias:.6f}")
        except:
            self.bias_log_h[1] = 0.0
            
        # 2. Global Cap (99.5th percentile of raw data)
        try:
            flat_targets = Yh_raw.flatten()
            self.global_cap = float(np.percentile(flat_targets, 99.5))
        except:
            self.global_cap = 1000.0
        print(f"  -> Global Cap: {self.global_cap:.2f}")

    def _train_direct_horizons(self, X, Yh, meta, df, split_idx):
        # Training Split (reuse indices)
        train_idx = range(split_idx)
        val_idx = range(split_idx, len(X))
        
        # Prepare Features Once
        Xtr_df, _, _ = self._assemble_features(X[train_idx], meta[train_idx], df)
        Xval_df, _, _ = self._assemble_features(X[val_idx], meta[val_idx], df)
        
        for h in range(1, Cfg.HORIZON + 1):
            print(f"     Training Horizon H={h}...")
            y_h = Yh[:, h-1] # Target at horizon h
            y_h_log = np.log1p(y_h)
            
            ytr = y_h_log[train_idx]
            yval = y_h_log[val_idx]
            
            # Train
            model_h = self._train_lgb_single(Xtr_df, ytr, Xval_df, yval)
            self.models[h] = model_h
            
            # Compute Bias for this horizon
            try:
                preds = model_h.predict(Xval_df)
                bias = float(np.mean(preds - yval))
            except: 
                bias = 0.0
            self.bias_log_h[h] = bias
            
            # Save
            model_h.save_model(os.path.join(Cfg.OUT_DIR_FORECAST, f"lgb_direct_h{h}.txt"))

    # --- Evaluation Logic ---
    def _evaluate(self, df):
        df_pairs = df.groupby(['store_id','product_id'], observed=True)
        eval_pairs_agg = df.groupby(['store_id','product_id'])['y'].count()
        eval_pairs = eval_pairs_agg[eval_pairs_agg >= Cfg.MIN_DAYS_PAIR].index.tolist()
        
        # New: List to store every single prediction point with metadata
        results_detailed = [] 
        
        np.random.seed(Cfg.SEED)
        sample_pairs = eval_pairs
        eval_count = 0
        limit = Cfg.MAX_SAMPLES // Cfg.HORIZON
        
        for store, prod in sample_pairs:
            if eval_count >= limit: break
            
            ser = df_pairs.get_group((store, prod)).sort_values('dt')
            ser = ser.set_index('dt')['y'].asfreq('D', fill_value=0.0)
            n_days = len(ser)
            if n_days < Cfg.SEQ_LEN + Cfg.HORIZON: continue
            
            anchor_idx = n_days - Cfg.HORIZON - 1
            forecast_start_date = ser.index[anchor_idx] + pd.Timedelta(days=1)
            last_window = ser.iloc[anchor_idx - Cfg.SEQ_LEN + 1 : anchor_idx + 1].values.astype(float)
            truth = ser.iloc[anchor_idx + 1 : anchor_idx + 1 + Cfg.HORIZON].values.astype(float)
            if len(truth) < Cfg.HORIZON: continue

            # Predict Multi-step
            for h in range(1, Cfg.HORIZON + 1):
                pred_date = forecast_start_date + pd.Timedelta(days=h-1)
                feat_row = self._assemble_single_row(last_window, pred_date, store, prod, df)
                
                if h in self.models:
                    y_log = self.models[h].predict(feat_row)[0]
                    bias = self.bias_log_h.get(h, 0.0)
                    y_pred = np.expm1(y_log - bias)
                else: y_pred = 0.0
                
                recent_mean = last_window[-7:].mean() if len(last_window)>=7 else last_window.mean()
                max_allowed = min(self.global_cap * 1.2, max(1e-6, recent_mean) * 6.0)
                y_pred = float(np.clip(y_pred, 0.0, max_allowed))
                
                # --- COLLECT DETAILED DATA ---
                # Resolve product metadata safely
                try:
                    pid_int = int(prod)
                    meta = self.product_meta.get(pid_int, {'name': f'Unknown_{prod}', 'category_id': 999})
                except:
                    meta = {'name': str(prod), 'category_id': 999}

                results_detailed.append({
                    'store_id': store,
                    'product_id': prod,
                    'product_name': meta['name'],
                    'category_id': meta['category_id'],
                    'horizon': h,
                    'date': pred_date,
                    'y_true': float(truth[h-1]),
                    'y_pred': y_pred
                })
                # -----------------------------
                
            eval_count += 1
            
        # Convert to DataFrame
        res_df = pd.DataFrame(results_detailed)
        
        # 1. Overall Metrics
        overall_metrics = self._calc_grouped_metrics(res_df)
        
        # 2. Per-Horizon Metrics
        per_horizon = []
        for h in range(1, Cfg.HORIZON+1):
            sub = res_df[res_df['horizon'] == h]
            m = self._calc_grouped_metrics(sub)
            m['horizon'] = h
            per_horizon.append(m)
        pd.DataFrame(per_horizon).to_csv(os.path.join(Cfg.OUT_DIR_FORECAST, "per_horizon_metrics.csv"), index=False)
        
        # 3. Per-Product Metrics (The "Root Cause" Analysis)
        product_metrics = []
        # Group by Product ID to calculate metrics per product
        for pid, sub in res_df.groupby('product_id'):
            # Basic info
            prod_name = sub['product_name'].iloc[0]
            
            # Metrics calculation
            y_true = sub['y_true'].values
            y_pred = sub['y_pred'].values
            
            # Mean Sales (Daily Average)
            mean_sales = np.mean(y_true)
            
            # RMSE
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            
            # MAPE (Handle division by zero)
            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan
                
            # OOS Rate (Cần lấy từ data gốc hoặc tính xấp xỉ từ Reconstruction)
            # Ở đây ta lấy từ dữ liệu Reconstruction đã merge
            # Cách tốt nhất: Merge lại với thông tin OOS từ Part 1/2
            product_metrics.append({
                'Product ID': pid,
                'Product Name': prod_name,
                'Mean Sales': round(mean_sales, 2),
                'RMSE': round(rmse, 2),
                'MAPE (%)': round(mape, 1) if not np.isnan(mape) else 'N/A',
                'Samples': len(sub)
            })
            
        # Save Product Metrics Table
        prod_metrics_df = pd.DataFrame(product_metrics).sort_values('Mean Sales', ascending=False)
        out_path = os.path.join(Cfg.OUT_DIR_FORECAST, "product_level_accuracy.csv")
        prod_metrics_df.to_csv(out_path, index=False)
        
        print(f"\n     [Product Evaluation] Detailed metrics saved to {out_path}")
        print("     Top 5 Products by Sales Volume:")
        print(prod_metrics_df.head(5).to_string(index=False))
        # -------------------------------------------------------------------
        
        return {
            "overall": overall_metrics,
            "per_horizon": per_horizon
        }

    def _assemble_single_row(self, window, pred_date, store, prod, df_features):
        # Helper for Inference (creating 1-row DataFrame)
        feat = {}
        seq_len = len(window)
        for j in range(seq_len):
            feat[f'lag_{j+1}'] = float(window[-1-j])
        feat['lag_mean_7'] = float(window[-7:].mean())
        feat['lag_std_7'] = float(window[-7:].std())
        
        # External features lookup
        # (This is a simplified lookup, ideally optimized index lookup)
        row = df_features[
            (df_features['store_id']==store) & 
            (df_features['product_id']==prod) & 
            (df_features['dt']==pred_date)
        ]
        
        feat['wday'] = int(pred_date.weekday())
        feat['month'] = int(pred_date.month)
        
        if not row.empty:
            feat['promo_bin'] = int(row['promo_bin'].iloc[0])
            feat['is_event'] = int(row['is_event'].iloc[0])
            feat['discount'] = float(row['discount'].iloc[0])
            feat['avg_temp'] = float(row['avg_temperature'].iloc[0])
        else:
            feat['promo_bin'] = 0; feat['is_event'] = 0; feat['discount'] = 0.0; feat['avg_temp'] = 0.0
            
        feat['store_code'] = self.store_map.get(store, 0)
        feat['prod_code'] = self.prod_map.get(prod, 0)
        
        return pd.DataFrame([feat])

    def _calc_grouped_metrics(self, df):
        """Helper to calc metrics for any dataframe slice"""
        if df.empty: return {'WAPE': np.nan, 'RMSE': np.nan, 'MAE': np.nan}
        
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        denom = np.sum(np.abs(y_true))
        wape = (np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
        
        return {
            "WAPE": float(wape), 
            "RMSE": float(rmse), 
            "MAE": float(mae),
            "n_samples": len(df)
        }