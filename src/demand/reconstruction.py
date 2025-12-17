# src/demand/reconstruction.py
import os
import json
import numpy as np
import pandas as pd
from config.settings import ProjectConfig as Cfg

class DemandReconstructor:
    """
    Class thực hiện quy trình Demand Reconstruction (Part 2):
    1. Tính toán Hourly CDF ở các cấp độ (L1->L4).
    2. Shrinkage & Blending các CDF để tìm ra phân phối tối ưu cho từng SKU.
    3. Khôi phục nhu cầu (Reconstruction) cho những ngày bị Stockout.
    4. Đánh giá chất lượng (QA Recensor) và Correlation Analysis (rhoDS).
    """

    def __init__(self):
        # Do not create Cfg.OUT_DIR_PART2 here, it will be handled by run method
        self.HOURLY_FLOOR = None # Sẽ được tính toán động

        # Định nghĩa các Keys cho từng Level (giữ nguyên logic gốc)
        self.KEYS_L1_full = ["store_id","product_id","wday","promo_bin"] + (["is_event"] if Cfg.USE_EVENT_KEY else [])
        self.KEYS_L1_promo = ["store_id","product_id","wday","promo_bin"]
        self.KEYS_L1_plain = ["store_id","product_id","wday"]

        self.KEYS_L2_promo = ["product_id","wday","promo_bin"]
        self.KEYS_L2_plain = ["product_id","wday"]

        self.KEYS_L3 = ["wday"]
        self.KEYS_L4 = ["wday"] # Global logic

        # Dictionaries lưu trữ CDF đã tính
        self.D_L1_full = {}
        self.D_L1_promo = {}
        self.D_L1_plain = {}
        self.D_L2_promo = {}
        self.D_L2_plain = {}
        self.D_L3 = {}
        self.D_L4 = {}

    def run(self, input_path=None, output_dir=None):
        print("\n[Reconstruction] Starting Demand Reconstruction Pipeline...")

        # Determine output directory
        self.current_output_dir = output_dir if output_dir else Cfg.OUT_DIR_PART2
        os.makedirs(self.current_output_dir, exist_ok=True)
        print(f" -> Output artifacts will be saved to: {self.current_output_dir}")

        # 1. Load Data
        if input_path is None:
            input_path = os.path.join(Cfg.ARTIFACTS_DIR, "preprocessed.parquet")

        df = pd.read_parquet(input_path)
        df.sort_values(["store_id","product_id","dt"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Ensure correct types
        df["wday"] = df["wday"].astype("int8")
        if "is_promo" not in df: df["is_promo"] = 0
        if "is_event" not in df: df["is_event"] = ((df.get("holiday_num", 0) > 0) | (df.get("is_weekend", 0) == 1)).astype("int8")

        # 2. Identify Non-OOS days (Training Data for CDFs)
        print(" -> Identifying Non-OOS days...")
        df["non_oos16"] = df["s16"].apply(self._is_non_oos_day)
        nonso = df[df["non_oos16"]].copy()
        nonso["iso_year"] = nonso["dt"].dt.isocalendar().year.astype("int16")
        nonso["iso_week"] = nonso["dt"].dt.isocalendar().week.astype("int16")

        # 3. Compute CDFs at all levels
        print(" -> Computing Hierarchical CDFs (L1, L2, L3, L4)...")
        self._compute_all_cdfs(nonso)

        # 4. Compute Hourly Floor (from stable L1)
        print(" -> Computing Hourly Floor...")
        self._compute_hourly_floor()

        # 5. Reconstruction (Shrinkage + Uncensoring)
        print(" -> Applying Shrinkage and Reconstructing Demand...")
        df = self._apply_reconstruction(df)

        # 6. Evaluation (QA)
        print(" -> Running Validation: Random Censoring Test (Global WAPE Approach)")
        qa_metrics = self._evaluate_quality_per_product(df)

        # 7. Correlation Analysis (Fisher Weighted)
        print(" -> Running Correlation Analysis (rhoDS Weighted)...")
        self._analyze_correlations(df)

        # 8. Save
        out_path = os.path.join(self.current_output_dir, "part2_reconstructed.parquet")
        df.to_parquet(out_path, index=False)

        qa_path = os.path.join(self.current_output_dir, "part2_quality.json")
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa_metrics, f, indent=2)

        print(f" -> Reconstructed data saved to {out_path}")
        print(f" -> QA metrics saved to {qa_path}")

        return df

    # --- Helpers Logic ---
    @staticmethod
    def _is_non_oos_day(s16):
        s = np.asarray(s16, np.float32)
        # Non-OOS means finite and NO stockout hour detected
        return np.isfinite(s).all() and (s == Cfg.FLAG_STOCKOUT_VAL).sum() == 0

    @staticmethod
    def _share(y16):
        y = np.nan_to_num(np.asarray(y16, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return y / (float(y.sum()) + 1e-8)

    # --- CDF Computation Logic ---
    def _weekly_mean_cdf(self, gdf: pd.DataFrame):
        gdf = gdf.reset_index(drop=True)
        shares = np.stack([self._share(y) for y in gdf["y16"].to_list()], axis=0)
        cdfs = np.cumsum(shares, axis=1)
        out = []
        for (yy, ww), sub in gdf.groupby(["iso_year","iso_week"], sort=False):
            out.append(((int(yy), int(ww)), cdfs[sub.index.values].mean(axis=0)))
        return out

    def _drift_metric(self, week_cdfs):
        if len(week_cdfs) < 2: return None
        week_cdfs = sorted(week_cdfs, key=lambda z: (z[0][0], z[0][1]))
        dists = [float(np.max(np.abs(week_cdfs[i][1] - week_cdfs[i-1][1]))) for i in range(1, len(week_cdfs))]
        return float(np.median(dists)) if dists else None

    def _agg_cdf(self, tbl: pd.DataFrame, keys, compute_drift: bool):
        recs = []
        # Groupby observed=True is faster for categoricals
        for k, g in tbl.groupby(keys, observed=True, sort=False):
            arr = np.stack([self._share(x) for x in g["y16"]], axis=0)
            cdf = np.nanmean(np.cumsum(arr, axis=1), axis=0).astype(np.float32)
            n_days = int(g.shape[0])
            n_weeks = int(g.drop_duplicates(["iso_year","iso_week"]).shape[0])
            med_ks = None
            if compute_drift:
                wk = self._weekly_mean_cdf(g)
                med_ks = self._drift_metric(wk)

            if not isinstance(k, tuple): k = (k,)

            rec = dict(zip(keys, k))
            rec.update({
                "cdf": cdf,
                "n_days": n_days,
                "n_weeks": n_weeks,
                "median_ks": (np.nan if med_ks is None else float(med_ks))
            })
            recs.append(rec)
        return pd.DataFrame(recs)

    def _dictify(self, tab, keys, fields):
        d = {}
        if tab is None or tab.empty: return d
        for r in tab.to_dict("records"):
            k = tuple(r[kname] for kname in keys)
            d[k] = {f: r[f] for f in fields}
        return d

    def _compute_all_cdfs(self, nonso):
        # L1
        cdf_L1_full = self._agg_cdf(nonso, self.KEYS_L1_full, True)
        cdf_L1_promo = self._agg_cdf(nonso, self.KEYS_L1_promo, True)
        cdf_L1_plain = self._agg_cdf(nonso, self.KEYS_L1_plain, True)

        # L2
        cdf_L2_promo = self._agg_cdf(nonso, self.KEYS_L2_promo, False)
        cdf_L2_plain = self._agg_cdf(nonso, self.KEYS_L2_plain, False)

        # L3
        cdf_L3 = self._agg_cdf(nonso, self.KEYS_L3, False)

        # L4 (Global Weekday)
        # Logic đặc biệt cho L4 như script gốc
        cdf_L4_rows = []
        for k, g in nonso.groupby(self.KEYS_L4, observed=True):
             arr = np.stack([self._share(x) for x in g["y16"]], axis=0)
             cdf = np.nanmean(np.cumsum(arr, axis=1), axis=0).astype(np.float32)
             cdf_L4_rows.append({
                 "wday": k if isinstance(k, (int, float)) else k[0], # Handle tuple/scalar key
                 "cdf": cdf, "n_days": len(g), "n_weeks": g["iso_week"].nunique()
             })
        cdf_L4 = pd.DataFrame(cdf_L4_rows)

        # Convert to Dicts for O(1) Lookup
        self.D_L1_full = self._dictify(cdf_L1_full, self.KEYS_L1_full, ["cdf","n_days","n_weeks","median_ks"])
        self.D_L1_promo = self._dictify(cdf_L1_promo, self.KEYS_L1_promo, ["cdf","n_days","n_weeks","median_ks"])
        self.D_L1_plain = self._dictify(cdf_L1_plain, self.KEYS_L1_plain, ["cdf","n_days","n_weeks","median_ks"])
        self.D_L2_promo = self._dictify(cdf_L2_promo, self.KEYS_L2_promo, ["cdf","n_days","n_weeks"])
        self.D_L2_plain = self._dictify(cdf_L2_plain, self.KEYS_L2_plain, ["cdf","n_days","n_weeks"])
        self.D_L3 = self._dictify(cdf_L3, self.KEYS_L3, ["cdf","n_days","n_weeks"])
        self.D_L4 = self._dictify(cdf_L4, self.KEYS_L4, ["cdf","n_days","n_weeks"])

    def _compute_hourly_floor(self):
        pool = []
        # Collect stable L1 CDFs
        for D in (self.D_L1_full, self.D_L1_promo, self.D_L1_plain):
            for stats in D.values():
                n_w = int(stats.get("n_weeks", 0) or 0)
                n_d = int(stats.get("n_days", 0) or 0)
                ks = float(stats.get("median_ks", np.nan))
                if (n_w >= Cfg.MIN_WEEKS_KEY) and (n_d >= Cfg.MIN_DAYS_KEY) and (ks <= Cfg.KS_THR_SOFT):
                    pool.append(np.asarray(stats["cdf"], dtype=np.float32))

        H = len(Cfg.HOURS)
        if len(pool) >= 50:
            cdf_stack = np.stack(pool, axis=0)
            self.HOURLY_FLOOR = np.percentile(cdf_stack, Cfg.HOURLY_FLOOR_PCTL, axis=0).astype(np.float32)
            self.HOURLY_FLOOR = np.maximum.accumulate(np.clip(self.HOURLY_FLOOR, 1e-3, 0.995))
        else:
            self.HOURLY_FLOOR = np.linspace(0.05, 0.95, H).astype(np.float32)

        print(f"     Hourly Floor Head: {np.round(self.HOURLY_FLOOR[:4], 3)}")

    # --- Shrinkage Helpers ---
    def _w_from_days(self, n, K): return float(np.clip(n / (n + K), 0.0, 1.0))
    def _w_from_drift(self, ks):
        if ks is None or not np.isfinite(ks): return 0.0
        return float(np.clip(1.0 - ks / max(Cfg.KS_THR_SOFT, 1e-6), 0.0, 1.0))

    def _l1_weight(self, stats):
        return float(np.clip(
            self._w_from_days(float(stats.get("n_days", 0)), Cfg.SHRINK_K_L1) *
            self._w_from_drift(stats.get("median_ks", np.nan)), 0.0, 1.0))

    def _l2_weight(self, stats):
        return float(np.clip(self._w_from_days(float(stats.get("n_days", 0)), Cfg.SHRINK_K_L2), 0.0, 1.0))

    def _fetch_cdf_shrunk(self, row):
        def _k(keys): return tuple(getattr(row, k) for k in keys)
        def _enough(st): return (st.get("n_weeks",0) >= Cfg.MIN_WEEKS_KEY) and (st.get("n_days",0) >= Cfg.MIN_DAYS_KEY)

        # 1. Get Best L1
        st1, tag1 = None, None
        k_full = _k(self.KEYS_L1_full); st = self.D_L1_full.get(k_full)
        if st and _enough(st): st1, tag1 = st, "L1_full"
        else:
            k_promo = _k(self.KEYS_L1_promo); st = self.D_L1_promo.get(k_promo)
            if st and _enough(st): st1, tag1 = st, "L1_promo"
            else:
                k_plain = _k(self.KEYS_L1_plain); st = self.D_L1_plain.get(k_plain)
                if st: st1, tag1 = st, "L1_plain"

        # 2. Get Best L2
        st2, tag2 = None, None
        k_p = _k(self.KEYS_L2_promo); st = self.D_L2_promo.get(k_p)
        if st and _enough(st): st2, tag2 = st, "L2_promo"
        else:
            k_pl = _k(self.KEYS_L2_plain); st = self.D_L2_plain.get(k_pl)
            if st: st2, tag2 = st, "L2_plain"

        # 3. Get L3, L4
        st3 = self.D_L3.get(_k(self.KEYS_L3))
        st4 = self.D_L4.get(_k(self.KEYS_L4))

        # 4. Shrinkage Bottom-Up
        # Base: L3 -> L4
        below, tag_below = None, ""
        if st3:
            w2 = self._l2_weight(st3)
            c3 = st3["cdf"]
            if st4:
                c4 = st4["cdf"]
                below = (w2*c3 + (1-w2)*c4).astype(np.float32)
                tag_below = f"L3~L4"
            else:
                below = c3; tag_below = "L3"
        elif st4:
            below = st4["cdf"]; tag_below = "L4"
        else:
            below = np.linspace(0, 1, len(Cfg.HOURS), dtype=np.float32)
            tag_below = "Fallback"

        # Mid: L2 -> Below
        w2_val = 0.0
        if st2:
            w2_val = self._l2_weight(st2)
            below = (w2_val * st2["cdf"] + (1-w2_val) * below).astype(np.float32)
            tag_below = f"{tag2}~{tag_below}"

        # Top: L1 -> Mid
        w1_val = 0.0
        if st1:
            w1_val = self._l1_weight(st1)
            final_cdf = (w1_val * st1["cdf"] + (1-w1_val) * below).astype(np.float32)
            tag_final = f"{tag1} -> {tag_below}"
        else:
            final_cdf = below
            tag_final = tag_below

        return final_cdf, tag_final, {"w1": w1_val, "w2": w2_val}

    def _reconstruct_day(self, y16, s16, cdf):
        y = np.asarray(y16, np.float32)
        s = np.asarray(s16, np.float32)
        if not (np.isfinite(y).all() and np.isfinite(s).all()): return np.nan

        # If no stockout, return observed sum
        if (s == Cfg.FLAG_STOCKOUT_VAL).sum() == 0:
            return float(y.sum())

        # If stockout detected
        good = np.where(s != Cfg.FLAG_STOCKOUT_VAL)[0]
        if good.size == 0: return float(y.sum()) # All stockout? return observed (safe fallback)

        last = int(good.max())
        floor_h = float(self.HOURLY_FLOOR[last])

        # --- CRITICAL FIX: ROBUST DENOMINATOR ---
        # Áp dụng ngưỡng cắt tối thiểu để tránh lỗi phóng đại (Positive Bias)
        denom = float(max(cdf[last], floor_h, Cfg.CDF_MIN_CLIP))

        return float(y[:last+1].sum() / denom)

    def _apply_reconstruction(self, df):
        levels, w1s, w2s, cdflist, Dhat = [], [], [], [], np.zeros(len(df), np.float32)

        # Iteration
        for i, r in enumerate(df.itertuples(index=False)):
            cdf, lvl, info = self._fetch_cdf_shrunk(r)
            levels.append(lvl)
            w1s.append(info["w1"])
            w2s.append(info["w2"])
            cdflist.append(cdf)
            Dhat[i] = self._reconstruct_day(r.y16, r.s16, cdf)

        df["cdf_level"] = np.array(levels, dtype=object)
        df["w1_l1_shrink"] = np.array(w1s, np.float32)
        df["w2_l2_shrink"] = np.array(w2s, np.float32)
        df["cdf_used"] = cdflist
        df["D_recon"] = Dhat

        # Filter finite results
        df = df[np.isfinite(df["D_recon"])].reset_index(drop=True)
        return df

    # --- Evaluation Logic (Updated: Global WAPE) ---
    def _evaluate_quality_per_product(self, df):
        rs = np.random.default_rng(Cfg.SEED)
        valid_df = df[df["non_oos16"]].copy()

        # Global Accumulators for WAPE
        total_abs_err = 0.0
        total_actual = 0.0

        results = []
        for pid, group in valid_df.groupby("product_id"):
            n_samples = min(50, len(group))
            if n_samples < 5: continue

            sample_indices = rs.choice(group.index, size=n_samples, replace=False)
            samples = group.loc[sample_indices]

            y_trues = []
            y_preds = []

            for _, row in samples.iterrows():
                y_real = np.asarray(row["y16"], float)
                cdf = np.asarray(row["cdf_used"], float)

                # Random censoring simulation
                cut = int(rs.integers(3, len(Cfg.HOURS)-2))

                # Use strict denominator for evaluation simulation
                denom = max(float(cdf[cut]), Cfg.CDF_MIN_CLIP)

                y_recon = float(y_real[:cut+1].sum() / denom)
                y_true = float(y_real.sum())

                y_trues.append(y_true)
                y_preds.append(y_recon)

            y_trues = np.array(y_trues)
            y_preds = np.array(y_preds)

            # Product level stats (for reference)
            mape = np.mean(np.abs(y_trues - y_preds) / np.maximum(y_trues, 1e-6)) * 100
            rmse = np.sqrt(np.mean((y_trues - y_preds)**2))
            mean_sales = np.mean(y_trues)

            # Update Global
            total_abs_err += np.sum(np.abs(y_trues - y_preds))
            total_actual += np.sum(y_trues)

            results.append({
                "product_id": pid,
                "Recon_MAPE": round(mape, 1),
                "Recon_RMSE": round(rmse, 2),
                "Mean_Sales": round(mean_sales, 1),
                "Samples": n_samples
            })

        res_df = pd.DataFrame(results).sort_values("Mean_Sales", ascending=False)
        out_csv = os.path.join(self.current_output_dir, "reconstruction_accuracy_by_product.csv")
        res_df.to_csv(out_csv, index=False)

        # Calculate Global WAPE
        global_wape = (total_abs_err / max(total_actual, 1e-6)) * 100

        print(f"     [Validation] Product-level accuracy saved. Global WAPE: {global_wape:.2f}%")
        # --- Fix: Show top rows ---
        print("     Detailed Product Accuracy (Top 20 by Volume):")
        print(res_df.head(20).to_string(index=False))

        return {
            "Total_Products_Evaluated": len(res_df),
            "Overall_Global_WAPE": float(global_wape), # Primary Metric
            "Overall_Mean_MAPE": float(res_df["Recon_MAPE"].mean()),
            "Overall_Mean_RMSE": float(res_df["Recon_RMSE"].mean())
        }

    # --- Correlation Analysis (Updated: Fisher Weighted) ---
    def _analyze_correlations(self, df):
        # 1. Prepare data
        def _oos_frac(s): return float((np.asarray(s)==Cfg.FLAG_STOCKOUT_VAL).mean())
        df["OOS_frac"] = df["s16"].apply(_oos_frac)
        df["Y_obs"] = df["y16"].apply(lambda x: np.sum(x))

        relevant = df[["store_id", "product_id", "OOS_frac", "D_recon", "Y_obs"]].copy()

        # 2. Compute pair-wise correlations
        def pair_corr(g, xcol, ycol):
            # Require minimum samples and variance
            if len(g) < 10 or g[xcol].std() < 1e-9 or g[ycol].std() < 1e-9:
                return np.nan, 0, 0

            r = float(np.corrcoef(g[xcol], g[ycol])[0,1])
            n = len(g)
            mu = float(g["Y_obs"].mean()) # Mean Volume (Weight)
            return r, n, mu

        print("     Calculating pair-wise correlations...")
        stats_list = []
        for (sid, pid), g in relevant.groupby(["store_id", "product_id"], observed=True):
            r_rec, n, mu = pair_corr(g, "OOS_frac", "D_recon")
            r_raw, _, _  = pair_corr(g, "OOS_frac", "Y_obs")

            if not np.isnan(r_rec) and not np.isnan(r_raw):
                stats_list.append({
                    "r_rec": r_rec, "r_raw": r_raw,
                    "n": n, "mu": mu
                })

        stats_df = pd.DataFrame(stats_list)

        if stats_df.empty:
            print("     [Warning] No valid correlations computed.")
            return

        # 3. Fisher Z-Transformation Weighted Average
        def fisher_avg(r_values, weights):
            # Clip r to range [-0.9999, 0.9999] to avoid inf in arctanh
            r_clipped = np.clip(r_values, -0.9999, 0.9999)
            z_values = np.arctanh(r_clipped)

            # Weighted average of z
            z_bar = np.average(z_values, weights=weights)

            # Inverse transform
            return np.tanh(z_bar)

        rho_rec_weighted = fisher_avg(stats_df["r_rec"], weights=stats_df["mu"])
        rho_raw_weighted = fisher_avg(stats_df["r_raw"], weights=stats_df["mu"])

        print("\n" + "="*50)
        print(" CORRELATION ANALYSIS (Volume Weighted)")
        print("="*50)
        print(f" Pairs Analyzed: {len(stats_df)}")
        print(f" Raw Sales Decoupling (Weighted):   {rho_raw_weighted:.3f} (Expect ~ -0.57)")
        print(f" Reconstructed Decoupling (Weighted): {rho_rec_weighted:.3f} (Target: Near 0.0)")
        print("="*50)