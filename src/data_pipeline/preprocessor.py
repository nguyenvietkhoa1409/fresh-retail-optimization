# src/data_pipeline/preprocessor.py
import os
import gc
import numpy as np
import pandas as pd

# 1. Import Config
from config.settings import ProjectConfig as Cfg

# 2. Import datasets
from datasets import load_dataset
from src.utils.common import DataUtils

class FrnPreprocessor:
    """
    Class xử lý dữ liệu FreshRetailNet-50K (Optimized for Memory):
    1. Load & Clean (Downcast ASAP)
    2. Feature Engineering (Avoid String Conversion)
    3. Stockout Flag Detection (Heuristic)
    4. Coverage Filtering
    
    UPDATES (v4):
    - Removed '_compute_sku_economics': Economics are now handled by CatalogEnricher.
    """

    def __init__(self):
        self.flag_stockout = 0
        os.makedirs(Cfg.ARTIFACTS_DIR, exist_ok=True)

    def run(self):
        print("\n[Preprocessor] Starting Data Preprocessing Pipeline...")
        
        # 1. Load Data
        df = self._load_data()
        
        # 2. Feature Engineering
        df = self._feature_engineering(df)
        
        # 3. Parse Hourly Arrays (String -> Numpy)
        df = self._parse_hourly_data(df)
        
        # 4. Stockout Detection Heuristic
        self.flag_stockout = self._detect_stockout_flag(df)
        print(f"  -> Detected Global STOCKOUT FLAG: {self.flag_stockout}")
        
        # 5. Subset & Coverage Filter
        df = self._filter_coverage(df)
        
        # 6. Save Final Parquet
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "preprocessed.parquet")
        df.to_parquet(out_path)
        print(f"  -> Preprocessed data saved to {out_path}")
        
        # Clean up
        del df; gc.collect()
        
        return pd.read_parquet(out_path)

    def _load_data(self):
        print("  -> Loading FRN-50K Dataset...")
        try:
            ds = load_dataset(Cfg.HF_DATASET)
        except Exception as e:
            print(f"    [Error] Could not load dataset from HuggingFace: {e}")
            raise e
        
        expected_cols = [
            "store_id","product_id","dt","sale_amount",
            "hours_sale","hours_stock_status",
            "discount","holiday_flag","precpt","avg_temperature"
        ]
        
        # Load từng phần và convert ngay để tiết kiệm RAM
        print("     Loading train split...")
        train = ds["train"].select_columns(expected_cols).to_pandas()
        train = self._downcast_numerics(train)
        
        print("     Loading eval split...")
        eval_df = ds["eval"].select_columns(expected_cols).to_pandas()
        eval_df = self._downcast_numerics(eval_df)
        
        print("     Concatenating...")
        raw = pd.concat([train, eval_df], ignore_index=True)
        del ds, train, eval_df; gc.collect()
        
        # Normalize types
        print("  -> Normalizing data types...")
        raw["dt"] = pd.to_datetime(raw["dt"], errors="coerce").dt.tz_localize(None)
        
        for c in ["store_id","product_id"]:
            if c in raw: 
                raw[c] = raw[c].astype(str)
            
        gc.collect()
        return raw

    def _downcast_numerics(self, df):
        """Helper để giảm dung lượng RAM ngay khi load"""
        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns
        
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype('float32')
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype('int32')
            
        # Xử lý các cột object nếu có thể chuyển về số
        for col in ["sale_amount", "discount", "precpt", "avg_temperature"]:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                
        return df

    def _feature_engineering(self, df):
        print("  -> Feature engineering (Optimized)...")
        # Sort in-place
        df.sort_values(["store_id","product_id","dt"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Date features
        df["wday"] = df["dt"].dt.weekday.astype("int8")
        df["month"] = df["dt"].dt.month.astype("int8")
        df["is_weekend"] = df["wday"].isin([5,6]).astype("int8")
        
        # Trig features -> float32
        pi_val = np.float32(np.pi)
        wday_vals = df["wday"].values.astype(np.float32)
        month_vals = (df["month"].values - 1).astype(np.float32)
        
        df["dow_sin"] = np.sin(2 * pi_val * wday_vals / 7).astype("float32")
        df["dow_cos"] = np.cos(2 * pi_val * wday_vals / 7).astype("float32")
        df["mon_sin"] = np.sin(2 * pi_val * month_vals / 12).astype("float32")
        df["mon_cos"] = np.cos(2 * pi_val * month_vals / 12).astype("float32")
        
        # --- OPTIMIZED HOLIDAY & PROMO LOGIC ---
        
        # 1. Discount / Promo
        if "discount" in df:
            df["discount"] = df["discount"].fillna(0).astype("float32")
            df["discount"] = df["discount"].clip(lower=0)
            df["is_promo"] = (df["discount"] > 0).astype("int8")
            
            df["promo_bin"] = 0
            mask_promo = df["discount"] > 0
            if mask_promo.sum() > 50:
                try:
                    bins = pd.qcut(df.loc[mask_promo, "discount"], q=3, labels=[1, 2, 3], duplicates="drop")
                    df.loc[mask_promo, "promo_bin"] = bins.astype("int8")
                except:
                    df.loc[mask_promo, "promo_bin"] = 1
            else:
                df.loc[mask_promo, "promo_bin"] = 1
                
            df["promo_bin"] = df["promo_bin"].astype("int8")
        else:
            df["is_promo"] = 0
            df["promo_bin"] = 0
            df[["is_promo", "promo_bin"]] = df[["is_promo", "promo_bin"]].astype("int8")

        # 2. Holiday Flag
        if "holiday_flag" in df:
            if df["holiday_flag"].dtype == 'object':
                 df["holiday_flag"] = pd.to_numeric(df["holiday_flag"], errors='coerce')
            
            df["holiday_flag"] = df["holiday_flag"].fillna(0).astype("int8")
            df["holiday_num"] = df["holiday_flag"] 
        else:
            df["holiday_flag"] = 0
            df["holiday_num"] = 0
            
        df["holiday_num"] = df["holiday_num"].astype("int8")
        df["holiday_flag"] = df["holiday_flag"].astype("int8")

        # 3. Is Event
        df["is_event"] = ((df["holiday_num"] > 0) | (df["is_weekend"] == 1)).astype("int8")
        
        gc.collect()
        return df

    def _parse_hourly_data(self, df):
        print("  -> Parsing hourly data (y16, s16)...")
        h0, h1 = min(Cfg.HOURS), max(Cfg.HOURS)
        
        # 1. Hours Sale
        y24 = df["hours_sale"].apply(DataUtils.parse24)
        df.drop(columns=["hours_sale"], inplace=True)
        
        df["y16"] = y24.apply(lambda x: DataUtils.hours_to_16(x, h0, h1))
        del y24; gc.collect()
        
        # 2. Stock Status
        s24 = df["hours_stock_status"].apply(DataUtils.parse24)
        df.drop(columns=["hours_stock_status"], inplace=True)
        
        df["s16"] = s24.apply(lambda x: DataUtils.hours_to_16(x, h0, h1))
        del s24; gc.collect()
        
        return df

    def _detect_stockout_flag(self, df):
        print("  -> Running Stockout Detection Heuristic...")
        B = 5000 
        N = len(df)
        z_m0 = z_m1 = n0 = n1 = 0.0
        
        for s in range(0, N, B):
            e = min(N, s + B)
            y_blk = np.stack(df["y16"].iloc[s:e].tolist())
            st_blk = np.stack(df["s16"].iloc[s:e].tolist())
            
            st_finite = np.isfinite(st_blk)
            st_round = np.where(st_finite, np.rint(st_blk), np.nan)
            finite_y = np.isfinite(y_blk)
            
            valid0 = finite_y & (st_round == 0)
            valid1 = finite_y & (st_round == 1)
            zeros = (y_blk <= 1e-9)
            
            z_m0 += (zeros & valid0).sum()
            n0 += valid0.sum()
            z_m1 += (zeros & valid1).sum()
            n1 += valid1.sum()
            
            del y_blk, st_blk, st_finite, st_round, finite_y, valid0, valid1, zeros
            if s % (B*10) == 0: gc.collect()
        
        gc.collect()

        p0 = float(z_m0) / max(float(n0), 1.0)
        p1 = float(z_m1) / max(float(n1), 1.0)
        
        print(f"     Stats: p(zero|0)={p0:.3f}, p(zero|1)={p1:.3f}")
        return 0 if p0 > p1 else 1

    def _filter_coverage(self, df):
        if not Cfg.KEEP_SUBSET:
            return df
            
        print("  -> Filtering subset based on coverage...")
        
        stockout_val = self.flag_stockout
        
        def is_non_oos_day(s16_arr):
            s = np.asarray(s16_arr, dtype=np.float32)
            if not np.isfinite(s).all(): return 0
            return int((s == stockout_val).sum() == 0)

        df["iso_week"] = df["dt"].dt.isocalendar().week.astype("int16")
        df["non_oos"] = df["s16"].apply(is_non_oos_day).astype("int8")
        
        has_promo = (df.get("is_promo", 0) == 1)
        has_bin = (df.get("promo_bin", 0) > 0)
        df["promo_day"] = (has_promo | has_bin).astype("int8")
        
        g_cov = df.groupby(["store_id","product_id"], observed=True).agg(
            days=("dt", "nunique"),
            weeks=("iso_week", "nunique"),
            non_oos_days=("non_oos", "sum"),
            promo_days=("promo_day", "sum"),
            vol=("sale_amount", "sum")
        ).reset_index()
        
        df.drop(columns=["iso_week", "non_oos", "promo_day"], inplace=True)
        gc.collect()

        rich = g_cov[
            (g_cov["days"] >= Cfg.COV_MIN_DAYS) &
            (g_cov["weeks"] >= Cfg.COV_MIN_WEEKS) &
            (g_cov["non_oos_days"] >= Cfg.COV_MIN_NONOOS) &
            (g_cov["promo_days"] >= Cfg.COV_MIN_PROMO)
        ].copy()
        
        if Cfg.COV_VERBOSE:
            print(f"     Coverage candidates: Total pairs={len(g_cov)} | Rich pairs={len(rich)}")

        keep_pairs = set()
        mode = "volume-share"
        
        if not rich.empty:
            mode = "coverage-first"
            rich_sorted = rich.sort_values("vol", ascending=False)
            if Cfg.PAIR_LIMIT is not None:
                rich_sorted = rich_sorted.iloc[:Cfg.PAIR_LIMIT]
            keep_pairs = set(map(tuple, rich_sorted[["store_id","product_id"]].to_numpy()))
        else:
            g = df.groupby(["store_id","product_id"], observed=True)["sale_amount"].sum().sort_values(ascending=False)
            if Cfg.PAIR_LIMIT is not None:
                g = g.iloc[:Cfg.PAIR_LIMIT]
            csum = g.cumsum()
            limit = Cfg.VOLUME_SHARE * g.sum()
            k = min(int(np.searchsorted(csum.values, limit, side="right")) + 1, len(g))
            keep_pairs = set(g.index[:k])

        before_rows = len(df)
        df_filtered = df.merge(pd.DataFrame(list(keep_pairs), columns=["store_id","product_id"]),
                               on=["store_id","product_id"], how="inner")
        
        print(f"     Subset ({mode}): {before_rows} -> {len(df_filtered)} rows | pairs={len(keep_pairs)}")
        
        del df; gc.collect()
        return df_filtered