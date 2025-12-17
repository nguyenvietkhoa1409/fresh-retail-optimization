import os
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from scipy.stats import entropy
from datasets import load_dataset
from config.settings import ProjectConfig as Cfg  # Import central config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import ProjectConfig as Cfg

class FreshRetailAnalyzer:
    def __init__(self):
        os.makedirs(Cfg.OUT_DIR_ANALYSIS, exist_ok=True)
        print(f"[Init] Analyzer initialized. Output dir: {Cfg.OUT_DIR_ANALYSIS}")

    def run(self):
        """Orchestrates the full analysis pipeline."""
        print("\n>>> PHASE 1: Data Retrieving & Feature Extraction (Chunked)...")
        raw_stats = self._extract_features_chunked()
        
        print("\n>>> PHASE 2: Aggregation & Metrics Calculation...")
        sku_stats = self._compute_sku_metrics(raw_stats)
        del raw_stats; gc.collect()
        
        print("\n>>> PHASE 3: Classification (SBC & Magnitude)...")
        classified_df = self._classify_series(sku_stats)
        
        print("\n>>> PHASE 4: Visualizing Insights...")
        self._visualize_insights(classified_df)
        
        print("\n>>> PHASE 5: Scenario Simulation & Recommendation...")
        recommendation = self._simulate_scenarios(classified_df)
        
        # Save final artifacts
        out_csv = os.path.join(Cfg.OUT_DIR_ANALYSIS, "final_sku_stats.csv")
        classified_df.to_csv(out_csv, index=False)
        print(f"\n[Done] Full analysis saved to {out_csv}")
        
        return recommendation

    def _parse_hourly_batch(self, batch_df):
        try:
            if isinstance(batch_df["hours_sale"].iloc[0], str):
                arr_24 = np.stack(batch_df["hours_sale"].apply(json.loads).values)
            else:
                arr_24 = np.stack(batch_df["hours_sale"].values)
            
            # Use Config for hours
            arr_16 = arr_24[:, Cfg.INTRA_HOUR_START : Cfg.INTRA_HOUR_END + 1]
            
            daily_vol = np.sum(arr_16, axis=1)
            sums = daily_vol[:, np.newaxis]
            probs = np.divide(arr_16, sums, out=np.zeros_like(arr_16), where=sums > 1e-9)
            ent = entropy(probs, axis=1)
            ent = np.nan_to_num(ent, nan=0.0)
            
            return daily_vol, ent
        except Exception as e:
            print(f"Error parsing batch: {e}")
            return np.zeros(len(batch_df)), np.zeros(len(batch_df))

    def _extract_features_chunked(self):
        ds = load_dataset(Cfg.HF_DATASET)
        cols = ["store_id", "product_id", "dt", "sale_amount", "hours_sale", "discount"]
        aggregated_chunks = []
        
        for split in ["train", "eval"]:
            print(f" -> Processing split: {split}...")
            # Use streaming or load based on RAM availability. 
            # Assuming sufficient RAM as discussed:
            df_split = ds[split].select_columns(cols).to_pandas()
            
            df_split["sale_amount"] = pd.to_numeric(df_split["sale_amount"], errors='coerce').fillna(0).astype("float32")
            df_split["discount"] = pd.to_numeric(df_split["discount"], errors='coerce').fillna(0).astype("float32")
            
            d_vol, d_ent = self._parse_hourly_batch(df_split)
            
            df_split["daily_vol_calc"] = d_vol
            df_split["daily_entropy"] = d_ent
            df_split["is_promo"] = (df_split["discount"] > 0).astype("int8")
            df_split["is_nonzero"] = (df_split["daily_vol_calc"] > 0).astype("int8")
            
            df_split.drop(columns=["hours_sale"], inplace=True)
            
            chunk_agg = df_split.groupby(["store_id", "product_id"], observed=True).agg(
                total_days=("dt", "count"),
                nonzero_days=("is_nonzero", "sum"),
                total_vol=("daily_vol_calc", "sum"),
                sum_entropy=("daily_entropy", "sum"),
                promo_days=("is_promo", "sum"),
                sum_sq_vol=("daily_vol_calc", lambda x: np.sum(x**2))
            ).reset_index()
            
            aggregated_chunks.append(chunk_agg)
            del df_split, d_vol, d_ent; gc.collect()
            
        print(" -> Merging chunks...")
        full_df = pd.concat(aggregated_chunks, ignore_index=True)
        
        final_agg = full_df.groupby(["store_id", "product_id"], observed=True).agg(
            total_days=("total_days", "sum"),
            nonzero_days=("nonzero_days", "sum"),
            total_vol=("total_vol", "sum"),
            sum_entropy=("sum_entropy", "sum"),
            promo_days=("promo_days", "sum"),
            sum_sq_vol=("sum_sq_vol", "sum")
        ).reset_index()
        
        return final_agg

    def _compute_sku_metrics(self, df):
        df["avg_daily_vol"] = df["total_vol"] / df["total_days"]
        df["adi"] = df["total_days"] / df["nonzero_days"].replace(0, 1)
        
        mean_nonzero = df["total_vol"] / df["nonzero_days"].replace(0, 1)
        var = (df["sum_sq_vol"] / df["nonzero_days"].replace(0, 1)) - (mean_nonzero ** 2)
        var = var.clip(lower=0)
        std = np.sqrt(var)
        
        df["cv2"] = (std / mean_nonzero.replace(0, np.nan)) ** 2
        df["cv2"] = df["cv2"].fillna(0)
        
        df["avg_entropy"] = df["sum_entropy"] / df["nonzero_days"].replace(0, 1)
        df["promo_share"] = df["promo_days"] / df["total_days"]
        return df

    def _classify_series(self, df):
        conditions = [
            (df["adi"] < Cfg.SBC_ADI_THRESHOLD) & (df["cv2"] < Cfg.SBC_CV2_THRESHOLD),
            (df["adi"] < Cfg.SBC_ADI_THRESHOLD) & (df["cv2"] >= Cfg.SBC_CV2_THRESHOLD),
            (df["adi"] >= Cfg.SBC_ADI_THRESHOLD) & (df["cv2"] < Cfg.SBC_CV2_THRESHOLD),
            (df["adi"] >= Cfg.SBC_ADI_THRESHOLD) & (df["cv2"] >= Cfg.SBC_CV2_THRESHOLD)
        ]
        choices = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
        df["sbc_class"] = np.select(conditions, choices, default="Unknown")
        
        print("\n[SBC Classification Report]")
        print(df["sbc_class"].value_counts())
        return df

    def _visualize_insights(self, df):
        # Using the corrected visualization logic from previous turn
        palette = {"Smooth": "orange", "Erratic": "blue", "Intermittent": "green", "Lumpy": "red", "Unknown": "gray"}

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="adi", y="cv2", hue="sbc_class", palette=palette, alpha=0.3, s=15, edgecolor=None)
        plt.axvline(Cfg.SBC_ADI_THRESHOLD, color='black', ls='--', lw=1)
        plt.axhline(Cfg.SBC_CV2_THRESHOLD, color='black', ls='--', lw=1)
        plt.xscale('log'); plt.yscale('log')
        plt.title("SBC Matrix (Demand Pattern)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{Cfg.OUT_DIR_ANALYSIS}/sbc_matrix.png")
        plt.close()

        smooth_erratic = df[df["sbc_class"].isin(["Smooth", "Erratic"])]
        plt.figure(figsize=(10, 6))
        sns.histplot(smooth_erratic["avg_daily_vol"], bins=100, log_scale=True, kde=True, color='skyblue')
        plt.axvline(1, color='red', ls='--', label='Risk Threshold (1 unit)')
        plt.axvline(5, color='green', ls='--', label='Safe Threshold (5 units)')
        plt.title("Volume Magnitude Distribution")
        plt.legend()
        plt.savefig(f"{Cfg.OUT_DIR_ANALYSIS}/magnitude_dist.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=smooth_erratic.sample(min(5000, len(smooth_erratic))), 
                        x="avg_daily_vol", y="avg_entropy", hue="sbc_class", palette=palette, alpha=0.4)
        plt.xscale('log')
        plt.title("Intraday Entropy vs Volume")
        plt.savefig(f"{Cfg.OUT_DIR_ANALYSIS}/entropy_vs_vol.png")
        plt.close()

    def _simulate_scenarios(self, df):
        base_pool = df[df["sbc_class"].isin(["Smooth", "Erratic"])].copy()
        base_pool = base_pool.sort_values("total_vol", ascending=False)
        total_system_vol = df["total_vol"].sum()
        
        results = []
        for v_thr in Cfg.SIMULATION_VOL_THRESHOLDS:
            subset = base_pool[base_pool["avg_daily_vol"] >= v_thr]
            count = len(subset)
            vol_covered = subset["total_vol"].sum()
            perc_covered = vol_covered / total_system_vol
            mean_entropy = subset["avg_entropy"].mean()
            
            risk = "High" if (v_thr < 2) else ("Medium" if v_thr < 5 else "Safe")
            
            results.append({
                "Min_Daily_Vol": v_thr,
                "Pair_Limit": count,
                "System_Vol_Covered": perc_covered,
                "Avg_Intraday_Entropy": mean_entropy,
                "Risk_Level": risk
            })
            
        res_df = pd.DataFrame(results)
        print("\n" + "="*60)
        print(" SCENARIO SIMULATION (Data-Driven Trade-offs)")
        print("="*60)
        print(res_df.to_string(index=False, formatters={'System_Vol_Covered': '{:.1%}'.format, 'Avg_Intraday_Entropy': '{:.3f}'.format}))
        
        safe_options = res_df[res_df["Risk_Level"] != "High"]
        best_scenario = safe_options.iloc[0] if not safe_options.empty else res_df.iloc[0]
            
        print("\n" + "="*60)
        print(" FINAL RECOMMENDATION (GOLDEN SUBSET)")
        print("="*60)
        print(f"Recommended PAIR_LIMIT: {int(best_scenario['Pair_Limit'])}")
        print(f" - Reason: Balances high coverage with Intraday Stability.")
        
        return int(best_scenario['Pair_Limit'])