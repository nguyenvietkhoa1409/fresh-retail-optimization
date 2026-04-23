import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure python finds modules in src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ProjectConfig as Cfg

def plot_results(out_dir, df_metrics, df_all):
    # 1. Bar Chart WAPE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x='WAPE (%)', y='Model', palette='viridis')
    plt.title('WAPE Comparison across Models (Lower is Better)')
    for i, v in enumerate(df_metrics['WAPE (%)']):
        plt.text(v + 0.5, i, f"{v:.1f}%", va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_wape_bar.png"), dpi=300)
    plt.close()
    
    # 2. Time Series Overlay cho item tốt nhất
    try:
        top_item = df_all.groupby(['store_id', 'product_id'])['y_true'].sum().idxmax()
        df_sub = df_all[(df_all['store_id'] == top_item[0]) & (df_all['product_id'] == top_item[1])]
        
        if not df_sub.empty:
            plt.figure(figsize=(12, 6))
            
            # Vẽ actual demand
            method_any = df_sub['method'].iloc[0]
            df_true = df_sub[df_sub['method'] == method_any].sort_values('date')
            plt.plot(df_true['date'], df_true['y_true'], label='Actual Demand', color='black', linewidth=2.5, marker='o')
            
            # Vẽ Proposed ML nếu có
            if 'LightGBM (Proposed)' in df_sub['method'].unique():
                df_m = df_sub[df_sub['method'] == 'LightGBM (Proposed)'].sort_values('date')
                plt.plot(df_m['date'], df_m['y_pred'], label='LightGBM (Proposed)', color='blue', linewidth=2, linestyle='-')
            
            # Vẽ các models khác
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
            plt.savefig(os.path.join(out_dir, "comparison_time_series_overlay.png"), dpi=300)
            plt.close()
            print(f"  -> Đã tạo biểu đồ tại: {out_dir}")
    except Exception as e:
        print(f"  [Warn] Failed to plot overlay: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Forecast Comparison Report")
    parser.add_argument("--out_dir", type=str, default=None, 
                        help="Thư mục chứa comparison_results_raw.csv (vd: /content/drive/MyDrive/...)")
    parser.add_argument("--ml_residuals_path", type=str, default=None,
                        help="Đường dẫn đến file forecast_residuals.parquet của mô hình LightGBM gốc")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else Cfg.FC_COMPARISON_OUT_DIR
    raw_results_path = os.path.join(out_dir, "comparison_results_raw.csv")
    
    if not os.path.exists(raw_results_path):
        print(f"[Error] Không tìm thấy file kết quả tại: {raw_results_path}")
        return

    print("Đang đọc file kết quả CSV...")
    df_res = pd.read_csv(raw_results_path)
    
    # Load kết quả LightGBM chính
    ml_path = args.ml_residuals_path if args.ml_residuals_path else os.path.join(Cfg.ARTIFACTS_DIR, "forecast_residuals.parquet")
    if os.path.exists(ml_path):
        print(f"Đang ghép nối với kết quả LightGBM từ: {ml_path}")
        full_res = pd.read_parquet(ml_path)
        ml_test_res = full_res[full_res['method'] == 'ml'].copy()
        ml_test_res = ml_test_res[['store_id', 'product_id', 'method', 'horizon', 'date', 'y_true', 'y_pred']]
        ml_test_res['method'] = 'LightGBM (Proposed)'
        
        df_all = pd.concat([df_res, ml_test_res], ignore_index=True)
    else:
        print(f"  [Warn] Không tìm thấy file LightGBM tại {ml_path}. Bỏ qua LightGBM.")
        df_all = df_res

    print("\nĐang tính toán metrics...")
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
    
    summary_path = os.path.join(out_dir, "forecast_comparison_summary.csv")
    metrics_path = os.path.join(out_dir, "forecast_comparison_metrics.json")
    
    df_metrics.to_csv(summary_path, index=False)
    with open(metrics_path, 'w') as f:
        json.dump(df_metrics.to_dict(orient='records'), f, indent=2)
        
    print("\n" + "="*60)
    print(" KẾT QUẢ SO SÁNH CÁC MÔ HÌNH DỰ BÁO (ML-Eligible subset)")
    print("="*60)
    print(df_metrics.to_string(index=False))
    print("="*60)
    
    plot_results(out_dir, df_metrics, df_all)
    print(f"\nHoàn tất! Kết quả đã được lưu tại {out_dir}")

if __name__ == "__main__":
    main()
