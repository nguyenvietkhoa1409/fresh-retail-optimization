"""
Standalone runner cho Forecast Comparison Engine
Chạy dễ dàng trên Google Colab.
"""
import os
import sys
import time

# Ensure python finds modules in src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.demand.forecast_comparison import ForecastComparisonEngine

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Forecast Comparison Engine")
    parser.add_argument("--out_dir", type=str, default=None, 
                        help="Optional external output directory (e.g. /content/drive/MyDrive/...)")
    args = parser.parse_args()

    t0 = time.time()
    print("Khởi chạy Forecast Comparison Engine...")
    
    if args.out_dir:
        print(f"Sử dụng thư mục lưu trữ tùy chỉnh: {args.out_dir}")
        
    engine = ForecastComparisonEngine(out_dir=args.out_dir)
    engine.run()
    dur = (time.time() - t0) / 60
    print(f"\nHoàn thành trong {dur:.1f} phút.")

if __name__ == "__main__":
    main()
