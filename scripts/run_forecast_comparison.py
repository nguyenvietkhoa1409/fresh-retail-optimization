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

def main():
    t0 = time.time()
    print("Khởi chạy Forecast Comparison Engine...")
    engine = ForecastComparisonEngine()
    engine.run()
    dur = (time.time() - t0) / 60
    print(f"\nHoàn thành trong {dur:.1f} phút.")

if __name__ == "__main__":
    main()
