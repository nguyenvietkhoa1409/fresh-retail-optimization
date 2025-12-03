# main_forecast.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.demand.forecasting import DemandForecaster

def main():
    print("=== STEP 3: DEMAND FORECASTING STARTED ===")
    
    forecaster = DemandForecaster()
    
    # 1. Run Pipeline: Load -> Train (H1..H7) -> Evaluate
    metrics = forecaster.run()
    
    # 2. Print Summary
    print("\n--- Final Metrics Summary ---")
    print(f"Overall WAPE: {metrics['overall']['WAPE']:.2f}%")
    print(f"Overall RMSE: {metrics['overall']['RMSE']:.4f}")
    
    print("\n=== STEP 3: COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()