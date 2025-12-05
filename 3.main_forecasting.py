# main_forecasting.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.demand.forecasting import DemandForecaster

if __name__ == "__main__":
    print("=== STEP 3: DEMAND FORECASTING STARTED ===")
    forecaster = DemandForecaster()
    forecaster.run()
    print("\n=== STEP 3: COMPLETED SUCCESSFULLY ===")