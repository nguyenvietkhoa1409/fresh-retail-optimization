# main_demand.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.demand.reconstruction import DemandReconstructor

if __name__ == "__main__":
    print("=== STEP 2: DEMAND RECONSTRUCTION STARTED ===")
    reconstructor = DemandReconstructor()
    reconstructor.run()
    print("\n=== STEP 2: COMPLETED SUCCESSFULLY ===")