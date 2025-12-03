# main_demand.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.demand.reconstruction import DemandReconstructor

def main():
    print("=== STEP 2: DEMAND RECONSTRUCTION STARTED ===")
    
    reconstructor = DemandReconstructor()
    
    # Run the full pipeline
    # Input tự động lấy từ data/artifacts/part1/preprocessed.parquet (mặc định trong code)
    df_recon = reconstructor.run()
    
    print("\n=== STEP 2: COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()