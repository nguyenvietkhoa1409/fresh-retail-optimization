# main_data.py
import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline.generator import SyntheticGenerator
from src.data_pipeline.preprocessor import FrnPreprocessor

def main():
    print("=== STEP 1: DATA PIPELINE STARTED ===")
    
    # 1. Preprocessing (Process FRN-50K data first to get store IDs)
    processor = FrnPreprocessor()
    df_clean = processor.run()
    
    # 2. Synthetic Logistics Data Generation
    # We need real store IDs from the clean data to map logistics correctly
    if "store_id" in df_clean.columns:
        unique_stores = df_clean["store_id"].unique()
    else:
        print("Warning: store_id not found in clean data, using dummy IDs.")
        unique_stores = [f"store_{i}" for i in range(20)]
        
    gen = SyntheticGenerator()
    gen.generate_all(unique_stores)
    
    print("\n=== STEP 1: COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()