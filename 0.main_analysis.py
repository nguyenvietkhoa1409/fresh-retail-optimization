import sys
import os
# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.analysis.dataset_analyzer import FreshRetailAnalyzer

if __name__ == "__main__":
    print("=== STEP 0: DATASET DEEP DIVE ANALYSIS STARTED ===")
    
    analyzer = FreshRetailAnalyzer()
    recommended_limit = analyzer.run()
    
    print("\n[NEXT STEPS]")
    print(f"Please update 'PAIR_LIMIT' in config/settings.py to: {recommended_limit}")
    print("Then run '1.main_data.py'.")
    print("=== STEP 0: COMPLETED SUCCESSFULLY ===")