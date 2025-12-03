# main_procurement.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.procurement import ProcurementOptimizer

def main():
    print("=== STEP 5: PROCUREMENT OPTIMIZATION STARTED ===")
    
    optimizer = ProcurementOptimizer()
    optimizer.run()
    
    print("\n=== STEP 5: COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()