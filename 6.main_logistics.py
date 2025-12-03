# main_logistics.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.logistics import LogisticsManager

def main():
    print("=== STEP 6: LOGISTICS VRP OPTIMIZATION STARTED ===")
    
    manager = LogisticsManager()
    manager.run()
    
    print("\n=== STEP 6: COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()