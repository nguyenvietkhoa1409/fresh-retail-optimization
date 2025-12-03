# main_inventory.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inventory.planner import InventoryPlanner

def main():
    print("=== STEP 4: INVENTORY PLANNING STARTED ===")
    
    planner = InventoryPlanner()
    planner.run()
    
    print("\n=== STEP 4: COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()