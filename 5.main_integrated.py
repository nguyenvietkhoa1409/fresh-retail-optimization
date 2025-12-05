# main_integrated.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the class from the file we created in src
from src.optimization.integrated_solver import IntegratedSolver

def main():
    print("=== STEP 5: PROCUREMENT + LOGISTICS: INTEGRATED SOLVER STARTED ===")
    solver = IntegratedSolver()
    solver.run()
    print("\n=== STEP 5: COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()