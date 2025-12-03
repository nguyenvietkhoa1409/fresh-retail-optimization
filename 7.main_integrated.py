# main_integrated.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.integrated_solver import IntegratedSolver

def main():
    print("=== FINAL CAPSTONE STEP: INTEGRATED SOLVER ===")
    
    # Run the adaptive loop
    solver = IntegratedSolver()
    solver.run()
    
    print("\n=== SYSTEM OPTIMIZATION COMPLETE ===")

if __name__ == "__main__":
    main()