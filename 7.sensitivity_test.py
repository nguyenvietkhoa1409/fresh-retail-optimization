import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.analysis.sensitivity import SensitivityAnalyzer
if __name__ == "__main__":
    print("=== STEP 7: SENSITIVITY TEST STARTED ===")
    analyzer = SensitivityAnalyzer(fast_mode=True)
    analyzer.run()
    print("\n=== STEP 7: COMPLETED SUCCESSFULLY ===")