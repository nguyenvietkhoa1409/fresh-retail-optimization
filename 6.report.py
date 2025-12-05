# report.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.analysis.reporter import PipelineReporter

if __name__ == "__main__":
    print("=== STEP 6: REPORTING STARTED ===")
    reporter = PipelineReporter()
    reporter.run()
    print("\n=== STEP 6: COMPLETED SUCCESSFULLY ===")