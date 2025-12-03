import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.reporter import PipelineReporter
print("STEP 7: ANALYSIS & REPORTING")
reporter = PipelineReporter()
reporter.run()