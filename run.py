import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.settings import ProjectConfig as Cfg
from src.analysis.reporter import PipelineReporter
reporter = PipelineReporter()
reporter.run()