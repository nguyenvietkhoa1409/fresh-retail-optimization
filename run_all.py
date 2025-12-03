# run_all.py
import sys
import os
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import ProjectConfig as Cfg
from src.data_pipeline.preprocessor import FrnPreprocessor
from src.data_pipeline.generator import SyntheticGenerator
from src.demand.reconstruction import DemandReconstructor
from src.demand.forecasting import DemandForecaster
from src.inventory.planner import InventoryPlanner
from src.optimization.procurement import ProcurementOptimizer
from src.optimization.logistics import LogisticsManager
from src.optimization.integrated_solver import IntegratedSolver
from src.analysis.reporter import PipelineReporter
from src.analysis.sensitivity import SensitivityAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_sep(name):
    print("\n" + "="*60)
    print(f"🚀 STARTING {name}")
    print("="*60)

def run_pipeline():
    t0 = time.time()
    try:
        # 1. Data
        print_sep("STEP 1: DATA")
        preprocessor = FrnPreprocessor()
        df_clean = preprocessor.run()
        stores = df_clean["store_id"].unique() if "store_id" in df_clean.columns else [f"s_{i}" for i in range(20)]
        SyntheticGenerator().generate_all(stores)

        # 2. Reconstruction
        print_sep("STEP 2: RECONSTRUCTION")
        DemandReconstructor().run()

        # 3. Forecasting
        print_sep("STEP 3: FORECASTING")
        DemandForecaster().run()

        # 4. Inventory
        print_sep("STEP 4: INVENTORY")
        InventoryPlanner().run()

        # 5. Procurement (Static Baseline)
        print_sep("STEP 5: PROCUREMENT (STATIC)")
        ProcurementOptimizer().run()

        # 6. Logistics (Static Baseline)
        print_sep("STEP 6: LOGISTICS (STATIC)")
        LogisticsManager().run()

        # 6.5 Integrated
        print_sep("STEP 6.5: INTEGRATED OPTIMIZATION")
        IntegratedSolver().run()

        # 7. Sensitivity (Scientific Bonus)
        print_sep("STEP 7: SENSITIVITY ANALYSIS")
        # Runs 18 iterations, might take 5-10 mins
        SensitivityAnalyzer().run()

        # 8. Reporting
        print_sep("STEP 8: REPORTING")
        PipelineReporter().run()

        dur = (time.time() - t0) / 60
        print(f"\n✅ ALL SYSTEMS GO. Duration: {dur:.1f} min.")

    except Exception as e:
        logger.exception("❌ PIPELINE FAILED")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()