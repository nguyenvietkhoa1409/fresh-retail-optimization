# run_all.py
import os
import sys
import time


# Ensure python finds modules in src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import ProjectConfig as Cfg

try:
    # --- MODULE IMPORTS ---
    # 1. Pipeline Start
    from src.analysis.dataset_analyzer import FreshRetailAnalyzer
    from src.data_pipeline.preprocessor import FrnPreprocessor
    
    # 2. Network & Enrichment (NEW)
    from src.data_pipeline.generator import EnhancedSupplierGenerator
    from src.data_pipeline.catalog_enricher import CatalogEnricher # <--- NEW
    
    # 3. Demand Side
    from src.demand.reconstruction import DemandReconstructor 
    from src.demand.forecasting import DemandForecaster
    
    # 4. Planning & Optimization
    from src.inventory.planner import InventoryPlanner # <--- Updated SAA Version
    from src.inventory.evaluator import PolicyEvaluator 
    from src.optimization.integrated_solver import IntegratedSolver
    
    # 5. Reporting
    from src.analysis.reporter import PipelineReporter
    from src.analysis.baselines import BaselineFramework
    
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import modules: {e}")
    sys.exit(1)

def run_pipeline():
    t0 = time.time()
    print("\n" + "="*60)
    print("   FRESH RETAIL OPTIMIZATION - DATA DRIVEN PIPELINE (v4)")
    print("   Workflow: Analyze -> Enrich -> Forecast -> SAA Plan -> Optimize")
    print("="*60 + "\n")

    # # ---------------------------------------------------------
    # # STEP 1: ANALYSIS & PREPROCESSING
    # # ---------------------------------------------------------
    # print(">>> STEP 1: ANALYSIS & PREPROCESSING")
    
    # # Analyze Dataset
    # # analyzer = FreshRetailAnalyzer()
    # # recommended_limit = analyzer.run()
    # # # (Optional: Update Cfg.PAIR_LIMIT dynamically if needed)
    
    # # Download & Clean
    # preprocessor = FrnPreprocessor()
    # df_clean = preprocessor.run()
    
    # # Get active stores
    # real_stores = []
    # if df_clean is not None and "store_id" in df_clean.columns:
    #     real_stores = df_clean["store_id"].unique().tolist()
    
    # print("    [✓] Step 1 Complete.\n")

    # # ---------------------------------------------------------
    # # STEP 2: MASTER DATA & ENRICHMENT (CORRECTED ORDER)
    # # ---------------------------------------------------------
    # print(">>> STEP 2: MASTER DATA & ENRICHMENT")
    
    # # 1. Create Catalog FIRST (Enricher reads Preprocessed Data -> Creates Catalog)
    # print("    -> Generating Master Product Catalog...")
    # enricher = CatalogEnricher()
    # enricher.run()
    
    # # 2. Generate Supply Network SECOND (Generator reads Catalog -> Creates Suppliers)
    # print("    -> Generating Supply Chain Network...")
    # unique_stores = real_stores if real_stores else [f"store_{i:03d}" for i in range(1, 21)]
    # gen = EnhancedSupplierGenerator()
    # gen.generate_all(unique_stores)
    
    # print("    [✓] Step 2 Complete: Physical & Financial Data Ready.\n")

    # ---------------------------------------------------------
    # STEP 3: DEMAND MODELING
    # ---------------------------------------------------------
    print(">>> STEP 3: DEMAND RECONSTRUCTION & FORECASTING")
    
    # print("    -> Demand Reconstruction...")
    # DemandReconstructor().run()
    
    print("    -> Demand Forecasting (Exporting SAA Residuals)...")
    DemandForecaster().run()
    
    print("    [✓] Step 3 Complete.\n")

    # ---------------------------------------------------------
    # STEP 4: INVENTORY PLANNING (SAA)
    # ---------------------------------------------------------
    print(">>> STEP 4: DATA-DRIVEN INVENTORY PLANNING")
    # Uses Forecasts + Residuals + Risk Catalog to output Order Qty
    planner = InventoryPlanner()
    planner.run()
    print("    [✓] Step 4 Complete: SAA Policies Generated.\n")

    evaluator = PolicyEvaluator()
    evaluator.run()

    # # ---------------------------------------------------------
    # # STEP 5: INTEGRATED OPTIMIZATION
    # # ---------------------------------------------------------
    # print(">>> STEP 5: INTEGRATED PROCUREMENT & LOGISTICS")
    # print("    Running Iterative Solver...")
    
    # solver = IntegratedSolver()
    # solver.run()
    # print("    [✓] Step 5 Complete.\n")

    # # ---------------------------------------------------------
    # # STEP 6: REPORTING
    # # ---------------------------------------------------------
    # print(">>> STEP 6: REPORTING & BENCHMARKING")
    
    # reporter = PipelineReporter()
    # reporter.run()

    # # Optional Baseline
    # # try:
    # #     baseline = BaselineFramework()
    # #     results = baseline.run_all_baselines()
    # #     baseline.visualize_comparison(results)
    # # except Exception as e:
    # #     print(f"    [Info] Baseline skipped: {e}")

    # dur = (time.time() - t0) / 60
    # print("\n" + "="*60)
    # print(f"   PIPELINE SUCCESS. Duration: {dur:.1f} min.")
    # print(f"   Artifacts: {Cfg.ARTIFACTS_DIR}")
    # print("="*60)

if __name__ == "__main__":
    run_pipeline()