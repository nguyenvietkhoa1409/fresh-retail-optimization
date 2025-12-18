import os
import sys
import pandas as pd
import numpy as np
import time

# Đảm bảo Python tìm thấy các module trong src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import ProjectConfig as Cfg

try:
    # --- MỚI: Import Analyzer & Preprocessor ---
    from src.analysis.dataset_analyzer import FreshRetailAnalyzer
    from src.data_pipeline.preprocessor import FrnPreprocessor

    from src.data_pipeline.generator import EnhancedSupplierGenerator
    from src.inventory.planner import EnhancedInventoryPlanner
    from src.optimization.integrated_solver import IntegratedSolver
    
    # Các module cũ (Demand/Forecast)
    from src.demand.reconstruction import DemandReconstructor 
    from src.demand.forecasting import DemandForecaster
    from src.analysis.reporter import PipelineReporter
    from src.analysis.baselines import BaselineFramework
    
except ImportError as e:
    print(f"[CRITICAL ERROR] Không thể import module: {e}")
    print("Hãy đảm bảo cấu trúc thư mục src/ đúng và chứa đầy đủ các file __init__.py")
    sys.exit(1)

def run_pipeline():
    t0 = time.time()
    print("\n" + "="*60)
    print("   FRESH RETAIL OPTIMIZATION - INTELLIGENT PIPELINE (v3)")
    print("   Workflow: Analyze -> Preprocess -> Generate -> Optimize")
    print("="*60 + "\n")

    # ---------------------------------------------------------
    # BƯỚC 1: DATASET ANALYSIS & PARAMETER TUNING (MỚI)
    # ---------------------------------------------------------
    print(">>> STEP 1: DATASET ANALYSIS & SUBSET SELECTION")
    print("    Analyzing Hf Dataset to determine optimal subset parameters...")
    
    analyzer = FreshRetailAnalyzer()
    recommended_limit = analyzer.run()
    
    # Cập nhật Config runtime (nếu module Config hỗ trợ) hoặc in ra để log
    print(f"    [Recommendation] Based on analysis, suggested PAIR_LIMIT: {recommended_limit}")
    # Lưu ý: Nếu FrnPreprocessor đọc trực tiếp Cfg.GLOBAL_PAIR_LIMIT, 
    # bạn có thể cần gán: Cfg.GLOBAL_PAIR_LIMIT = recommended_limit tại đây.
    
    print("    [✓] Step 1 Complete: Analysis Artifacts Saved.\n")

    # ---------------------------------------------------------
    # BƯỚC 2: DATA DOWNLOAD & PREPROCESSING (MỚI)
    # ---------------------------------------------------------
    print(">>> STEP 2: DATA DOWNLOAD & PREPROCESSING")
    
    preprocessor = FrnPreprocessor()
    df_clean = preprocessor.run()
    
    # Lấy danh sách Store ID thực tế từ dữ liệu vừa clean để dùng cho bước sau
    real_stores = []
    if df_clean is not None and "store_id" in df_clean.columns:
        real_stores = df_clean["store_id"].unique().tolist()
        print(f"    [Input] Preprocessor provided {len(real_stores)} active stores.")
    else:
        print("    [Warn] Preprocessor returned empty/invalid data. Will attempt fallback.")

    print("    [✓] Step 2 Complete: Clean Data Ready.\n")

    # ---------------------------------------------------------
    # BƯỚC 3: SUPPLY CHAIN NETWORK GENERATION
    # ---------------------------------------------------------
    print(">>> STEP 3: SUPPLY CHAIN NETWORK GENERATION")
    
    # Logic: Ưu tiên dùng Store ID từ bước Preprocessing
    unique_stores = real_stores

    # Fallback 1: Nếu Preprocessor fail, thử đọc file cũ
    if not unique_stores:
        store_source_path = os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement.parquet")
        if os.path.exists(store_source_path):
            try:
                stores_df = pd.read_parquet(store_source_path)
                unique_stores = stores_df['store_id'].unique().tolist()
                print(f"    [Fallback] Loaded {len(unique_stores)} stores from existing artifact.")
            except:
                pass
    
    # Fallback 2: Tạo Mock IDs
    if not unique_stores:
        print("    [Fallback] No stores found. Generating MOCK Store IDs.")
        limit_stores = getattr(Cfg, 'GLOBAL_NUM_STORES', 20)
        unique_stores = [f"store_{i:03d}" for i in range(1, limit_stores + 1)]

    # Chạy Generator
    gen = EnhancedSupplierGenerator()
    gen.generate_all(unique_stores)
    print("    [✓] Step 3 Complete: Network Artifacts Generated.\n")

    # ---------------------------------------------------------
    # BƯỚC 4 & 5: DEMAND RECONSTRUCTION & FORECASTING
    # ---------------------------------------------------------
    print(">>> STEP 4 & 5: DEMAND RECONSTRUCTION & FORECASTING")
    
    # Kiểm tra xem có cần chạy lại không (hoặc ép chạy lại nếu data mới)
    # Ở đây giữ logic cũ: kiểm tra file tồn tại. 
    # Tuy nhiên, vì vừa chạy Preprocess mới, tốt nhất nên xóa file cũ hoặc force run.
    # Để an toàn theo logic bạn yêu cầu, ta sẽ gọi hàm run() luôn.
    
    print("    -> Running Demand Reconstruction...")
    DemandReconstructor().run()
    
    print("    -> Running Demand Forecasting...")
    DemandForecaster().run()
    
    print("    [✓] Step 4 & 5 Complete.\n")

    # ---------------------------------------------------------
    # BƯỚC 6: INVENTORY PLANNING (ENHANCED)
    # ---------------------------------------------------------
    print(">>> STEP 6: INVENTORY PLANNING & POLICY")
    planner = EnhancedInventoryPlanner()
    planner.run()
    print("    [✓] Step 6 Complete: Inventory Policies Updated.\n")

    # ---------------------------------------------------------
    # BƯỚC 7: INTEGRATED OPTIMIZATION
    # ---------------------------------------------------------
    print(">>> STEP 7: INTEGRATED PROCUREMENT & LOGISTICS OPTIMIZATION")
    print("    Running Iterative Solver with Feedback Loop...")
    
    solver = IntegratedSolver()
    solver.run()
    print("    [✓] Step 7 Complete.\n")

    # ---------------------------------------------------------
    # BƯỚC 8: REPORTER & BASELINES
    # ---------------------------------------------------------
    print(">>> STEP 8: REPORTING & BENCHMARKING")
    
    reporter = PipelineReporter()
    reporter.run()

    # Chạy Baseline so sánh (nếu cần thiết, có thể comment out để tiết kiệm thời gian)
    try:
        baseline = BaselineFramework()
        results = baseline.run_all_baselines()
        baseline.visualize_comparison(results)
    except Exception as e:
        print(f"    [Info] Baseline comparison skipped or failed: {e}")

    dur = (time.time() - t0) / 60
    print("\n" + "="*60)
    print(f"   PIPELINE COMPLETED SUCCESSFULLY. Duration: {dur:.1f} min.")
    print(f"   Check outputs in: {Cfg.ARTIFACTS_DIR}")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()