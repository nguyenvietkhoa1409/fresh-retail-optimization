import os
import pandas as pd
import numpy as np
import math
from copy import deepcopy
import sys

# 1. CẤU HÌNH OVERRIDE
from config.settings import ProjectConfig as Cfg
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Tăng giới hạn khoảng cách & thời gian
Cfg.VRP_MAX_ROUTE_DISTANCE_KM = 5000 
Cfg.WAREHOUSE_WINDOW = (0, 1440)     

# Import các class gốc
from src.optimization.logistics import LogisticsManager, IntegratedVRPSolver
from src.utils.vrp_visualization import RouteMapVisualizer

# --- 2. CLASS PATCHED (SỬA LỖI CAPACITY) ---
class SmartLogisticsManager(LogisticsManager):
    """
    Phiên bản nâng cấp của LogisticsManager.
    Tự động chia nhỏ (Split) các đơn hàng quá khổ để vừa với xe tải.
    """
    
    def _get_max_vehicle_capacity(self):
        """Tìm tải trọng xe lớn nhất trong đội xe"""
        if hasattr(Cfg, 'VEHICLE_FLEET_DEFINITIONS'):
            return max(v['capacity'] for v in Cfg.VEHICLE_FLEET_DEFINITIONS)
        return 5000 # Fallback an toàn

    def _split_oversized_demands(self, df_demand, max_cap):
        """
        Logic chia nhỏ đơn hàng:
        Nếu Demand = 15000kg, MaxCap = 8000kg
        -> Tách thành: 8000kg, 7000kg
        """
        new_rows = []
        split_count = 0
        
        for _, row in df_demand.iterrows():
            qty = row['demand_kg']
            if qty <= max_cap:
                new_rows.append(row)
                continue
            
            # Bắt đầu chia
            while qty > 0:
                chunk = min(qty, max_cap)
                new_r = row.copy()
                new_r['demand_kg'] = chunk
                new_rows.append(new_r)
                qty -= chunk
                split_count += 1
                
        if split_count > 0:
            print(f"   [SmartLogistics] Đã chia nhỏ {split_count} đơn hàng quá khổ ( > {max_cap}kg)!")
            
        return pd.DataFrame(new_rows)

    def run(self):
        """Override phương thức run để chèn logic chia nhỏ"""
        print("[SmartLogistics] Đang chạy với logic xử lý đơn hàng lớn...")
        
        # --- PHẦN 1: CHUẨN BỊ DỮ LIỆU (Giống code gốc) ---
        agg_path = os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv")
        if not os.path.exists(agg_path): 
            print("Không tìm thấy file summary procurement.")
            return []
            
        df_agg = pd.read_csv(agg_path)
        df_sup = pd.read_csv(os.path.join(Cfg.ARTIFACTS_DIR, "suppliers.csv"))
        df_store = pd.read_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement.parquet"))
        
        # --- PHẦN 2: XỬ LÝ INBOUND (Có can thiệp) ---
        # 2.1 Gom nhóm ban đầu
        in_dem = df_agg.groupby('supplier_id')['total_kg'].sum().reset_index()
        in_dem['supplier_id'] = in_dem['supplier_id'].astype(int)
        df_sup['supplier_id'] = df_sup['supplier_id'].astype(int)
        
        # 2.2 Merge thông tin địa lý
        in_dem = in_dem.merge(df_sup, on='supplier_id').rename(
            columns={'supplier_id': 'id', 'sup_lat': 'lat', 'sup_lon': 'lon', 'total_kg': 'demand_kg'}
        )
        
        # 2.3 [NEW] CHIA NHỎ ĐƠN HÀNG QUÁ KHỔ
        max_v_cap = self._get_max_vehicle_capacity()
        in_dem = self._split_oversized_demands(in_dem, max_v_cap)
        
        # 2.4 Chạy Solver Inbound
        print(f"   -> Solving Inbound: {len(in_dem)} stops (sau khi split)")
        in_solver = IntegratedVRPSolver(self.wh_loc, in_dem, "Inbound", 240)
        in_routes = in_solver.solve()
        
        # Tính thời gian Crossdock (như cũ)
        last_arr = max([r['steps'][-1]['arrival_time_min'] for r in in_routes]) if in_routes else 240
        out_start = last_arr + Cfg.SERVICE_TIME_CROSSDOCK_MINS
        if out_start > 540: out_start = 360 # Next day logic
        
        # --- PHẦN 3: XỬ LÝ OUTBOUND (Có can thiệp tương tự) ---
        df_store['store_id'] = df_store['store_id'].astype(str)
        df_agg['store_id'] = df_agg['store_id'].astype(str)
        cols_to_merge = ['store_id','store_lat','store_lon','tw_open','tw_close', 'service_time']
        
        out_dem = df_agg.groupby('store_id')['total_kg'].sum().reset_index().merge(
            df_store[cols_to_merge].drop_duplicates(), on='store_id'
        ).rename(columns={'store_id': 'id', 'store_lat': 'lat', 'store_lon': 'lon', 'total_kg': 'demand_kg'})
        
        # [NEW] Chia nhỏ Outbound nếu cần
        out_dem = self._split_oversized_demands(out_dem, max_v_cap)
        
        print(f"   -> Solving Outbound: {len(out_dem)} stops (sau khi split)")
        out_solver = IntegratedVRPSolver(self.wh_loc, out_dem, "Outbound", out_start)
        out_routes = out_solver.solve()
        
        # --- [CRITICAL UPDATE] ---
        # Lưu file csv kết quả vào đúng thư mục output chỉ định
        all_routes = in_routes + out_routes
        if all_routes:
            # Lưu file chi tiết tuyến đường
            pd.DataFrame(all_routes).to_csv(os.path.join(self.out_dir, "vrp_routes_solution.csv"), index=False)
            
            # Tính tổng chi phí để lưu summary
            total_cost = sum(r['cost'] for r in all_routes)
            pd.DataFrame([{
                'Cost_USD': total_cost, 
                'Crossdock_Ready_Time': f"{int(out_start//60)}:{int(out_start%60):02d}",
                'Is_Next_Day': out_start > 540
            }]).to_csv(os.path.join(self.out_dir, "vrp_summary.csv"), index=False)
            
        return all_routes

# --- 3. HÀM CHẠY CHÍNH ---
def run_fix():
    # Tên file plan tốt nhất của bạn
    TARGET_FILE = 'procurement_plan_Local-Batch_P2_U5.csv' 
    SCENARIO_NAME = 'Local-Batch' # Giữ nguyên tên gốc để khớp logic report
    
    print(f"\n>>> RUNNING VRP FIX FOR: {SCENARIO_NAME}")
    
    # 1. Tạo input giả lập
    input_path = os.path.join(Cfg.OUT_DIR_ANALYSIS, TARGET_FILE)
    if not os.path.exists(input_path):
        print(f"[Error] Không tìm thấy: {input_path}")
        return

    df_plan = pd.read_csv(input_path)
    weight_col = 'order_weight_kg' if 'order_weight_kg' in df_plan.columns else 'total_weight'
    agg_df = df_plan.groupby(['supplier_id', 'store_id'])[weight_col].sum().reset_index()
    agg_df.rename(columns={weight_col: 'total_kg'}, inplace=True)
    
    temp_path = os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv")
    agg_df.to_csv(temp_path, index=False)

    # 2. Chạy Smart Logistics Manager
    # Lưu vào thư mục riêng để kiểm tra trước
    output_dir = os.path.join("data", "artifacts", "vrp_fixed_runs", SCENARIO_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    log_manager = SmartLogisticsManager(override_output_dir=output_dir)
    routes = log_manager.run()

    # 3. Kết quả & Visualization
    inbound_count = len([r for r in routes if r['role'] == 'Inbound'])
    print(f" -> RESULT: Inbound={inbound_count}, Outbound={len(routes)-inbound_count}")
    
    if inbound_count > 0:
        # Vẽ bản đồ
        print(" -> SUCCESS! Generating Maps...")
        viz = RouteMapVisualizer(save_dir=output_dir)
        viz.visualize_scenario(
            routes=routes, 
            warehouse_loc=(Cfg.CENTER_LAT, Cfg.CENTER_LON), 
            scenario_name=SCENARIO_NAME
        )
        
        # --- [NEW LOGIC] XUẤT FILE CHO REPORT ---
        # Xuất ra file routes_{Name}.csv vào thư mục logistics chính (để Report đọc được)
        main_logistics_dir = getattr(Cfg, 'OUT_DIR_LOGISTICS', 'data/artifacts/vrp_route_maps')
        os.makedirs(main_logistics_dir, exist_ok=True)
        
        final_route_path = os.path.join(main_logistics_dir, f"routes_{SCENARIO_NAME}.csv")
        pd.DataFrame(routes).to_csv(final_route_path, index=False)
        print(f" -> [IMPORTANT] Saved fixed routes to: {final_route_path}")
        print("    Bạn có thể chạy lại report.py ngay bây giờ.")
        
    else:
        print(" -> STILL FAILED. Có thể Capacity vẫn chưa phải là nguyên nhân duy nhất, hoặc file config xe quá nhỏ.")

if __name__ == "__main__":
    run_fix()