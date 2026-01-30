# src/visualization/route_map.py
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

# Đảm bảo import được module từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class RouteMapVisualizer:
    """
    Advanced Visualization for VRP Routes.
    High contrast, clear annotations, and bigger markers for visibility.
    """
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Cấu hình style: Dùng ggplot hoặc seaborn-whitegrid cho nền sáng, dễ nhìn
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('ggplot')

    def visualize_scenario(self, routes, warehouse_loc, scenario_name):
        """
        Main entry point to visualize both Inbound and Outbound.
        """
        inbound_routes = [r for r in routes if r['role'] == 'Inbound']
        outbound_routes = [r for r in routes if r['role'] == 'Outbound']
        
        # 1. Visualize Inbound
        if inbound_routes:
            self._plot_layer(
                routes=inbound_routes, 
                warehouse_loc=warehouse_loc, 
                title=f"Inbound Logistics - {scenario_name}", 
                filename=f"map_inbound_{scenario_name}.png",
                node_label="Supplier",
                marker_style='^' # Tam giác
            )
            
        # 2. Visualize Outbound
        if outbound_routes:
            self._plot_layer(
                routes=outbound_routes, 
                warehouse_loc=warehouse_loc, 
                title=f"Outbound Logistics - {scenario_name}", 
                filename=f"map_outbound_{scenario_name}.png",
                node_label="Store",
                marker_style='o' # Hình tròn
            )

    def _plot_layer(self, routes, warehouse_loc, title, filename, node_label, marker_style):
        # Tăng kích thước ảnh lớn hơn nữa để các node không bị đè
        plt.figure(figsize=(16, 12))
        
        # 1. Vẽ Kho Trung Tâm (Depot) - To và nổi bật
        wh_lat, wh_lon = warehouse_loc
        plt.scatter(wh_lon, wh_lat, c='black', s=500, marker='*', 
                   edgecolors='white', linewidth=2, zorder=30, label='Center Warehouse')
        
        # 2. Setup Colors - Dùng TAB10 (Cũ) cho độ tương phản cao nhất
        num_routes = len(routes)
        # Sử dụng plt.get_cmap để tránh warning deprecation
        try:
            cmap = plt.get_cmap('tab10')
        except:
            cmap = cm.get_cmap('tab10')
            
        total_km = 0
        total_cost = 0
        
        # 3. Vẽ từng tuyến
        for idx, route in enumerate(routes):
            # Lặp lại màu nếu quá 10 xe (tab10 chỉ có 10 màu)
            color = cmap(idx % 10)
            
            # Thông tin hiển thị Legend: Loại xe + % Tải
            veh_type = route.get('vehicle_type', 'Truck')
            load_pct = route.get('utilization_pct', 0)
            label_str = f"{veh_type} #{idx+1} ({load_pct}%)"
            
            # Tọa độ
            lats = [s['lat'] for s in route['steps']]
            lons = [s['lon'] for s in route['steps']]
            
            # --- A. Vẽ Đường (Lines) ---
            # Nét đậm (linewidth=3) để dễ nhìn đường đi
            plt.plot(lons, lats, color=color, linewidth=3, alpha=0.7, linestyle='-', zorder=1)
            
            # --- B. Vẽ Điểm Dừng (Stops) ---
            stop_lats = lats[1:-1]
            stop_lons = lons[1:-1]
            
            if stop_lats:
                # Marker to (s=250) để chứa được số thứ tự bên trong
                plt.scatter(stop_lons, stop_lats, c=[color], s=250, marker=marker_style, 
                           edgecolors='white', linewidth=1.5, zorder=20, label=label_str)
                
                # Số thứ tự: Màu trắng, Font to (11), Đậm
                for i, (lat, lon) in enumerate(zip(stop_lats, stop_lons)):
                    plt.text(lon, lat, str(i+1), fontsize=11, fontweight='bold', 
                             color='white', ha='center', va='center', zorder=21)

            # Cộng dồn chỉ số
            total_km += route.get('distance_km', 0)
            total_cost += route.get('cost', 0)

        # 4. Trang trí
        plt.title(f"{title}\nTotal Cost: ${total_cost:,.0f} | Total Distance: {total_km:,.1f} km", 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Longitude", fontsize=14)
        plt.ylabel("Latitude", fontsize=14)
        
        # --- C. Xử lý Legend ---
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Sắp xếp: Kho lên đầu
        wh_items = [(h, l) for h, l in zip(handles, labels) if 'Warehouse' in l]
        route_items = [(h, l) for h, l in zip(handles, labels) if 'Warehouse' not in l]
        
        # Giới hạn legend nếu quá nhiều
        MAX_LEGEND = 12
        final_handles = [x[0] for x in wh_items] + [x[0] for x in route_items[:MAX_LEGEND]]
        final_labels = [x[1] for x in wh_items] + [x[1] for x in route_items[:MAX_LEGEND]]
        
        if len(route_items) > MAX_LEGEND:
            final_handles.append(plt.Line2D([0], [0], color='white'))
            final_labels.append("... (others)")

        # Đặt legend ra ngoài bên phải
        plt.legend(final_handles, final_labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
                   title="Vehicle & Load", fontsize=12, title_fontsize=13, frameon=True, shadow=True)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # 5. Save
        out_path = os.path.join(self.save_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    -> Map saved: {out_path}")