import pandas as pd
import folium
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Cập nhật đường dẫn file csv của bạn tại đây nếu có thay đổi
STORES_PATH = 'data/artifacts/part1/stores.csv'
SUPPLIERS_PATH = 'data/artifacts/part1/suppliers.csv'
# Nếu bạn có file riêng cho kho tổng, hãy load nó. 
# Ở đây tôi giả định kho tổng nằm ở vị trí trung tâm hoặc được định nghĩa trong settings.
# Tạm thời ta lấy trung bình cộng tọa độ các cửa hàng làm tâm kho (Center of Gravity)
# hoặc bạn có thể nhập tọa độ cứng vào biến bên dưới.

OUTPUT_FILE = 'supply_chain_map.html'

def load_data():
    """Load dữ liệu và kiểm tra cột cần thiết"""
    if not os.path.exists(STORES_PATH) or not os.path.exists(SUPPLIERS_PATH):
        print(f"Error: Không tìm thấy file dữ liệu tại {STORES_PATH} hoặc {SUPPLIERS_PATH}")
        sys.exit(1)
        
    df_stores = pd.read_csv(STORES_PATH)
    df_suppliers = pd.read_csv(SUPPLIERS_PATH)
    
    # Chuẩn hóa tên cột về dạng chuẩn (Latitude, Longitude) nếu cần
    # Logic này giúp code chạy được ngay cả khi tên cột là 'lat', 'LAT', 'y', v.v.
    def normalize_cols(df):
        col_map = {c: c for c in df.columns}
        for c in df.columns:
            lower_c = c.lower()
            if 'lat' in lower_c: col_map[c] = 'Latitude'
            if 'lon' in lower_c or 'lng' in lower_c: col_map[c] = 'Longitude'
            if 'id' in lower_c and 'store' in lower_c: col_map[c] = 'ID'
            if 'id' in lower_c and 'supplier' in lower_c: col_map[c] = 'ID'
            if 'archetype' in lower_c or 'type' in lower_c: col_map[c] = 'Type'
        return df.rename(columns=col_map)

    df_stores = normalize_cols(df_stores)
    df_suppliers = normalize_cols(df_suppliers)
    
    return df_stores, df_suppliers

def create_map(df_stores, df_suppliers):
    """Tạo bản đồ Folium"""
    
    # 1. Tính toán tâm bản đồ (Center Warehouse location assumption)
    center_lat = df_stores['Latitude'].mean()
    center_lon = df_stores['Longitude'].mean()
    
    print(f"Center Warehouse (Estimated): Lat {center_lat:.4f}, Lon {center_lon:.4f}")
    
    # Khởi tạo bản đồ nền (OpenStreetMap)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB positron')

    # 2. Vẽ điểm Kho Tổng (Warehouse) - Màu Đen
    folium.Marker(
        location=[center_lat, center_lon],
        popup=folium.Popup('<b>Center Warehouse</b><br>Hub trung chuyển', max_width=300),
        icon=folium.Icon(color='black', icon='home', prefix='fa')
    ).add_to(m)

    # 3. Vẽ các Cửa hàng (Stores) - Màu Xanh Dương
    # Sử dụng FeatureGroup để có thể bật/tắt layer trên bản đồ
    store_group = folium.FeatureGroup(name="Retail Stores")
    
    for _, row in df_stores.iterrows():
        # Lấy thông tin hiển thị popup
        store_id = row.get('ID', 'N/A')
        store_type = row.get('Type', 'Store')
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"<b>Store:</b> {store_id}<br>Type: {store_type}",
            color='#3186cc',      # Viền xanh
            fill=True,
            fill_color='#3186cc', # Nền xanh
            fill_opacity=0.7
        ).add_to(store_group)
    
    store_group.add_to(m)

    # 4. Vẽ các Nhà cung cấp (Suppliers) - Màu Đỏ/Cam
    supplier_group = folium.FeatureGroup(name="Suppliers/Farms")
    
    for _, row in df_suppliers.iterrows():
        sup_id = row.get('ID', 'N/A')
        sup_type = row.get('Type', 'Supplier')
        
        # Chọn màu dựa trên loại Supplier (nếu có)
        color = '#e31a1c' # Mặc định đỏ
        if 'farm' in str(sup_type).lower():
            color = '#33a02c' # Xanh lá cho Farm
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"<b>Supplier:</b> {sup_id}<br>Type: {sup_type}",
            icon=folium.Icon(color='red', icon='leaf', prefix='fa')
        ).add_to(supplier_group)
        
    supplier_group.add_to(m)

    # 5. Thêm công cụ điều khiển layer
    folium.LayerControl().add_to(m)
    
    # 6. Thêm chú thích (Legend)
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 160px; height: 130px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.9;">
     &nbsp;<b>Supply Chain Legend</b> <br>
     &nbsp;<i class="fa fa-home" style="color:black"></i>&nbsp; Warehouse (Center) <br>
     &nbsp;<i class="fa fa-circle" style="color:#3186cc"></i>&nbsp; Retail Stores <br>
     &nbsp;<i class="fa fa-map-marker" style="color:red"></i>&nbsp; Suppliers <br>
     &nbsp;<i class="fa fa-leaf" style="color:green"></i>&nbsp; Farms <br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

if __name__ == "__main__":
    print("--- Starting Visualization ---")
    try:
        stores, suppliers = load_data()
        my_map = create_map(stores, suppliers)
        my_map.save(OUTPUT_FILE)
        print(f"Success! Map saved to: {os.path.abspath(OUTPUT_FILE)}")
        print("Open this file in your web browser to view the interactive map.")
    except Exception as e:
        print(f"An error occurred: {e}")