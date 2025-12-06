# End-to-End Fresh Retail Supply Chain Optimization
*(Hệ thống Tối ưu hóa Chuỗi cung ứng Bán lẻ Thực phẩm Tươi sống Khép kín)*

> **Capstone Project** | **Data Science & Operations Research**

## 1. Giới thiệu (Overview)

Dự án này xây dựng một **Hệ thống Hỗ trợ Ra quyết định (Decision Support System - DSS)** tự động hóa dành cho chuỗi bán lẻ thực phẩm tươi sống (Fresh Retail). Hệ thống giải quyết bài toán cốt lõi: cân bằng giữa **Chi phí Vận hành (TCO)** và **Chất lượng Sản phẩm (Freshness)** thông qua một quy trình xử lý dữ liệu khép kín từ khôi phục nhu cầu ẩn đến tối ưu hóa logistics.

Hệ thống áp dụng phương pháp luận **Adaptive Sequential Decision-Making**, tích hợp các mô hình Học máy (Machine Learning) và Vận trù học (Operations Research) để đưa ra các quyết định đặt hàng và vận chuyển tối ưu.

## 2. Tính năng Cốt lõi (Key Features)

Hệ thống bao gồm 4 module xử lý chính:

* **📈 Module A: Khôi phục Nhu cầu Ẩn (Latent Demand Reconstruction)**
    * Sử dụng thuật toán *Non-parametric Hierarchical Shrinkage*.
    * Tự động phát hiện và khôi phục nhu cầu trong những ngày hết hàng (Stockout), loại bỏ thiên kiến dữ liệu.

* **🔮 Module B: Dự báo Đa kỳ hạn (Multi-Horizon Forecasting)**
    * Sử dụng *LightGBM* với chiến lược *Direct Multi-Horizon*.
    * Dự báo nhu cầu chính xác cho 7 ngày tới ($t+1 \dots t+7$), tích hợp các yếu tố mùa vụ, khuyến mãi.

* **📦 Module C: Hoạch định Tồn kho (Inventory Planning)**
    * Mô hình *Smart Newsvendor* với mức độ phục vụ động (Risk-based Service Level).
    * Tính toán Tồn kho an toàn (Safety Stock) và Điểm đặt hàng lại (ROP) dựa trên rủi ro và hạn sử dụng (Shelf-life).

* **🚚 Module D: Tối ưu hóa Tích hợp (Integrated Procurement & VRP)**
    * **Procurement:** Sử dụng quy hoạch tuyến tính nguyên (MILP) để chọn nhà cung cấp tối ưu chi phí và thời gian ($P$).
    * **Logistics:** Sử dụng Constraint Programming (Google OR-Tools) để giải bài toán định tuyến xe (CVRPTW) với mô hình Cross-docking.
    * **Simulation:** Mô phỏng các kịch bản chiến lược ($P, U$) để tìm ra điểm cân bằng tối ưu.

## 3. Cấu trúc Dự án (Project Structure)

```text
fresh-retail-optimization/
├── config/
│   └── settings.py             # Cấu hình toàn cục (Hyperparameters, Constants)
├── data/
│   ├── artifacts/              # Thư mục chứa dữ liệu đầu ra (Parquet, CSV, Images)
│   └── ...                     # Dữ liệu đầu vào (FreshRetailNet-50K)
├── src/
│   ├── data_pipeline/
│   │   ├── generator.py        # Sinh dữ liệu giả lập (Stores, Suppliers Locations)
│   │   └── preprocessor.py     # Làm sạch dữ liệu & Gán nhãn Stockout (s16)
│   ├── demand/
│   │   ├── reconstruction.py   # Thuật toán Khôi phục nhu cầu (Hierarchical Shrinkage)
│   │   └── forecasting.py      # Mô hình dự báo LightGBM
│   ├── inventory/
│   │   └── planner.py          # Tính toán Safety Stock & ROP
│   ├── optimization/
│   │   ├── procurement.py      # Tối ưu hóa mua hàng (MILP - PuLP)
│   │   ├── logistics.py        # Tối ưu hóa vận tải (VRP - OR-Tools)
│   │   ├── integrated_solver.py # Vòng lặp mô phỏng chiến lược (Core Logic)
│   │   └── cost_evaluator.py   # Tính toán TCO & Freshness Penalty
│   ├── analysis/
│   │   ├── sensitivity.py      # Kiểm định độ nhạy (Sensitivity Test)
│   │   └── reporter.py         # Sinh báo cáo & Biểu đồ
│   └── utils/                  # Các hàm tiện ích (Geo, Common)
├── 1.main_data.py              # Script chạy bước Data Pipeline
├── 2.main_demand.py            # Script chạy bước Reconstruction
├── 3.main_forecasting.py       # Script chạy bước Forecasting
├──
