# src/data_pipeline/catalog_enricher.py
import os
import numpy as np
import pandas as pd
from config.settings import ProjectConfig as Cfg

class CatalogEnricher:
    """
    Module: Master Data Enrichment (Risk-Aware Version)
    
    CẢI TIẾN:
    Thay vì gán category ngẫu nhiên, ta phân tích độ biến động (CV) của từng sản phẩm.
    - Sản phẩm Rủi ro cao (High CV / Low Volume): Gán vào nhóm High Margin (Seafood/Meat) 
      để đảm bảo Cu >> Co, tránh việc Inventory Model trả về 0.
    - Sản phẩm Ổn định: Gán vào nhóm Low/Medium Margin.
    """

    def __init__(self):
        os.makedirs(Cfg.ARTIFACTS_DIR, exist_ok=True)
        # Profile cấu hình: (min_margin, max_margin)
        self.cat_profiles = {
            'Stable':  {'name': 'Vegetables',   'shelf_life': 3, 'margin_range': (0.30, 0.45), 'waste_adder': 0.5},
            'Normal':  {'name': 'Fruits',       'shelf_life': 5, 'margin_range': (0.40, 0.55), 'waste_adder': 0.5},
            'Risky':   {'name': 'Meat',         'shelf_life': 2, 'margin_range': (0.60, 0.75), 'waste_adder': 0.8},
            'HighRisk':{'name': 'Seafood',      'shelf_life': 1, 'margin_range': (0.75, 0.90), 'waste_adder': 1.5} 
            # Note: Margin HighRisk rất cao (75-90%) để bù đắp rủi ro hủy hàng
        }

    def run(self):
        print("\n[CatalogEnricher] Starting Risk-Aware Master Data Generation...")

        # 1. Load Preprocessed Data (để tính toán rủi ro lịch sử)
        prep_path = os.path.join(Cfg.ARTIFACTS_DIR, "preprocessed.parquet")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Not found {prep_path}. Run Preprocessor first.")
        
        # Chỉ load cột cần thiết để tiết kiệm RAM
        df = pd.read_parquet(prep_path, columns=['product_id', 'sale_amount'])
        
        # 2. Analyze Risk Profile (Tính Mean, Std, CV cho từng SKU)
        print("  -> Analyzing Historical Risk Profiles (CV calculation)...")
        stats = df.groupby('product_id', observed=True)['sale_amount'].agg(['mean', 'std']).reset_index()
        stats['cv'] = stats['std'] / (stats['mean'] + 1e-6) # Tránh chia cho 0
        
        # Phân loại rủi ro
        # Logic: CV càng cao hoặc Mean càng thấp -> Rủi ro càng cao
        def classify_risk(row):
            if row['mean'] < 2.0 or row['cv'] > 0.8: return 'HighRisk'
            if row['cv'] > 0.5: return 'Risky'
            if row['cv'] > 0.3: return 'Normal'
            return 'Stable'

        stats['risk_group'] = stats.apply(classify_risk, axis=1)
        print(f"  -> Risk Distribution:\n{stats['risk_group'].value_counts()}")

        # 3. Generate Economics based on Risk Group
        catalog_rows = []
        np.random.seed(Cfg.SEED)
        
        # Map product_id -> risk_group
        risk_map = dict(zip(stats['product_id'], stats['risk_group']))
        unique_products = stats['product_id'].values

        for pid in unique_products:
            risk_group = risk_map.get(pid, 'Normal')
            profile = self.cat_profiles[risk_group]
            
            # Gán Category ID giả lập dựa trên Risk Group
            cat_map = {'Stable': 1, 'Normal': 2, 'Risky': 3, 'HighRisk': 4}
            cat_id = cat_map[risk_group]

            # Sinh Giá bán (Price)
            # Hàng HighRisk thường đắt tiền hơn (VD: Hải sản vs Rau)
            base_price = {'Stable': 15.0, 'Normal': 30.0, 'Risky': 80.0, 'HighRisk': 150.0}
            price_mean = base_price[risk_group]
            price = max(5.0, np.random.normal(price_mean, price_mean * 0.1))
            
            # Sinh Margin % (Quan trọng nhất)
            margin_pct = np.random.uniform(*profile['margin_range'])
            
            # Tính Cost & Profit
            # Cost = Price * (1 - Margin)
            cost = price * (1 - margin_pct)
            
            # Params Newsvendor
            Cu = price - cost # Profit
            
            # Co = Cost + Disposal. 
            # Với hàng rủi ro cao, Disposal Cost cũng thường cao (xử lý môi trường)
            Co = cost + profile['waste_adder']
            
            # Target Service Level (SL*)
            target_sl = Cu / (Cu + Co)
            
            catalog_rows.append({
                'product_id': str(pid),
                'risk_group': risk_group, # Lưu lại để debug
                'category_id': cat_id,
                'category_name': profile['name'],
                'price': round(price, 2),
                'cost': round(cost, 2),
                'holding_cost': round(cost * 0.05, 3),
                'shelf_life': profile['shelf_life'],
                'Cu': round(Cu, 2),
                'Co': round(Co, 2),
                'target_sl_theoretical': round(target_sl, 4),
                'historical_cv': round(stats.loc[stats['product_id']==pid, 'cv'].values[0], 2)
            })

        # 4. Save Catalog
        df_catalog = pd.DataFrame(catalog_rows)
        out_path = os.path.join(Cfg.ARTIFACTS_DIR, "master_product_catalog.parquet")
        df_catalog.to_parquet(out_path)
        
        # Export CSV sample để bạn kiểm tra
        sample_path = os.path.join(Cfg.ARTIFACTS_DIR, "master_product_catalog_check.csv")
        df_catalog.sort_values('historical_cv', ascending=False).head(50).to_csv(sample_path, index=False)
        
        print(f" -> Catalog saved to {out_path}")
        print(" -> Logic Verification (High Risk should have High SL):")
        print(df_catalog.groupby('risk_group')[['historical_cv', 'Cu', 'Co', 'target_sl_theoretical']].mean())
        
        return df_catalog