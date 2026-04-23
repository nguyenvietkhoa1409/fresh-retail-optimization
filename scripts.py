import sys
# Đảm bảo đường dẫn này trỏ đúng vào thư mục gốc của project trên Colab
sys.path.insert(0, '/content/fresh-retail-optimization') 

# Khởi tạo và chạy Preprocessor
from src.data_pipeline.preprocessor import FrnPreprocessor
print("Đang tạo lại preprocessed.parquet...")
preprocessor = FrnPreprocessor()
df_clean = preprocessor.run()

# (Tuỳ chọn) Cập nhật lại Catalog nếu cần thiết
from src.data_pipeline.catalog_enricher import CatalogEnricher
print("Đang cập nhật lại Catalog...")
enricher = CatalogEnricher()
enricher.run()

print("Hoàn tất!")
