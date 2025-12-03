# src/utils/common.py
import numpy as np
import pandas as pd
import ast
import re

class DataUtils:
    @staticmethod
    def parse24(x) -> np.ndarray:
        """
        Chuyển chuỗi biểu diễn mảng 24 giờ (từ dataset raw) thành numpy array float32.
        Xử lý cả format list string lẫn string phân cách bởi dấu phẩy/khoảng trắng.
        """
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 24:
            return np.asarray(x, dtype=np.float32)
        if isinstance(x, str):
            s = x.strip()
            # Trường hợp string list chuẩn "[1, 2, ...]"
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = ast.literal_eval(s)
                    if isinstance(arr, (list, tuple)) and len(arr) == 24:
                        return np.asarray(arr, dtype=np.float32)
                except Exception:
                    pass
            # Trường hợp string phân cách lỏng lẻo
            toks = [t for t in re.split(r"[\s,;|]+", s.strip("[]")) if t != ""]
            if len(toks) == 24:
                try:
                    return np.asarray([float(t) for t in toks], dtype=np.float32)
                except Exception:
                    pass
        return np.full(24, np.nan, dtype=np.float32)

    @staticmethod
    def hours_to_16(arr24: np.ndarray, h0: int, h1: int) -> np.ndarray:
        """Cắt mảng 24h xuống khung giờ hoạt động (ví dụ 6h-22h)."""
        return np.asarray(arr24, dtype=np.float32)[h0:h1+1]

    @staticmethod
    def safe_to_float32(s):
        """Chuyển đổi an toàn sang float32, ép lỗi thành NaN."""
        return pd.to_numeric(s, errors="coerce").astype("float32")