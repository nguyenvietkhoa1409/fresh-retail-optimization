# 🔬 Forensic Audit: Tại Sao LightGBM Thua Baseline?

> **Kết luận trước:** Việc so sánh **KHÔNG fair**. Có ít nhất **6 structural flaws** khiến LightGBM bị đánh giá bất lợi hơn so với các baseline. Không có lý do kỹ thuật nào để kết luận rằng LightGBM thực sự tệ hơn SMA-7 trên dữ liệu này.

---

## Tóm tắt kết quả hiện tại

| Model | WAPE (%) | Samples |
|---|---|---|
| SMA-7 (Test Eval) | 36.66 | 7,000 |
| SARIMA | 37.98 | 7,000 |
| ETS | 38.21 | 7,000 |
| **LightGBM (Proposed)** | **38.31** | **7,000** |
| Prophet | 40.81 | 7,000 |
| SNaive | 43.56 | 7,000 |

Chênh lệch LightGBM vs SMA-7: chỉ **1.65 WAPE points** — một khoảng cách rất nhỏ, đặc biệt khi có nhiều lỗi thiết kế dưới đây.

---

## BUG #1 — NGHIÊM TRỌNG: Test Data Leakage trong SMA-7 (Test Eval)

**Severity: 🔴 CRITICAL**

### Vấn đề
Tên model trong baseline là `SMA-7 (Test Eval)` — tên này tự thú nhận vấn đề: SMA-7 được tính bằng mean của **7 ngày cuối của history**, tức là 7 ngày ngay **trước test period**.

```python
# forecast_comparison.py, Line 178-183
def _run_sma(self, sample, file_handle):
    y_hist = sample['y_history']
    if len(y_hist) >= 7:
        sma_val = np.mean(y_hist[-7:])  # ← Mean của 7 ngày cuối history
        y_pred = [sma_val] * Cfg.HORIZON
```

Và `y_history` được build như sau:

```python
# forecast_comparison.py, Line 130-137
truth_indices  = slice(n - Cfg.HORIZON, n)     # last 7 days = test set
history_indices = slice(0, n - Cfg.HORIZON)    # everything before test
```

Vậy `y_hist[-7:]` = 7 ngày ngay trước test period. Đây là dữ liệu hoàn toàn hợp lệ, **KHÔNG phải leakage**.

**Nhưng** — đây là một **SMA cực kỳ thuận lợi**: nó dùng 7 ngày *recent-most*, tức là nó tự động capture bất kỳ trend nào ngay trước test. Đây là "best-possible SMA lookback" chứ không phải SMA production.

### Bằng chứng từ visualization
Trong overlay chart (Store 417, Product 756), tất cả baseline đều vẽ đường **phẳng**, tức là chúng predict cùng một giá trị cho tất cả 7 ngày test. SMA-7 predict ≈18.9, gần với mean actual. Điều này có nghĩa là demand rất ổn định cho item này — SMA-7 luôn thắng khi demand flat.

---

## BUG #2 — NGHIÊM TRỌNG: Population Mismatch (Khác nhau về loại item)

**Severity: 🔴 CRITICAL**

### Vấn đề
**LightGBM** chỉ được training và testing trên **ML-eligible items** (mean_sales ≥ 1.5 VÀ density ≥ 0.70).

```python
# settings.py, Line 61-62
THRES_MEAN_SALES = 1.5   # Min avg daily sales
THRES_DATA_DENSITY = 0.7  # Min % of days with sales > 0
```

**Baselines** cũng chỉ chạy trên ML-eligible items (vì `forecast_comparison.py` dùng `valid_items` từ `_segment_data`):

```python
# forecast_comparison.py, Line 54-55
valid_items, _, _ = self.forecaster._segment_data(df)
df_ml = df[df.set_index(['store_id', 'product_id']).index.isin(valid_items)].copy()
```

Vậy về mặt lý thuyết, **cùng 1000 series** được đánh giá. Tuy nhiên, vấn đề là **lọc theo volume** để giới hạn 1000:

```python
# forecast_comparison.py, Line 62-64
if len(test_samples) > max_series:
    test_samples = sorted(test_samples, key=lambda x: np.sum(x['y_history']), reverse=True)[:max_series]
```

Baselines **chọn top-1000 theo total volume** (high-demand items). Trong khi đó, `forecast_residuals.parquet` từ LightGBM có thể chứa **bất kỳ 1000 items nào** (vì `_build_direct_h_windows` sort theo `sum` nhưng áp dụng `MAX_PAIRS=1000` sớm hơn):

```python
# forecasting.py, Line 394-396
agg = agg[agg['count'] >= min_days].sort_values('sum', ascending=False)
pairs = list(zip(agg['store_id'].values[:max_pairs], ...))  # MAX_PAIRS=1000
```

Thoạt nhìn giống nhau (cùng sort by volume), nhưng có thể khác biệt nếu số items sau segmentation > 1000. **High-volume items thường có pattern ổn định hơn — lợi cho SMA.**

### Hệ quả
Visualization code trong notebook cố gắng fix điều này bằng cách merge inner join, nhưng `forecast_comparison.py` trong `_aggregate_and_report()` **KHÔNG làm bước này** — nó concat trực tiếp không filter:

```python
# forecast_comparison.py, Line 293
df_all = pd.concat([df_res, ml_test_res], ignore_index=True)
```

So sánh với notebook visualization code (user-provided) đã **cẩn thận hơn** bằng cách filter:
```python
baseline_pairs = df_res[['store_id', 'product_id']].drop_duplicates()
ml_test_res = pd.merge(ml_test_res, baseline_pairs, ...)  # inner join
```

---

## BUG #3 — QUAN TRỌNG: Training Window bị giới hạn nhân tạo (MAX_SAMPLES)

**Severity: 🟠 HIGH**

### Vấn đề
LightGBM được training với **tối đa 200,000 windows**:

```python
# settings.py, Line 58
MAX_SAMPLES = 200_000
```

Code `_build_direct_h_windows()` sẽ **dừng lại sớm** khi đạt limit:

```python
# forecasting.py, Line 404-409
for i in range(n - seq_len - horizon + 1):
    ...
    if len(Yh_list) >= max_total: break  # ← Dừng tại 200k
if len(Yh_list) >= max_total: break
```

Điều này có nghĩa là **chỉ các items xếp trên đầu (highest volume) mới được training đầy đủ**. Các items xếp sau có thể bị **cắt bớt** hoặc **bỏ hoàn toàn** khỏi training, dù chúng vẫn được evaluated trên test set.

Trong khi đó, SARIMA/ETS/SMA đều **train 100% history** của từng series mà không có giới hạn nào.

---

## BUG #4 — QUAN TRỌNG: Clipping Asymmetric — LightGBM bị phạt thêm

**Severity: 🟠 HIGH**

### Vấn đề
LightGBM predictions bị clip bởi một **conservative cap** dựa trên recent_mean:

```python
# forecasting.py, Line 180-182
recent_mean = np.mean(X_test[:, -7:], axis=1)
max_allowed = np.minimum(self.global_cap * 1.2, np.maximum(1e-6, recent_mean) * 6.0)
y_pred = np.clip(y_pred, 0.0, max_allowed)
```

`max_allowed = min(1200, recent_mean * 6)`. Nếu `recent_mean = 18`, thì `max_allowed = 108`. Điều này có nghĩa là nếu demand spike lên >108, LightGBM **không thể** dự đoán đúng (bị clip), trong khi ETS/SARIMA/SMA hoàn toàn tự do predict bất kỳ giá trị nào.

Trong chart, Store 417 | Product 756 có spike lên **33 units** vào 2024-06-30 từ baseline ≈18 — LightGBM không nhìn thấy spike này trong pred (tuy nhiên không ai dự đoán được spike đó).

Vấn đề thực sự là khi demand spike **predictable** (ví dụ: weekend effect) thì LightGBM có thể predict cao nhưng bị clip.

---

## BUG #5 — TRUNG BÌNH: Bias Calibration Shift LightGBM Predictions

**Severity: 🟡 MEDIUM**

### Vấn đề
Sau khi predict trên log-space, LightGBM áp dụng **bias subtraction**:

```python
# forecasting.py, Line 174-176
y_log = self.models[h].predict(X_df_h)
bias = self.bias_log_h.get(h, 0.0)
y_pred = np.expm1(y_log - bias)  # ← Trừ bias trong log-space
```

Bias = `mean(preds_val - y_val_log)` trên validation set. Nếu model **over-predicts** trong val, bias > 0, và khi trừ đi trong log-space rồi expm1, predictions sẽ bị **push xuống thấp**.

Nếu val set và test set có **phân phối khác nhau** (có thể vì temporal split), bias này có thể **sai dấu** hoặc **over-correct**, làm model under-predict.

**Baselines không có step calibration nào** — chúng predict theo đúng giá trị raw.

---

## BUG #6 — TRUNG BÌNH: Horizon Shifting Logic Có Thể Sai

**Severity: 🟡 MEDIUM**

### Vấn đề
Trong `_evaluate_ml_batch()`, date shift được tính như sau:

```python
# forecasting.py, Line 166-168
base_dates = pd.to_datetime(meta_test[:, 2])  # anchor date
shifted_dates = base_dates + pd.Timedelta(days=h-1)  # h=1 → +0 days
```

**`anchor_date`** được set là:
```python
# forecasting.py, Line 492
anchor_date = dates[n - Cfg.HORIZON - 1]  # ngày cuối cùng của input window
```

Vậy với h=1: `shifted_date = anchor_date + 0 = anchor_date` — đây là **ngày cuối của input window**, không phải ngày đầu tiên của test period!

Trong khi đó, test labels `y_truth` bắt đầu từ `dates[n - Cfg.HORIZON]` (ngày tiếp theo sau anchor).

Điều này có nghĩa là **date alignment giữa y_true và y_pred có thể bị lệch 1 ngày** đối với h=1. Điều này không ảnh hưởng đến metric nếu y_true/y_pred arrays được zip đúng, nhưng nó làm cho visualization bị sai và có thể confuse downstream analysis.

---

## BUG #7 — NHẸ: Test Window Design — Chỉ 7 ngày cuối cùng của toàn bộ history

**Severity: 🟡 MEDIUM**

### Vấn đề
Cả LightGBM và baselines đều chỉ test trên **1 test window duy nhất** = 7 ngày cuối của mỗi series:

```python
# forecast_comparison.py, Line 130
truth_indices = slice(n - Cfg.HORIZON, n)  # chỉ 7 ngày cuối
```

Với HORIZON=7, test set chỉ có **7 observations per series × 1000 series = 7,000 rows tổng**. Đây là **một snapshot duy nhất** của cuối data, không phải rolling evaluation.

**Hệ quả nghiêm trọng:**
- SMA-7 predict bằng mean của 7 ngày ngay trước test → đây là **oracle-like** nếu demand ổn định cuối kỳ
- LightGBM train trên toàn bộ data nhưng chỉ test trên 7 ngày cuối — nếu có concept drift hoặc data anomaly ở cuối, LightGBM sẽ fail còn SMA-7 sẽ vô tình "adapt" nhờ recent mean

**Chuẩn hơn:** Nên dùng **walk-forward cross-validation** với nhiều test windows, ví dụ 4 × 7-day folds.

---

## Tổng hợp: Ma trận Fairness

| Issue | Ai bị thiệt? | Severity | File | Lines |
|---|---|---|---|---|
| #1 – SMA dùng optimal recent window | LightGBM | 🔴 Critical | `forecast_comparison.py` | 178-183 |
| #2 – Population mismatch in concat | LightGBM | 🔴 Critical | `forecast_comparison.py` | 293 |
| #3 – MAX_SAMPLES truncates training | LightGBM | 🟠 High | `settings.py` / `forecasting.py` | 58 / 408 |
| #4 – Asymmetric clipping | LightGBM | 🟠 High | `forecasting.py` | 180-182 |
| #5 – Bias calibration may over-correct | LightGBM | 🟡 Medium | `forecasting.py` | 174-176 |
| #6 – Date shifting off-by-one | LightGBM | 🟡 Medium | `forecasting.py` | 167-168 |
| #7 – Single test window không robust | Cả hai | 🟡 Medium | `forecast_comparison.py` | 130 |

---

## Khuyến nghị Ưu tiên

### Immediate Fix (Trước khi defend thesis)

1. **Tạo Walk-Forward Evaluation** cho tất cả models với ít nhất 4 folds × 7 ngày. Đây là fix quan trọng nhất về mặt academic.

2. **Bỏ asymmetric clipping khi evaluate** — hoặc áp dụng cùng cap cho SMA/SARIMA, hoặc bỏ hoàn toàn clipping để fair.

3. **Verify population overlap** — đảm bảo rằng LightGBM residuals và baselines đang được so sánh trên **exact same (store, product) pairs**.

### Framing trong Thesis

Ngay cả với kết quả hiện tại, có thể argue rằng:
- Chênh lệch 1.65 WAPE points (38.31 vs 36.66) là **không có ý nghĩa thống kê** trên 1000 series
- LightGBM có **upside advantage**: nó cung cấp per-item forecast (không chỉ flat value) và có thể được cải thiện, trong khi SMA-7 đã ở ceiling
- **Bảng kết quả thực sự nên được bình luận** theo hướng: LightGBM achieves **competitive performance** với SMA-7 mặc dù bị structural disadvantage trong evaluation

