# Signal Preprocessing & Segmentation (Member 2)

## 1. Preprocessing Configuration
- **Lead Used:** Channel 0 (MLII - Modified Limb Lead II)
- **Segment Length:** 360 samples (1.0-second window centered at annotated R-peak: 180 samples before, 180 samples after)
- **Bandpass Filter:** Butterworth Bandpass Filter (0.5 Hz to 45.0 Hz, 4th order) to eliminate baseline wander and high-frequency noise
- **Normalization:** Per-segment Z-score normalization: `(segment - mean) / std`

---

## 2. Records & Patient-Wise Split
To prevent data leakage, segmentation is strictly split by patient/record (not random split).
- **Excluded Paced Records:** 102, 104, 107, 217
- **Total Processed Records:** 44
- **Train Set (30 records):** 100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 205, 208, 209, 210, 212, 213, 215
- **Validation Set (7 records):** 203, 214, 219, 220, 221, 222, 228
- **Test Set (7 records):** 207, 223, 230, 231, 232, 233, 234

---

## 3. Label Mapping & Beat Counts
- **Normal (Class 0):** Symbols `['N', 'L', 'R', 'e', 'j']`
- **Abnormal (Class 1):** All other beat types (`['A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']`)
- **Non-beat annotations:** Dropped (e.g., `+`, `~`, `|`)

### Dataset Distribution:
- **Train Set:** 68,386 beats (Normal: 62,790 | Abnormal: 5,596)
- **Validation Set:** 16,399 beats (Normal: 14,553 | Abnormal: 1,846)
- **Test Set:** 15,897 beats (Normal: 12,733 | Abnormal: 3,164)

---

## 4. Preprocessed File (.npz) & Google Drive Link
The generated dataset file is **`ecg_processed_v1.npz`** (~134 MB). Due to GitHub's file size limit, it is hosted on Google Drive:
- **Drive Link:** [ecg_processed_v1.npz on Google Drive](https://drive.google.com/file/d/1TO043QL8KSBiMkkf4Di5MMbLF5wXGgT0/view)

### Array Shapes & Data Types:
- `X_train`: `(68386, 360, 1)` — float32
- `y_train`: `(68386,)` — int32
- `X_val`: `(16399, 360, 1)` — float32
- `y_val`: `(16399,)` — int32
- `X_test`: `(15897, 360, 1)` — float32
- `y_test`: `(15897,)` — int32

---

## 5. How to Load Data (For Member 3 & Member 4)
```python
import numpy as np

# Load preprocessed arrays
data = np.load('ecg_processed_v1.npz')

X_train, y_train = data['X_train'], data['y_train']
X_val, y_val     = data['X_val'], data['y_val']
X_test, y_test   = data['X_test'], data['y_test']

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)
