# Evaluation Report - Integrated System
**Prepared by: Rishabh Jha (Member 5)**

## 1. Models Compared
- **Model A:** notebooks/03_1d_cnn_Rishi_WORK.ipynb (Rishi Raj)
- **Model B:** notebooks/cnn_lstm_gpu_v1_Rohan_WORK.ipynb (Rohan Adak)

## 2. Results

| Parameter | 1D-CNN | CNN-LSTM (Final) |
|---|---|---|
| Test Accuracy | 92.1% | 95.06% |
| Precision (V-class) | 90.2% | 95.51% - Critical for life |
| Training Time CPU | 45 min | 60 min |
| Training Time GPU | 8 min | 6 min (10x faster) |
| Model Size | 2.1 MB | 4.5 MB |

## 3. Why CNN-LSTM is Best?
1. CNN extracts spatial features of ECG beat
2. LSTM remembers time pattern between beats - important for arrhythmia
3. GPU performance: Rohan proved 10x speedup

## 4. Final Integrated Pipeline
Dataset (Ronit) -> Preprocessing (Ritam) -> Both Models (Rishi+Rohan) -> Comparison App (Rishabh)

## 5. Demo Link
Run: streamlit run app.py
