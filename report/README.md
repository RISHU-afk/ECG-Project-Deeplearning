# Report (Group 7 - ECG Deep Learning Project)

**Title:** GPU-Accelerated Deep Learning-Based ECG Signal Classification for Cardiac Abnormality Detection

This folder holds the final project report. This README is the report outline: what each section must contain, which files to take the numbers and figures from, and who writes it. Every number below is copied from `results/*.csv` and `results/*.json`. Nothing is estimated. If an experiment is run again, update the numbers here from the new CSV files.

Model output is a classification result (Normal / Abnormal), not a clinical diagnosis.

## Status

| Item | Status |
|---|---|
| Report outline with final numbers (this file) | Done |
| Report document (report_v1.docx / .pdf) | Done |
| Presentation (presentation/ folder) | To do |

## Report structure

| # | Section | What to write | Source files | Main member |
|---|---|---|---|---|
| 1 | Introduction | Problem, why deep learning for ECG, what the project demonstrates | Project brief PDF | Member 5 |
| 2 | Dataset | MIT-BIH source, 360 Hz, 44 of 48 records used, labels, class imbalance | dataset/README.md, graphs/class_distribution.png, ecg_normal.png, ecg_abnormal_V.png, ecg_abnormal_A.png, results/dataset_stats_per_record.csv | Member 1 |
| 3 | Preprocessing | Bandpass filter, 360-sample segments, z-score, record-wise split, leakage | preprocessing/README.md, preprocessing/02_preprocessing_v1.py | Member 2 |
| 4 | 1D-CNN | Architecture, settings, training curves, test metrics | notebooks/03_..., results/1d_cnn_*.csv, graphs/1d_cnn_*.png | Member 3 |
| 5 | CNN-LSTM | Architecture, settings, training curves, test metrics | notebooks/04_..., results/cnn_lstm_*.csv, graphs/cnn_lstm_*.png | Member 4 |
| 6 | CPU vs GPU | Experiment design, hardware, timing, speedup | results/cnn_lstm_cpu_gpu_timing_v1.csv, hardware_info_v1.json, graphs/cnn_lstm_cpu_vs_gpu_v1.png | Member 4 |
| 7 | Model comparison and discussion | 1D-CNN vs CNN-LSTM, weaknesses, limitations | graphs/final_model_comparison.png | Member 5 |
| 8 | Demo | Input ECG -> preprocessing -> models -> Normal / Abnormal | demo/README.md, demo/app.py | Member 5 |
| 9 | Conclusion and future work | Short summary based only on measured values | - | All |
| 10 | References | PhysioNet / MIT-BIH, WFDB, TensorFlow/Keras | - | All |

## 2. Dataset - key facts

| Item | Value |
|---|---|
| Source | MIT-BIH Arrhythmia Database (PhysioNet) |
| Sampling rate | 360 Hz |
| Records downloaded / used | 48 / 44 (paced records 102, 104, 107, 217 excluded) |
| Lead used | MLII (channel 0) |
| Total beats | 100,733 |
| Normal / Abnormal | 90,125 / 10,608 (10.5% abnormal) |

Labels: **Normal (0)** = N, L, R, e, j. **Abnormal (1)** = A, a, J, S, V, E, F, /, f, Q.

Point to explain: L and R (bundle branch block) beats are labelled Normal by the team's definition.

## 3. Preprocessing - key facts

- Bandpass filter: Butterworth, 0.5 to 45 Hz, order 4 (applied to the whole record first)
- Segment: 360 samples centred on each R peak (180 before, 180 after)
- Normalization: per-segment z-score
- 100,733 annotated beats become 100,682 segments (beats too close to the record edges are dropped)
- Split is record-wise, so no record appears in two splits

| Split | Records | Beats | Normal | Abnormal | Abnormal % |
|---|---|---|---|---|---|
| Train | 30 | 68,386 | 62,790 | 5,596 | 8.2% |
| Validation | 7 | 16,399 | 14,553 | 1,846 | 11.3% |
| Test | 7 | 15,897 | 12,733 | 3,164 | 19.9% |

## 4 and 5. Models

**1D-CNN:** ECG (360 x 1) -> Conv1D(32, k=5) + MaxPool -> Conv1D(64, k=5) + MaxPool -> Flatten -> Dense(64) -> Dropout(0.5) -> Sigmoid. 379,265 parameters.

**CNN-LSTM:** ECG (360 x 1) -> 3 x (Conv1D + MaxPool) -> LSTM(64) -> Dense(32) -> Dropout(0.3) -> Sigmoid.

| Setting | 1D-CNN | CNN-LSTM |
|---|---|---|
| Data | ecg_processed_v1.npz | ecg_processed_v1.npz |
| Seed / batch size | 42 / 128 | 42 / 128 |
| Optimizer | Adam, 0.001 | Adam, 0.001 |
| Loss | binary cross-entropy | binary cross-entropy |
| Class weights | 0.5446 / 6.1103 | 0.5446 / 6.1103 |
| Max epochs | 30 | 15 |
| Early stopping (val_loss) | patience 6 | patience 4 |
| ReduceLROnPlateau | yes | no |
| Epochs run / best epoch | 7 / 1 | 13 / 9 |
| Threshold | 0.5 | 0.5 |

Note: max epochs, patience and ReduceLROnPlateau differ between the two models. Say this in the report.

## 7. Final model comparison (test set, 15,897 beats from 7 unseen records)

| Metric | 1D-CNN | CNN-LSTM |
|---|---|---|
| Accuracy | 0.8066 | 0.7605 |
| Precision (abnormal) | 0.5139 | 0.4138 |
| Recall (abnormal) | 0.5193 | 0.4880 |
| F1-score (abnormal) | 0.5166 | 0.4479 |
| Specificity | 0.8780 | 0.8282 |
| ROC-AUC | 0.8448 | 0.7418 |
| Confusion matrix (TN / FP / FN / TP) | 11179 / 1554 / 1521 / 1643 | 10546 / 2187 / 1620 / 1544 |
| Training time | 462.6 s (7 epochs, no GPU visible) | 112.11 s (13 epochs, Tesla T4 GPU) |

**Do not compare the two training times directly.** The runs used different hardware, different epoch limits and different early stopping.

## 6. CPU vs GPU (CNN-LSTM, same model, data, seed, batch size, 5 epochs, no early stopping)

| Metric | CPU | GPU | Speedup (CPU / GPU) |
|---|---|---|---|
| Total training time (5 epochs) | 297.20 s | 33.94 s | 8.76x |
| Average epoch time (epoch 2 onward) | 60.64 s | 6.22 s | 9.75x |
| First epoch (includes warm-up) | 54.47 s | 8.80 s | - |
| Inference, full test set | 3.563 s | 0.479 s | 7.43x |
| Inference per beat | 0.2241 ms | 0.0302 ms | 7.42x |
| Test accuracy after 5 epochs | 0.7632 | 0.7671 | - |

**Runtime (recorded 2026-10-04):** Google Colab, Tesla T4 GPU (15360 MiB), NVIDIA driver 580.82.07, CUDA 12.5.1, cuDNN 9, Intel Xeon CPU @ 2.00GHz (2 logical cores), about 12.7 GiB RAM, Python 3.13.15, TensorFlow 2.20.0, Keras 3.13.2.

The CPU run uses `tf.device('/CPU:0')` inside the same Colab GPU runtime. Epoch time includes validation time. Each timing experiment was run once (TIMING_REPEATS = 1).

## Discussion points (based only on the measured values)

- The 1D-CNN is better than the CNN-LSTM on every metric in the comparison table. The extra LSTM layer did not help on this dataset.
- Both models are weak on the abnormal class (F1 about 0.45 to 0.52).
- Both models overfit the training records. 1D-CNN: best validation loss at epoch 1, train accuracy about 98.8% later. CNN-LSTM: train accuracy about 98%, unstable validation loss.
- Validation recall (CNN-LSTM 0.85) is much higher than test recall (0.49). The test set has 19.9% abnormal beats against 8.2% in train, and the test records are unseen patients.
- Three test records (232, 233, 223) hold 2,791 of the 3,164 abnormal test beats, so the test result depends heavily on a few records.
- GPU gives about 7.4x to 9.8x speedup for the CNN-LSTM, depending on what is measured (inference, total training, steady epoch).
- Single runs only (no repeats), so small differences between models or devices should not be over-interpreted.

## Figures to include

| Figure | File |
|---|---|
| Class distribution | graphs/class_distribution.png |
| Normal ECG example | graphs/ecg_normal.png |
| Abnormal ECG examples (V, A) | graphs/ecg_abnormal_V.png, graphs/ecg_abnormal_A.png |
| 1D-CNN accuracy / loss | graphs/1d_cnn_accuracy.png, graphs/1d_cnn_loss.png |
| 1D-CNN confusion matrix | graphs/1d_cnn_confusion_matrix.png |
| CNN-LSTM training curves | graphs/cnn_lstm_training_curves_v1.png |
| CNN-LSTM confusion matrix | graphs/cnn_lstm_confusion_matrix_v1.png |
| CPU vs GPU timing | graphs/cnn_lstm_cpu_vs_gpu_v1.png |
| Final model comparison | graphs/final_model_comparison.png |

## Rules for the report

- Every number must come from a real experiment (results/*.csv, results/*.json).
- Describe the output as a prediction/classification result, not a clinical diagnosis.
- Every member's contribution must be represented.
- Save versions as report_v1, report_v2 and keep one approved final version.
