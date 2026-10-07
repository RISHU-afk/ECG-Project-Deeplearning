ECG Deep Learning Project

GPU-Accelerated Deep Learning-Based ECG Signal Classification for Cardiac Abnormality Detection

- **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
- **Models:** 1D-CNN and CNN-LSTM
- **Goal:** Classify ECG beats as normal or abnormal, and compare CPU vs GPU training performance
- **Environment:** Google Colab (Python, TensorFlow/Keras)

## Quick start (run the demo)

    pip install -r requirements.txt
    streamlit run demo/app.py

Before running: download ecg_processed_v1.npz from Google Drive (link in the "Processed data" section below) and put it in the project root or in a data/ folder. Full demo guide: demo/README.md

## Repository structure

    ECG-Project-Deeplearning-main/
    |-- README.md                  this file
    |-- requirements.txt           packages for the demo (TensorFlow 2.20.0, Keras 3.13.2)
    |-- dataset/                   dataset README (Member 1)
    |-- preprocessing/             preprocessing code + README (Member 2)
    |-- notebooks/                 Colab notebooks 01 to 04
    |-- models/                    1d_cnn_model.keras, cnn_lstm_v1.keras
    |-- results/                   metrics, timing, settings, hardware CSV/JSON
    |-- graphs/                    all plots
    |-- demo/                      app.py (Streamlit demo) + README (Member 5)
    |-- report/                    report README / outline

## Team

| Member | Role | Name |
|---|---|---|
| Member 1 | Dataset + ECG Analysis | Ronit Jana |
| Member 2 | Data Preprocessing | Ritam Jana |
| Member 3 | 1D-CNN Model | Rishi Raj |
| Member 4 | CNN-LSTM + GPU Performance | Rohan Adak |
| Member 5 | Integration + Evaluation + Demo | Rishabh Jha |

## Progress

| Stage | Member | Status |
|---|---|---|
| Dataset + ECG analysis | Member 1 | Done |
| Preprocessing + split | Member 2 | Done |
| 1D-CNN | Member 3 | Done |
| CNN-LSTM + CPU/GPU | Member 4 | Done |
| Integration + evaluation + demo | Member 5 | Demo app and results comparison done; final report in progress |

Update your row when your part is finished.

## Dataset (Member 1)

The raw dataset is **not** stored in this repo. Download it in Colab:

    !pip -q install wfdb
    import wfdb
    wfdb.dl_database('mitdb', dl_dir='/content/Group_7_ECG_Project/dataset/mitdb')

| Item | Value |
|---|---|
| Source | MIT-BIH Arrhythmia Database (PhysioNet) |
| Records downloaded | 48 |
| Records used | 44 (paced records 102, 104, 107, 217 excluded) |
| Sampling rate | 360 Hz |
| Leads | MLII (channel 0), V5 (channel 1) |
| Total beats | 100,733 |
| Normal beats | 90,125 |
| Abnormal beats | 10,608 |
| Abnormal % | 10.5% |

The dataset is imbalanced (normal beats are much more than abnormal). Use a record-wise split to avoid data leakage. Details: dataset/README.md

## Label definition (agreed by the team)

- **Normal (0):** N, L, R, e, j
- **Abnormal (1):** A, a, J, S, V, E, F, /, f, Q
- Paced records 102, 104, 107, 217 are excluded

## Preprocessing and split (Member 2)

- **Lead:** channel 0 (MLII)
- **Segment:** 360 samples, centered on each R peak (180 before, 180 after)
- **Filter:** Butterworth bandpass, 0.5 to 45 Hz, order 4
- **Normalization:** per-segment z-score
- **Split:** record-wise (no record appears in two splits)
- Beats too close to the start or end of a record (window does not fit) are dropped, so 100,733 annotated beats become 100,682 segments.

| Split | Records | Beats | Normal | Abnormal |
|---|---|---|---|---|
| Train | 30 | 68,386 | 62,790 | 5,596 |
| Validation | 7 | 16,399 | 14,553 | 1,846 |
| Test | 7 | 15,897 | 12,733 | 3,164 |

Record lists and full details: preprocessing/README.md

## Processed data (for Member 3 and Member 4)

File: **ecg_processed_v1.npz** (about 134 MB, hosted on Google Drive because it is too big for GitHub)

Drive link: https://drive.google.com/file/d/1TO043QL8KSBiMkkf4Di5MMbLF5wXGgT0/view

| Array | Shape | Type |
|---|---|---|
| X_train | (68386, 360, 1) | float32 |
| y_train | (68386,) | int32 |
| X_val | (16399, 360, 1) | float32 |
| y_val | (16399,) | int32 |
| X_test | (15897, 360, 1) | float32 |
| y_test | (15897,) | int32 |

Load in Colab:

    !pip -q install gdown
    !gdown 1TO043QL8KSBiMkkf4Di5MMbLF5wXGgT0 -O ecg_processed_v1.npz

    import numpy as np
    d = np.load('ecg_processed_v1.npz')
    X_train, y_train = d['X_train'], d['y_train']
    X_val, y_val = d['X_val'], d['y_val']
    X_test, y_test = d['X_test'], d['y_test']

Train has far fewer abnormal beats (8.2%) than test (19.9%), so use class weights and judge models with recall, F1 and the confusion matrix, not accuracy alone.

## CNN-LSTM and CPU/GPU results (Member 4)

**Model:** ECG (360 x 1) -> 3 x (Conv1D + MaxPool) -> LSTM(64) -> Dense(32) -> Dropout(0.3) -> Sigmoid (Normal / Abnormal)

**Files:** notebooks/04_cnn_lstm_gpu_v1_Rohan_WORK.ipynb, models/cnn_lstm_v1.keras, results/cnn_lstm_*.csv, graphs/cnn_lstm_*.png

**Experiment settings** (Member 3 should use the same settings so the comparison is fair):

| Setting | Value |
|---|---|
| Data | ecg_processed_v1.npz (same file as Member 3) |
| Seed | 42 |
| Batch size | 128 |
| Optimizer / learning rate | Adam, 0.001 |
| Loss | binary cross-entropy |
| Class weights | Normal 0.5446, Abnormal 6.1103 (balanced, from train set) |
| Epochs | max 15, early stopping on val_loss (patience 4, best weights restored) |
| Threshold | 0.5 (probability >= 0.5 means abnormal) |
| Precision | float32 (no mixed precision) |

**Test set results (Experiment A, trained on GPU):**

| Metric | CNN-LSTM |
|---|---|
| Accuracy | 0.7605 |
| Precision (abnormal) | 0.4138 |
| Recall (abnormal) | 0.4880 |
| F1-score (abnormal) | 0.4479 |
| Specificity | 0.8282 |
| ROC-AUC | 0.7418 |
| Confusion matrix (TN / FP / FN / TP) | 10546 / 2187 / 1620 / 1544 |
| Epochs run / best epoch | 13 / 9 |
| Training time (GPU, 13 epochs) | 112.11 s (about 8.59 s per epoch) |

**CPU vs GPU (Experiment B, same model, data, seed, batch size, 5 epochs, no early stopping):**

| Metric | CPU | GPU | Speedup (CPU / GPU) |
|---|---|---|---|
| Total training time (5 epochs) | 297.20 s | 33.94 s | 8.76x |
| Average epoch time (epoch 2 onward) | 60.64 s | 6.22 s | 9.75x |
| First epoch (includes warm-up) | 54.47 s | 8.80 s | - |
| Inference, full test set (15,897 beats) | 3.563 s | 0.479 s | 7.43x |
| Inference per beat | 0.2241 ms | 0.0302 ms | 7.42x |
| Test accuracy after 5 epochs | 0.7632 | 0.7671 | - |

**Runtime used (recorded 2026-10-04):** Google Colab, Tesla T4 GPU (15360 MiB), NVIDIA driver 580.82.07, CUDA 12.5.1, cuDNN 9, Intel Xeon CPU @ 2.00GHz (2 logical cores), about 12.7 GiB RAM, Python 3.13.15, TensorFlow 2.20.0, Keras 3.13.2. Full details: results/hardware_info_v1.json

**Notes and limitations:**

- Validation recall is high (0.85) but test recall is much lower (0.49). Validation loss is unstable and train accuracy is about 98%, so the model fits the training records much better than new records. Records are split record-wise, so test records are unseen patients.
- Each timing experiment was run once (TIMING_REPEATS = 1). Colab hardware can be different in another session, so timings should be compared only with runs on the same hardware.
- The CPU run uses tf.device('/CPU:0') inside the same Colab GPU runtime, with the same settings as the GPU run. Epoch time includes validation time.
- The CPU and GPU accuracy values differ slightly (0.7632 vs 0.7671) because they are separate 5-epoch runs with different compute kernels. They are not the same as the main model in the test results table.

## 1D-CNN results (Member 3)

**Model:** ECG (360 x 1) -> Conv1D(32, k=5) + MaxPool -> Conv1D(64, k=5) + MaxPool -> Flatten -> Dense(64) -> Dropout(0.5) -> Sigmoid (Normal / Abnormal). Total 379,265 parameters.

**Files:** notebooks/03_1d_cnn_Rishi_WORK.ipynb, models/1d_cnn_model.keras, results/1d_cnn_metrics.csv, results/1d_cnn_classification_report.csv, graphs/1d_cnn_accuracy.png, 1d_cnn_loss.png, 1d_cnn_confusion_matrix.png

**Settings:**

| Setting | Value |
|---|---|
| Data | ecg_processed_v1.npz (same file as Member 4, shapes checked in the notebook) |
| Seed | 42 |
| Batch size | 128 |
| Optimizer / learning rate | Adam, 0.001 (ReduceLROnPlateau: factor 0.5, patience 2, min 1e-5) |
| Loss | binary cross-entropy |
| Class weights | Normal 0.5446, Abnormal 6.1103 (balanced, from train set) |
| Epochs | max 30, early stopping on val_loss (patience 6, best weights restored) |
| Threshold | 0.5 (probability >= 0.5 means abnormal) |
| Precision | float32 |

Training used only X_train / X_val. The test set was used once, for the final evaluation of the best checkpoint.

**Test set results:**

| Metric | 1D-CNN |
|---|---|
| Accuracy | 0.8066 |
| Precision (abnormal) | 0.5139 |
| Recall (abnormal) | 0.5193 |
| F1-score (abnormal) | 0.5166 |
| Macro F1 | 0.6978 |
| Specificity | 0.8780 |
| ROC-AUC | 0.8448 |
| Test loss | 0.5674 |
| Confusion matrix (TN / FP / FN / TP) | 11179 / 1554 / 1521 / 1643 |
| Epochs run / best epoch | 7 / 1 |
| Training time (7 epochs, no GPU visible in the run) | 462.6 s |

**1D-CNN vs CNN-LSTM (test set, threshold 0.5):**

| Metric | 1D-CNN | CNN-LSTM |
|---|---|---|
| Accuracy | 0.8066 | 0.7605 |
| Precision (abnormal) | 0.5139 | 0.4138 |
| Recall (abnormal) | 0.5193 | 0.4880 |
| F1-score (abnormal) | 0.5166 | 0.4479 |
| Specificity | 0.8780 | 0.8282 |
| ROC-AUC | 0.8448 | 0.7418 |

**Notes and limitations:**

- Best validation loss (0.4543, val accuracy 0.8145) came at epoch 1. After that, train accuracy rose to about 98.8% while validation loss went up (0.62 to 0.78), so the model overfits the training records. Early stopping restored epoch 1 weights.
- Test records are unseen patients (record-wise split), and abnormal share is 19.9% in test vs 8.2% in train, so abnormal precision and recall are only about 0.51 to 0.52.
- Settings are the same as Member 4 for data, seed, batch size, optimizer, loss, class weights and threshold. Differences: max epochs 30 (vs 15), early stopping patience 6 (vs 4), and ReduceLROnPlateau is used.
- The 1D-CNN was run once; no repeated runs, so small differences between models should not be over-interpreted.

## Integration, final comparison and demo (Member 5)

**Files:** demo/app.py, demo/README.md, requirements.txt, report/README.md. All numbers below are copied from results/*.csv and results/*.json, nothing is estimated.

**Demo flow:** input ECG beat -> preprocessing (bandpass 0.5 to 45 Hz + per-segment z-score, same as Member 2) -> 1D-CNN and CNN-LSTM (one sigmoid output each, threshold 0.5) -> predicted class: Normal or Abnormal.

**Final model comparison (test set, 15,897 beats from 7 unseen records):**

| Metric | 1D-CNN (Member 3) | CNN-LSTM (Member 4) |
|---|---|---|
| Accuracy | 0.8066 | 0.7605 |
| Precision (abnormal) | 0.5139 | 0.4138 |
| Recall (abnormal) | 0.5193 | 0.4880 |
| F1-score (abnormal) | 0.5166 | 0.4479 |
| Specificity | 0.8780 | 0.8282 |
| ROC-AUC | 0.8448 | 0.7418 |
| Confusion matrix (TN / FP / FN / TP) | 11179 / 1554 / 1521 / 1643 | 10546 / 2187 / 1620 / 1544 |

**Final CNN-LSTM CPU vs GPU (Tesla T4 vs 2-core Xeon):**

| Metric | CPU | GPU | Speedup |
|---|---|---|---|
| Total training time (5 epochs) | 297.20 s | 33.94 s | 8.76x |
| Average epoch time (epoch 2 onward) | 60.64 s | 6.22 s | 9.75x |
| Inference, full test set | 3.563 s | 0.479 s | 7.43x |

**Conclusions (based only on the measured values above):**

- On this test set the 1D-CNN is better than the CNN-LSTM on every metric in the comparison table. The extra LSTM layer did not improve results here.
- Both models are weak on the abnormal class (F1 about 0.45 to 0.52). The test set has 19.9% abnormal beats while train has 8.2%, and test records are unseen patients.
- GPU gives about 7.4x to 9.8x speedup for the CNN-LSTM, depending on what is measured (inference, total training, steady epoch).
- Training time of the two models must not be compared directly: the 1D-CNN run (462.6 s, 7 epochs) had no GPU visible and used early stopping with max 30 epochs, while the CNN-LSTM main run (112.11 s, 13 epochs) used the T4 GPU.
- Each timing experiment was run once, so small differences should not be over-interpreted.

**Demo:** run `streamlit run demo/app.py`. Step 1 choose a test beat (or upload an ECG file), Step 2 see the preprocessing, Step 3 see both model predictions with P(abnormal), Step 4 see the evaluation tables, which the app reads directly from results/*.csv.

## Folder guide

| Folder | What goes here | Main member |
|---|---|---|
| dataset/ | Dataset README, labels, notes | Member 1 |
| preprocessing/ | Preprocessing code and notes | Member 2 |
| notebooks/ | Colab notebooks (.ipynb) | All |
| models/ | Trained models | Member 3, 4 |
| results/ | Metrics, timing tables, CSV files | All |
| graphs/ | Plots and charts | All |
| demo/ | Final demo (app.py) and demo guide | Member 5 |
| report/ | Project report | All |

## Member 1 files

- results/dataset_stats_per_record.csv - beat counts per record
- graphs/class_distribution.png, ecg_normal.png, ecg_abnormal_V.png, ecg_abnormal_A.png
- notebooks/01_dataset_ecg_analisis_Ronit_WORK.ipynb - dataset analysis notebook

## Member 2 files

- notebooks/02_preprocessing_v1_Ritam_WORK.ipynb - preprocessing and split
- preprocessing/ - code and README

## Member 3 files

- notebooks/03_1d_cnn_Rishi_WORK.ipynb - 1D-CNN training and evaluation
- models/1d_cnn_model.keras - trained 1D-CNN model (best checkpoint, epoch 1)
- results/1d_cnn_metrics.csv, 1d_cnn_classification_report.csv
- graphs/1d_cnn_accuracy.png, 1d_cnn_loss.png, 1d_cnn_confusion_matrix.png

## Member 4 files

- notebooks/04_cnn_lstm_gpu_v1_Rohan_WORK.ipynb - CNN-LSTM training, CPU/GPU experiments
- models/cnn_lstm_v1.keras - trained CNN-LSTM model
- results/cnn_lstm_metrics_v1.csv, cnn_lstm_cpu_gpu_timing_v1.csv, cnn_lstm_timing_raw_v1.csv, cnn_lstm_history_v1.csv, cnn_lstm_experiment_settings_v1.json, hardware_info_v1.json
- graphs/cnn_lstm_training_curves_v1.png, cnn_lstm_confusion_matrix_v1.png, cnn_lstm_cpu_vs_gpu_v1.png

## Member 5 files

- demo/app.py - Streamlit demo (input ECG, preprocessing, both models, evaluation tables)
- demo/README.md - how to run the demo
- requirements.txt - packages for the demo
- report/README.md - report outline with the final numbers
- README.md and the README.md of every folder - kept up to date

## Rules

- Upload your files only to your own folder.
- Add a version to file names (preprocess_v1.py, cnn_model_v1.ipynb).
- If preprocessing changes, inform everyone and save it as a new version.
- Every number in the report must come from a real experiment.
- Model output is a classification result, not a clinical diagnosis.

## How to Run Final Integrated Demo (Member 5)

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

Needs `models/` (both .keras files), `results/` and `ecg_processed_v1.npz` (Google Drive link above). Models were saved with Keras 3.13.2, so use TensorFlow 2.20.0 as in requirements.txt. More details and troubleshooting: demo/README.md
