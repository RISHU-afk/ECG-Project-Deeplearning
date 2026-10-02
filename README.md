ECG Deep Learning Project

GPU-Accelerated Deep Learning-Based ECG Signal Classification for Cardiac Abnormality Detection

- **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
- **Models:** 1D-CNN and CNN-LSTM
- **Goal:** Classify ECG beats as normal or abnormal, and compare CPU vs GPU training performance
- **Environment:** Google Colab (Python, TensorFlow/Keras)

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
| 1D-CNN | Member 3 | Pending |
| CNN-LSTM + CPU/GPU | Member 4 | Pending |
| Integration + demo | Member 5 | Pending |

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

## Folder guide

| Folder | What goes here | Main member |
|---|---|---|
| dataset/ | Dataset README, labels, notes | Member 1 |
| preprocessing/ | Preprocessing code and notes | Member 2 |
| notebooks/ | Colab notebooks (.ipynb) | All |
| models/ | Trained models | Member 3, 4 |
| results/ | Metrics, timing tables, CSV files | All |
| graphs/ | Plots and charts | All |
| demo/ | Final demo files | Member 5 |
| report/ | Project report | All |
| presentation/ | PPT | All |

## Member 1 files

- results/dataset_stats_per_record.csv - beat counts per record
- graphs/class_distribution.png, ecg_normal.png, ecg_abnormal_V.png, ecg_abnormal_A.png
- notebooks/ - dataset analysis notebook

## Member 2 files

- notebooks/02_preprocessing_v1.ipynb - preprocessing and split
- preprocessing/ - code and README

## Rules

- Upload your files only to your own folder.
- Add a version to file names (preprocess_v1.py, cnn_model_v1.ipynb).
- If preprocessing changes, inform everyone and save it as a new version.
- Every number in the report must come from a real experiment.
- Model output is a classification result, not a clinical diagnosis.
