ECG Deep Learning Project

GPU-Accelerated Deep Learning-Based ECG Signal Classification for Cardiac Abnormality Detection

- **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
- **Models:** 1D-CNN and CNN-LSTM
- **Goal:** Classify ECG beats as normal or abnormal, and compare CPU vs GPU training performance
- **Environment:** Google Colab (Python, TensorFlow/Keras)

## Team

| Member | Role | Name |
|---|---|---|
| Member 1 | Dataset + ECG Analysis | Rohan Adak |
| Member 2 | Data Preprocessing | [Name] |
| Member 3 | 1D-CNN Model | [Name] |
| Member 4 | CNN-LSTM + GPU Performance | [Name] |
| Member 5 | Integration + Evaluation + Demo | [Name] |

## Progress

| Stage | Member | Status |
|---|---|---|
| Dataset + ECG analysis | Member 1 | Done |
| Preprocessing + split | Member 2 | Pending |
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
| Total beats | [from notebook output] |
| Normal beats | [from notebook output] |
| Abnormal beats | [from notebook output] |
| Abnormal % | [from notebook output] |

The dataset is imbalanced (normal beats are much more than abnormal). Use a record-wise split to avoid data leakage.

## Label definition (agreed by the team)

- **Normal (0):** N, L, R, e, j
- **Abnormal (1):** A, a, J, S, V, E, F, /, f, Q
- Paced records 102, 104, 107, 217 are excluded

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

## Rules

- Upload your files only to your own folder.
- Add a version to file names (preprocess_v1.py, cnn_model_v1.ipynb).
- If preprocessing changes, inform everyone and save it as a new version.
- Every number in the report must come from a real experiment.
- Model output is a classification result, not a clinical diagnosis.
