# Dataset README - Group 7 ECG Project

## Source
MIT-BIH Arrhythmia Database, PhysioNet (https://physionet.org/content/mitdb/1.0.0/).
Downloaded with the WFDB Python package (wfdb.dl_database).

## Folder structure
The raw records are not stored in this GitHub repo (download them with the code in the main README). In Colab, dataset/mitdb/ has 48 records, each with 3 files:
- .dat = ECG signal, .hea = header (sampling rate, leads, gain), .atr = beat annotations
Total files: 144

## Signal information
- Sampling rate: 360 Hz
- Leads: MLII (channel 0) and V5 (channel 1) in record 100
- Record length: about 30 min (650,000 samples per record)
- Recommended lead: channel 0 (MLII)

## Labels
Annotation symbols are placed at each R peak. Non-beat symbols (like '+') are ignored.
- Normal (label 0): N, L, R, e, j (AAMI class N)
- Abnormal (label 1): A, a, J, S, V, E, F, /, f, Q

## Dataset statistics (paced records excluded)
- Records used: 44 (excluded paced records: 102, 104, 107, 217)
- Total beats: 100733
- Normal: 90125
- Abnormal: 10608 (10.5%)

Beat symbol counts:
N    74546
L     8075
R     7259
V     6903
A     2546
F      803
j      229
a      150
E      106
J       83
e       16
Q       15
S        2
f        0
/        0

Per-record counts: results/dataset_stats_per_record.csv

## Known issues
- Class imbalance: normal beats are much more than abnormal beats.
- Split must be record-wise (patient-wise) so beats from the same record are not in both train and test.
- Model output is a classification result, not a clinical diagnosis.

## Files from Member 1
- graphs/class_distribution.png, ecg_normal.png, ecg_abnormal_V.png, ecg_abnormal_A.png
- results/dataset_stats_per_record.csv

## Used later in the project
- Member 2 turned the 44 usable records into ecg_processed_v1.npz (see preprocessing/README.md).
- The demo (demo/app.py) uses beats from that processed file, not the raw records.
