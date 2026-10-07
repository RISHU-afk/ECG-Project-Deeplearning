# Report (all members, coordinated by Member 5)

The final report goes in this folder. Rule: every number must come from a real experiment (results/*.csv, results/*.json). Do not invent or estimate numbers.

## Planned outline

| Section | Content | Main member | Source files |
|---|---|---|---|
| 1. Introduction | Problem, goal: classify ECG beats as normal or abnormal, GPU speedup | Member 5 | README.md |
| 2. Dataset and ECG analysis | MIT-BIH, 48 records downloaded, 44 used, 100,733 beats, class imbalance, ECG examples | Member 1 | dataset/README.md, results/dataset_stats_per_record.csv, graphs/class_distribution.png, ecg_normal.png, ecg_abnormal_V.png, ecg_abnormal_A.png |
| 3. Preprocessing | Bandpass 0.5 to 45 Hz, 360-sample window, z-score, record-wise split 30/7/7 records | Member 2 | preprocessing/README.md, preprocessing/02_preprocessing_v1.py |
| 4. 1D-CNN | Architecture, settings, test metrics, curves, confusion matrix | Member 3 | results/1d_cnn_metrics.csv, results/1d_cnn_classification_report.csv, graphs/1d_cnn_*.png |
| 5. CNN-LSTM + GPU | Architecture, settings, test metrics, CPU vs GPU experiment, hardware | Member 4 | results/cnn_lstm_*.csv, results/hardware_info_v1.json, graphs/cnn_lstm_*.png |
| 6. Comparison and discussion | Model comparison, CPU/GPU table, limitations | Member 5 | README.md (Member 5 section) |
| 7. Demo | Demo flow and screenshots | Member 5 | demo/README.md |
| 8. Conclusion | Findings, limitations, future work | All | - |

## Final numbers to use (copied from results/)

Test set: 15,897 beats (12,733 normal, 3,164 abnormal), 7 unseen records, threshold 0.5.

| Metric | 1D-CNN | CNN-LSTM |
|---|---|---|
| Accuracy | 0.8066 | 0.7605 |
| Precision (abnormal) | 0.5139 | 0.4138 |
| Recall (abnormal) | 0.5193 | 0.4880 |
| F1-score (abnormal) | 0.5166 | 0.4479 |
| ROC-AUC | 0.8448 | 0.7418 |

| CNN-LSTM, 5 epochs | CPU | GPU | Speedup |
|---|---|---|---|
| Total training time | 297.20 s | 33.94 s | 8.76x |
| Average epoch time (epoch 2 onward) | 60.64 s | 6.22 s | 9.75x |
| Inference, full test set | 3.563 s | 0.479 s | 7.43x |

Hardware: Google Colab, Tesla T4 GPU, Intel Xeon 2.00GHz (2 logical cores), TensorFlow 2.20.0, Keras 3.13.2.

## Limitations to mention

- Abnormal class is imbalanced (8.2% in train, 19.9% in test) and both models have F1 of only about 0.45 to 0.52 on it.
- 1D-CNN best epoch was 1 and it overfits afterwards; CNN-LSTM validation recall (0.85) is much higher than test recall (0.49).
- Training times of the two models are not directly comparable (1D-CNN ran without GPU for 7 epochs, CNN-LSTM main run used the T4 GPU for 13 epochs).
- Each timing experiment was run once.
- Model output is a classification result, not a clinical diagnosis.

## Status

Report not written yet. Each member writes their own section from the table above; Member 5 merges and checks that every number matches results/.
