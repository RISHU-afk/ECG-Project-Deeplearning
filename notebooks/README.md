# Notebooks

| File | Member | Description |
|---|---|---|
| 01_dataset_ecg_analisis_Ronit_WORK.ipynb | Member 1 | MIT-BIH download, label counts, ECG plots, README |
| 02_preprocessing_v1_Ritam_WORK.ipynb | Member 2 | Preprocessing |
| 03_1d_cnn_Rishi_WORK.ipynb | Member 3 | 1D-CNN training, evaluation, metrics, graphs, saved model (loads ecg_processed_v1.npz; asks for upload in Colab if not found) |
| 04_cnn_lstm_gpu_v1_Rohan_WORK.ipynb | Member 4 | CNN-LSTM training, CPU vs GPU timing, hardware info (run on Colab T4 GPU runtime) |

Run notebooks in Google Colab. File naming: name_v1.ipynb, name_v2.ipynb.

Member 4 notebook: upload ecg_processed_v1.npz to Colab (or let the notebook download it from Drive), select a GPU runtime, then Runtime > Run all. It saves models, results and graphs in /content/Group_7_ECG_Project/ and downloads them as one zip.
