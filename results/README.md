# Results

| File | Member | Description |
|---|---|---|
| dataset_stats_per_record.csv | Member 1 | Beat counts per record (normal/abnormal) |
| 1d_cnn_metrics.csv | Member 3 | 1D-CNN test metrics (accuracy, precision, recall, F1, macro F1, ROC-AUC), confusion matrix counts, best epoch, epochs run, training time, parameters |
| 1d_cnn_classification_report.csv | Member 3 | 1D-CNN per-class precision, recall, F1 and support on the test set |
| cnn_lstm_metrics_v1.csv | Member 4 | CNN-LSTM main run (GPU): test and validation metrics, confusion matrix counts, training time, epochs run |
| cnn_lstm_cpu_gpu_timing_v1.csv | Member 4 | CPU vs GPU summary: training time, epoch time, inference time, accuracy (5 epochs each) |
| cnn_lstm_timing_raw_v1.csv | Member 4 | Raw CPU/GPU timing for each repeat |
| cnn_lstm_history_v1.csv | Member 4 | Per-epoch train/validation loss and accuracy of the main run |
| cnn_lstm_experiment_settings_v1.json | Member 4 | Seed, batch size, learning rate, epochs, class weights, threshold |
| hardware_info_v1.json | Member 4 | Actual Colab runtime: Tesla T4 GPU, Xeon CPU, TensorFlow/CUDA versions |

Column notes for cnn_lstm_cpu_gpu_timing_v1.csv: epoch1_s includes warm-up, steady_epoch_avg_s is the average from epoch 2 onward, inference_test_set_s is the time to predict the full test set (15,897 beats).

Add your result files here (metrics, timing tables, confusion matrices).

## How the numbers are used (Member 5)

- demo/app.py reads 1d_cnn_metrics.csv, cnn_lstm_metrics_v1.csv, cnn_lstm_cpu_gpu_timing_v1.csv and hardware_info_v1.json and shows them in its evaluation step.
- graphs/final_model_comparison.png is drawn from the same CSV files.
- CPU/GPU speedups (CPU time / GPU time) from cnn_lstm_cpu_gpu_timing_v1.csv: total training 8.76x, steady epoch 9.75x, inference on the test set 7.43x.
- Use cnn_lstm_metrics_v1.csv (main run) for model comparison and cnn_lstm_cpu_gpu_timing_v1.csv only for timing. The 5-epoch timing runs have slightly different accuracy (0.7632 CPU, 0.7671 GPU) than the main run (0.7605).
- 1d_cnn_metrics.csv: the 1D-CNN training_time_sec (462.6 s, 7 epochs) comes from a run with no GPU visible, so it is not directly comparable with the CNN-LSTM GPU time.
