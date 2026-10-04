# Results

| File | Member | Description |
|---|---|---|
| dataset_stats_per_record.csv | Member 1 | Beat counts per record (normal/abnormal) |
| cnn_lstm_metrics_v1.csv | Member 4 | CNN-LSTM main run (GPU): test and validation metrics, confusion matrix counts, training time, epochs run |
| cnn_lstm_cpu_gpu_timing_v1.csv | Member 4 | CPU vs GPU summary: training time, epoch time, inference time, accuracy (5 epochs each) |
| cnn_lstm_timing_raw_v1.csv | Member 4 | Raw CPU/GPU timing for each repeat |
| cnn_lstm_history_v1.csv | Member 4 | Per-epoch train/validation loss and accuracy of the main run |
| cnn_lstm_experiment_settings_v1.json | Member 4 | Seed, batch size, learning rate, epochs, class weights, threshold |
| hardware_info_v1.json | Member 4 | Actual Colab runtime: Tesla T4 GPU, Xeon CPU, TensorFlow/CUDA versions |

Column notes for cnn_lstm_cpu_gpu_timing_v1.csv: epoch1_s includes warm-up, steady_epoch_avg_s is the average from epoch 2 onward, inference_test_set_s is the time to predict the full test set (15,897 beats).

Add your result files here (metrics, timing tables, confusion matrices).
