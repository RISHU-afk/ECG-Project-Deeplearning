# Models

| File | Member | Description |
|---|---|---|
| cnn_lstm_v1.keras | Member 4 | CNN-LSTM trained on ecg_processed_v1.npz (GPU, best epoch 9 of 13). Test accuracy 0.7605, recall 0.4880, F1 0.4479. |
| 1d_cnn_model.keras | Member 3 | 1D-CNN trained on ecg_processed_v1.npz (best checkpoint, epoch 1 of 7 run, 379,265 parameters). Test accuracy 0.8066, recall 0.5193, F1 0.5166, ROC-AUC 0.8448. |

Load a model:

    from tensorflow import keras
    model = keras.models.load_model('cnn_lstm_v1.keras')   # or '1d_cnn_model.keras'
    prob = model.predict(X_test)        # probability of abnormal
    pred = (prob >= 0.5).astype(int)    # 0 = Normal, 1 = Abnormal

Model output is a classification result, not a clinical diagnosis.
