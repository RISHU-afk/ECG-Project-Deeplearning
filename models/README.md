# Models

| File | Member | Description |
|---|---|---|
| cnn_lstm_v1.keras | Member 4 | CNN-LSTM trained on ecg_processed_v1.npz (GPU, best epoch 9 of 13). Test accuracy 0.7605, recall 0.4880, F1 0.4479. |
| (Member 3 file - pending) | Member 3 | 1D-CNN |

Load a model:

    from tensorflow import keras
    model = keras.models.load_model('cnn_lstm_v1.keras')
    prob = model.predict(X_test)        # probability of abnormal
    pred = (prob >= 0.5).astype(int)    # 0 = Normal, 1 = Abnormal

Model output is a classification result, not a clinical diagnosis.
