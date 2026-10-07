# Demo (Member 5)

Streamlit demo that connects the work of all members:

    input ECG beat -> preprocessing -> 1D-CNN + CNN-LSTM -> predicted class (Normal / Abnormal)

| File | Description |
|---|---|
| app.py | Streamlit app (run from the project root) |
| ../requirements.txt | Packages: Streamlit, TensorFlow 2.20.0, Keras 3.13.2, NumPy, pandas, Matplotlib, SciPy |

## What the app does

1. **Step 1 - Input:** choose a beat from the test set of ecg_processed_v1.npz (filter: Any / Normal / Abnormal, the true label is shown), or upload your own ECG file (.csv, .txt, .npy) as raw signal or already preprocessed.
2. **Step 2 - Preprocessing (Member 2):** Butterworth bandpass 0.5 to 45 Hz (order 4, 360 Hz), 360-sample window, per-segment z-score. Demo beats from the .npz are already preprocessed, so this step is skipped for them.
3. **Step 3 - Prediction (Member 3 and Member 4):** both models give one sigmoid output, P(abnormal). Threshold 0.5: P >= 0.5 is Abnormal, otherwise Normal. The app also says whether the two models agree.
4. **Step 4 - Evaluation:** metrics, confusion matrix counts, CPU vs GPU table and speedups are read from results/*.csv and results/hardware_info_v1.json. No number is typed into the code. The "best model" line is chosen automatically from the abnormal-class F1 in the CSV files.

## How to run

1. Install (Python 3.10 to 3.13):

       pip install -r requirements.txt

2. Download **ecg_processed_v1.npz** (about 134 MB, too big for GitHub) from Google Drive:
   https://drive.google.com/file/d/1TO043QL8KSBiMkkf4Di5MMbLF5wXGgT0/view
   Put it in the project root (next to requirements.txt) or in a data/ folder. If the app cannot find it, it shows the link and an upload box.
3. Run from the project root:

       streamlit run demo/app.py

   The browser opens at http://localhost:8501

## Needed files

- models/1d_cnn_model.keras and models/cnn_lstm_v1.keras
- results/ folder (CSV and JSON files)
- ecg_processed_v1.npz

## Notes and troubleshooting

- Both models were saved with Keras 3.13.2 (.keras format) in Colab with TensorFlow 2.20.0. Older TensorFlow (for example 2.15, which uses Keras 2) can fail to load them, so keep the versions in requirements.txt.
- Own ECG files: for raw signals give at least 360 samples at 360 Hz. Longer signals are filtered as a whole and the centre 360 samples are used as the beat. Files with fewer than 360 samples give an error message.
- The models were trained on MIT-BIH beats (MLII lead, 360 Hz) with the preprocessing above. Other leads, sampling rates or filters give unreliable predictions.
- Results are weak for the abnormal class (F1 about 0.45 to 0.52 on unseen records), so the demo can make mistakes. It is an educational project demo, not a medical device.
