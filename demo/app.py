"""ECG Normal/Abnormal Detection - Integrated Demo (Group 7)

Flow: input ECG -> preprocessing (Member 2) -> 1D-CNN (Member 3) + CNN-LSTM (Member 4)
      -> predicted Normal / Abnormal.

All numbers shown in the evaluation section are read from results/*.csv
(real experiment outputs). Nothing is hardcoded.
Run (from the project root):  streamlit run demo/app.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(start):
    """Project root = the folder that contains models/ and results/ (works if app.py is in demo/)."""
    p = start
    for _ in range(3):
        if os.path.isdir(os.path.join(p, "models")) and os.path.isdir(os.path.join(p, "results")):
            return p
        p = os.path.dirname(p)
    return start


BASE = _find_root(HERE)
MODEL_DIR = os.path.join(BASE, "models")
RESULT_DIR = os.path.join(BASE, "results")
DATA_NAME = "ecg_processed_v1.npz"
DRIVE_LINK = "https://drive.google.com/file/d/1TO043QL8KSBiMkkf4Di5MMbLF5wXGgT0/view"
DATA_CANDIDATES = [os.path.join(b, p, DATA_NAME) for b in dict.fromkeys([BASE, HERE, os.getcwd()])
                   for p in ("", "data", "dataset", "demo")]

MODEL_FILES = {
    "1D-CNN": os.path.join(MODEL_DIR, "1d_cnn_model.keras"),
    "CNN-LSTM": os.path.join(MODEL_DIR, "cnn_lstm_v1.keras"),
}
WINDOW = 360       # 180 left + 180 right of the beat (1 s at 360 Hz)
FS = 360.0
THRESHOLD = 0.5    # same as THRESHOLD in cnn_lstm_experiment_settings_v1.json

st.set_page_config(page_title="ECG Project - Integrated Demo", page_icon="❤️", layout="wide")
st.title("❤️ ECG Normal / Abnormal Detection - Deep Learning")
st.caption("Group 7 | MIT-BIH Arrhythmia subset | 1D-CNN vs CNN-LSTM")


# ---------------------------------------------------------------- preprocessing
def bandpass(sig, low=0.5, high=45.0, fs=FS, order=4):
    """Same filter as Member 2 (preprocessing/02_preprocessing_v1.py)."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def zscore(seg):
    """Same per-segment normalization as Member 2."""
    std = np.std(seg)
    return (seg - np.mean(seg)) / std if std > 0 else seg - np.mean(seg)


def preprocess(signal, already_processed):
    """Return a (360,) float32 beat ready for the models."""
    signal = np.asarray(signal, dtype=np.float64).flatten()
    if len(signal) < WINDOW:
        raise ValueError(f"Need at least {WINDOW} samples, got {len(signal)}.")
    if not already_processed:
        signal = bandpass(signal)              # filter the whole signal first
    start = (len(signal) - WINDOW) // 2        # beat window = centre 360 samples
    seg = signal[start:start + WINDOW]
    if not already_processed:
        seg = zscore(seg)
    return seg.astype(np.float32)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    import tensorflow as tf
    models, errors = {}, {}
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            errors[name] = f"file not found: {os.path.relpath(path, BASE)}"
            continue
        try:
            models[name] = tf.keras.models.load_model(path, compile=False)
        except Exception as e:  # noqa: BLE001
            errors[name] = str(e)
    return models, errors


def read_csv(name):
    p = os.path.join(RESULT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_results():
    cnn = read_csv("1d_cnn_metrics.csv")
    lstm = read_csv("cnn_lstm_metrics_v1.csv")
    timing = read_csv("cnn_lstm_cpu_gpu_timing_v1.csv")
    return cnn, lstm, timing


def comparison_table(cnn, lstm):
    c, l = cnn.iloc[0], lstm.iloc[0]
    rows = [
        ("Test accuracy", c.test_accuracy, l.test_accuracy),
        ("Precision (abnormal)", c.precision_abnormal, l.test_precision),
        ("Recall (abnormal)", c.recall_abnormal, l.test_recall),
        ("F1 (abnormal)", c.f1_abnormal, l.test_f1),
        ("ROC-AUC", c.roc_auc, l.test_roc_auc),
        ("TN", c.TN, l.test_TN),
        ("FP", c.FP, l.test_FP),
        ("FN", c.FN, l.test_FN),
        ("TP", c.TP, l.test_TP),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "1D-CNN", "CNN-LSTM"])
    fmt = lambda v, m: f"{int(v)}" if m in ("TN", "FP", "FN", "TP") else f"{v:.4f}"
    df["1D-CNN"] = [fmt(v, m) for m, v in zip(df.Metric, df["1D-CNN"])]
    df["CNN-LSTM"] = [fmt(v, m) for m, v in zip(df.Metric, df["CNN-LSTM"])]
    return df


# ---------------------------------------------------------------- Step 1: input
st.subheader("Step 1: ECG input")
mode = st.radio(
    "Input source",
    ["Demo beat (already preprocessed)", "Upload ECG file (.csv / .txt / .npy)"],
    horizontal=True,
)

signal, already_processed, true_label = None, True, None

if mode.startswith("Demo"):
    local = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    uploaded_npz = None
    if local is None:
        st.info(
            f"`{DATA_NAME}` (~134 MB) is hosted on Google Drive because of GitHub's size limit: "
            f"[download it here]({DRIVE_LINK}). Put it in the project folder next to app.py "
            "(or in data/) and refresh, or upload it below."
        )
        uploaded_npz = st.file_uploader(DATA_NAME, type=["npz"])
    src = uploaded_npz if uploaded_npz is not None else local
    if src is not None:
        d = np.load(src)
        X_test, y_test = d["X_test"], d["y_test"]          # only the test split is read
        which = st.radio("Beat type", ["Any", "Normal", "Abnormal"], horizontal=True)
        pool = np.arange(len(y_test)) if which == "Any" else np.where(y_test == (which == "Abnormal"))[0]
        pos = st.slider("Test beat number", 0, len(pool) - 1, 0)
        idx = int(pool[pos])
        signal = X_test[idx].flatten()
        true_label = int(y_test[idx])
        already_processed = True
else:
    kind = st.radio(
        "File content",
        ["Raw ECG (mV, 360 Hz) - will be filtered + normalized",
         "Already preprocessed (filtered + z-score, 360 samples)"],
    )
    already_processed = kind.startswith("Already")
    f = st.file_uploader("ECG file", type=["csv", "txt", "npy"])
    if f is not None:
        try:
            signal = np.load(f) if f.name.endswith(".npy") else np.loadtxt(f, delimiter=",")
            signal = np.asarray(signal).flatten()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read file: {e}")
            signal = None
    st.caption("For raw input give >= 360 samples (the centre 360 samples are used as the beat). "
               "Longer raw signals are filtered as a whole first, like in training.")

# ---------------------------------------------------------------- Step 2: preprocessing
beat = None
if signal is not None:
    try:
        beat = preprocess(signal, already_processed)
    except ValueError as e:
        st.error(str(e))

if beat is not None:
    st.subheader("Step 2: Preprocessing")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.plot(signal[:max(len(signal), WINDOW)] if len(signal) <= 2000 else signal[:2000], lw=0.8)
        ax.set_title("Input signal")
        ax.set_xlabel("Sample")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.plot(beat, lw=0.8, color="tab:red")
        ax.set_title("Model input (360 samples, bandpass 0.5-45 Hz + z-score)")
        ax.set_xlabel("Sample")
        st.pyplot(fig)
    if true_label is not None:
        st.caption(f"Ground-truth label of this demo beat: **{'Abnormal' if true_label else 'Normal'}**")

    # ------------------------------------------------------------ Step 3: model
    st.subheader("Step 3: Prediction")
    if st.button("Run both models", type="primary"):
        models, errors = load_models()
        for n, e in errors.items():
            st.error(f"{n} could not be loaded: {e}")
        if models:
            x = beat.reshape(1, WINDOW, 1)
            cols = st.columns(len(models))
            preds = {}
            for col, (name, m) in zip(cols, models.items()):
                p = float(m.predict(x, verbose=0).ravel()[0])   # sigmoid P(abnormal)
                preds[name] = p
                with col:
                    st.markdown(f"### {name}")
                    st.metric("Prediction", "Abnormal" if p >= THRESHOLD else "Normal",
                              f"P(abnormal) = {p:.3f}")
                    st.progress(min(max(p, 0.0), 1.0))
                    st.caption(f"Threshold = {THRESHOLD}")
            if len(preds) == 2:
                labels = {n: p >= THRESHOLD for n, p in preds.items()}
                if len(set(labels.values())) == 1:
                    st.success(f"Both models agree: **{'Abnormal' if next(iter(labels.values())) else 'Normal'}**")
                else:
                    st.warning("The two models disagree on this beat.")

# ---------------------------------------------------------------- Step 4: evaluation
st.divider()
st.subheader("Step 4: Evaluation (from results/*.csv)")
cnn, lstm, timing = load_results()
if cnn is None or lstm is None:
    st.warning("results/1d_cnn_metrics.csv or results/cnn_lstm_metrics_v1.csv not found.")
else:
    st.table(comparison_table(cnn, lstm).set_index("Metric"))
    f1_cnn, f1_lstm = cnn.iloc[0].f1_abnormal, lstm.iloc[0].test_f1
    best = "1D-CNN" if f1_cnn >= f1_lstm else "CNN-LSTM"
    st.info(f"Best model on the test set by abnormal-class F1: **{best}** "
            f"(1D-CNN {f1_cnn:.4f} vs CNN-LSTM {f1_lstm:.4f}).")

if timing is not None and {"CPU", "GPU"} <= set(timing.device):
    t = timing.set_index("device")
    cpu, gpu = t.loc["CPU"], t.loc["GPU"]
    tt = pd.DataFrame({
        "Metric": ["Total training time (s)", "Steady epoch avg (s)", "Inference, test set (s)",
                   "Inference per beat (ms)"],
        "CPU": [cpu.train_time_total_s, cpu.steady_epoch_avg_s, cpu.inference_test_set_s, cpu.inference_ms_per_beat],
        "GPU": [gpu.train_time_total_s, gpu.steady_epoch_avg_s, gpu.inference_test_set_s, gpu.inference_ms_per_beat],
    })
    tt["Speed-up (CPU/GPU)"] = [f"{c / g:.2f}x" for c, g in zip(tt.CPU, tt.GPU)]
    tt["CPU"] = tt.CPU.map("{:.4f}".format)
    tt["GPU"] = tt.GPU.map("{:.4f}".format)
    st.markdown("**CNN-LSTM: CPU vs GPU** "
                f"({int(cpu.epochs)} epochs, batch {int(cpu.batch_size)}, "
                f"{int(cpu.n_train_samples)} train / {int(cpu.n_test_samples)} test beats)")
    st.table(tt.set_index("Metric"))
    hw = os.path.join(RESULT_DIR, "hardware_info_v1.json")
    if os.path.exists(hw):
        import json
        h = json.load(open(hw))
        st.caption(f"Hardware: {h.get('gpu_name')} ({h.get('gpu_memory')}), CPU {h.get('cpu_model')} "
                   f"x{h.get('cpu_logical_cores')} cores, TensorFlow {h.get('tensorflow')}, Keras {h.get('keras')}")

st.divider()
st.caption("Educational project demo - not a medical device. "
           "Dataset: MIT-BIH Arrhythmia Database (binary Normal vs Abnormal).")
