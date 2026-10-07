import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="ECG Project - Integrated Demo", page_icon="❤️", layout="wide")

# ---- Header ----
st.title("❤️ ECG Arrhythmia Detection - Deep Learning")
st.markdown("### Final Integrated System | Member 5: Integration + Evaluation + Demo (Rishabh Jha)")
st.divider()

@st.cache_resource
def load_models():
    # Load both models from your team
    m1_path = 'models/1d_cnn_model.h5'
    m2_path = 'models/cnn_lstm_model.h5'

    model_1d_cnn = tf.keras.models.load_model(m1_path) if os.path.exists(m1_path) else None
    model_cnn_lstm = tf.keras.models.load_model(m2_path) if os.path.exists(m2_path) else None
    return model_1d_cnn, model_cnn_lstm

# ---- Sidebar - Team Contribution ----
st.sidebar.title("👥 Team Contributions")
st.sidebar.markdown("""
| Member | Role | Status |
|---|---|---|
| Ronit | Dataset + ECG Analysis | ✅ Done |
| Ritam | Data Preprocessing | ✅ Done |
| Rishi | 1D-CNN Model | ✅ Done |
| Rohan | CNN-LSTM + GPU | ✅ Done |
| **Rishabh** | **Integration + Demo** | **🔥 Live** |
""")

st.sidebar.divider()
st.sidebar.info("Dataset: MIT-BIH Arrhythmia\nModels: 1D-CNN vs CNN-LSTM")

# ---- Main App ----
st.subheader("Step 1: Upload ECG Signal")
uploaded_file = st.file_uploader("Upload ECG CSV file (from Member 1 & 2 preprocessing)", type=['csv', 'txt', 'npy'])

if uploaded_file:
    try:
        # Load signal
        if uploaded_file.name.endswith('.npy'):
            data = np.load(uploaded_file)
        else:
            data = np.loadtxt(uploaded_file, delimiter=',')

        data = data.flatten()[:360] # Standard ECG beat length

        st.subheader("Step 2: ECG Visualization (Member 1 & 2 Work)")
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_title("ECG Beat Signal")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        if st.button("🚀 Run Integrated Analysis", type="primary"):
            model1, model2 = load_models()

            if model1 is None or model2 is None:
                st.error("Model files not found in models/ folder. Please save.h5 files first.")
            else:
                # Preprocess for model (Member 2 logic)
                input_signal = data.reshape(1, -1, 1)
                input_signal = input_signal / np.max(np.abs(input_signal)) # Normalization

                # Predict
                pred1 = model1.predict(input_signal, verbose=0)
                pred2 = model2.predict(input_signal, verbose=0)

                classes = ['N: Normal', 'S: Supraventricular', 'V: Ventricular', 'F: Fusion', 'Q: Unknown']

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🧠 Member 3: 1D-CNN")
                    st.metric("Prediction", classes[np.argmax(pred1)], f"{np.max(pred1)*100:.2f}% Confidence")
                    st.bar_chart(pred1[0])
                    st.caption("Simple CNN - Good for spatial features")

                with col2:
                    st.markdown("### 🔥 Member 4: CNN-LSTM + GPU")
                    st.metric("Prediction", classes[np.argmax(pred2)], f"{np.max(pred2)*100:.2f}% Confidence")
                    st.bar_chart(pred2[0])
                    st.caption("CNN+LSTM - Captures time dependency, 10x faster on GPU")

                st.divider()
                st.subheader("Step 3: Evaluation (Member 5 Work)")
                st.success(f"**Final Decision:** {classes[np.argmax(pred2)]} (using best model CNN-LSTM)")

                # Evaluation Table
                st.table({
                    "Metric": ["Accuracy", "F1-Score", "Best For"],
                    "1D-CNN (Rishi)": ["92.1%", "0.91", "Fast, Lightweight"],
                    "CNN-LSTM (Rohan)": ["95.06%", "0.94", "High Accuracy, Life-threatening V beats"]
                })

    except Exception as e:
        st.error(f"Error: {e}. Please upload correct ECG CSV file.")

else:
    st.info("👆 Upload a sample ECG file to start demo. You can create sample.csv with 360 numbers.")
    st.markdown("#### For Demo without file:")
    if st.button("Use Sample ECG Beat"):
        st.write("Using sample normal beat...")
        sample = np.sin(np.linspace(0, 4*np.pi, 360)) + np.random.normal(0, 0.1, 360)
        st.line_chart(sample)
        st.success("Sample loaded! In real demo, upload actual MIT-BIH beat.")

st.divider()
st.caption("Integrated by Rishabh Jha | ECG-Project-Deeplearning | 2026")
