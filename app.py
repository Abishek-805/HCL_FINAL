import streamlit as st
import numpy as np
import joblib
import json
import os
import shutil
import tempfile
import h5py
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
def _remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj = {k: _remove_quantization_config(v) for k, v in obj.items() if k != "quantization_config"}
        return obj
    if isinstance(obj, list):
        return [_remove_quantization_config(item) for item in obj]
    return obj


def _load_model_compat(model_path):
    try:
        return load_model(model_path, compile=False)
    except Exception as err:
        if "quantization_config" not in str(err):
            raise

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            patched_path = tmp.name

        shutil.copyfile(model_path, patched_path)

        try:
            with h5py.File(patched_path, "r+") as h5f:
                raw_config = h5f.attrs.get("model_config")
                if raw_config is None:
                    raise

                if isinstance(raw_config, bytes):
                    config_text = raw_config.decode("utf-8")
                elif hasattr(raw_config, "decode"):
                    config_text = raw_config.decode("utf-8")
                else:
                    config_text = str(raw_config)

                config_json = json.loads(config_text)
                cleaned_config = _remove_quantization_config(config_json)
                h5f.attrs.modify("model_config", json.dumps(cleaned_config).encode("utf-8"))

            return load_model(patched_path, compile=False)
        finally:
            try:
                os.remove(patched_path)
            except OSError:
                pass


model = _load_model_compat("dl_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Waste AI", layout="centered")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.big-number {
    font-size: 60px;
    font-weight: bold;
    color: #00FFA6;
    text-align: center;
    margin-top: 20px;
}
.title-center {
    text-align: center;
}
.card {
    padding: 25px;
    border-radius: 15px;
    background-color: #111;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<h1 class="title-center">🧠 Smart Waste Predictor</h1>', unsafe_allow_html=True)
st.caption("Simple inputs • Smart prediction")

# =========================
# INPUT SECTION
# =========================
st.subheader("Input Data")

day = st.selectbox(
    "Select Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

# FULL RANGE SLIDERS
yesterday = st.slider(
    "Yesterday Waste (kg)",
    0, 50000, 25000, step=1
)

weekly_avg = st.slider(
    "Weekly Average (kg)",
    0, 50000, 30000, step=1
)

# =========================
# DAY FEATURES
# =========================
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

day_num = day_map[day]
is_weekend = 1 if day_num >= 5 else 0

# =========================
# AUTO LAST WEEK ESTIMATE
# =========================
def estimate_last_week(avg, day_num):
    return avg * 1.7 if day_num >= 5 else avg * 0.95

last_week = estimate_last_week(weekly_avg, day_num)

# 🔥 NEW FEATURE (MATCHES MODEL)
weekend_signal = is_weekend * weekly_avg

# =========================
# SESSION STATE
# =========================
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.is_weekend = None

# =========================
# BUTTON
# =========================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Waste", use_container_width=True):

    input_data = np.array([[ 
        yesterday,
        last_week,
        weekly_avg,
        0,              # rolling_std placeholder
        day_num,
        is_weekend,
        weekend_signal
    ]])

    # SCALE
    input_scaled = scaler_X.transform(input_data)

    # PREDICT
    pred_scaled = model.predict(input_scaled, verbose=0)

    prediction = scaler_y.inverse_transform(pred_scaled)[0][0]

    # SAFETY
    prediction = max(0, prediction)

    # =========================
    # 🔥 FINAL WEEKEND SAFETY BOOST (SMALL)
    # =========================
    if is_weekend:
        prediction += 0.08 * weekly_avg   # small + safe

    # SAVE
    st.session_state.prediction = prediction
    st.session_state.is_weekend = is_weekend

# =========================
# RESULT DISPLAY
# =========================
if st.session_state.prediction is not None:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Predicted Waste")

    st.markdown(
        f"""
        <div class="big-number">
            {st.session_state.prediction:,.0f} kg
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.is_weekend:
        st.success("📈 Weekend → Higher waste expected")
    else:
        st.info("📉 Weekday → Normal waste")

    st.markdown('</div>', unsafe_allow_html=True)