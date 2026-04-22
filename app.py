import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("dl_model.h5", compile=False)
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