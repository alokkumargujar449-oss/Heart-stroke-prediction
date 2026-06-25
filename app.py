"""
app.py  –  Heart Disease Prediction  (Streamlit)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model            = joblib.load("knn.pkl")
    scaler           = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    "Fill in the patient details below and click **Predict** "
    "to estimate the risk of heart disease."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    age             = st.slider("Age", 18, 100, 45)
    sex             = st.selectbox("Sex", ["M", "F"])
    chest_pain      = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp      = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol     = st.number_input("Cholesterol (mg/dL)", 100, 603, 200)
    fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    resting_ecg     = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr          = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
    oldpeak         = st.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0, step=0.1)
    st_slope        = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):

    # Build a one-row dict with every one-hot column defaulting to 0
    raw = {col: 0 for col in expected_columns}

    # Numerical features
    raw['Age']         = age
    raw['RestingBP']   = resting_bp
    raw['Cholesterol'] = cholesterol
    raw['FastingBS']   = fasting_bs
    raw['MaxHR']       = max_hr
    raw['Oldpeak']     = oldpeak

    # One-hot features (drop_first=True means the *first* category is the
    # reference level and has no column; only non-reference dummies appear)
    for prefix, value in [
        ("Sex",            sex),
        ("ChestPainType",  chest_pain),
        ("RestingECG",     resting_ecg),
        ("ExerciseAngina", exercise_angina),
        ("ST_Slope",       st_slope),
    ]:
        col_name = f"{prefix}_{value}"
        if col_name in raw:          # skip if it's the reference category
            raw[col_name] = 1

    input_df     = pd.DataFrame([raw], columns=expected_columns)
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    proba        = model.predict_proba(scaled_input)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Heart Disease**  \n"
            f"Confidence: {proba[1]*100:.1f}%"
        )
    else:
        st.success(
            f"✅ **Low Risk of Heart Disease**  \n"
            f"Confidence: {proba[0]*100:.1f}%"
        )

    with st.expander("📊 Input summary sent to model"):
        st.dataframe(input_df)

st.caption("⚕️ This tool is for educational purposes only and is not a substitute for medical advice.")
