import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="⚠️",
    layout="wide"
)

# ============================================================
# BACK TO MAIN DASHBOARD
# ============================================================

if st.button(
    "🏠 Back to Main Dashboard",
    width="stretch"
):
    st.switch_page("main.py")

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

st.title("⚠️ Machine Anomaly Detection")

st.write(
    "Detect unusual operating conditions using the "
    "unsupervised Isolation Forest model."
)

st.divider()


# ---------------- LOAD ----------------

@st.cache_resource
def load_anomaly_system():

    model = joblib.load(
        MODEL_DIR /
        "anomaly_isolation_forest.joblib"
    )

    scaler = joblib.load(
        MODEL_DIR /
        "anomaly_scaler.joblib"
    )

    return model, scaler


model, scaler = load_anomaly_system()


# ---------------- INPUT ----------------

st.subheader("Machine Sensor Readings")

c1, c2, c3 = st.columns(3)

with c1:

    air_temp = st.number_input(
        "Air Temperature [K]",
        250.0,
        350.0,
        300.0
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        250.0,
        400.0,
        310.0
    )


with c2:

    speed = st.number_input(
        "Rotational Speed [rpm]",
        0.0,
        5000.0,
        1500.0
    )

    torque = st.number_input(
        "Torque [Nm]",
        0.0,
        200.0,
        40.0
    )


with c3:

    tool_wear = st.number_input(
        "Tool Wear [min]",
        0.0,
        500.0,
        100.0
    )


input_df = pd.DataFrame({

    "Air temperature [K]": [
        air_temp
    ],

    "Process temperature [K]": [
        process_temp
    ],

    "Rotational speed [rpm]": [
        speed
    ],

    "Torque [Nm]": [
        torque
    ],

    "Tool wear [min]": [
        tool_wear
    ]
})


st.divider()


# ---------------- ANALYSIS ----------------

if st.button(
    "🔍 Detect Anomaly",
    type="primary",
    use_container_width=True
):

    try:

        scaled = scaler.transform(
            input_df
        )

        prediction = model.predict(
            scaled
        )[0]

        decision = model.decision_function(
            scaled
        )[0]

        anomaly_score = -decision


        st.subheader(
            "Anomaly Analysis"
        )

        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.4f}"
            )


        with c2:

            status = (
                "ANOMALY"
                if prediction == -1
                else "NORMAL"
            )

            st.metric(
                "Operating Status",
                status
            )


        if prediction == -1:

            st.error(
                "⚠️ Unusual operating behaviour detected."
            )

            st.write(
                "The current combination of sensor "
                "measurements differs significantly "
                "from typical machine operating conditions."
            )

        else:

            st.success(
                "✅ Operating conditions are within "
                "the learned normal region."
            )


        st.info(
            "Anomaly detection does not automatically mean "
            "machine failure. It identifies unusual operating "
            "patterns that may require further investigation."
        )


    except Exception as e:

        st.error(
            f"Anomaly detection error: {e}"
        )