import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Failure Prediction",
    page_icon="🔴",
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

st.title("🔴 Machine Failure Prediction")
st.write(
    "Enter the current operating conditions of the machine "
    "to estimate failure risk."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_failure_model():

    model_path = MODEL_DIR / "failure_xgboost.joblib"

    if not model_path.exists():
        st.error(
            f"Model not found: {model_path}"
        )
        st.stop()

    return joblib.load(model_path)


failure_model = load_failure_model()


# --------------------------------------------------
# LOAD CONFIGURATION
# --------------------------------------------------

config_path = MODEL_DIR / "failure_model_config.json"

failure_threshold = 0.29

if config_path.exists():

    with open(config_path, "r") as file:
        config = json.load(file)

    # Try common configuration keys
    failure_threshold = config.get(
        "threshold",
        config.get(
            "decision_threshold",
            config.get(
                "failure_threshold",
                0.29
            )
        )
    )


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Machine Operating Conditions")

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Product Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

with col2:

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0.0,
        max_value=5000.0,
        value=1500.0,
        step=10.0
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=1.0
    )


# --------------------------------------------------
# CREATE INPUT DATA
# --------------------------------------------------

input_data = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temperature],
    "Process temperature [K]": [process_temperature],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear]
})


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "Analyze Machine",
    type="primary",
    use_container_width=True
):

    try:

        failure_probability = (
            failure_model.predict_proba(input_data)[0][1]
        )

        prediction = int(
            failure_probability >= failure_threshold
        )

        st.subheader("Analysis Result")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Failure Probability",
                f"{failure_probability * 100:.2f}%"
            )

        with m2:

            st.metric(
                "Decision Threshold",
                f"{failure_threshold:.2f}"
            )

        with m3:

            if prediction == 1:

                st.metric(
                    "Machine Status",
                    "HIGH RISK"
                )

            else:

                st.metric(
                    "Machine Status",
                    "NORMAL"
                )

        st.progress(
            min(
                float(failure_probability),
                1.0
            )
        )

        if prediction == 1:

            st.error(
                "⚠️ Elevated machine failure risk detected. "
                "Further inspection and fault diagnosis are recommended."
            )

        else:

            st.success(
                "✅ Current operating conditions are classified "
                "as lower failure risk."
            )

        st.caption(
            "This prediction is a machine-learning estimate "
            "for maintenance decision support."
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )