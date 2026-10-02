import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Fault Diagnosis",
    page_icon="🛠️",
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

# ---------------- PATHS ----------------

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

# ---------------- UI ----------------

st.title("🛠️ Machine Fault Diagnosis")

st.write(
    "Analyze machine operating conditions to identify supported "
    "failure modes using trained machine-learning models."
)

st.divider()

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():

    return {
        "HDF": joblib.load(
            MODEL_DIR / "diagnosis_hdf.joblib"
        ),

        "PWF": joblib.load(
            MODEL_DIR / "diagnosis_pwf.joblib"
        ),

        "OSF": joblib.load(
            MODEL_DIR / "diagnosis_osf.joblib"
        )
    }


models = load_models()

# ---------------- INPUT ----------------

st.subheader("Machine Operating Conditions")

c1, c2, c3 = st.columns(3)

with c1:

    machine_type = st.selectbox(
        "Product Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        250.0,
        350.0,
        300.0
    )

with c2:

    process_temp = st.number_input(
        "Process Temperature [K]",
        250.0,
        400.0,
        310.0
    )

    speed = st.number_input(
        "Rotational Speed [rpm]",
        0.0,
        5000.0,
        1500.0
    )

with c3:

    torque = st.number_input(
        "Torque [Nm]",
        0.0,
        200.0,
        40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        0.0,
        500.0,
        100.0
    )


input_df = pd.DataFrame({

    "Type": [machine_type],

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

# ---------------- DIAGNOSIS ----------------

if st.button(
    "🔍 Diagnose Machine",
    type="primary",
    use_container_width=True
):

    try:

        results = {}

        for fault, model in models.items():

            probability = (
                model.predict_proba(
                    input_df
                )[0][1]
            )

            results[fault] = probability


        st.subheader("Fault Diagnosis Results")

        c1, c2, c3 = st.columns(3)

        fault_names = {
            "HDF": "Heat Dissipation",
            "PWF": "Power Failure",
            "OSF": "Overstrain"
        }

        columns = [c1, c2, c3]

        for column, fault in zip(
            columns,
            ["HDF", "PWF", "OSF"]
        ):

            probability = results[fault]

            with column:

                st.metric(
                    fault,
                    f"{probability * 100:.2f}%"
                )

                st.caption(
                    fault_names[fault]
                )


        highest_fault = max(
            results,
            key=results.get
        )

        highest_probability = (
            results[highest_fault]
        )

        st.divider()

        st.write(
            "**Highest detected fault probability:**",
            fault_names[highest_fault],
            f"({highest_probability * 100:.2f}%)"
        )

        if highest_probability >= 0.50:

            st.error(
                "⚠️ Significant fault pattern detected. "
                "Machine inspection is recommended."
            )

        else:

            st.success(
                "✅ No supported fault currently exceeds "
                "the 50% display threshold."
            )


        st.info(
            "TWF is not included as a reliable diagnosis because "
            "its training data were extremely limited. RNF is also "
            "excluded because there were insufficient positive samples."
        )

    except Exception as e:

        st.error(
            f"Diagnosis error: {e}"
        )