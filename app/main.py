import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page width */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    /* Main title */
    .main-title {
        font-size: 40px;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 17px;
        color: #667085;
        line-height: 1.6;
        margin-bottom: 22px;
    }

    /* Green system status */
    .status-box {
        border-radius: 10px;
        padding: 15px 20px;
        background-color: #e9f8ef;
        color: #137333;
        font-size: 16px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* Module cards */
    .module-card {
        border: 1px solid #e2e6ec;
        border-radius: 12px;
        padding: 22px;
        min-height: 190px;
        margin-bottom: 8px;
    }

    .module-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .module-text {
        font-size: 15px;
        line-height: 1.6;
        color: #667085;
    }

    /* Workflow box */
    .workflow-box {
        border: 1px solid #e2e6ec;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        font-size: 16px;
        line-height: 1.8;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8a8f98;
        font-size: 13px;
        margin-top: 30px;
        padding-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        ⚙️ Predictive Maintenance & Failure Forecasting System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Machine-learning based decision-support system for machine
        condition monitoring, failure prediction, fault diagnosis,
        Remaining Useful Life estimation and anomaly detection.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    """
    <div class="status-box">
        ✓ System Ready — Failure Prediction, Fault Diagnosis,
        RUL Prediction and Anomaly Detection modules are available.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.subheader("System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Core ML Modules",
        value="4"
    )

with col2:
    st.metric(
        label="AI4I Dataset",
        value="10,000 rows"
    )

with col3:
    st.metric(
        label="RUL Dataset",
        value="NASA FD001"
    )

with col4:
    st.metric(
        label="Application",
        value="Streamlit"
    )

st.divider()


# ============================================================
# PREDICTIVE MAINTENANCE MODULES
# ============================================================

st.subheader("Predictive Maintenance Modules")

st.write(
    "Select a module below to perform machine-health analysis."
)


# ============================================================
# ROW 1
# FAILURE PREDICTION + FAULT DIAGNOSIS
# ============================================================

left, right = st.columns(2)


# ---------------- FAILURE PREDICTION ----------------

with left:

    st.markdown(
        """
        <div class="module-card">

            <div class="module-title">
                🔴 Failure Prediction
            </div>

            <div class="module-text">
                Estimates machine-failure probability from current
                operating conditions using the trained XGBoost
                classifier.
                <br><br>
                Inputs include product type, temperature,
                rotational speed, torque and tool wear.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Open Failure Prediction →",
        width="stretch",
        key="failure_button"
    ):

        st.switch_page(
            "pages/1_Failure_Prediction.py"
        )


# ---------------- FAULT DIAGNOSIS ----------------

with right:

    st.markdown(
        """
        <div class="module-card">

            <div class="module-title">
                🛠️ Fault Diagnosis
            </div>

            <div class="module-text">
                Diagnoses supported machine-failure mechanisms
                using dedicated trained machine-learning models.
                <br><br>
                Supported diagnosis includes HDF, PWF and OSF
                failure mechanisms.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Open Fault Diagnosis →",
        width="stretch",
        key="diagnosis_button"
    ):

        st.switch_page(
            "pages/2_Fault_Diagnosis.py"
        )


st.write("")


# ============================================================
# ROW 2
# RUL + ANOMALY DETECTION
# ============================================================

left2, right2 = st.columns(2)


# ---------------- RUL PREDICTION ----------------

with left2:

    st.markdown(
        """
        <div class="module-card">

            <div class="module-title">
                ⏳ Remaining Useful Life
            </div>

            <div class="module-text">
                Estimates the number of operating cycles remaining
                before the end of useful engine life.
                <br><br>
                Uses NASA C-MAPSS FD001 data, XGBoost regression
                and 5-cycle rolling sensor statistics.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Open RUL Prediction →",
        width="stretch",
        key="rul_button"
    ):

        st.switch_page(
            "pages/3_RUL_Prediction.py"
        )


# ---------------- ANOMALY DETECTION ----------------

with right2:

    st.markdown(
        """
        <div class="module-card">

            <div class="module-title">
                🔍 Anomaly Detection
            </div>

            <div class="module-text">
                Identifies unusual machine operating conditions
                using an unsupervised Isolation Forest model.
                <br><br>
                This complements supervised failure prediction by
                detecting abnormal operating patterns.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Open Anomaly Detection →",
        width="stretch",
        key="anomaly_button"
    ):

        st.switch_page(
            "pages/4_Anomaly_Detection.py"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("Validated Model Performance")

p1, p2, p3, p4 = st.columns(4)


with p1:

    st.metric(
        label="Failure Recall",
        value="82%"
    )


with p2:

    st.metric(
        label="Failure F1 Score",
        value="0.70"
    )


with p3:

    st.metric(
        label="RUL Test MAE",
        value="12.86 cycles"
    )


with p4:

    st.metric(
        label="RUL Test R²",
        value="0.805"
    )


st.caption(
    "Failure metrics correspond to the selected recall-oriented "
    "operating threshold. RUL metrics correspond to the final "
    "NASA FD001 capped-target test evaluation."
)


# ============================================================
# SYSTEM WORKFLOW
# ============================================================

st.divider()

st.subheader("System Workflow")

st.markdown(
    """
    <div class="workflow-box">

        <b>Machine / Sensor Data</b>

        <br>
        ↓
        <br>

        <b>Data Preprocessing & Feature Engineering</b>

        <br>
        ↓
        <br>

        <b>Machine Learning Models</b>

        <br>
        ↓
        <br>

        <b>
        Failure Risk &nbsp; • &nbsp;
        Fault Diagnosis &nbsp; • &nbsp;
        RUL Forecast &nbsp; • &nbsp;
        Anomaly Status
        </b>

        <br>
        ↓
        <br>

        <b>Maintenance Decision Support</b>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASETS
# ============================================================

st.divider()

st.subheader("Datasets")

dataset1, dataset2 = st.columns(2)


with dataset1:

    st.markdown(
        """
        ### AI4I 2020 Predictive Maintenance

        **Used for:**

        - Machine failure prediction
        - Fault diagnosis
        - Anomaly detection

        The dataset contains **10,000 machine observations**.

        Important operating variables include:

        - Product type
        - Air temperature
        - Process temperature
        - Rotational speed
        - Torque
        - Tool wear
        """
    )


with dataset2:

    st.markdown(
        """
        ### NASA C-MAPSS FD001

        **Used for:**

        - Remaining Useful Life prediction
        - Engine sensor trend analysis

        The dataset contains run-to-failure turbofan engine
        measurements collected over multiple operating cycles.

        The RUL model uses current sensor measurements together
        with rolling sensor statistics.
        """
    )


# ============================================================
# PROJECT OBJECTIVE
# ============================================================

st.divider()

st.subheader("Project Objective")

st.write(
    """
    The objective of this project is to develop an end-to-end
    predictive-maintenance and failure-forecasting system capable
    of analyzing machine operating conditions and generating
    multiple maintenance-support outputs.

    Instead of relying on a single prediction, the system combines
    supervised classification, fault-specific diagnosis,
    regression-based Remaining Useful Life estimation and
    unsupervised anomaly detection.
    """
)


# ============================================================
# CAPABILITIES
# ============================================================

st.subheader("System Capabilities")

cap1, cap2 = st.columns(2)


with cap1:

    st.markdown(
        """
        **Machine Condition Analysis**

        - Failure probability estimation
        - Risk classification
        - Fault-specific diagnosis
        - Abnormal-condition detection
        """
    )


with cap2:

    st.markdown(
        """
        **Maintenance Forecasting**

        - Remaining Useful Life estimation
        - Sensor-history visualization
        - Near-failure identification
        - Maintenance decision support
        """
    )


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

st.info(
    """
    This application is an academic machine-learning
    decision-support prototype. Model predictions should be used
    together with engineering inspection and should not be
    interpreted as guaranteed equipment failure dates.
    """
)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.divider()

st.subheader("Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)


with tech1:

    st.markdown(
        """
        **Programming**

        Python
        """
    )


with tech2:

    st.markdown(
        """
        **Machine Learning**

        Scikit-learn  
        XGBoost
        """
    )


with tech3:

    st.markdown(
        """
        **Data Analysis**

        Pandas  
        NumPy  
        SHAP
        """
    )


with tech4:

    st.markdown(
        """
        **Application**

        Streamlit  
        Plotly  
        Joblib
        """
    )


# ============================================================
# PROJECT STRUCTURE INFORMATION
# ============================================================

st.divider()

with st.expander("Project Architecture"):

    st.code(
        """
HCL_PROJECT/
│
├── app/
│   ├── main.py
│   ├── utils.py
│   │
│   └── pages/
│       ├── 1_Failure_Prediction.py
│       ├── 2_Fault_Diagnosis.py
│       ├── 3_RUL_Prediction.py
│       └── 4_Anomaly_Detection.py
│
├── data/
│   ├── processed/
│   │
│   └── raw/
│       ├── ai4i2020.csv
│       │
│       └── cmapss/
│           ├── train_FD001.txt
│           ├── test_FD001.txt
│           └── RUL_FD001.txt
│
├── models/
│   ├── failure_xgboost.joblib
│   ├── failure_model_config.json
│   ├── diagnosis_hdf.joblib
│   ├── diagnosis_pwf.joblib
│   ├── diagnosis_osf.joblib
│   ├── diagnosis_config.json
│   ├── rul_xgboost.joblib
│   ├── anomaly_isolation_forest.joblib
│   ├── anomaly_scaler.joblib
│   └── anomaly_config.json
│
├── notebooks/
├── src/
└── requirements.txt
        """,
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Predictive Maintenance and Failure Forecasting System
        <br>
        Machine Learning Based Maintenance Decision Support
    </div>
    """,
    unsafe_allow_html=True
)