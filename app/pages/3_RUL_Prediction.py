import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RUL Prediction",
    page_icon="⏳",
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


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("⏳ Remaining Useful Life Prediction")

st.write(
    "Estimate the Remaining Useful Life (RUL) of turbofan engines "
    "using the trained NASA C-MAPSS FD001 XGBoost model."
)

st.divider()


# ============================================================
# LOAD SAVED RUL MODEL
# ============================================================

@st.cache_resource
def load_rul_model():

    model_path = MODEL_DIR / "rul_xgboost.joblib"

    if not model_path.exists():

        st.error(
            f"RUL model not found: {model_path}"
        )

        st.stop()

    return joblib.load(model_path)


model = load_rul_model()


# ============================================================
# NASA C-MAPSS COLUMN NAMES
# ============================================================

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("Engine Dataset")

uploaded_file = st.file_uploader(
    "Upload NASA C-MAPSS FD001 engine data",
    type=["txt"]
)

st.caption(
    "Use test_FD001.txt from the data/raw/cmapss folder."
)


# ============================================================
# PROCESS DATASET
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ DATASET
        # ----------------------------------------------------

        df = pd.read_csv(
            uploaded_file,
            sep=r"\s+",
            header=None
        )

        # NASA FD001 contains exactly 26 required columns
        df = df.iloc[:, :26]

        if df.shape[1] != 26:

            st.error(
                f"Invalid NASA C-MAPSS file. "
                f"Expected 26 columns but found {df.shape[1]}."
            )

            st.stop()


        df.columns = columns


        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        number_engines = df["engine_id"].nunique()

        st.success(
            f"Dataset loaded successfully — "
            f"{len(df):,} observations and "
            f"{number_engines} engines detected."
        )


        # ----------------------------------------------------
        # ENGINE SELECTION
        # ----------------------------------------------------

        engine_ids = sorted(
            df["engine_id"].unique()
        )

        selected_engine = st.selectbox(
            "Select Engine ID",
            engine_ids
        )


        # ----------------------------------------------------
        # SELECT ENGINE HISTORY
        # ----------------------------------------------------

        engine_df = (
            df[
                df["engine_id"] == selected_engine
            ]
            .copy()
            .sort_values("cycle")
            .reset_index(drop=True)
        )


        latest_cycle = int(
            engine_df["cycle"].max()
        )


        # ----------------------------------------------------
        # ENGINE INFORMATION
        # ----------------------------------------------------

        info1, info2 = st.columns(2)

        with info1:

            st.metric(
                "Selected Engine",
                selected_engine
            )

        with info2:

            st.metric(
                "Latest Observed Cycle",
                latest_cycle
            )


        st.divider()


        # ====================================================
        # FEATURES USED FOR ROLLING STATISTICS
        # ====================================================

        rolling_features = [

            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_6",
            "sensor_7",
            "sensor_8",
            "sensor_9",
            "sensor_11",
            "sensor_12",
            "sensor_13",
            "sensor_14",
            "sensor_15",
            "sensor_17",
            "sensor_20",
            "sensor_21"
        ]


        # ====================================================
        # CREATE EXACT TRAINING ROLLING FEATURES
        # ====================================================

        rolling_window = 5


        for feature in rolling_features:

            # ------------------------------
            # 5-CYCLE ROLLING MEAN
            # ------------------------------

            engine_df[
                f"{feature}_roll_mean_5"
            ] = (
                engine_df[feature]
                .rolling(
                    window=rolling_window,
                    min_periods=1
                )
                .mean()
            )


            # ------------------------------
            # 5-CYCLE ROLLING STD
            # ------------------------------

            engine_df[
                f"{feature}_roll_std_5"
            ] = (
                engine_df[feature]
                .rolling(
                    window=rolling_window,
                    min_periods=1
                )
                .std()
                .fillna(0)
            )


        # ====================================================
        # GET LATEST ENGINE CONDITION
        # ====================================================

        latest_record = (
            engine_df
            .iloc[[-1]]
            .copy()
        )


        # ====================================================
        # GET EXACT MODEL FEATURES
        # ====================================================

        if hasattr(
            model,
            "feature_names_in_"
        ):

            required_features = list(
                model.feature_names_in_
            )

        else:

            st.error(
                "The saved RUL model does not contain "
                "feature_names_in_. Exact training features "
                "cannot be verified."
            )

            st.stop()


        # ====================================================
        # CHECK FOR MISSING FEATURES
        # ====================================================

        missing_features = [

            feature

            for feature in required_features

            if feature not in latest_record.columns
        ]


        if missing_features:

            st.error(
                "The application cannot reproduce some "
                "features used during RUL model training."
            )

            st.write(
                "**Missing features:**"
            )

            st.code(
                "\n".join(
                    missing_features
                )
            )

            st.stop()


        # ====================================================
        # BUILD MODEL INPUT
        # ====================================================

        X_latest = latest_record[
            required_features
        ].copy()


        # Convert everything to numeric
        X_latest = X_latest.apply(
            pd.to_numeric,
            errors="coerce"
        )


        # ----------------------------------------------------
        # CHECK FOR NaN VALUES
        # ----------------------------------------------------

        if X_latest.isnull().any().any():

            st.error(
                "NaN values were found after feature "
                "engineering. Prediction stopped."
            )

            st.stop()


        # ====================================================
        # PREDICTION BUTTON
        # ====================================================

        if st.button(
            "🔍 Predict Remaining Useful Life",
            type="primary",
            width="stretch"
        ):

            # ------------------------------------------------
            # MAKE RUL PREDICTION
            # ------------------------------------------------

            predicted_rul = float(
                model.predict(
                    X_latest
                )[0]
            )


            # RUL cannot logically be negative
            predicted_rul = max(
                0.0,
                predicted_rul
            )


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader(
                "RUL Forecast"
            )


            result1, result2, result3 = st.columns(3)


            with result1:

                st.metric(
                    "Engine ID",
                    selected_engine
                )


            with result2:

                st.metric(
                    "Current Cycle",
                    latest_cycle
                )


            with result3:

                st.metric(
                    "Predicted RUL",
                    f"{predicted_rul:.1f} cycles"
                )


            # =================================================
            # MAINTENANCE INTERPRETATION
            # =================================================

            if predicted_rul <= 30:

                st.error(
                    "🔴 CRITICAL CONDITION — "
                    "Predicted RUL is 30 cycles or less. "
                    "Maintenance should be prioritized."
                )


            elif predicted_rul <= 60:

                st.warning(
                    "🟠 WARNING — "
                    "The engine is approaching the "
                    "near-failure operating region."
                )


            else:

                st.success(
                    "🟢 STABLE — "
                    "Predicted RUL is currently above "
                    "the near-failure threshold."
                )


            st.info(
                "Remaining Useful Life is an ML estimate "
                "of the number of operating cycles remaining. "
                "It should be used as maintenance decision "
                "support rather than an exact failure date."
            )


            # =================================================
            # RECENT SENSOR HISTORY
            # =================================================

            st.subheader(
                "Recent Engine Sensor History"
            )


            display_columns = [

                "cycle",

                "sensor_2",
                "sensor_3",
                "sensor_4",

                "sensor_7",
                "sensor_11",
                "sensor_12",

                "sensor_15",
                "sensor_20",
                "sensor_21"
            ]


            st.dataframe(
                engine_df[
                    display_columns
                ].tail(10),

                width="stretch",

                hide_index=True
            )


            # =================================================
            # SENSOR 11 TREND
            # ============================================================

            st.subheader(
                "Sensor 11 Operating Trend"
            )


            chart_data = (
                engine_df[
                    [
                        "cycle",
                        "sensor_11"
                    ]
                ]
                .set_index(
                    "cycle"
                )
            )


            st.line_chart(
                chart_data,
                width="stretch"
            )


            # =================================================
            # MODEL INFORMATION
            # =================================================

            with st.expander(
                "Model Information"
            ):

                st.write(
                    "**Dataset:** NASA C-MAPSS FD001"
                )

                st.write(
                    "**Model:** Tuned XGBoost Regressor"
                )

                st.write(
                    "**Feature Engineering:** "
                    "Current sensor readings + "
                    "5-cycle rolling statistics"
                )

                st.write(
                    "**Final NASA FD001 Test Performance:**"
                )


                performance = pd.DataFrame({

                    "Metric": [
                        "MAE",
                        "RMSE",
                        "R²"
                    ],

                    "Capped RUL": [
                        12.8588,
                        17.6897,
                        0.8051
                    ],

                    "Original RUL": [
                        13.9288,
                        18.8852,
                        0.7935
                    ]

                })


                st.dataframe(
                    performance,
                    width="stretch",
                    hide_index=True
                )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            f"RUL prediction error: {e}"
        )