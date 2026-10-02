# Predictive Maintenance and Failure Forecasting System

An end-to-end machine-learning based predictive maintenance system for **machine failure prediction, fault diagnosis, Remaining Useful Life (RUL) forecasting, and anomaly detection**.

The project combines supervised classification, regression, unsupervised anomaly detection, feature engineering, model evaluation, and an interactive **Streamlit dashboard** to provide maintenance decision support.

---

## 1. Project Objective

Traditional reactive maintenance repairs equipment only after a failure occurs, which can lead to unexpected downtime, production losses, and increased maintenance costs.

This project develops a predictive-maintenance system capable of analyzing machine and sensor data before maintenance decisions are made.

The system provides four major capabilities:

1. Machine Failure Prediction
2. Fault Diagnosis
3. Remaining Useful Life Prediction
4. Anomaly Detection

The final models are integrated into an interactive Streamlit application.

---

## 2. System Architecture

```text
Machine / Sensor Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
┌───────────────────────────────┐
│ Machine Learning Layer        │
│                               │
│ • Failure Prediction          │
│ • Fault Diagnosis             │
│ • RUL Prediction              │
│ • Anomaly Detection           │
└───────────────────────────────┘
        │
        ▼
Machine Health Information
        │
        ▼
Maintenance Decision Support
```

---

## 3. Main Modules

### 3.1 Machine Failure Prediction

The first module predicts whether the current machine operating condition indicates an elevated risk of machine failure.

Important input variables include:

- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Multiple approaches were evaluated, including:

- Dummy Classifier
- Logistic Regression
- Random Forest
- XGBoost

The final implementation uses a tuned **XGBoost classifier**.

Because machine failures are rare compared with normal observations, accuracy alone was not used as the primary evaluation criterion.

Important metrics included:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

A recall-oriented decision threshold was selected for the deployed failure-detection configuration.

Final failure-class performance was approximately:

| Metric | Result |
|---|---:|
| Precision | 0.61 |
| Recall | 0.82 |
| F1 Score | 0.70 |
| Overall Accuracy | 0.98 |

---

## 4. Fault Diagnosis

After detecting potential machine failure, the system can analyze supported failure mechanisms.

The AI4I dataset contains the following fault labels:

- TWF — Tool Wear Failure
- HDF — Heat Dissipation Failure
- PWF — Power Failure
- OSF — Overstrain Failure
- RNF — Random Failure

Due to the very small number of positive examples for some fault categories, the final interactive diagnosis system focuses on:

- HDF
- PWF
- OSF

### Fault Diagnosis Test Results

| Fault | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| HDF | 0.9545 | 0.9130 | 0.9333 | 0.9941 | 0.9448 |
| PWF | 0.6429 | 0.9474 | 0.7660 | 0.9984 | 0.8269 |
| OSF | 0.7727 | 0.8500 | 0.8095 | 0.9987 | 0.9255 |

TWF and RNF were not presented as reliable final diagnosis outputs because of their limited positive training samples and weak validated performance.

---

## 5. Remaining Useful Life Prediction

The RUL module estimates how many operating cycles may remain before the end of useful engine life.

### Dataset

**NASA C-MAPSS FD001**

The dataset contains turbofan engine degradation trajectories with:

- Engine ID
- Operating Cycle
- Operational Settings
- 21 Sensor Measurements

For each training engine:

```text
RUL = Maximum Engine Cycle - Current Cycle
```

### Feature Engineering

Constant features were removed.

The model uses informative sensor measurements together with **5-cycle rolling statistics**, including rolling mean and rolling standard deviation.

Example engineered features:

```text
sensor_2_roll_mean_5
sensor_2_roll_std_5
sensor_3_roll_mean_5
sensor_3_roll_std_5
...
```

### Models Evaluated

- Dummy Regressor
- Random Forest Regressor
- XGBoost Regressor
- Tuned XGBoost Regressor

### Final NASA FD001 Test Performance

#### Capped RUL Target

| Metric | Result |
|---|---:|
| MAE | 12.8588 cycles |
| RMSE | 17.6897 cycles |
| R² | 0.8051 |

#### Original RUL Target

| Metric | Result |
|---|---:|
| MAE | 13.9288 cycles |
| RMSE | 18.8852 cycles |
| R² | 0.7935 |

The application also displays recent engine sensor measurements and a Sensor 11 operating trend.

---

## 6. Anomaly Detection

The project includes an unsupervised anomaly-detection module using:

**Isolation Forest**

The purpose of this module is to identify operating conditions that differ from the patterns learned from machine data.

The anomaly detector is used as an auxiliary monitoring mechanism rather than as a replacement for supervised failure prediction.

During evaluation, the anomaly score achieved approximately:

```text
ROC-AUC = 0.8202
PR-AUC  = 0.1462
```

The positive failure baseline was approximately:

```text
0.0339
```

This demonstrates that the anomaly score contains useful failure-related information, although the supervised XGBoost classifier remains substantially more appropriate for direct failure prediction.

---

## 7. Datasets

### AI4I 2020 Predictive Maintenance Dataset

Used for:

- Failure Prediction
- Fault Diagnosis
- Anomaly Detection

Dataset size:

```text
10,000 rows
14 columns
```

Machine-failure distribution:

```text
Normal  = 9661
Failure = 339
```

Failure rate:

```text
3.39%
```

The severe class imbalance was explicitly considered during model selection and evaluation.

### NASA C-MAPSS FD001

Used for:

- Remaining Useful Life Prediction
- Engine degradation analysis
- Sensor trend analysis

The training data contains run-to-failure histories, while the test data contains partially observed engine histories.

---

## 8. Technology Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Explainability and Analysis

- SHAP

### Visualization

- Matplotlib
- Plotly
- Streamlit

### Model Persistence

- Joblib

### Development Environment

- Visual Studio Code
- Jupyter Notebook
- Python Virtual Environment

---

## 9. Project Structure

```text
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
├── requirements.txt
└── README.md
```

---

## 10. Installation

### Step 1 — Open the Project

```powershell
cd "D:\COLLEGE WORK\HCL_PROJECT"
```

### Step 2 — Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 3 — Activate It

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(.venv) PS D:\COLLEGE WORK\HCL_PROJECT>
```

### Step 4 — Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 11. Running the Application

From the project root:

```powershell
streamlit run app/main.py
```

Streamlit will provide a local address, normally:

```text
http://localhost:8501
```

Open it in a web browser.

---

## 12. Using the Application

### Failure Prediction

Enter:

- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Click:

```text
Analyze Machine Condition
```

The system displays failure probability and machine risk status.

### Fault Diagnosis

Enter current machine operating conditions and run the diagnosis.

The system displays probabilities for:

```text
HDF
PWF
OSF
```

### RUL Prediction

Upload:

```text
data/raw/cmapss/test_FD001.txt
```

Select an Engine ID and click:

```text
Predict Remaining Useful Life
```

The application displays:

- Engine ID
- Current Cycle
- Predicted RUL
- Maintenance interpretation
- Recent sensor history
- Sensor 11 operating trend

### Anomaly Detection

Enter machine sensor measurements and run anomaly detection.

The application reports whether the current operating condition is classified as:

```text
NORMAL
```

or:

```text
ANOMALY
```

---

## 13. Machine-Learning Workflow

The complete development workflow was:

```text
Problem Definition
        ↓
Dataset Collection
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Class-Imbalance Analysis
        ↓
Train/Test Splitting
        ↓
Baseline Models
        ↓
Advanced Models
        ↓
Cross-Validation
        ↓
Hyperparameter Tuning
        ↓
Threshold Optimization
        ↓
Model Evaluation
        ↓
Feature Engineering
        ↓
Model Persistence
        ↓
Streamlit Integration
        ↓
Final Predictive Maintenance Application
```

---

## 14. Key Challenges

### Class Imbalance

Only approximately 3.39% of AI4I observations represent machine failures.

Therefore, high accuracy alone could be misleading.

Precision, recall, F1 and PR-AUC were emphasized during evaluation.

### Rare Fault Categories

Some failure mechanisms contained very few positive observations.

This made reliable modeling difficult for TWF and especially RNF.

The final diagnosis interface therefore focuses on the fault categories with stronger validated performance.

### RUL Feature Engineering

Raw sensor readings alone do not completely represent degradation history.

Rolling sensor statistics were therefore introduced to capture short-term temporal behavior.

### Anomaly Detection

Anomalies are not necessarily failures.

The Isolation Forest module is therefore interpreted as an auxiliary abnormal-condition detector rather than a direct failure classifier.

---

## 15. Advantages

- Early machine-failure risk identification
- Fault-specific diagnosis
- Remaining-life forecasting
- Unsupervised abnormal-condition detection
- Multiple ML techniques in one system
- Interactive web-based dashboard
- Saved reusable models
- Maintenance-oriented outputs
- Modular architecture

---

## 16. Limitations

- AI4I is a synthetic predictive-maintenance dataset.
- Some fault categories contain very few positive observations.
- NASA C-MAPSS represents simulated turbofan degradation.
- RUL predictions contain unavoidable prediction error.
- Anomalies do not automatically represent failures.
- The system is an academic prototype rather than a production industrial monitoring platform.

---

## 17. Future Scope

Possible future extensions include:

- Real-time IoT sensor integration
- Streaming machine-health monitoring
- Automated maintenance alerts
- Cloud deployment
- Historical machine-health database
- Maintenance scheduling
- Additional industrial datasets
- Deep-learning based time-series models
- LSTM/GRU/Transformer-based RUL forecasting
- Fleet-level monitoring
- Automated reporting
- Integration with industrial maintenance systems

---

## 18. Conclusion

This project demonstrates a complete machine-learning workflow for predictive maintenance and failure forecasting.

Rather than relying on a single model, the system combines:

```text
Failure Prediction
        +
Fault Diagnosis
        +
Remaining Useful Life Prediction
        +
Anomaly Detection
```

The final Streamlit application converts these trained machine-learning models into an interactive maintenance decision-support system.

The project demonstrates how machine-learning techniques can be applied to machine-condition data to support earlier failure detection, fault identification, degradation forecasting, and maintenance planning.

---

## Disclaimer

This project is developed for academic and experimental purposes.

Model predictions are estimates based on the available datasets and should not replace professional engineering inspection, safety procedures, or manufacturer maintenance requirements.