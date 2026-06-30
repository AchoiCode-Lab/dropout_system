# EduPredict — Student Dropout Prediction System
## FYP: An Explainable Machine Learning Model for Early Prediction of Student Dropout in Secondary Schools

**Student:** Muhammad Harith Irfan bin Yahya (2023663582)  
**Supervisor:** Madam Nor Fadilah Binti Tahar @ Yusoff  
**University:** Universiti Teknologi MARA (UiTM)  
**Program:** Bachelor of Computer Science (Hons.)  

---

## System Overview

This is a **complete web-based student dropout prediction system** using:
- **Random Forest** classifier for dropout risk prediction
- **SHAP (SHapley Additive exPlanations)** for explainable AI
- **Flask** web framework with SQLite database
- **Plotly** interactive dashboards
- **3,000+ student records** (649 original + synthesized data)

## System Features

| Page | Description |
|------|-------------|
| Login | User authentication with username/password |
| Register | Account creation for teachers/counselors/admins |
| Dashboard | Interactive Plotly charts: pie, box plot, bar, bubble, heatmap, gauges |
| Student Records | Browse, search, filter all 3,000 students |
| New Prediction | Manual data entry form for dropout risk prediction |
| Prediction Result | Risk gauge, SHAP explanation, suggested interventions |
| Student Detail | Individual analysis with radar chart and SHAP breakdown |
| SHAP Explanation | Global feature importance, category contribution |
| Reports | Model evaluation (accuracy, precision, recall, F1, confusion matrix) |

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 98.47% |
| Precision | 95.74% |
| Recall | 94.74% |
| F1-Score | 95.24% |
| ROC-AUC | 99.87% |

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (if models/ folder is empty)
python train_model.py

# 3. Run the web application
python app.py
```

### Access the System
- Open browser: **http://localhost:5000**
- Login credentials:
  - Username: `admin`
  - Password: `admin123`

## Project Structure

```
dropout_system/
├── app.py                          # Main Flask application
├── train_model.py                  # Model training pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── student_dropout_original.csv    # Original 649-row dataset
│   ├── student_dropout_3000.csv        # Expanded 3000-row dataset
│   ├── student_dropout_processed.csv   # Encoded dataset
│   ├── synthesize_data.py              # Data synthesis script
│   └── preprocessing_3000.ipynb        # Complete preprocessing notebook
├── models/
│   ├── rf_model.pkl                # Trained Random Forest model
│   ├── scaler.pkl                  # StandardScaler
│   ├── label_encoders.pkl          # Label encoders
│   ├── shap_explainer.pkl          # SHAP TreeExplainer
│   ├── feature_names.pkl           # Feature column names
│   ├── numerical_cols.pkl          # Numerical columns list
│   └── metrics.json                # Evaluation metrics
├── templates/
│   ├── base.html                   # Base template with sidebar
│   ├── login.html                  # Login page
│   ├── register.html               # Registration page
│   ├── dashboard.html              # Dashboard with Plotly charts
│   ├── students.html               # Student records list
│   ├── predict.html                # Prediction input form
│   ├── prediction_result.html      # Prediction result + SHAP
│   ├── student_detail.html         # Individual student analysis
│   ├── explanation.html            # SHAP explanation page
│   └── report.html                 # Model evaluation report
└── static/
    ├── css/
    ├── js/
    └── img/
```

## Dataset Details

- **Source:** Kaggle Student Dropout Analysis and Prediction Dataset
- **Original:** 649 records, 34 features
- **Expanded:** 3,000 records (synthesized using statistical noise injection)
- **Features:** Academic (9), Attendance (1), Behavioral (7), Demographic (16)
- **Target:** Dropped_Out (True/False → At-Risk/Not At-Risk)

## Preprocessing Pipeline

1. Data loading (3,000 records)
2. Missing values check (0 found)
3. Duplicate removal
4. Target analysis (83.9% Not At-Risk, 16.1% At-Risk)
5. Feature categorization (3 main categories + demographic)
6. Label encoding (13 binary features)
7. One-hot encoding (4 nominal features)
8. StandardScaler on 16 numerical features only
9. 80/20 train-test split (stratified)
10. Random Forest training (200 trees, balanced weights)
11. SHAP TreeExplainer computation

## Supervisor Feedback Addressed

1. ✅ **Dataset expanded to 3,000+** records (from 649)
2. ✅ **Feature scaling explanation** — why StandardScaler transforms data (mean=0, std=1) and why only 16 numerical features are scaled (not binary/one-hot)
3. ✅ **Heatmap explanation** — what correlation values mean, color interpretation, key observations
4. ✅ **Box plot explanation** — what each component shows (median, Q1, Q3, whiskers, outliers), observations per feature

---

