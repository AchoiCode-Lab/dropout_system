"""
Complete Preprocessing + Model Training Pipeline
Student Dropout Prediction System - FYP
"""
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)
import shap

print("="*60)
print("STUDENT DROPOUT PREDICTION - MODEL TRAINING PIPELINE")
print("="*60)

# ============================================================
# 1. LOAD THE 3000-ROW DATASET
# ============================================================
df = pd.read_csv('data/student_dropout_3000.csv')
print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ============================================================
# 2. DATA INSPECTION
# ============================================================
print(f"\n[2] Data Inspection:")
print(f"    Missing values: {df.isnull().sum().sum()}")
print(f"    Duplicates: {df.duplicated().sum()}")

# Remove duplicates if any
df = df.drop_duplicates().reset_index(drop=True)
print(f"    After cleaning: {df.shape[0]} rows")

# ============================================================
# 3. TARGET VARIABLE
# ============================================================
target_counts = df['Dropped_Out'].value_counts()
print(f"\n[3] Target Distribution:")
print(f"    Not At-Risk: {target_counts.get(False, 0)} ({target_counts.get(False, 0)/len(df)*100:.1f}%)")
print(f"    At-Risk:     {target_counts.get(True, 0)} ({target_counts.get(True, 0)/len(df)*100:.1f}%)")

# ============================================================
# 4. FEATURE CATEGORIZATION
# ============================================================
academic_features = ['Grade_1', 'Grade_2', 'Final_Grade', 'Number_of_Failures',
                     'Study_Time', 'School_Support', 'Family_Support',
                     'Extra_Paid_Class', 'Wants_Higher_Education']

attendance_features = ['Number_of_Absences']

behavioral_features = ['Extra_Curricular_Activities', 'Free_Time', 'Going_Out',
                       'Weekday_Alcohol_Consumption', 'Weekend_Alcohol_Consumption',
                       'Health_Status', 'In_Relationship']

demographic_features = ['School', 'Gender', 'Age', 'Address', 'Family_Size',
                        'Parental_Status', 'Mother_Education', 'Father_Education',
                        'Mother_Job', 'Father_Job', 'Reason_for_Choosing_School',
                        'Guardian', 'Travel_Time', 'Family_Relationship',
                        'Attended_Nursery', 'Internet_Access']

print(f"\n[4] Feature Categories:")
print(f"    Academic:    {len(academic_features)} features")
print(f"    Attendance:  {len(attendance_features)} features")
print(f"    Behavioral:  {len(behavioral_features)} features")
print(f"    Demographic: {len(demographic_features)} features")

# ============================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================
df_processed = df.copy()

binary_cols = ['School', 'Gender', 'Address', 'Family_Size', 'Parental_Status',
               'School_Support', 'Family_Support', 'Extra_Paid_Class',
               'Extra_Curricular_Activities', 'Attended_Nursery',
               'Wants_Higher_Education', 'Internet_Access', 'In_Relationship']

label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

nominal_cols = ['Mother_Job', 'Father_Job', 'Reason_for_Choosing_School', 'Guardian']
df_processed = pd.get_dummies(df_processed, columns=nominal_cols, drop_first=True)

dummy_cols = [c for c in df_processed.columns if any(c.startswith(n + '_') for n in nominal_cols)]
df_processed[dummy_cols] = df_processed[dummy_cols].astype(int)
df_processed['Dropped_Out'] = df_processed['Dropped_Out'].map({True: 1, False: 0, 'True': 1, 'False': 0})
df_processed['Dropped_Out'] = df_processed['Dropped_Out'].astype(int)

print(f"\n[5] After encoding: {df_processed.shape[1]} columns")

# ============================================================
# 6. FEATURE SCALING (StandardScaler on numerical features only)
# ============================================================
X = df_processed.drop('Dropped_Out', axis=1)
y = df_processed['Dropped_Out']

numerical_to_scale = ['Age', 'Mother_Education', 'Father_Education', 'Travel_Time',
                      'Study_Time', 'Number_of_Failures', 'Family_Relationship',
                      'Free_Time', 'Going_Out', 'Weekend_Alcohol_Consumption',
                      'Weekday_Alcohol_Consumption', 'Health_Status',
                      'Number_of_Absences', 'Grade_1', 'Grade_2', 'Final_Grade']

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_to_scale] = scaler.fit_transform(X[numerical_to_scale])

print(f"\n[6] Feature Scaling applied to {len(numerical_to_scale)} numerical features")
print(f"    Feature matrix: {X_scaled.shape}")

# ============================================================
# 7. TRAIN-TEST SPLIT (80/20, stratified)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n[7] Train-Test Split:")
print(f"    Training: {X_train.shape[0]} samples")
print(f"    Testing:  {X_test.shape[0]} samples")

# ============================================================
# 8. RANDOM FOREST MODEL TRAINING
# ============================================================
print(f"\n[8] Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("    Model trained successfully!")

# ============================================================
# 9. MODEL EVALUATION
# ============================================================
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n[9] Model Evaluation:")
print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"    Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"    F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"    ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")

print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not At-Risk', 'At-Risk']))

cm = confusion_matrix(y_test, y_pred)
print(f"    Confusion Matrix:")
print(f"    {cm}")

# ============================================================
# 10. SHAP EXPLAINABILITY
# ============================================================
print(f"\n[10] Computing SHAP values...")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # At-Risk class
elif shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]  # 3D: (samples, features, classes)
else:
    shap_vals = shap_values

# Feature importance from SHAP
mean_abs_shap = np.abs(shap_vals).mean(axis=0).flatten()
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'shap_importance': mean_abs_shap
}).sort_values('shap_importance', ascending=False)

print("    Top 10 Most Important Features (SHAP):")
for i, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']}: {row['shap_importance']:.4f}")

# ============================================================
# 11. SAVE ALL ARTIFACTS
# ============================================================
print(f"\n[11] Saving model artifacts...")

# Save model
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save SHAP explainer
with open('models/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X_train.columns), f)

# Save numerical columns to scale
with open('models/numerical_cols.pkl', 'wb') as f:
    pickle.dump(numerical_to_scale, f)

# Save evaluation metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'roc_auc': float(roc_auc),
    'confusion_matrix': cm.tolist(),
    'dataset_size': int(len(df)),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'n_features': int(X_train.shape[1]),
    'feature_importance': feature_importance.head(15).to_dict('records')
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save processed dataset for the web app
df_processed.to_csv('data/student_dropout_processed.csv', index=False)

# Also save raw 3000 dataset
df.to_csv('data/student_dropout_3000.csv', index=False)

print("    All artifacts saved!")
print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
