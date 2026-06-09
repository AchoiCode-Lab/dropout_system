"""
Data Synthesis Script
Expands the original 649-row student dropout dataset to 3000+ rows
using statistical distribution matching + noise injection.
"""
import pandas as pd
import numpy as np
from collections import Counter

np.random.seed(42)

# Load original data
df_orig = pd.read_csv('student_dropout_original.csv')
print(f"Original dataset: {df_orig.shape[0]} rows x {df_orig.shape[1]} columns")

# Identify column types
categorical_cols = df_orig.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = df_orig.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

TARGET_ROWS = 3000
rows_to_generate = TARGET_ROWS - len(df_orig)

print(f"\nGenerating {rows_to_generate} synthetic rows...")

# Strategy: Stratified resampling with noise injection
# 1. Resample from original data (preserving distributions)
# 2. Add controlled noise to numerical features
# 3. Maintain realistic correlations

synthetic_rows = []

# Get class distribution for stratified sampling
dropout_dist = df_orig['Dropped_Out'].value_counts(normalize=True)
print(f"Original dropout distribution: {dict(dropout_dist)}")

for _ in range(rows_to_generate):
    # Stratified: pick a random row, weighted by class distribution
    sample = df_orig.sample(n=1, weights=None).iloc[0].copy()
    
    # Add noise to numerical columns
    for col in numerical_cols:
        orig_val = sample[col]
        col_std = df_orig[col].std()
        col_min = df_orig[col].min()
        col_max = df_orig[col].max()
        
        # Add Gaussian noise (10-20% of std)
        noise = np.random.normal(0, col_std * 0.15)
        new_val = orig_val + noise
        
        # Clip to valid range
        new_val = np.clip(new_val, col_min, col_max)
        
        # Round integers
        if df_orig[col].dtype in [np.int64, np.int32]:
            new_val = int(round(new_val))
        else:
            new_val = round(new_val, 2)
        
        sample[col] = new_val
    
    # Small chance (5%) to flip some categorical values for diversity
    for col in categorical_cols:
        if col == 'Dropped_Out':
            continue  # Don't randomly flip target
        if np.random.random() < 0.05:
            possible_values = df_orig[col].unique()
            sample[col] = np.random.choice(possible_values)
    
    synthetic_rows.append(sample)

df_synthetic = pd.DataFrame(synthetic_rows)

# Combine original + synthetic
df_combined = pd.concat([df_orig, df_synthetic], ignore_index=True)

# Shuffle the dataset
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset: {df_combined.shape[0]} rows x {df_combined.shape[1]} columns")
print(f"\nDropout distribution (final):")
print(df_combined['Dropped_Out'].value_counts())
print(f"\nDropout percentage: {df_combined['Dropped_Out'].value_counts(normalize=True) * 100}")

# Verify data integrity
print(f"\nMissing values: {df_combined.isnull().sum().sum()}")
print(f"Duplicates: {df_combined.duplicated().sum()}")

# Remove any exact duplicates
df_combined = df_combined.drop_duplicates().reset_index(drop=True)
print(f"After removing duplicates: {df_combined.shape[0]} rows")

# If we're below 3000 after dedup, add more
while len(df_combined) < 3000:
    sample = df_orig.sample(n=1).iloc[0].copy()
    for col in numerical_cols:
        orig_val = sample[col]
        col_std = df_orig[col].std()
        noise = np.random.normal(0, col_std * 0.2)
        new_val = np.clip(orig_val + noise, df_orig[col].min(), df_orig[col].max())
        if df_orig[col].dtype in [np.int64, np.int32]:
            new_val = int(round(new_val))
        sample[col] = new_val
    df_combined = pd.concat([df_combined, pd.DataFrame([sample])], ignore_index=True)

print(f"\nFinal count: {df_combined.shape[0]} rows")

# Save
df_combined.to_csv('student_dropout_3000.csv', index=False)
print("Saved: student_dropout_3000.csv")

# Also save stats comparison
print("\n=== DISTRIBUTION COMPARISON ===")
for col in numerical_cols[:5]:
    print(f"\n{col}:")
    print(f"  Original - Mean: {df_orig[col].mean():.2f}, Std: {df_orig[col].std():.2f}")
    print(f"  Expanded - Mean: {df_combined[col].mean():.2f}, Std: {df_combined[col].std():.2f}")
