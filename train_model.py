"""
Quick Model Training Script
This script trains the perfect rainfall prediction model
Run this if you prefer command line over Jupyter notebook
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PERFECT RAINFALL PREDICTION MODEL - TRAINING SCRIPT")
print("="*80)

# 1. LOAD DATA
print("\n[1/9] Loading data...")
df = pd.read_csv(r"C:\Users\Admin\Desktop\-rainfall-pradction-ml\eth_householdgeovariables_y5.csv")
print(f"   Initial dataset shape: {df.shape}")

# 2. DATA CLEANING - Missing Values
print("\n[2/9] Handling missing values...")
missing_data = (df.isnull().sum() / len(df)) * 100
high_missing_cols = missing_data[missing_data > 80].index.tolist()
print(f"   Dropping {len(high_missing_cols)} columns with >80% missing data")
df_cleaned = df.drop(columns=high_missing_cols)

# Drop rows with missing coordinates
df_cleaned = df_cleaned.dropna(subset=['lat_dd_mod', 'lon_dd_mod'])
print(f"   Dataset shape after cleaning: {df_cleaned.shape}")

# 3. REMOVE DUPLICATES
print("\n[3/9] Removing duplicates...")
initial_rows = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates()
print(f"   Removed {initial_rows - len(df_cleaned)} duplicate rows")
print(f"   Final dataset shape: {df_cleaned.shape}")

# 4. FEATURE PREPARATION
print("\n[4/9] Preparing features...")
# Remove non-informative columns
cols_to_drop = ['household_id', 'suppress']
df_cleaned = df_cleaned.drop(columns=[col for col in cols_to_drop if col in df_cleaned.columns])

# Separate target and features
target = 'af_bio_12_x'
y = df_cleaned[target]
X = df_cleaned.drop(columns=[target])

# Encode categorical variables
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
if len(categorical_features) > 0:
    print(f"   Encoding {len(categorical_features)} categorical features")
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print(f"   Total features: {X.shape[1]}")

# 5. FEATURE SELECTION BY CORRELATION
print("\n[5/9] Selecting relevant features...")
X_with_target = X.copy()
X_with_target['rainfall'] = y
correlation_with_target = X_with_target.corr()['rainfall'].drop('rainfall')

threshold = 0.05
low_corr_features = correlation_with_target[correlation_with_target.abs() < threshold].index.tolist()
X = X.drop(columns=low_corr_features)
print(f"   Removed {len(low_corr_features)} features with low correlation")
print(f"   Features remaining: {X.shape[1]}")

# 6. TRAIN-TEST SPLIT
print("\n[6/9] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# 7. SCALING
print("\n[7/9] Scaling features with RobustScaler...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. MODEL TRAINING AND COMPARISON
print("\n[8/9] Training and comparing models...")
models = {
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42)
}

results = {}
best_score = -np.inf
best_model_name = None

for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred_test = model.predict(X_test_scaled)
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'model': model
    }
    
    print(f"     Test R²: {test_r2:.4f}")
    print(f"     Test RMSE: {test_rmse:.2f} mm")
    print(f"     Test MAE: {test_mae:.2f} mm")
    
    if test_r2 > best_score:
        best_score = test_r2
        best_model_name = name

print(f"\n   🏆 Best model: {best_model_name} (R² = {best_score:.4f})")

# 9. HYPERPARAMETER TUNING
print(f"\n[9/9] Tuning hyperparameters for {best_model_name}...")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7]
    }
    base_model = GradientBoostingRegressor(random_state=42)
else:
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    base_model = Ridge(random_state=42) if best_model_name == 'Ridge Regression' else Lasso(random_state=42)

grid_search = GridSearchCV(
    base_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)

print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best CV score: {grid_search.best_score_:.4f}")

# Final model evaluation
final_model = grid_search.best_estimator_
y_pred_train = final_model.predict(X_train_scaled)
y_pred_test = final_model.predict(X_test_scaled)

print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE")
print("="*80)
print(f"\nModel: {best_model_name}")
print(f"Parameters: {grid_search.best_params_}")
print(f"\nTraining Set:")
print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f} mm")
print(f"  MAE: {mean_absolute_error(y_train, y_pred_train):.2f} mm")
print(f"\nTest Set:")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} mm")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_test):.2f} mm")

# SAVE MODEL
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

joblib.dump(final_model, 'rainfall_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')
print("✅ Saved: rainfall_model.pkl")
print("✅ Saved: scaler.pkl")
print("✅ Saved: feature_names.pkl")

# Save metrics
metrics = {
    'model_name': best_model_name,
    'best_params': grid_search.best_params_,
    'train_r2': float(r2_score(y_train, y_pred_train)),
    'test_r2': float(r2_score(y_test, y_pred_test)),
    'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
    'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
    'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
    'num_features': len(X_train.columns),
    'num_training_samples': len(X_train),
    'num_test_samples': len(X_test)
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Saved: model_metrics.json")

print("\n" + "="*80)
print("✨ MODEL TRAINING COMPLETE! ✨")
print("="*80)
print("\nYou can now use the trained model in your Flask app!")
print("Run: python app.py")
print("\nOr make predictions with:")
print("  model = joblib.load('rainfall_model.pkl')")
print("  scaler = joblib.load('scaler.pkl')")
print("="*80)
