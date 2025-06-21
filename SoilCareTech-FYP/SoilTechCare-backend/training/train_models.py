import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Create directory if not exists
model_dir = r'E:\projects\backup project\SoilTechCare-backend\backend\models'
os.makedirs(model_dir, exist_ok=True)

# Load updated dataset
df = pd.read_csv(r'outputs\merged_features.csv')

# Automatically detect feature columns (excluding target)
target_column = "Degradation-Level"
if target_column not in df.columns:
    raise ValueError("❌ 'Degradation-Level' column is missing from dataset!")

feature_columns = [col for col in df.columns if col != target_column]

# Log dataset structure
print(f"✅ Loaded dataset with {df.shape[0]} rows & {df.shape[1]} columns.")
print(f"✅ Features used for training: {feature_columns}")

# Handle missing values
df.fillna(df.median(), inplace=True)

# Split features & target
X = df[feature_columns]
y = df[target_column]

# Save feature names (for prediction consistency)
feature_names_path = os.path.join(model_dir, "feature_names.npy")
np.save(feature_names_path, np.array(feature_columns))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (only for numerical features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_path = os.path.join(model_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)

# Initialize models
models = {
    "XGBoost": xgb.XGBRegressor(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMRegressor()
}

# Train and save models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    model_path = os.path.join(model_dir, f"{name}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ {name} model trained & saved at {model_path}")

print("\n✅ All models, scaler, & feature names saved successfully!")
