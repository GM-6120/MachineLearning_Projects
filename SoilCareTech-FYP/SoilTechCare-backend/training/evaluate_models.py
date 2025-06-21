import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define model directory
model_dir = r'E:\projects\backup project\SoilTechCare-backend\backend\models'

# Load dataset for evaluation
df = pd.read_csv(r'outputs\merged_features.csv')

# Load feature names
feature_names_path = os.path.join(model_dir, "feature_names.npy")
feature_columns = np.load(feature_names_path, allow_pickle=True)

# Extract features & target
X = df[list(feature_columns)]
y = df["Degradation-Level"]

# Load scaler and transform data
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
X_scaled = scaler.transform(X)

# Load trained models
models = {
    "XGBoost": joblib.load(os.path.join(model_dir, "XGBoost.pkl")),
    "RandomForest": joblib.load(os.path.join(model_dir, "RandomForest.pkl")),
    "LightGBM": joblib.load(os.path.join(model_dir, "LightGBM.pkl"))
}

# Evaluate each model
results = {}
plt.figure(figsize=(15, 5))

for idx, (name, model) in enumerate(models.items(), 1):
    # Predictions
    y_pred = model.predict(X_scaled)
    
    # Metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    results[name] = {"MAE": mae, "MSE": mse, "R2 Score": r2}
    
    # Scatter Plot
    plt.subplot(1, 3, idx)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='dashed')  # Ideal line
    plt.xlabel("Actual Degradation Level")
    plt.ylabel("Predicted Degradation Level")
    plt.title(f"{name} Model")
    
# Save and show plot
plot_path = os.path.join(model_dir, "model_evaluation.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.show()

# Print results
for model, metrics in results.items():
    print(f"\n🔹 {model} Model Evaluation:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

print(f"\n✅ Scatter plot saved at: {plot_path}")
