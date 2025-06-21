import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load preprocessed text dataset
data_path = "dataset/processed_soil_data.csv"  # Update with your actual path
df = pd.read_csv(data_path)

# Drop unnecessary columns
df_features = df.drop(columns=['ID', 'Latitude', 'Longitude'], errors='ignore')

# Standardize numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features.drop(columns=['Degradation-Level'], errors='ignore'))

# Save extracted text features
scaled_text_features_path = "outputs/scaled_text_features.csv"
scaled_df = pd.DataFrame(scaled_features, columns=df_features.drop(columns=['Degradation-Level']).columns)
scaled_df["Degradation-Level"] = df_features["Degradation-Level"]  # Add target column back
scaled_df.to_csv(scaled_text_features_path, index=False)

print(f"✅ Text data features saved to {scaled_text_features_path}")
