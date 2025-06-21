import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# Input file path
input_file = "dataset/soil_data.csv"
output_file = "dataset/processed_soil_data.csv"

try:
    # Read the CSV file with a fallback to different encodings
    try:
        data = pd.read_csv(input_file, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(input_file, encoding="latin-1")  # Fallback encoding

    # Drop unnecessary columns (e.g., Region)
    data = data.drop(columns=["Region"], errors="ignore")

    # Normalize numeric columns (excluding categorical or ID columns)
    numeric_cols = ["pH", "Organic Matter (%)", "Compaction (g/cm³)", "Temperature (°C)", "Moisture (%)"]
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Save the preprocessed data
    data.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Preprocessed data saved to: {output_file}")

except Exception as e:
    print(f"Error during preprocessing: {e}")
