import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
DATA_PATH = "outputs/merged_features.csv"
data = pd.read_csv(DATA_PATH)

# Load trained model & scaler
MODEL_PATH = "backend/models/LightGBM.pkl"
SCALER_PATH = "backend/models/scaler.pkl"
FEATURE_NAMES_PATH = "backend/models/feature_names.npy"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True)

# Function to compute erosion (dummy implementation)
def compute_erosion(pH, organic_matter, compaction, temperature):
    """
    Compute erosion based on soil parameters (dummy formula for demonstration)
    Higher values mean more erosion
    """
    erosion = (10 - pH) * 0.5 + (3 - organic_matter) * 0.3 + compaction * 0.2 + temperature * 0.01
    return np.clip(erosion, 0, 10)  # Keep between 0-10

# Function to get closest row and compute additional parameters
def get_soil_data(latitude, longitude):
    # Find closest location
    data["Distance"] = np.sqrt((data["Latitude"] - latitude) ** 2 + (data["Longitude"] - longitude) ** 2)
    closest_row = data.loc[data["Distance"].idxmin()]
    
    # Get actual values from dataset
    temperature = closest_row["Temperature (°C)"]
    moisture = closest_row["Moisture (%)"]
    degradation = closest_row["Degradation-Level"]
    
    # Compute erosion (dummy data)
    erosion = compute_erosion(
        closest_row["pH"],
        closest_row["Organic Matter (%)"],
        closest_row["Compaction (g/cm³)"],
        temperature
    )
    
    return {
        "latitude": latitude,
        "longitude": longitude,
        "temperature": temperature,
        "moisture": moisture,
        "erosion": erosion,
        "degradation_raw": degradation,
        "degradation_level": int(round(degradation)),  # Rounded to nearest integer
        "features": closest_row[feature_names].values.reshape(1, -1)
    }

# Function to interpret degradation level
def interpret_degradation(level):
    if level <= 1.5:
        return "Low", 1
    elif level <= 2.5:
        return "Moderate", 2
    elif level <= 3.5:
        return "High", 3
    else:
        return "Very High", 4

# Main prediction function
def predict_soil_health(latitude, longitude):
    # Find closest location
    data["Distance"] = np.sqrt((data["Latitude"] - latitude) ** 2 + (data["Longitude"] - longitude) ** 2)
    closest_row = data.loc[data["Distance"].idxmin()].copy()
    
    # Store original values BEFORE scaling
    original_values = {
        "temperature": closest_row["Temperature (°C)"],
        "moisture": closest_row["Moisture (%)"],
        "ph": closest_row["pH"],
        "organic_matter": closest_row["Organic Matter (%)"],
        "compaction": closest_row["Compaction (g/cm³)"],
        "degradation": closest_row["Degradation-Level"]
    }
    
    # Compute erosion (using original values)
    erosion = compute_erosion(
        original_values["ph"],
        original_values["organic_matter"],
        original_values["compaction"],
        original_values["temperature"]
    )
    
    # Scale features for prediction
    features = closest_row[feature_names].values.reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    # Interpret prediction
    degradation_label, degradation_category = interpret_degradation(prediction)
    
    return {
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude,
            "nearest_latitude": closest_row["Latitude"],
            "nearest_longitude": closest_row["Longitude"]
        },
        "parameters": {
            "temperature": round(float(original_values["temperature"]), 2),
            "moisture": round(float(original_values["moisture"]), 2),
            "erosion": round(float(erosion), 2),
            "ph": round(float(original_values["ph"]), 2),
            "organic_matter": round(float(original_values["organic_matter"]), 2),
            "compaction": round(float(original_values["compaction"]), 2)
        },
        "degradation": {
            "raw_value": round(float(prediction), 4),
            "category": degradation_category,
            "label": degradation_label,
            "original_value": round(float(original_values["degradation"]), 4)
        },
        "distance": round(float(closest_row["Distance"]), 6)
    }

# Example usage
if __name__ == "__main__":
    # Get user input
    latitude = float(input("Enter Latitude: "))
    longitude = float(input("Enter Longitude: "))
    
    # Get prediction
    result = predict_soil_health(latitude, longitude)
    
    # Print results
    print("\nSoil Health Analysis Results:")
    print("---------------------------")
    print(f"Location: {result['coordinates']['latitude']}, {result['coordinates']['longitude']}")
    print(f"Temperature: {result['parameters']['temperature']} °C")
    print(f"Moisture: {result['parameters']['moisture']} %")
    print(f"Erosion: {result['parameters']['erosion']} (0-10 scale)")
    print(f"pH: {result['parameters']['ph']}")
    print(f"Organic Matter: {result['parameters']['organic_matter']} %")
    print(f"Compaction: {result['parameters']['compaction']} g/cm³")
    print("\nDegradation Analysis:")
    print(f"Raw Value: {result['degradation']['raw_value']}")
    print(f"Category: {result['degradation']['category']} ({result['degradation']['label']})")