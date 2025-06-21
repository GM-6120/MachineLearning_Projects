from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load data and models
DATA_PATH = "outputs/merged_features.csv"
MODEL_PATH = "backend/models/LightGBM.pkl"
SCALER_PATH = "backend/models/scaler.pkl"
FEATURE_NAMES_PATH = "backend/models/feature_names.npy"

data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True)

def get_soil_analysis(latitude, longitude):
    # Find nearest coordinates
    data['Distance'] = np.sqrt((data['Latitude']-latitude)**2 + (data['Longitude']-longitude)**2)
    closest = data.loc[data['Distance'].idxmin()]
    
    # Get actual values from dataset
    temp = closest['Temperature (°C)']
    moisture = closest['Moisture (%)']
    
    # Predict degradation
    features = closest[feature_names].values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    degradation = float(model.predict(features_scaled)[0])
    
    # Calculate erosion based on degradation
    erosion = calculate_erosion_level(degradation)
    
    return {
        'temperature': round(temp, 1),
        'moisture': round(moisture, 1),
        'erosion': erosion,
        'degradation': classify_degradation(degradation),
        'coordinates': {
            'searched': [latitude, longitude],
            'matched': [float(closest['Latitude']), float(closest['Longitude'])]
        }
    }

def calculate_erosion_level(degradation):
    """Returns erosion level based on degradation"""
    if degradation < 1.5: return "Low"
    elif degradation < 2.5: return "Moderate"
    else: return "High"

def classify_degradation(value):
    """Classify degradation into categories"""
    if value < 1.5: return {'level': 1, 'label': 'Low', 'value': round(value, 2)}
    elif value < 2.5: return {'level': 2, 'label': 'Moderate', 'value': round(value, 2)}
    else: return {'level': 3, 'label': 'High', 'value': round(value, 2)}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json()
        lat, lng = float(req_data['lat']), float(req_data['lng'])
        result = get_soil_analysis(lat, lng)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)