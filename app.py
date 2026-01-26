from flask import Flask, render_template, request, jsonify
import random
import os

app = Flask(__name__)

# Mock features for the form
FEATURES = [
    "Average Temperature (C)",
    "Humidity (%)",
    "Wind Speed (km/h)",
    "Precipitation (mm)",
    "Soil Moisture",
    "Vegetation Index (NDVI)",
    "Altitude (m)",
    "Latitude",
    "Longitude",
    "Distance to Water Body (km)",
    "Slope",
    "Aspect",
    "Cloud Cover (%)",
    "Pressure (hPa)",
    "Solar Radiation",
    "Evapotranspiration",
    "Land Use Code",
    "Seasonality Index",
    "Historical Rainfall Avg",
    "Region Code"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html', features=FEATURES)

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    # Mock prediction logic
    # In reality, we would valid input and load model here
    
    # Simulate processing time
    import time
    time.sleep(1)
    
    # Generate random rainfall between 0 and 500 mm
    predicted_rainfall = round(random.uniform(0, 300), 2)
    
    # Mock confidence intervals or feature importance
    response = {
        "prediction": predicted_rainfall,
        "unit": "mm",
        "confidence": "High",
        "note": "Based on mock random forest logic.",
        "feature_importance": {
            "Humidity": 0.35,
            "Temperature": 0.25,
            "Wind Speed": 0.15,
            "Pressure": 0.10,
            "Others": 0.15
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
