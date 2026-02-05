from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variables for model, scaler, and features
model = None
scaler = None
feature_names = None

def load_model_artifacts():
    """Load the trained model, scaler, and feature names"""
    global model, scaler, feature_names
    
    try:
        if os.path.exists('rainfall_model.pkl'):
            model = joblib.load('rainfall_model.pkl')
            scaler = joblib.load('scaler.pkl')
            feature_names = joblib.load('feature_names.pkl')
            print("Model loaded successfully!")
            return True
        else:
            print("Model files not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Try to load model on startup
model_loaded = load_model_artifacts()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    # Get feature names for the form
    if model_loaded and feature_names:
        features = feature_names
    else:
        # Fallback to basic features
        features = [
            "dist_road", "dist_market", "dist_border", "dist_popcenter",
            "dist_admhq", "af_bio_1_x", "af_bio_8_x", "af_bio_13_x",
            "af_bio_16_x", "afmnslp_pct", "srtm_1k", "popdensity",
            "cropshare", "twi_ne", "pct_urban_cluster", "pct_urban_center",
            "h2021_tot", "h2021_wetQstart", "h2021_wetQ", "anntot_avg",
            "wetQ_avgstart", "wetQ_avg", "eviarea_avg", "evimax_avg",
            "grn_avg", "sen_avg", "h2021_eviarea", "h2021_evimax",
            "h2021_grn", "h2021_sen", "lat_dd_mod", "lon_dd_mod"
        ]
    
    return render_template('predict.html', features=features)

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for rainfall prediction
    Expects JSON data with feature values
    """
    try:
        if not model_loaded:
            return jsonify({
                "error": "Model not loaded. Please train the model first.",
                "success": False
            }), 500
        
        data = request.json
        
        # Extract feature values from request
        # Expecting a dictionary with feature names as keys
        if 'features' not in data:
            return jsonify({
                "error": "Missing 'features' in request data",
                "success": False
            }), 400
        
        # Create DataFrame with the same features used during training
        input_data = pd.DataFrame([data['features']])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_data.columns)
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {list(missing_features)}",
                "success": False
            }), 400
        
        # Reorder columns to match training data
        input_data = input_data[feature_names]
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            # Get top 10 most important features
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            feature_importance = {
                feature_names[i]: float(importances[i])
                for i in top_indices
            }
        
        response = {
            "success": True,
            "prediction": float(round(prediction, 2)),
            "unit": "mm",
            "model_info": {
                "type": type(model).__name__,
                "features_used": len(feature_names)
            },
            "feature_importance": feature_importance
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """
    Return information about the loaded model
    """
    if not model_loaded:
        return jsonify({
            "loaded": False,
            "message": "Model not loaded"
        })
    
    info = {
        "loaded": True,
        "model_type": type(model).__name__,
        "num_features": len(feature_names),
        "features": feature_names
    }
    
    # Try to load metrics if available
    if os.path.exists('model_metrics.json'):
        import json
        with open('model_metrics.json', 'r') as f:
            info['metrics'] = json.load(f)
    
    return jsonify(info)

@app.route('/api/sample_prediction', methods=['GET'])
def sample_prediction():
    """
    Make a sample prediction using median values from training data
    Useful for testing the API
    """
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "success": False
        }), 500
    
    # Create sample input with reasonable default values
    # You should replace these with actual median/mean values from your training data
    sample_features = {feature: 0.0 for feature in feature_names}
    
    # Make prediction
    input_df = pd.DataFrame([sample_features])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    return jsonify({
        "success": True,
        "prediction": float(round(prediction, 2)),
        "unit": "mm",
        "note": "This is a sample prediction using default feature values"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
