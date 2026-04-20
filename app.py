import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Create flask app
app = Flask(__name__)

# Load model and transformer
model = pickle.load(open("linear_regression_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features
        age = float(data['Age'])
        gender = data['Gender']
        blood_type = data['Blood Type']
        medical_condition = data['Medical Condition']
        
        # Create DataFrame with the same structure as training data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Blood Type': [blood_type],
            'Medical Condition': [medical_condition]
        })
        
        # Apply the same transformation (one-hot encoding)
        transformed_input = transformer.transform(input_data)
        
        # Make prediction
        prediction = model.predict(transformed_input)
        
        return jsonify({
            'success': True,
            'predicted_amount': float(prediction[0])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Health check endpoint for Render
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
