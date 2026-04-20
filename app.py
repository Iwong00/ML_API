import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Ganti nama variabel menjadi 'app' (bukan 'flask_app')
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
        data = request.get_json()
        
        age = float(data['Age'])
        gender = data['Gender']
        blood_type = data['Blood Type']
        medical_condition = data['Medical Condition']
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Blood Type': [blood_type],
            'Medical Condition': [medical_condition]
        })
        
        transformed_input = transformer.transform(input_data)
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

if __name__ == "__main__":
    app.run(debug=True)
