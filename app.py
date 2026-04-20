import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd  # PERLU: untuk membuat DataFrame

flask_app = Flask(__name__)

# PERUBAHAN 1: Load transformer juga!
model = pickle.load(open("linear_regression_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))  # <-- INI PENTING!

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # PERUBAHAN 2: Ambil data sesuai format asli
    age = float(request.form.get('Age'))
    gender = request.form.get('Gender')
    blood_type = request.form.get('Blood Type')
    medical_condition = request.form.get('Medical Condition')
    
    # PERUBAHAN 3: Buat DataFrame seperti waktu training
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Blood Type': [blood_type],
        'Medical Condition': [medical_condition]
    })
    
    # PERUBAHAN 4: Transform data (one-hot encoding)
    transformed_input = transformer.transform(input_data)  # <-- INI PENTING!
    
    # PERUBAHAN 5: Prediksi
    prediction = model.predict(transformed_input)
    
    return render_template("index.html", 
                          prediction_text=f"Predicted Billing Amount: ${prediction[0]:,.2f}")
