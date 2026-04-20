import numpy as np
from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # Abaikan warning versi

# Create flask app
app = Flask(__name__)

# Load model and transformer
model = pickle.load(open("linear_regression_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

# HTML Template (embedded agar lebih simple)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Healthcare Billing Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 500px;
            margin: 50px auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        label {
            display: block;
            margin: 15px 0 5px;
            font-weight: bold;
            color: #555;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 8px;
            text-align: center;
        }
        .result h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .amount {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏥 Healthcare Billing Predictor</h2>
        <form id="predictForm">
            <label>Age:</label>
            <input type="number" id="age" required min="0" max="120">
            
            <label>Gender:</label>
            <select id="gender" required>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>
            
            <label>Blood Type:</label>
            <select id="blood_type" required>
                <option value="A+">A+</option>
                <option value="A-">A-</option>
                <option value="B+">B+</option>
                <option value="B-">B-</option>
                <option value="O+">O+</option>
                <option value="O-">O-</option>
                <option value="AB+">AB+</option>
                <option value="AB-">AB-</option>
            </select>
            
            <label>Medical Condition:</label>
            <select id="condition" required>
                <option value="Arthritis">Arthritis</option>
                <option value="Asthma">Asthma</option>
                <option value="Cancer">Cancer</option>
                <option value="Diabetes">Diabetes</option>
                <option value="Hypertension">Hypertension</option>
                <option value="Obesity">Obesity</option>
            </select>
            
            <button type="submit">Predict Billing Amount 💰</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const data = {
                Age: parseInt(document.getElementById('age').value),
                Gender: document.getElementById('gender').value,
                'Blood Type': document.getElementById('blood_type').value,
                'Medical Condition': document.getElementById('condition').value
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('result').innerHTML = `
                        <div class="result">
                            <h3>Predicted Billing Amount:</h3>
                            <div class="amount">$${result.predicted_amount.toLocaleString()}</div>
                        </div>
                    `;
                } else {
                    document.getElementById('result').innerHTML = `
                        <div class="result error">
                            Error: ${result.error}
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `
                    <div class="result error">
                        Error: Failed to get prediction
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
'''

@app.route("/")
def home():
    return HTML_TEMPLATE

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        input_data = pd.DataFrame({
            'Age': [float(data['Age'])],
            'Gender': [data['Gender']],
            'Blood Type': [data['Blood Type']],
            'Medical Condition': [data['Medical Condition']]
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
    app.run(debug=True, host='0.0.0.0', port=10000)
