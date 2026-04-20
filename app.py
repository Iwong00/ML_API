import numpy as np
from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import os  # <- IMPORTANT: untuk baca environment variable PORT
import warnings
warnings.filterwarnings('ignore')

# Create flask app
app = Flask(__name__)

# Load model and transformer
model = pickle.load(open("linear_regression_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
