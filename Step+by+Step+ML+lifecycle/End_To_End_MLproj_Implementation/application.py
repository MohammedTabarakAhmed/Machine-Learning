import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extract form data
        data = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['ISI']),
            float(request.form['Classes']),
            float(request.form['Region'])
        ]

        # Scale and predict
        scaled_data = standard_scaler.transform([data])
        result = ridge_model.predict(scaled_data)[0]

        # Return result to home.html
        return render_template('home.html', result=result)
    else:
        # If GET request, just show the form
        return render_template('home.html')

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)