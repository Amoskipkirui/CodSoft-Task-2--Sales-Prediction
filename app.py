from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained XGBoost model
xgb_model = joblib.load('Sales1_model.pkl')

# Load the scaler used for feature scaling
scaler = joblib.load('scaler.pkl')

def predict_sales(tv, radio, newspaper):
    input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
    input_data_scaled = scaler.transform(input_data)
    prediction = xgb_model.predict(input_data_scaled)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tv = float(request.form['TV'])
    radio = float(request.form['Radio'])
    newspaper = float(request.form['Newspaper'])
    predicted_sales = predict_sales(tv, radio, newspaper)
    return render_template('results.html', prediction=predicted_sales)

if __name__ == "__main__":
    app.run(debug=True)
