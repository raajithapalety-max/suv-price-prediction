from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

FOLDER_PATH = r"C:\Users\Hello\OneDrive\Desktop\ml"
model = joblib.load(os.path.join(FOLDER_PATH, "rf_model.pkl"))
model_columns = joblib.load(os.path.join(FOLDER_PATH, "model_columns.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Create a template row of zeros
        input_row = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Fill numeric fields
        input_row['Car_Age'] = 2026 - int(data['year'])
        input_row['Kilometers_Driven'] = int(data['km'])
        
        # Fill categorical fields efficiently
        for key in ['brand', 'fuel', 'transmission']:
            col_name = f"{key.capitalize()}_{data[key].lower()}"
            if col_name in input_row.columns:
                input_row[col_name] = 1

        # Predict
        raw_prediction = model.predict(input_row)[0]
        
        # Formatting: Ensure price isn't negative and round it
        final_price = max(raw_prediction, 50000)
        return jsonify({'price': f"₹ {round(final_price):,}"})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)