from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# LOAD DATASET + MODEL
df = pd.read_csv("5.feature_engineered.csv")
model = joblib.load("xgboost_suv_price_model.pkl")

# LABEL ENCODERS
cat_cols = ["Brand", "Model", "Fuel_Type", "Transmission"]

encoders = {}

for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    encoders[c] = le


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    brand = encoders["Brand"].transform([data["brand"]])[0]
    model_name = encoders["Model"].transform([data["model"]])[0]
    fuel = encoders["Fuel_Type"].transform([data["fuel"]])[0]
    transmission = encoders["Transmission"].transform([data["transmission"]])[0]

    input_data = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Car_Age": [data["car_age"]],
        "Kilometers_Driven": [data["km"]],
        "Fuel_Type": [fuel],
        "Transmission": [transmission],
        "Owner_Count": [data["owner"]],
        "Estimated_Original_Price": [data["original_price"]],
        "Core_Systems_Score": [data["core_score"]],
        "Supporting_Systems_Score": [data["support_score"]],
        "Interiors_AC_Score": [data["interior_score"]],
        "Exteriors_Lights_Score": [data["exterior_score"]],
        "Wear_Tear_Parts_Score": [data["wear_score"]]
    })

    prediction = model.predict(input_data)[0]

    return jsonify({
        "prediction": round(prediction, 0)
    })


if __name__ == "__main__":
    app.run(debug=True)