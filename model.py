import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb
import catboost as cb

# --- 1. SETTINGS & LOAD DATA ---
folder_path = r"C:\Users\Hello\OneDrive\Desktop\ml"
csv_path = os.path.join(folder_path, "ready_for_training.csv")
df = pd.read_csv(csv_path)

# --- 2. PREPROCESSING ---
# Creating Car_Age and handling categorical columns
if 'Year' in df.columns:
    df['Car_Age'] = 2026 - df['Year']

# Standardize text for One-Hot Encoding
for col in ['Brand', 'Fuel_Type', 'Transmission']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

# Defining Features and Target
target_column_name = 'Resale_Price' if 'Resale_Price' in df.columns else 'Price'
features_to_use = ['Car_Age', 'Kilometers_Driven', 'Brand', 'Fuel_Type', 'Transmission']

# Filter only existing features
existing_features = [f for f in features_to_use if f in df.columns]
X = pd.get_dummies(df[existing_features])
y = df[target_column_name]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. DEFINE REGRESSION MODELS ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
    "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1),
    "CatBoost": cb.CatBoostRegressor(random_state=42, verbose=0)
}

# --- 4. EVALUATE MODELS ---
results = []
feature_columns = X.columns.tolist()

print(f"🚀 Starting Evaluation on {len(models)} models...\n")

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()

    # Calculate MAPE (Avoid division by zero)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    results.append({
        "Model": name,
        "R² Score": round(r2_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "MAPE (%)": round(mape, 2),
        "Time (s)": round(end - start, 3)
    })

    # Explainability
    print(f"--- Model Explainability: {name} ---")
    try:
        if hasattr(model, 'coef_'):
            imp = pd.Series(model.coef_, index=feature_columns).sort_values(ascending=False)
            print(imp.head(5).to_string()) # Show top 5
        elif hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
            print(imp.head(5).to_string()) # Show top 5
        else:
            print("No direct importance available.")
    except:
        print("Could not extract importance.")
    print("-" * 30)

# --- 5. SUMMARY & SAVING BEST ---
results_df = pd.DataFrame(results).sort_values(by="MAE")

print("\n🔍 MODEL EVALUATION SUMMARY:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 The best performing model is: {best_model_name}")

# Save the best model for the Web App
best_model = models[best_model_name]
joblib.dump(best_model, os.path.join(folder_path, 'car_price_predictor.pkl'))
joblib.dump(feature_columns, os.path.join(folder_path, 'model_columns.pkl'))
print(f"💾 Best model ({best_model_name}) saved for Deployment!")