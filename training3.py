import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Disable scientific notation in Pandas
pd.options.display.float_format = '{:.2f}'.format

# --- 1. SETUP PATHS ---
folder_path = r"C:\Users\Hello\OneDrive\Desktop\ml"
input_file = os.path.join(folder_path, "ready_for_training.csv")
model_save_path = os.path.join(folder_path, "car_price_predictor.pkl")
columns_save_path = os.path.join(folder_path, "model_columns.pkl")

def train_high_performance():
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return
    
    df = pd.read_csv(input_file)

    # --- 2. FEATURE ENGINEERING ---
    score_cols = ['Core_Systems_Score', 'Supporting_Systems_Score', 'Interiors_AC_Score', 
                  'Exteriors_Lights_Score', 'Wear_Tear_Parts_Score']
    df['Overall_Condition'] = df[score_cols].mean(axis=1)

    # --- 3. LOG TRANSFORMATION ---
    y_log = np.log1p(df['Price'])

    # --- 4. ENCODING ---
    df_ml = pd.get_dummies(df, columns=['Brand', 'Fuel_Type', 'Transmission'], drop_first=True)
    cols_to_drop = ['Price', 'Model'] + score_cols
    X = df_ml.drop(columns=cols_to_drop)

    # --- 5. SPLIT DATA ---
    X_train, X_test, y_train, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # --- 6. TUNED MODEL ---
    model = RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- 7. EVALUATION ---
    log_preds = model.predict(X_test)
    final_preds = np.expm1(log_preds)
    actual_prices = np.expm1(y_test_log)

    r2 = r2_score(actual_prices, final_preds)
    mae = mean_absolute_error(actual_prices, final_preds)

    print("\n" + "="*40)
    print("🏆 HIGH-PERFORMANCE RESULTS")
    print("="*40)
    print(f"✅ R2 Score: {r2:.4f}")
    print(f"💸 Final Avg Error (MAE): ₹{mae:,.0f}")
    print("="*40 + "\n")

    # Show actual vs predicted without scientific notation
    check = pd.DataFrame({
        'Actual_Price': actual_prices, 
        'Predicted_Price': final_preds,
        'Difference': np.abs(actual_prices - final_preds)
    }).reset_index(drop=True)
    
    print("📊 Comparison of Test Cars (Normal Numbers):")
    # Formats the dataframe to show integers for easier reading
    print(check.head().astype(int))

    # --- 8. SAVE ---
    joblib.dump(model, model_save_path)
    joblib.dump(X.columns.tolist(), columns_save_path)
    print(f"\n💾 Model & Columns saved to: {folder_path}")

if __name__ == "__main__":
    train_high_performance()