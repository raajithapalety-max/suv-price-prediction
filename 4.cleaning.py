import pandas as pd
import numpy as np
import re
import os

# --- 1. FILE PATHS (Set to your Desktop folder) ---
# I'm using your specific path from the screenshots
folder_path = r"C:\Users\Hello\OneDrive\Desktop\ml"
input_path = os.path.join(folder_path, "spinny_gemini_results.csv")
output_path = os.path.join(folder_path, "ready_for_training.csv")

def clean_spinny_data():
    if not os.path.exists(input_path):
        print(f"❌ Error: Could not find {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"📊 Loading {len(df)} cars for cleaning...")

    # --- 2. CLEAN PRICE (Target Variable) ---
    def parse_price(x):
        if pd.isna(x) or x == "N/A": return np.nan
        x = str(x).lower().replace('₹', '').replace(',', '').strip()
        if 'lakh' in x:
            return float(x.replace('lakh', '').strip()) * 100000
        return float(x)

    df['Price'] = df['Price'].apply(parse_price)

    # --- 3. CLEAN KILOMETERS (e.g., '45K km' -> 45000) ---
    def clean_km(x):
        if pd.isna(x): return 0
        x = str(x).lower()
        num_match = re.search(r'(\d+)', x)
        if num_match:
            val = float(num_match.group(1))
            if 'k' in x: val *= 1000
            return val
        return 0

    df['Kilometers_Driven'] = df['Kilometers_Driven'].apply(clean_km)

    # --- 4. CLEAN MAKE_YEAR & CREATE AGE ---
    def get_age(x):
        # Extract last 2 digits from "Oct-22"
        match = re.search(r'(\d{2})$', str(x))
        if match:
            year = 2000 + int(match.group(1))
            return 2026 - year # Car age in 2026
        return 0

    df['Car_Age'] = df['Make_Year'].apply(get_age)

    # --- 5. CLEAN OWNER COUNT ---
    df['Owner_Count'] = df['Owner_Count'].str.extract(r'(\d)').astype(float).fillna(1)

    # --- 6. CLEAN SCORES (Convert to Floats) ---
    score_cols = [
        'Core_Systems_Score', 'Supporting_Systems_Score', 
        'Interiors_AC_Score', 'Exteriors_Lights_Score', 'Wear_Tear_Parts_Score'
    ]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill missing scores with the column average
        df[col] = df[col].fillna(df[col].mean())

    # --- 7. STANDARDIZE FUEL & TRANSMISSION ---
    df['Fuel_Type'] = df['Fuel_Type'].str.split(' ').str[0].str.strip().str.title()
    df['Transmission'] = df['Transmission'].str.strip().str.title()

    # --- 8. FINAL SELECTION ---
    # We remove text columns that are too unique for simple ML (like names/variants)
    # but we keep Brand/Model for the next step (Encoding)
    final_cols = [
        'Brand', 'Model', 'Car_Age', 'Kilometers_Driven', 'Fuel_Type', 
        'Transmission', 'Owner_Count', 'Price', 'Estimated_Original_Price'
    ] + score_cols
    
    df_final = df[final_cols].dropna(subset=['Price'])

    # Save the file
    df_final.to_csv(output_path, index=False)
    print(f"✅ Success! Your ML-ready file is at: {output_path}")

if __name__ == "__main__":
    clean_spinny_data()