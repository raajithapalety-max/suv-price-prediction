import pandas as pd
import google.generativeai as genai
import time
import re
import os

# --- 1. SETUP ---
GEMINI_API_KEY = "xx"
genai.configure(api_key=GEMINI_API_KEY)

# Use the most current 2026 Lite model for high-volume free tasks
MODEL_NAME = 'gemini-3.1-flash-lite-preview' 
model = genai.GenerativeModel(MODEL_NAME)

INPUT_FILE = "spinny_ml_ready_dataset.csv"
OUTPUT_FILE = "spinny_gemini_results.csv"

def get_price_with_retry(year, brand, model_name, variant, retries=3):
    prompt = f"Original India ex-showroom price for {year} {brand} {model_name} {variant}. Return ONLY the number."
    for i in range(retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                clean_price = re.sub(r'[^0-9]', '', response.text.strip())
                return int(clean_price) if clean_price else None
        except Exception as e:
            if "429" in str(e): # The 'Rate Limit' error you're getting
                wait_time = (i + 1) * 30 
                print(f"⚠️ Rate limit hit. Cooling down for {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Error: {e}")
                return None
    return None

# --- 2. RESUME LOGIC ---
if os.path.exists(OUTPUT_FILE):
    df = pd.read_csv(OUTPUT_FILE)
    print(f"🔄 Resuming! Found {df['Estimated_Original_Price'].notnull().sum()} already completed.")
else:
    df = pd.read_csv(INPUT_FILE)
    df['Estimated_Original_Price'] = None

# --- 3. EXECUTION ---
print(f"🚀 Starting process on {MODEL_NAME}...")

for index, row in df.iterrows():
    # SKIP if this row already has a price from a previous run
    if pd.notnull(row['Estimated_Original_Price']):
        continue

    # Clean the year
    yr_match = re.search(r'\d{4}', str(row['Make_Year']))
    yr = yr_match.group() if yr_match else "2022"
    
    print(f"[{index+1}/{len(df)}] {yr} {row['Brand']} {row['Model']}...", end=" ", flush=True)
    
    price = get_price_with_retry(yr, row['Brand'], row['Model'], row.get('Variant', ''))
    
    if price:
        df.at[index, 'Estimated_Original_Price'] = price
        print(f"✅ ₹{price:,}")
    else:
        print("❌ Failed")

    # The 7-second rule for Free Tier (Keeps you under 10 RPM)
    time.sleep(7)

    # Save after EVERY row so you never lose progress again
    df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Finished! Your data is in {OUTPUT_FILE}")