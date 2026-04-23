import pandas as pd
import re

print("Loading the scraped dataset...")
# Make sure this matches the file name outputted by your scraping script
df = pd.read_csv("spinny_full_dataset_final2.csv")

# A robust list of common words Spinny uses to describe the trim/variant
variant_keywords = [
    "SX", "SX(O)", "VXI", "ZXI", "LXI", "W8", "W6", "W4",
    "Top", "Model", "Automatic", "Manual", "AT", "MT", "AMT",
    "Diesel", "Petrol", "Turbo", "Edition", "Plus", "Premium", "Sport",
    "RXZ", "RXT", "RXE", "CVT", "DCT", "XMA", "XZA+", "XZA", "XZ", "XM", "XE", 
    "TDCi", "(O)", "GTX", "HTX", "HTK", "Alpha", "Zeta", "Delta", "Sigma"
]

def parse_car_name(name):
    # 1. Remove the 4-digit year from the beginning (e.g., "2021 Renault Kiger..." -> "Renault Kiger...")
    clean_name = re.sub(r'^\d{4}\s+', '', str(name))
    
    words = clean_name.split()
    if len(words) == 0 or str(name) == "N/A":
        return pd.Series(["Unknown", "Unknown", "Unknown"])

    # 2. First word is almost always the Brand
    brand = words[0].title()

    # 3. Separate the remaining words into Model and Variant
    model_words = []
    variant_words = []
    
    # We safely assume the second word is always part of the model name (e.g., "Kiger", "Creta")
    if len(words) > 1:
        model_words.append(words[1].title())
        
    # Check the rest of the words against our keyword list
    for word in words[2:]:
        clean_word = word.strip()
        # If the word matches our variant list, it's a trim level
        if clean_word.upper() in [x.upper() for x in variant_keywords]:
            variant_words.append(clean_word.upper())
        else:
            # Otherwise, assume it's part of a multi-word model name (like "Range Rover")
            model_words.append(clean_word.title())

    model = " ".join(model_words)
    variant = " ".join(variant_words)

    return pd.Series([brand, model, variant])

print("Splitting 'Name' into Brand, Model, and Variant...")

# Apply the parsing function to create the new columns
df[["Brand", "Model", "Variant"]] = df["Name"].apply(parse_car_name)

# Reorder columns to put the new ones right next to 'Name' for easy reading
cols = df.columns.tolist()
cols = ['Name', 'Brand', 'Model', 'Variant'] + [c for c in cols if c not in ['Name', 'Brand', 'Model', 'Variant']]
df = df[cols]

# Save to the specific CSV file name expected by the LLM Pricing Script
output_filename = "spinny_ml_ready_dataset.csv"
df.to_csv(output_filename, index=False)

print(f"\n✅ Success! Data split and saved to '{output_filename}'")
print("\n--- Quick Preview ---")
print(df[['Name', 'Brand', 'Model', 'Variant']].head())