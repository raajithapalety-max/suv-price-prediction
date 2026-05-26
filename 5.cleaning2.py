import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("4.ready_for_training.csv")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# 1. Depreciation Percentage
df["Depreciation_Percent"] = (
    (df["Estimated_Original_Price"] - df["Price"])
    / df["Estimated_Original_Price"]
) * 100

# 2. Price Lost Per Year
df["Price_Lost_Per_Year"] = (
    (df["Estimated_Original_Price"] - df["Price"])
    / df["Car_Age"]
)

# 3. Kilometers Driven Per Year
df["Km_Per_Year"] = (
    df["Kilometers_Driven"]
    / df["Car_Age"]
)

# 4. Ownership Impact
df["Ownership_Impact"] = (
    df["Owner_Count"]
    * df["Car_Age"]
)

# 5. Luxury Score
df["Luxury_Score"] = (
    df["Core_Systems_Score"]
    + df["Supporting_Systems_Score"]
    + df["Interiors_AC_Score"]
    + df["Exteriors_Lights_Score"]
) / 4

# 6. Wear vs Price Ratio
df["Wear_Price_Ratio"] = (
    df["Wear_Tear_Parts_Score"]
    / df["Price"]
)

# 7. Total Feature Score
df["Total_Feature_Score"] = (
    df["Core_Systems_Score"]
    + df["Supporting_Systems_Score"]
    + df["Interiors_AC_Score"]
    + df["Exteriors_Lights_Score"]
    + df["Wear_Tear_Parts_Score"]
)

# -----------------------------
# HANDLE DIVIDE BY ZERO
# -----------------------------

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# SAVE NEW DATASET
# -----------------------------

df.to_csv("5.feature_engineered.csv", index=False)

print("Feature Engineering Completed")
print(df.head())