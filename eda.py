import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. SETUP ---
folder_path = r"C:\Users\Hello\OneDrive\Desktop\ml"
file_path = os.path.join(folder_path, "ready_for_training.csv")

def run_eda():
    if not os.path.exists(file_path):
        print("❌ Error: Run the cleaning script first to create 'ready_for_training.csv'")
        return

    df = pd.read_csv(file_path)
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    # --- CHART 1: Price Distribution ---
    # Does your data have mostly cheap cars or luxury cars?
    plt.subplot(2, 2, 1)
    sns.histplot(df['Price'], kde=True, color='blue')
    plt.title("Distribution of Car Prices")

    # --- CHART 2: Age vs Price (Depreciation) ---
    # This should show a downward slope. If not, your data is messy!
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='Car_Age', y='Price', hue='Fuel_Type')
    plt.title("Car Age vs Price (By Fuel Type)")

    # --- CHART 3: Correlation Heatmap ---
    # Which features actually matter? 1.0 is a perfect match.
    plt.subplot(2, 2, 3)
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")

    # --- CHART 4: Brand vs Price ---
    # Which brand is the most expensive in your 265 records?
    plt.subplot(2, 2, 4)
    df.groupby('Brand')['Price'].mean().sort_values().plot(kind='barh', color='teal')
    plt.title("Average Price by Brand")

    plt.tight_layout()
    
    # Save the EDA results as an image
    plot_path = os.path.join(folder_path, "eda_report.png")
    plt.savefig(plot_path)
    print(f"✅ EDA Charts saved to: {plot_path}")
    plt.show()

    # --- 5. Quick Stats ---
    print("\n--- Top 5 Most Expensive Brands in your Data ---")
    print(df.groupby('Brand')['Price'].mean().sort_values(ascending=False).head())

if __name__ == "__main__":
    run_eda()