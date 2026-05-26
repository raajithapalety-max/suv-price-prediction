import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv("5.feature_engineered.csv")

print(df.head())

print(df.info())


# ------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# ------------------------------------------------

cat_cols = [

    "Brand",

    "Model",

    "Fuel_Type",

    "Transmission"

]

encoders = {}

for c in cat_cols:

    le = LabelEncoder()

    df[c] = le.fit_transform(df[c])

    encoders[c] = le


# ------------------------------------------------
# USE ONLY REALISTIC FEATURES
# ------------------------------------------------

feature_cols = [

    "Brand",

    "Model",

    "Car_Age",

    "Kilometers_Driven",

    "Fuel_Type",

    "Transmission",

    "Owner_Count",

    "Estimated_Original_Price",

    "Core_Systems_Score",

    "Supporting_Systems_Score",

    "Interiors_AC_Score",

    "Exteriors_Lights_Score",

    "Wear_Tear_Parts_Score"

]

X = df[feature_cols]

y = df["Price"]


# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


# ------------------------------------------------
# XGBOOST MODEL
# ------------------------------------------------

model = XGBRegressor(

    objective='reg:squarederror',

    n_estimators=200,

    learning_rate=0.05,

    max_depth=5,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42

)


# ------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------

print("\nTraining XGBoost Model...\n")

model.fit(X_train, y_train)

print("Training Completed")


# ------------------------------------------------
# PREDICTIONS
# ------------------------------------------------

pred = model.predict(X_test)


# ------------------------------------------------
# METRICS
# ------------------------------------------------

mae = mean_absolute_error(

    y_test,

    pred

)

mse = mean_squared_error(

    y_test,

    pred

)

rmse = np.sqrt(mse)

r2 = r2_score(

    y_test,

    pred

)


# ------------------------------------------------
# RESULTS
# ------------------------------------------------

print("\n===================================")

print("XGBOOST RESULTS")

print("===================================")

print(f"MAE   : {mae:.2f}")

print(f"MSE   : {mse:.2f}")

print(f"RMSE  : {rmse:.2f}")

print(f"R2    : {r2:.4f}")


# ------------------------------------------------
# CROSS VALIDATION
# ------------------------------------------------

print("\nRunning Cross Validation...\n")

cv_scores = cross_val_score(

    model,

    X,

    y,

    cv=5,

    scoring="r2"

)

print("Cross Validation Scores:")

print(cv_scores)

print(f"\nAverage CV R2 : {cv_scores.mean():.4f}")


# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------

importance = pd.DataFrame({

    "Feature": feature_cols,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n===================================")

print("FEATURE IMPORTANCE")

print("===================================")

print(importance)


# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------

joblib.dump(

    model,

    "xgboost_suv_price_model.pkl"

)

print("\nModel Saved Successfully")