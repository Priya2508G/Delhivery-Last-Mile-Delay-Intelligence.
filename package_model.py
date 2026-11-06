
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments


# --------------------------------------------------
# 1. CREATE FOLDERS
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

data = pd.read_csv("data/modelling_table.csv")
raw = pd.read_csv("data/raw_shipments.csv")

config = pd.read_csv(
    "outputs/part_c_final_config.csv"
)

print("\nData loaded successfully.")
print("Dataset shape:", data.shape)


# --------------------------------------------------
# 3. PREPARE FEATURES AND TARGET
# --------------------------------------------------

X = data[FEATURES].copy()

if "is_delayed" in data.columns:
    y = data["is_delayed"].astype(int)
else:
    y = (
        data["delivery_hours"] > data["promised_hours"]
    ).astype(int)

print("Number of features:", len(FEATURES))
print("Delayed shipments:", int(y.sum()))


# --------------------------------------------------
# 4. RECREATE EXACT PART C DATA SPLIT
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nData split recreated.")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# --------------------------------------------------
# 5. READ SELECTED MODEL CONFIGURATION
# --------------------------------------------------

settings = config.iloc[0]

selector_k = int(settings["selector_k"])
n_estimators = int(settings["n_estimators"])
min_samples_leaf = int(settings["min_samples_leaf"])

max_depth_value = settings["max_depth"]

if pd.isna(max_depth_value):
    max_depth = None
else:
    max_depth = int(max_depth_value)

threshold = float(settings["threshold"])

print("\nSelected model configuration:")
print("Selector K:", selector_k)
print("Number of trees:", n_estimators)
print("Maximum depth:", max_depth)
print("Minimum samples per leaf:", min_samples_leaf)
print("Threshold:", threshold)


# --------------------------------------------------
# 6. BUILD FINAL MODEL
# --------------------------------------------------

final_model = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "forest",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])


# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

final_model.fit(X_train, y_train)

print("\nFinal model trained successfully.")


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

model_path = "models/delay_model.pkl"

joblib.dump(
    final_model,
    model_path
)

print("\nModel saved:", model_path)


# --------------------------------------------------
# 9. SAVE THRESHOLD
# --------------------------------------------------

threshold_df = pd.DataFrame([{
    "threshold": threshold
}])

threshold_df.to_csv(
    "models/delay_threshold.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 10. SAVE FEATURE ORDER
# --------------------------------------------------

feature_order = pd.DataFrame({
    "feature": FEATURES
})

feature_order.to_csv(
    "models/feature_order.csv",
    index=False
)


# --------------------------------------------------
# 11. SAVE TRAINING IMPUTATION VALUES
# --------------------------------------------------

cleaned_raw, medians = clean_shipments(raw)

impute_values = pd.DataFrame([
    {
        "column": column,
        "median": value
    }
    for column, value in medians.items()
])

impute_values.to_csv(
    "models/impute_values.csv",
    index=False,
    float_format="%.4f"
)

print("\nImputation values saved:")
print(impute_values.to_string(index=False))


# --------------------------------------------------
# 12. RELOAD SAVED MODEL
# --------------------------------------------------

loaded_model = joblib.load(model_path)

print("\nSaved model reloaded successfully.")


# --------------------------------------------------
# 13. VERIFY PREDICTION CONSISTENCY
# --------------------------------------------------

original_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

reloaded_probabilities = loaded_model.predict_proba(
    X_test
)[:, 1]

max_difference = float(
    np.max(
        np.abs(
            original_probabilities - reloaded_probabilities
        )
    )
)

reload_check = pd.DataFrame([{
    "test_rows_scored": len(X_test),
    "reload_max_abs_probability_diff": max_difference,
    "threshold": threshold
}])

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False,
    float_format="%.10f"
)


# --------------------------------------------------
# 14. DISPLAY VERIFICATION
# --------------------------------------------------

print("\nReload verification:")
print(reload_check.to_string(index=False))

if max_difference <= 1e-9:
    print("\nModel reload verification PASSED.")
else:
    print("\nModel reload verification FAILED.")


# --------------------------------------------------
# 15. DISPLAY GENERATED FILES
# --------------------------------------------------

print("\nGenerated model files:")
print("1. models/delay_model.pkl")
print("2. models/delay_threshold.csv")
print("3. models/feature_order.csv")
print("4. models/impute_values.csv")
print("5. outputs/part_d_reload_check.csv")

print("\nPackage model process completed!")
