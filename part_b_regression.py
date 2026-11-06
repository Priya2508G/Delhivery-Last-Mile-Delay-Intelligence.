
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]

print("Dataset loaded:", df.shape)
print("Number of features:", len(FEATURES))


# --------------------------------------------------
# 2. TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("\nData split completed")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# --------------------------------------------------
# 3. SCALING CHECK
# --------------------------------------------------

diagnostic_scaler = StandardScaler()

X_train_scaled = diagnostic_scaler.fit_transform(X_train)
X_test_scaled = diagnostic_scaler.transform(X_test)

train_scaled_max_abs_mean = np.abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = np.abs(
    X_test_scaled.mean(axis=0)
).max()

part_b_checks = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {
        "metric": "train_scaled_max_abs_mean",
        "value": train_scaled_max_abs_mean
    },
    {
        "metric": "test_scaled_max_abs_mean",
        "value": test_scaled_max_abs_mean
    }
])

part_b_checks["value"] = part_b_checks["value"].round(4)
part_b_checks.to_csv("outputs/part_b_checks.csv", index=False)


# --------------------------------------------------
# 4. METRIC FUNCTION
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


# --------------------------------------------------
# 5. MEAN BASELINE
# --------------------------------------------------

baseline_prediction = np.full(
    len(y_val),
    y_train.mean()
)

baseline_metrics = calculate_metrics(
    y_val,
    baseline_prediction
)

validation_results = []

validation_results.append({
    "model": "mean_baseline",
    "split": "val",
    **baseline_metrics
})


# --------------------------------------------------
# 6. DECISION TREE DEPTH DIAGNOSTIC
# --------------------------------------------------

depth_values = [1, 2, 3, 4, 6, 8, 12, None]

tree_depth_results = []

for depth in depth_values:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )

    depth_name = "unlimited" if depth is None else str(depth)

    tree_depth_results.append({
        "max_depth": depth_name,
        "train_rmse": round(train_rmse, 4),
        "val_rmse": round(val_rmse, 4)
    })


tree_depth_df = pd.DataFrame(tree_depth_results)

tree_depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False
)

# Select the depth with the lowest validation RMSE.
# Ties are resolved by selecting the smaller depth.

best_tree_row = min(
    tree_depth_results,
    key=lambda row: (
        row["val_rmse"],
        float("inf") if row["max_depth"] == "unlimited"
        else int(row["max_depth"])
    )
)

best_tree_depth = (
    None if best_tree_row["max_depth"] == "unlimited"
    else int(best_tree_row["max_depth"])
)

print("\nSelected decision tree depth:", best_tree_row["max_depth"])


# --------------------------------------------------
# 7. BUILD FOUR REGRESSION MODELS
# --------------------------------------------------

models = {

    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_tree_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


# --------------------------------------------------
# 8. TRAIN AND EVALUATE ON VALIDATION DATA
# --------------------------------------------------

for model_name, model in models.items():

    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)

    metrics = calculate_metrics(
        y_val,
        val_prediction
    )

    validation_results.append({
        "model": model_name,
        "split": "val",
        **metrics
    })

    print(
        f"{model_name}: "
        f"Validation RMSE = {metrics['RMSE']}, "
        f"R2 = {metrics['R2']}"
    )


# --------------------------------------------------
# 9. GRADIENT BOOSTING LEARNING RATE COMPARISON
# --------------------------------------------------

learning_rate_results = []

for rate in [0.01, 0.1, 1.0]:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    prediction = gb_model.predict(X_val)

    metrics = calculate_metrics(y_val, prediction)

    learning_rate_results.append({
        "learning_rate": rate,
        "val_rmse": metrics["RMSE"],
        "val_r2": metrics["R2"]
    })


learning_rate_df = pd.DataFrame(learning_rate_results)

learning_rate_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)


# --------------------------------------------------
# 10. RANK FOUR CANDIDATE MODELS
# --------------------------------------------------

candidate_results = [
    row for row in validation_results
    if row["model"] != "mean_baseline"
]

ranked_models = sorted(
    candidate_results,
    key=lambda row: row["RMSE"]
)

print("\nValidation model ranking:")

for position, result in enumerate(ranked_models, start=1):
    print(
        position,
        result["model"],
        "RMSE:",
        result["RMSE"]
    )


# --------------------------------------------------
# 11. CROSS-VALIDATION FOR TOP TWO MODELS
# --------------------------------------------------

top_two_names = [
    result["model"] for result in ranked_models[:2]
]

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for model_name in top_two_names:

    model = models[model_name]

    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_results.append({
        "model": model_name,
        "mean_r2": round(scores.mean(), 4),
        "std_r2": round(scores.std(), 4)
    })


cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False
)


# --------------------------------------------------
# 12. SELECT FINAL MODEL
# --------------------------------------------------

winner_name = ranked_models[0]["model"]
winner_model = models[winner_name]

print("\nFinal selected model:", winner_name)


# --------------------------------------------------
# 13. FINAL TEST EVALUATION
# --------------------------------------------------

# The test set is used only once, after model selection.

test_prediction = winner_model.predict(X_test)

test_metrics = calculate_metrics(
    y_test,
    test_prediction
)

validation_results.append({
    "model": winner_name,
    "split": "test",
    **test_metrics
})


# --------------------------------------------------
# 14. SAVE ALL REGRESSION METRICS
# --------------------------------------------------

metrics_df = pd.DataFrame(validation_results)

metrics_df = metrics_df[
    ["model", "split", "MAE", "MSE", "RMSE", "R2"]
]

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)


# --------------------------------------------------
# 15. DISPLAY RESULTS
# --------------------------------------------------

print("\n===================================")
print("PART B REGRESSION COMPLETED")
print("===================================")

print("\nBaseline metrics:")
print(baseline_metrics)

print("\nGradient Boosting learning rates:")
print(learning_rate_df.to_string(index=False))

print("\nCross-validation results:")
print(cv_df.to_string(index=False))

print("\nFinal model:", winner_name)

print("\nFinal test metrics:")
print(test_metrics)

print("\nGenerated files:")
print("1. outputs/part_b_checks.csv")
print("2. outputs/part_b_tree_depth.csv")
print("3. outputs/part_b_gb_learning_rate.csv")
print("4. outputs/part_b_cv.csv")
print("5. outputs/part_b_metrics.csv")
