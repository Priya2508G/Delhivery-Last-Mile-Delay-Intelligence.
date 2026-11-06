
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")

from src.shipment_prep import FEATURES

X = df[FEATURES].copy()

if "is_delayed" in df.columns:
    y = df["is_delayed"].astype(int)
else:
    y = (df["delivery_hours"] > df["promised_hours"]).astype(int)

print("Dataset shape:", df.shape)
print("Features:", len(FEATURES))
print("Delayed shipments:", int(y.sum()))
print("On-time shipments:", int((y == 0).sum()))


# --------------------------------------------------
# 2. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
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

print("\nStratified split completed")
print("Training:", len(X_train), "Delayed:", int(y_train.sum()))
print("Validation:", len(X_val), "Delayed:", int(y_val.sum()))
print("Test:", len(X_test), "Delayed:", int(y_test.sum()))


# --------------------------------------------------
# 3. METRIC FUNCTION
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    return {
        "accuracy": round(accuracy_score(actual, predicted), 4),
        "precision": round(
            precision_score(actual, predicted, zero_division=0), 4
        ),
        "recall": round(
            recall_score(actual, predicted, zero_division=0), 4
        ),
        "f1": round(
            f1_score(actual, predicted, zero_division=0), 4
        )
    }


def get_confusion_values(actual, predicted):

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# --------------------------------------------------
# 4. BLIND ALL-ON-TIME BASELINE
# --------------------------------------------------

baseline_prediction = np.zeros(len(y_val), dtype=int)

baseline_metrics = calculate_metrics(
    y_val,
    baseline_prediction
)

baseline_confusion = get_confusion_values(
    y_val,
    baseline_prediction
)

model_results = []

model_results.append({
    "model": "blind_all_on_time",
    "split": "val",
    **baseline_metrics,
    **baseline_confusion
})


# --------------------------------------------------
# 5. LOGISTIC REGRESSION
# --------------------------------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_val)

logistic_metrics = calculate_metrics(
    y_val,
    logistic_prediction
)

model_results.append({
    "model": "logistic_regression",
    "split": "val",
    **logistic_metrics,
    **get_confusion_values(y_val, logistic_prediction)
})


# --------------------------------------------------
# 6. DECISION TREE DEPTH SELECTION
# --------------------------------------------------

tree_depths = [2, 3, 4, 5, 6, 8, 10]

tree_depth_results = []

for depth in tree_depths:

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    prediction = tree.predict(X_val)

    metrics = calculate_metrics(y_val, prediction)

    tree_depth_results.append({
        "max_depth": depth,
        **metrics
    })


# Highest F1 wins; ties go to smaller depth.

best_tree_result = max(
    tree_depth_results,
    key=lambda row: (row["f1"], -row["max_depth"])
)

best_tree_depth = best_tree_result["max_depth"]

decision_tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_tree_depth,
    random_state=42
)

decision_tree_model.fit(X_train, y_train)

tree_prediction = decision_tree_model.predict(X_val)

tree_metrics = calculate_metrics(
    y_val,
    tree_prediction
)

model_results.append({
    "model": "decision_tree",
    "split": "val",
    **tree_metrics,
    **get_confusion_values(y_val, tree_prediction)
})

print("\nSelected decision tree depth:", best_tree_depth)


# --------------------------------------------------
# 7. RANDOM FOREST BASELINE MODEL
# --------------------------------------------------

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_prediction = forest_model.predict(X_val)

forest_metrics = calculate_metrics(
    y_val,
    forest_prediction
)

model_results.append({
    "model": "random_forest",
    "split": "val",
    **forest_metrics,
    **get_confusion_values(y_val, forest_prediction)
})


# --------------------------------------------------
# 8. SAVE VALIDATION MODEL RESULTS
# --------------------------------------------------

models_df = pd.DataFrame(model_results)

models_df.to_csv(
    "outputs/part_c_models.csv",
    index=False
)

print("\nValidation model results:")
print(models_df.to_string(index=False))


# --------------------------------------------------
# 9. FEATURE IMPORTANCE AND F-SCORES
# --------------------------------------------------

forest_importance = forest_model.feature_importances_

feature_selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

feature_selector_all.fit(X_train, y_train)

f_scores = feature_selector_all.scores_

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "forest_importance": forest_importance,
    "f_score": f_scores
})

feature_scores["importance_rank"] = (
    feature_scores["forest_importance"]
    .rank(ascending=False, method="min")
    .astype(int)
)

feature_scores["f_score_rank"] = (
    feature_scores["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

noise_columns = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

feature_scores["is_noise"] = (
    feature_scores["feature"].isin(noise_columns)
)

feature_scores = feature_scores.sort_values(
    "importance_rank"
)

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)

noise_importance_sum = feature_scores.loc[
    feature_scores["is_noise"],
    "forest_importance"
].sum()

print("\nTop seven features:")
print(
    feature_scores.head(7)[
        ["feature", "forest_importance", "f_score"]
    ].to_string(index=False)
)

print("\nNoise feature importance sum:", round(noise_importance_sum, 4))


# --------------------------------------------------
# 10. SELECTKBEST FEATURE COUNT COMPARISON
# --------------------------------------------------

selectk_results = []

for k in [5, 7, 10, 12, 17]:

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )

    X_val_selected = selector.transform(X_val)

    selected_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    selected_forest.fit(
        X_train_selected,
        y_train
    )

    prediction = selected_forest.predict(
        X_val_selected
    )

    metrics = calculate_metrics(
        y_val,
        prediction
    )

    selectk_results.append({
        "k": k,
        **metrics
    })


selectk_df = pd.DataFrame(selectk_results)

selectk_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False
)

print("\nSelectKBest comparison:")
print(selectk_df.to_string(index=False))


# --------------------------------------------------
# 11. SELECT TOP SEVEN FEATURES FOR TUNING
# --------------------------------------------------

selector_7 = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_7 = selector_7.fit_transform(
    X_train,
    y_train
)

X_val_7 = selector_7.transform(X_val)

selected_feature_names = np.array(FEATURES)[
    selector_7.get_support()
]

print("\nSelected seven features:")
print(list(selected_feature_names))


# --------------------------------------------------
# 12. GRID SEARCH
# --------------------------------------------------

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train_7, y_train)

grid_prediction = grid_search.best_estimator_.predict(
    X_val_7
)

grid_metrics = calculate_metrics(
    y_val,
    grid_prediction
)

grid_best_cv_accuracy = grid_search.best_score_


# --------------------------------------------------
# 13. RANDOMIZED SEARCH
# --------------------------------------------------

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_7, y_train)

random_prediction = random_search.best_estimator_.predict(
    X_val_7
)

random_metrics = calculate_metrics(
    y_val,
    random_prediction
)

random_best_cv_accuracy = random_search.best_score_


# --------------------------------------------------
# 14. DEFAULT FOREST WITH SAME SEVEN FEATURES
# --------------------------------------------------

default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest.fit(X_train_7, y_train)

default_prediction = default_forest.predict(
    X_val_7
)

default_metrics = calculate_metrics(
    y_val,
    default_prediction
)


# --------------------------------------------------
# 15. SAVE TUNING COMPARISON
# --------------------------------------------------

tuning_results = [

    {
        "method": "grid_search",
        "candidate_count": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_cv_accuracy": round(grid_best_cv_accuracy, 4),
        "best_params": json.dumps(
            grid_search.best_params_,
            sort_keys=True
        ),
        "val_f1": grid_metrics["f1"]
    },

    {
        "method": "randomized_search",
        "candidate_count": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_cv_accuracy": round(random_best_cv_accuracy, 4),
        "best_params": json.dumps(
            random_search.best_params_,
            sort_keys=True
        ),
        "val_f1": random_metrics["f1"]
    },

    {
        "method": "default_forest_same_7_features",
        "candidate_count": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_cv_accuracy": np.nan,
        "best_params": "",
        "val_f1": default_metrics["f1"]
    }
]

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False
)

print("\nTuning comparison:")
print(tuning_df.to_string(index=False))


# --------------------------------------------------
# 16. SELECT FINAL TUNING METHOD
# --------------------------------------------------

candidates = [
    {
        "name": "default_forest_same_7_features",
        "priority": 0,
        "val_f1": default_metrics["f1"],
        "params": {
            "n_estimators": 100
        }
    },
    {
        "name": "grid_search",
        "priority": 1,
        "val_f1": grid_metrics["f1"],
        "params": grid_search.best_params_
    },
    {
        "name": "randomized_search",
        "priority": 2,
        "val_f1": random_metrics["f1"],
        "params": random_search.best_params_
    }
]

chosen_candidate = sorted(
    candidates,
    key=lambda row: (-row["val_f1"], row["priority"])
)[0]

chosen_from = chosen_candidate["name"]
chosen_params = chosen_candidate["params"]

print("\nChosen tuning method:", chosen_from)
print("Chosen parameters:", chosen_params)
print("Chosen validation F1:", chosen_candidate["val_f1"])


# --------------------------------------------------
# 17. BUILD FINAL MODEL WITH SELECTKBEST + RANDOM FOREST
# --------------------------------------------------

final_model = Pipeline([
    ("selector", SelectKBest(
        score_func=f_classif,
        k=7
    )),
    ("forest", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])

final_model.fit(X_train, y_train)


# --------------------------------------------------
# 18. VALIDATION PROBABILITIES AND THRESHOLD SELECTION
# --------------------------------------------------

val_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

threshold_results = []

for threshold in threshold_values:

    prediction = (
        val_probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_val,
        prediction
    )

    threshold_results.append({
        "threshold": threshold,
        **metrics
    })


threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)

eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    chosen_threshold = float(
        eligible_thresholds["threshold"].max()
    )
else:
    chosen_threshold = 0.1

print("\nChosen probability threshold:", chosen_threshold)


# --------------------------------------------------
# 19. FINAL TEST EVALUATION - ONCE ONLY
# --------------------------------------------------

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_prediction = (
    test_probabilities >= chosen_threshold
).astype(int)

final_test_metrics = calculate_metrics(
    y_test,
    test_prediction
)

final_test_confusion = get_confusion_values(
    y_test,
    test_prediction
)

final_test_row = {
    "threshold": chosen_threshold,
    **final_test_metrics,
    **final_test_confusion
}

pd.DataFrame([final_test_row]).to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)


# --------------------------------------------------
# 20. SAVE FINAL MODEL CONFIGURATION
# --------------------------------------------------

final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params.get("n_estimators", 100),
    "max_depth": chosen_params.get("max_depth", None),
    "min_samples_leaf": chosen_params.get("min_samples_leaf", 1),
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)


# --------------------------------------------------
# 21. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n========================================")
print("PART C CLASSIFICATION COMPLETED")
print("========================================")

print("\nFinal test metrics:")
print(final_test_metrics)

print("\nFinal test confusion matrix:")
print(final_test_confusion)

print("\nFinal configuration:")
print(final_config.to_string(index=False))

print("\nGenerated files:")
print("1. outputs/part_c_models.csv")
print("2. outputs/part_c_feature_scores.csv")
print("3. outputs/part_c_selectk.csv")
print("4. outputs/part_c_tuning.csv")
print("5. outputs/part_c_thresholds.csv")
print("6. outputs/part_c_final_test.csv")
print("7. outputs/part_c_final_config.csv")
