
import pandas as pd
import joblib
from pathlib import Path

from src.shipment_prep import clean_shipments, build_features


# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    # 1. Load new shipments and hub information
    raw_new = pd.read_csv(DATA_DIR / "new_shipments.csv")
    hubs = pd.read_csv(DATA_DIR / "hubs.csv")

    print("New shipment data loaded.")
    print("Number of shipments:", len(raw_new))

    # 2. Load saved imputation values
    impute_df = pd.read_csv(MODEL_DIR / "impute_values.csv")

    medians = dict(
        zip(impute_df["column"], impute_df["median"])
    )

    # 3. Clean new shipment data using training medians
    clean_new, _ = clean_shipments(
        raw_new,
        medians=medians
    )

    # 4. Build model features
    feature_table = build_features(clean_new, hubs)

    # 5. Load saved feature order
    feature_df = pd.read_csv(
        MODEL_DIR / "feature_order.csv"
    )

    feature_order = feature_df["feature"].tolist()

    missing_features = [
        col for col in feature_order
        if col not in feature_table.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X_new = feature_table[feature_order]

    # 6. Load trained model and threshold
    model = joblib.load(
        MODEL_DIR / "delay_model.pkl"
    )

    threshold_df = pd.read_csv(
        MODEL_DIR / "delay_threshold.csv"
    )

    threshold = float(
        threshold_df["threshold"].iloc[0]
    )

    # 7. Predict delay probabilities
    probabilities = model.predict_proba(X_new)[:, 1]

    flagged = probabilities >= threshold

    # 8. Load hub segments
    segments = pd.read_csv(
        OUTPUT_DIR / "hub_segments.csv"
    )

    if "segment" not in segments.columns:
        raise ValueError(
            "The hub_segments.csv file must contain a 'segment' column."
        )

    # 9. Prepare shipment action report
    results = pd.DataFrame({
        "shipment_id": clean_new["shipment_id"].values,
        "hub_id": clean_new["hub_id"].values,
        "delay_probability": probabilities.round(4),
        "flagged": flagged
    })

    results = results.merge(
        segments[["hub_id", "segment"]],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # 10. Assign operational actions
    def assign_action(row):

        if row["flagged"] and row["segment"] == "Escalate":
            return "Escalate to hub manager"

        elif row["flagged"]:
            return "Proactive customer call"

        else:
            return "No action"

    results["action"] = results.apply(
        assign_action,
        axis=1
    )

    # 11. Arrange required output columns
    results = results[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action"
        ]
    ]

    # 12. Validate predictions
    if results["shipment_id"].nunique() != len(results):
        raise ValueError("Duplicate shipment IDs found.")

    if results.isnull().any().any():
        raise ValueError("Missing values found in prediction output.")

    # 13. Save output
    output_file = OUTPUT_DIR / "new_shipment_actions.csv"

    results.to_csv(
        output_file,
        index=False,
        float_format="%.4f"
    )

    # 14. Display summary
    print("\nPrediction completed successfully!")

    print("\nPrediction summary:")
    print("Total shipments:", len(results))
    print("Flagged shipments:", int(results["flagged"].sum()))
    print("Not flagged:", int((~results["flagged"]).sum()))

    print("\nOperational actions:")
    print(results["action"].value_counts().to_string())

    print("\nFirst 10 shipment predictions:")
    print(results.head(10).to_string(index=False))

    print("\nOutput saved:", output_file)


if __name__ == "__main__":
    main()
