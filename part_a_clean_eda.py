
import pandas as pd
import numpy as np

from src.shipment_prep import clean_shipments, build_features


# ==========================================
# 1. READ INPUT DATA
# ==========================================

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


# ==========================================
# 2. INITIAL DATA CHECKS
# ==========================================

raw_rows = len(raw)

duplicate_rows = int(raw.duplicated().sum())

# Count missing distance after excluding duplicates
unique_raw = raw.drop_duplicates()

missing_distance = int(
    unique_raw["distance_km"].isna().sum()
)

# Exclude duplicates when counting missing weights
missing_weight = int(
    unique_raw["parcel_weight_kg"].isna().sum()
)

sentinel_weight = int(
    (unique_raw["parcel_weight_kg"] > 100).sum()
)

missing_value = int(
    unique_raw["declared_value"].isna().sum()
)

missing_payment = int(
    unique_raw["payment_mode"].isna().sum()
)

service_spellings_before = int(
    raw["service_type"].nunique()
)


# ==========================================
# 3. CLEAN DATA
# ==========================================

clean, medians = clean_shipments(raw)

service_spellings_after = int(
    clean["service_type"].nunique()
)


# ==========================================
# 4. BUILD FEATURES
# ==========================================

df = build_features(clean, hubs)


# ==========================================
# 5. CREATE TARGET VARIABLE
# ==========================================

df["is_delayed"] = (
    df["delivery_hours"] > df["promised_hours"]
).astype(int)


# ==========================================
# 6. SAVE MODELLING TABLE
# ==========================================

df.to_csv(
    "data/modelling_table.csv",
    index=False,
    float_format="%.4f"
)


# ==========================================
# 7. PART A CHECKS
# ==========================================

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows),
    ("rows_after_dedup", len(clean)),
    ("missing_distance_km_before", missing_distance),
    ("missing_parcel_weight_before", missing_weight),
    ("sentinel_weight_rows", sentinel_weight),
    ("missing_declared_value_before", missing_value),
    ("missing_payment_mode_before", missing_payment),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),

    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),

    ("rows_after_hub_merge", len(df)),

    (
        "nulls_in_features_and_target",
        int(
            df[
                [
                    "promised_hours",
                    "distance_km",
                    "parcel_weight_kg",
                    "log_declared_value",
                    "n_stops",
                    "rain_flag",
                    "sale_event_flag",
                    "is_cod",
                    "is_weekend",
                    "dispatch_hour",
                    "hub_tier",
                    "hub_daily_capacity",
                    "delivery_hours",
                    "is_delayed"
                ]
            ].isna().sum().sum()
        )
    ),

    ("delayed_count", int(df["is_delayed"].sum())),
    ("delay_rate", df["is_delayed"].mean()),

    ("hours_q1", df["delivery_hours"].quantile(0.25)),
    ("hours_q3", df["delivery_hours"].quantile(0.75))
]


# ==========================================
# 8. OUTLIER CHECK
# ==========================================

q1 = df["delivery_hours"].quantile(0.25)
q3 = df["delivery_hours"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = int(
    (
        (df["delivery_hours"] < lower_bound) |
        (df["delivery_hours"] > upper_bound)
    ).sum()
)

checks.extend([
    ("hours_lower_bound", lower_bound),
    ("hours_upper_bound", upper_bound),
    ("hours_outlier_count", outlier_count)
])


# ==========================================
# 9. DECLARED VALUE STATISTICS
# ==========================================

checks.extend([
    (
        "declared_value_mean_inr",
        clean["declared_value_inr"].mean()
    ),

    (
        "declared_value_median_inr",
        clean["declared_value_inr"].median()
    ),

    (
        "log_declared_value_mean",
        np.log(clean["declared_value_inr"]).mean()
    ),

    (
        "log_declared_value_median",
        np.log(clean["declared_value_inr"]).median()
    )
])


# Save checks
pd.DataFrame(
    checks,
    columns=["metric", "value"]
).to_csv(
    "outputs/part_a_checks.csv",
    index=False,
    float_format="%.4f"
)


# ==========================================
# 10. DISTANCE BANDS
# ==========================================

df["distance_band"] = pd.cut(
    df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=[
        "0-200",
        "201-400",
        "401-600",
        "601+"
    ],
    include_lowest=True
)


# ==========================================
# 11. DELAY BY GROUP
# ==========================================

group_results = []

for dimension in [
    "service_type",
    "hub_tier",
    "distance_band"
]:

    grouped = df.groupby(
        dimension,
        observed=True
    ).agg(
        shipments=("shipment_id", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    for _, row in grouped.iterrows():

        group_results.append({
            "dimension": dimension,
            "group": row[dimension],
            "shipments": row["shipments"],
            "delayed": row["delayed"],
            "delay_rate": row["delay_rate"],
            "mean_hours": row["mean_hours"]
        })


group_df = pd.DataFrame(group_results)

group_df.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False,
    float_format="%.4f"
)


# ==========================================
# 12. CORRELATION MATRIX
# ==========================================

correlations = df.select_dtypes(
    include=np.number
).corr()

correlations.to_csv(
    "outputs/part_a_correlations.csv",
    index=True,
    float_format="%.4f"
)


# ==========================================
# 13. FINAL RESULTS
# ==========================================

print("\nPART A COMPLETED SUCCESSFULLY!")

print("Modelling table shape:", df.shape)

print("Delayed shipments:", int(df["is_delayed"].sum()))

print("Delay rate:", round(df["is_delayed"].mean(), 4))

print("\nGenerated files:")
print("1. data/modelling_table.csv")
print("2. outputs/part_a_checks.csv")
print("3. outputs/part_a_delay_by_group.csv")
print("4. outputs/part_a_correlations.csv")
