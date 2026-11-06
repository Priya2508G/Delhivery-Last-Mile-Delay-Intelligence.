
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# --------------------------------------------------
# 1. FOLDERS AND INPUT FILES
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

data = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")


# --------------------------------------------------
# 2. CREATE HUB PROFILE
# --------------------------------------------------

hub_profile = (
    data.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
)

# Merge hub details
hub_profile = hub_profile.merge(
    hubs,
    on="hub_id",
    how="left"
)

hub_profile = (
    hub_profile.sort_values("hub_id")
    .reset_index(drop=True)
)

hub_profile.to_csv(
    "outputs/hub_profile.csv",
    index=False,
    float_format="%.4f"
)

print("\nHub profile created.")
print("Number of hubs:", len(hub_profile))


# --------------------------------------------------
# 3. PREPARE CLUSTERING FEATURES
# --------------------------------------------------

CLUSTER_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

X = hub_profile[CLUSTER_FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 4. CALCULATE CLUSTER SEPARATION
# --------------------------------------------------

separation_results = []

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    distances = pairwise_distances(X_scaled)

    within_distances = []
    across_distances = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            if labels[i] == labels[j]:
                within_distances.append(distances[i, j])
            else:
                across_distances.append(distances[i, j])

    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    ratio = mean_across / mean_within

    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True
    )

    separation_results.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(
            map(str, cluster_sizes)
        )
    })


separation_df = pd.DataFrame(separation_results)

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False,
    float_format="%.4f"
)

print("\nCluster separation results:")
print(separation_df.to_string(index=False))


# --------------------------------------------------
# 5. FINAL THREE-CLUSTER MODEL
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = kmeans.fit_predict(X_scaled)


# --------------------------------------------------
# 6. NAME SEGMENTS BY DELAY RATE
# --------------------------------------------------

cluster_order = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

segment_names = {
    cluster_order[0]: "Standard",
    cluster_order[1]: "Watch",
    cluster_order[2]: "Escalate"
}

hub_profile["segment"] = (
    hub_profile["cluster"].map(segment_names)
)


# --------------------------------------------------
# 7. SAVE HUB SEGMENTS
# --------------------------------------------------

hub_segments = hub_profile[
    ["hub_id", "segment"]
].copy()

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False
)


# --------------------------------------------------
# 8. CREATE SEGMENT PROFILE
# --------------------------------------------------

segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

segment_order = [
    "Standard",
    "Watch",
    "Escalate"
]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = (
    segment_profile.sort_values("segment")
    .reset_index(drop=True)
)

segment_profile["segment"] = (
    segment_profile["segment"].astype(str)
)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 9. FIND TIER VS SEGMENT MISMATCHES
# --------------------------------------------------

tier_to_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_profile["expected_segment"] = (
    hub_profile["hub_tier"].map(tier_to_segment)
)

mismatches = hub_profile[
    hub_profile["segment"] != hub_profile["expected_segment"]
].copy()

# Correct column name: hub_city
mismatch_columns = [
    "hub_id",
    "hub_city",
    "hub_tier",
    "segment",
    "expected_segment"
]

mismatches = (
    mismatches[mismatch_columns]
    .sort_values("hub_id")
)

mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False
)


# --------------------------------------------------
# 10. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\nHub segments:")
print(hub_segments.to_string(index=False))

print("\nSegment profile:")
print(segment_profile.to_string(index=False))

print("\nTier vs segment mismatches:")
print(mismatches.to_string(index=False))


# --------------------------------------------------
# 11. COMPLETION
# --------------------------------------------------

print("\nPart D completed successfully!")

print("\nGenerated files:")
print("1. outputs/hub_profile.csv")
print("2. outputs/part_d_separation.csv")
print("3. outputs/hub_segments.csv")
print("4. outputs/segment_profile.csv")
print("5. outputs/tier_vs_segment_mismatch.csv")
