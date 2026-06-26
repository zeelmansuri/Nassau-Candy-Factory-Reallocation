from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FACTORIES = pd.DataFrame(
    [
        {"Factory": "Lot's O' Nuts", "Latitude": 32.881893, "Longitude": -111.768036},
        {"Factory": "Wicked Choccy's", "Latitude": 32.076176, "Longitude": -81.088371},
        {"Factory": "Sugar Shack", "Latitude": 48.119140, "Longitude": -96.181150},
        {"Factory": "Secret Factory", "Latitude": 41.446333, "Longitude": -90.565487},
        {"Factory": "The Other Factory", "Latitude": 35.117500, "Longitude": -89.971107},
    ]
)

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

STATE_COORDS = {
    "Alabama": (32.806671, -86.791130),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}

SHIP_MODE_MULTIPLIER = {
    "Same Day": 0.55,
    "First Class": 0.72,
    "Second Class": 0.88,
    "Standard Class": 1.0,
}


@dataclass
class ModelBundle:
    model: Pipeline
    metrics: pd.DataFrame
    feature_columns: list[str]
    training_rows: int


def load_orders(path: str = "data/nassau_candy_orders.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days.clip(lower=0)
    for col in ["Sales", "Units", "Gross Profit", "Cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Current Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP).fillna("Unknown")
    df = df.merge(FACTORIES, left_on="Current Factory", right_on="Factory", how="left")
    state_lat_lon = df["State/Province"].map(STATE_COORDS)
    df["Destination Latitude"] = state_lat_lon.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    df["Destination Longitude"] = state_lat_lon.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    df["Distance Miles"] = haversine_series(
        df["Latitude"], df["Longitude"], df["Destination Latitude"], df["Destination Longitude"]
    )
    df["Margin Rate"] = np.where(df["Sales"] > 0, df["Gross Profit"] / df["Sales"], 0)
    df["Unit Price"] = np.where(df["Units"] > 0, df["Sales"] / df["Units"], 0)
    df["Month"] = df["Order Date"].dt.month.fillna(0).astype(int)
    df["Day Of Week"] = df["Order Date"].dt.dayofweek.fillna(0).astype(int)
    return df.dropna(subset=["Lead Time", "Product Name", "Region", "Ship Mode", "Distance Miles"])


def haversine_series(lat1, lon1, lat2, lon2) -> pd.Series:
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(3958.8 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))


def train_models(df: pd.DataFrame) -> ModelBundle:
    feature_columns = [
        "Product Name",
        "Division",
        "Region",
        "Ship Mode",
        "Current Factory",
        "Distance Miles",
        "Units",
        "Sales",
        "Gross Profit",
        "Margin Rate",
        "Month",
        "Day Of Week",
    ]
    model_df = remove_extreme_outliers(df, "Lead Time")
    x = model_df[feature_columns]
    y = model_df["Lead Time"]
    test_size = 0.25 if len(model_df) > 80 else 0.35
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    categorical = ["Product Name", "Division", "Region", "Ship Mode", "Current Factory"]
    numeric = [c for c in feature_columns if c not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=220, min_samples_leaf=4, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    rows = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        rows.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mean_absolute_error(y_test, preds),
                "R2": r2_score(y_test, preds),
            }
        )
        fitted[name] = pipe
    metrics = pd.DataFrame(rows).sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)
    best_model = fitted[metrics.iloc[0]["Model"]]
    return ModelBundle(best_model, metrics, feature_columns, len(model_df))


def remove_extreme_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1, q3 = df[column].quantile([0.02, 0.98])
    return df[df[column].between(q1, q3)].copy()


def route_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    grouped = (
        df.groupby(["Region", "Ship Mode", "Current Factory"], as_index=False)
        .agg(
            Avg_Lead_Time=("Lead Time", "mean"),
            Avg_Distance=("Distance Miles", "mean"),
            Avg_Margin=("Margin Rate", "mean"),
            Orders=("Order ID", "nunique"),
        )
        .query("Orders >= 2")
    )
    if len(grouped) < n_clusters:
        grouped["Cluster"] = 0
        return grouped
    features = StandardScaler().fit_transform(grouped[["Avg_Lead_Time", "Avg_Distance", "Avg_Margin", "Orders"]])
    grouped["Cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(features)
    cluster_speed = grouped.groupby("Cluster")["Avg_Lead_Time"].mean().sort_values(ascending=False)
    labels = {cluster: f"Cluster {i + 1}: {'Slow' if i == 0 else 'Stable'} routes" for i, cluster in enumerate(cluster_speed.index)}
    grouped["Cluster Label"] = grouped["Cluster"].map(labels)
    return grouped


def simulate_product(
    df: pd.DataFrame,
    bundle: ModelBundle,
    product_name: str,
    region: str,
    ship_mode: str,
    speed_weight: float,
) -> pd.DataFrame:
    product_rows = df[df["Product Name"] == product_name]
    if product_rows.empty:
        product_rows = df
    base = product_rows.iloc[-1].copy()
    if region != "All":
        region_rows = df[df["Region"] == region]
        if not region_rows.empty:
            base["Region"] = region
            base["State/Province"] = region_rows["State/Province"].mode().iloc[0]
            base["Destination Latitude"] = region_rows["Destination Latitude"].median()
            base["Destination Longitude"] = region_rows["Destination Longitude"].median()
    if ship_mode != "All":
        base["Ship Mode"] = ship_mode

    current_factory = PRODUCT_FACTORY_MAP.get(product_name, base.get("Current Factory", "Unknown"))
    current_row = scenario_row(base, current_factory)
    current_prediction = float(bundle.model.predict(current_row[bundle.feature_columns])[0])
    scenarios = []
    for _, factory in FACTORIES.iterrows():
        candidate = scenario_row(base, factory["Factory"])
        pred = float(bundle.model.predict(candidate[bundle.feature_columns])[0])
        distance = float(candidate["Distance Miles"].iloc[0])
        margin_rate = float(base.get("Margin Rate", 0))
        distance_delta = distance - float(current_row["Distance Miles"].iloc[0])
        logistics_penalty = max(distance_delta, 0) * 0.000035
        margin_after = max(margin_rate - logistics_penalty, 0)
        profit_stability = np.clip((margin_after / margin_rate) if margin_rate > 0 else 0.75, 0, 1.15)
        lead_reduction = current_prediction - pred
        lead_reduction_pct = (lead_reduction / current_prediction * 100) if current_prediction else 0
        confidence = confidence_score(product_rows, distance, pred, current_prediction)
        score = speed_weight * max(lead_reduction_pct, -50) + (1 - speed_weight) * profit_stability * 100 + confidence * 8
        risk = risk_label(lead_reduction_pct, profit_stability, confidence)
        scenarios.append(
            {
                "Factory": factory["Factory"],
                "Current Factory": current_factory,
                "Predicted Lead Time": round(max(pred, 0), 2),
                "Lead Time Reduction %": round(lead_reduction_pct, 2),
                "Distance Miles": round(distance, 0),
                "Profit Stability": round(profit_stability * 100, 1),
                "Scenario Confidence": round(confidence * 100, 1),
                "Recommendation Score": round(score, 2),
                "Risk": risk,
                "Is Current": factory["Factory"] == current_factory,
            }
        )
    return pd.DataFrame(scenarios).sort_values("Recommendation Score", ascending=False).reset_index(drop=True)


def scenario_row(base: pd.Series, factory_name: str) -> pd.DataFrame:
    row = base.copy()
    factory = FACTORIES[FACTORIES["Factory"] == factory_name].iloc[0]
    row["Current Factory"] = factory_name
    row["Factory"] = factory_name
    row["Latitude"] = factory["Latitude"]
    row["Longitude"] = factory["Longitude"]
    row["Distance Miles"] = haversine_miles(
        factory["Latitude"], factory["Longitude"], row["Destination Latitude"], row["Destination Longitude"]
    )
    row["Lead Time"] = max(
        1,
        row.get("Lead Time", 1)
        * (row["Distance Miles"] / max(base.get("Distance Miles", row["Distance Miles"]), 1)) ** 0.32
        * SHIP_MODE_MULTIPLIER.get(row.get("Ship Mode"), 1.0),
    )
    return pd.DataFrame([row])


def confidence_score(product_rows: pd.DataFrame, distance: float, predicted: float, baseline: float) -> float:
    support = min(len(product_rows) / 120, 1)
    variance = product_rows["Lead Time"].std()
    stability = 1 if pd.isna(variance) else 1 / (1 + variance / max(product_rows["Lead Time"].mean(), 1))
    extrapolation = 1 / (1 + abs(distance - product_rows["Distance Miles"].median()) / 1800)
    prediction_shift = 1 / (1 + abs(predicted - baseline) / max(baseline, 1))
    return float(np.clip(0.35 * support + 0.3 * stability + 0.2 * extrapolation + 0.15 * prediction_shift, 0.2, 0.98))


def risk_label(lead_reduction_pct: float, profit_stability: float, confidence: float) -> str:
    if profit_stability < 0.82 or confidence < 0.45 or lead_reduction_pct < -10:
        return "High"
    if profit_stability < 0.93 or confidence < 0.62 or lead_reduction_pct < 3:
        return "Medium"
    return "Low"


def recommendation_table(df: pd.DataFrame, bundle: ModelBundle, speed_weight: float, top_n: int = 20) -> pd.DataFrame:
    recs = []
    for product in sorted(df["Product Name"].dropna().unique()):
        product_df = df[df["Product Name"] == product]
        main_region = product_df["Region"].mode().iloc[0]
        main_ship_mode = product_df["Ship Mode"].mode().iloc[0]
        scenarios = simulate_product(df, bundle, product, main_region, main_ship_mode, speed_weight)
        best = scenarios[~scenarios["Is Current"]].iloc[0]
        current = scenarios[scenarios["Is Current"]].iloc[0]
        recs.append(
            {
                "Product Name": product,
                "Current Factory": current["Factory"],
                "Recommended Factory": best["Factory"],
                "Predicted Current Lead Time": current["Predicted Lead Time"],
                "Predicted Recommended Lead Time": best["Predicted Lead Time"],
                "Lead Time Reduction %": best["Lead Time Reduction %"],
                "Profit Stability": best["Profit Stability"],
                "Scenario Confidence": best["Scenario Confidence"],
                "Risk": best["Risk"],
                "Recommendation Score": best["Recommendation Score"],
            }
        )
    return pd.DataFrame(recs).sort_values("Recommendation Score", ascending=False).head(top_n)
