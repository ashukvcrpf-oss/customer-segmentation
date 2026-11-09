"""
segment.py — Customer Segmentation MLOps
Author: Ashutosh Behera | Data Scientist | Bandhan Bank Ltd

Loads K-Means centroids from S3.
Assigns segment to any new customer using centroid mapping.
No retraining needed — instant assignment in milliseconds.
This is for the Bandhan Bank employees to get the banking experience.
"""

import pandas as pd
import numpy as np
import boto3
import io
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()


# ── AWS Configuration — Secure ────────────────────────────────────────────

BUCKET = os.getenv(
    "S3_BUCKET",
    "customer-segmentation-2026-1"
)

REGION = os.getenv(
    "AWS_REGION",
    "ap-southeast-2"
)


# ── Segment Definitions ────────────────────────────────────────────────────
SEGMENT_STRATEGIES = {
    "Champions": {
        "description": "Best customers — buy frequently, high spend, very recent",
        "tone": "premium, exclusive, appreciative",
        "offer": "early access, VIP rewards, no discount needed",
        "urgency": "low",
        "channel": "WhatsApp",
        "color": "#1F4E79"
    },
    "Loyal Customers": {
        "description": "Regular buyers with good spend and recency",
        "tone": "warm, rewarding, friendly",
        "offer": "loyalty points, bundle deals, category expansion",
        "urgency": "low",
        "channel": "Email",
        "color": "#2e7d32"
    },
    "Potential Loyalists": {
        "description": "Recent buyers with low frequency — need nurturing",
        "tone": "educational, welcoming, helpful",
        "offer": "first-time category discount, onboarding series",
        "urgency": "medium",
        "channel": "Push notification",
        "color": "#C55A11"
    },
    "At Risk": {
        "description": "Was loyal — gone quiet — needs urgent win-back",
        "tone": "warm, concerned, personalised",
        "offer": "25% discount, personalised product reminder",
        "urgency": "HIGH",
        "channel": "WhatsApp",
        "color": "#C00000"
    },
    "Cannot Lose": {
        "description": "High-value customer gone silent — URGENT intervention",
        "tone": "personal, sincere, premium",
        "offer": "senior agent call, premium win-back, exclusive offer",
        "urgency": "CRITICAL",
        "channel": "Phone call + WhatsApp",
        "color": "#7030A0"
    },
    "Hibernating": {
        "description": "Low engagement, last bought 3-6 months ago",
        "tone": "re-engagement, value-focused",
        "offer": "reactivation discount, best-seller showcase",
        "urgency": "medium",
        "channel": "Email",
        "color": "#833C00"
    },
    "Lost Customers": {
        "description": "Not bought in 6+ months — last attempt before write-off",
        "tone": "genuine, humble, last chance",
        "offer": "40% off, no conditions — just come back",
        "urgency": "low",
        "channel": "Email",
        "color": "#4472C4"
    }
}

# ── Global cache ───────────────────────────────────────────────────────────
_cache = {}

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

def load_model_artifacts():
    """Load centroids, scaler, and feature list from S3. Cached after first load."""
    global _cache
    if _cache:
        return _cache

    print("Loading model artifacts from S3...")
    s3 = get_s3_client()

    # Load centroids
    obj = s3.get_object(Bucket=BUCKET, Key="models/kmeans_centroids.csv")
    centroids_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    segment_names = centroids_df["segment_name"].values
    centroid_matrix = centroids_df.drop(columns=["segment_name"]).values
    features = centroids_df.drop(columns=["segment_name"]).columns.tolist()

    # Load cluster profiles
    obj = s3.get_object(Bucket=BUCKET, Key="models/cluster_profiles.csv")
    profiles_df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    _cache = {
        "centroid_matrix": centroid_matrix,
        "segment_names":   segment_names,
        "features":        features,
        "profiles_df":     profiles_df,
    }

    print(f"Loaded {len(segment_names)} segment centroids with {len(features)} features")
    return _cache

def preprocess_customer(record: dict, features: list) -> np.ndarray:
    """
    Preprocess a single customer record to match training feature space.
    Applies same encoding and transformation as Notebook 6.
    """
    df = pd.DataFrame([record])

    # ── Ordinal encoding ───────────────────────────────────────────────────
    income_map = {"Low": 0, "Mid": 1, "High": 2, "Ultra": 3}
    tier_map   = {"Tier3": 0, "Tier2": 1, "Metro": 2}
    gender_map = {"Male": 0, "Female": 1, "Other": 2}

    df["income_enc"] = df.get("income_segment", pd.Series(["Mid"])).map(income_map).fillna(1)
    df["tier_enc"]   = df.get("city_tier", pd.Series(["Metro"])).map(tier_map).fillna(1)
    df["gender_enc"] = df.get("gender", pd.Series(["Male"])).map(gender_map).fillna(0)

    # ── Log transforms ─────────────────────────────────────────────────────
    for col in ["monetary_value", "avg_order_value", "tenure_months", "spend_consistency"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── Engineered features ────────────────────────────────────────────────
    df["weighted_rfm"] = (
        df.get("recency_score",   pd.Series([3])) * 0.25 +
        df.get("frequency_score", pd.Series([3])) * 0.35 +
        df.get("monetary_score",  pd.Series([3])) * 0.40
    )
    df["engagement_score"] = (
        (1 - df.get("discount_usage_rate", pd.Series([0.3]))) * 0.30 +
        df.get("loyalty_enrolled", pd.Series([0])) * 0.30 +
        df.get("app_user", pd.Series([0])) * 0.20 +
        df.get("response_to_campaign", pd.Series([0])) * 0.20
    )
    df["loyalty_rate"]       = df.get("frequency", pd.Series([12])) / df.get("tenure_months", pd.Series([12])).clip(lower=1)
    df["risk_score"]         = (df.get("recency_days", pd.Series([30])) / 365).clip(0, 1) * 0.50 + \
                               df.get("purchase_gap_trend", pd.Series([0])).clip(0, 1) * 0.30
    df["premium_engagement"] = (df.get("premium_brand_ratio", pd.Series([0.2])) +
                                df.get("organic_product_ratio", pd.Series([0.1]))) / 2

    # ── Frequency encoding ─────────────────────────────────────────────────
    df["log_city_freq"] = np.log1p(df.get("city_freq", pd.Series([5000])))
    df["log_cat_freq"]  = np.log1p(df.get("cat_freq", pd.Series([7000])))

    # ── Align to feature list ──────────────────────────────────────────────
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0

    X = df[features].fillna(0).values[0]
    return X

def standardize_features(X: np.ndarray, centroid_matrix: np.ndarray) -> np.ndarray:
    """
    Simple standardisation using centroid statistics.
    In production, load the saved StandardScaler from S3.
    """
    mean = centroid_matrix.mean(axis=0)
    std  = centroid_matrix.std(axis=0) + 1e-9
    X_scaled      = (X - mean) / std
    cents_scaled   = (centroid_matrix - mean) / std
    return X_scaled, cents_scaled

def assign_segment(record: dict) -> dict:
    """
    Assign a customer to a segment using centroid mapping.

    Input:  dict of customer features
    Output: dict with segment_name, confidence, distances, strategy

    This is the core production function — runs in milliseconds.
    No retraining ever needed.
    """
    artifacts = load_model_artifacts()
    features       = artifacts["features"]
    centroid_matrix = artifacts["centroid_matrix"]
    segment_names  = artifacts["segment_names"]

    # Preprocess
    X = preprocess_customer(record, features)

    # Standardise
    X_sc, cents_sc = standardize_features(X, centroid_matrix)

    # Euclidean distance to each centroid
    distances = np.sqrt(((cents_sc - X_sc) ** 2).sum(axis=1))

    # Assign to nearest centroid
    best_idx      = np.argmin(distances)
    segment_name  = segment_names[best_idx]
    confidence    = float(1 / (1 + distances[best_idx]))

    # Get strategy
    strategy = SEGMENT_STRATEGIES.get(segment_name, SEGMENT_STRATEGIES["Hibernating"])

    return {
        "segment_name":     str(segment_name),
        "confidence":       round(confidence, 4),
        "distance_to_centroid": round(float(distances[best_idx]), 4),
        "all_distances":    {str(seg): round(float(d), 4) for seg, d in zip(segment_names, distances)},
        "strategy":         strategy,
    }

def assign_batch(df_customers: pd.DataFrame) -> pd.DataFrame:
    """
    Assign segments to a full DataFrame of customers.
    Returns original DataFrame with segment columns added.
    """
    artifacts = load_model_artifacts()
    features  = artifacts["features"]
    cents     = artifacts["centroid_matrix"]
    segs      = artifacts["segment_names"]

    results = []
    for _, row in df_customers.iterrows():
        result = assign_segment(row.to_dict())
        results.append({
            "segment_name":         result["segment_name"],
            "confidence":           result["confidence"],
            "distance_to_centroid": result["distance_to_centroid"],
            "urgency":              result["strategy"]["urgency"],
            "recommended_channel":  result["strategy"]["channel"],
            "campaign_offer":       result["strategy"]["offer"],
        })

    result_df = pd.DataFrame(results)
    return pd.concat([df_customers.reset_index(drop=True), result_df], axis=1)

if __name__ == "__main__":
    # Quick test
    sample = {
        "recency_days": 3, "frequency": 52, "monetary_value": 55000,
        "avg_order_value": 1058, "rfm_score": 14, "recency_score": 5,
        "frequency_score": 5, "monetary_score": 5, "tenure_months": 36,
        "discount_usage_rate": 0.05, "loyalty_enrolled": 1, "app_user": 1,
        "income_segment": "High", "city_tier": "Metro", "gender": "Female",
    }
    result = assign_segment(sample)
    print(f"Segment: {result['segment_name']} | Confidence: {result['confidence']:.2%}")
