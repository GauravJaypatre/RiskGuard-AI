"""
RiskGuard AI — End-to-End Feature Engineering Pipeline

Combines all feature families into a single model-ready feature matrix.
This is the sole entry point for feature engineering.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR

logger = logging.getLogger(__name__)

# All feature families are computed inline below to keep Track B self-contained
# while consuming Track A's contracts.


def build_feature_matrix(
    transactions_df: pd.DataFrame,
    customers_df: Optional[pd.DataFrame] = None,
    devices_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the complete feature matrix from raw transactions.

    Returns (feature_df, feature_names) where feature_df is indexed by
    transaction_id and feature_names is the list of columns the model uses.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Raw transactions with all columns.
    customers_df : Optional[pd.DataFrame]
        Customer profiles (for account age).
    devices_df : Optional[pd.DataFrame]
        Device profiles (for device type encoding).

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (feature_matrix, feature_column_names)
    """
    df = transactions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("transaction_id", drop=False)

    logger.info("Building transaction features...")
    tx_feats = _transaction_features(df)

    logger.info("Building customer features...")
    cust_feats = _customer_features(df, customers_df)

    logger.info("Building device features...")
    dev_feats = _device_features(df)

    logger.info("Building temporal velocity features...")
    vel_feats = _temporal_velocity_features(df)

    logger.info("Building merchant baseline features...")
    merch_feats = _merchant_baseline_features(df)

    logger.info("Building relationship features...")
    rel_feats = _relationship_features(df)

    # Merge all features
    feature_df = pd.concat([tx_feats, cust_feats, dev_feats, vel_feats, merch_feats, rel_feats], axis=1)

    # Fill NaN with 0 (safe for tree models and regularized linear models)
    feature_df = feature_df.fillna(0)

    # Feature names (exclude label and ID columns)
    exclude_cols = {"transaction_id", "merchant_id", "customer_id", "device_id",
                    "is_fraudulent", "scenario_type", "timestamp", "currency",
                    "ip_address", "order_id", "refund_id", "dispute_id"}
    feature_names = [c for c in feature_df.columns if c not in exclude_cols]

    logger.info(f"Feature matrix: {feature_df.shape[0]} rows × {len(feature_names)} features")

    # Save to disk
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURES_DIR / "feature_matrix.csv")

    return feature_df, feature_names


# ═══════════════════════════════════════════════════════════════════════════
# Transaction Features
# ═══════════════════════════════════════════════════════════════════════════

def _transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amount, payment method, hour/day (cyclical), status."""
    feats = pd.DataFrame(index=df.index)

    # Amount (raw + log-transformed)
    feats["amount"] = df["amount"]
    feats["amount_log"] = np.log1p(df["amount"])

    # Payment method (one-hot)
    method_dummies = pd.get_dummies(df["payment_method"], prefix="method")
    feats = feats.join(method_dummies)

    # Hour of day (cyclical encoding)
    hour = df["timestamp"].dt.hour
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week
    dow = df["timestamp"].dt.dayofweek
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Is weekend
    feats["is_weekend"] = (dow >= 5).astype(int)

    # Status
    feats["status_failed"] = (df["status"] == "failed").astype(int)
    feats["status_disputed"] = (df["status"] == "disputed").astype(int)

    # Has refund / dispute
    feats["has_refund"] = df["refund_id"].notna().astype(int)
    feats["has_dispute"] = df["dispute_id"].notna().astype(int)

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Customer Features
# ═══════════════════════════════════════════════════════════════════════════

def _customer_features(df: pd.DataFrame, customers_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-customer aggregates joined back to each transaction."""
    feats = pd.DataFrame(index=df.index)

    # Aggregate customer stats
    cust_stats = df.groupby("customer_id").agg(
        cust_tx_count=("amount", "count"),
        cust_avg_amount=("amount", "mean"),
        cust_std_amount=("amount", "std"),
        cust_total_amount=("amount", "sum"),
        cust_failed_attempts=("status", lambda x: (x == "failed").sum()),
        cust_refund_count=("refund_id", lambda x: x.notna().sum()),
        cust_dispute_count=("dispute_id", lambda x: x.notna().sum()),
    )
    cust_stats["cust_std_amount"] = cust_stats["cust_std_amount"].fillna(0)

    # Join back to each transaction
    feats = feats.join(df[["customer_id"]].join(cust_stats, on="customer_id").drop("customer_id", axis=1))

    # Account age (if customer profiles available)
    if customers_df is not None and "account_created" in customers_df.columns:
        customers_df = customers_df.copy()
        customers_df["account_created"] = pd.to_datetime(customers_df["account_created"])
        acct_age = customers_df.set_index("customer_id")["account_created"]
        tx_dates = df[["customer_id", "timestamp"]].copy()
        tx_dates = tx_dates.join(acct_age, on="customer_id")
        feats["account_age_days"] = (tx_dates["timestamp"] - tx_dates["account_created"]).dt.days
        feats["account_age_days"] = feats["account_age_days"].clip(lower=0).fillna(30)
    else:
        feats["account_age_days"] = 30  # Default

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Device Features
# ═══════════════════════════════════════════════════════════════════════════

def _device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-device aggregates: tx count, unique customers, velocity."""
    feats = pd.DataFrame(index=df.index)

    dev_stats = df.groupby("device_id").agg(
        dev_tx_count=("amount", "count"),
        customers_per_device=("customer_id", "nunique"),
        dev_total_amount=("amount", "sum"),
    )

    feats = feats.join(df[["device_id"]].join(dev_stats, on="device_id").drop("device_id", axis=1))

    # Device velocity: transactions in the last hour (per-device)
    # Approximation: count of device's tx on the same calendar day
    day_device = df.groupby(["device_id", df["timestamp"].dt.date]).size().reset_index(name="device_daily_tx")
    day_device.columns = ["device_id", "date", "device_daily_tx"]
    df_with_date = df.copy()
    df_with_date["date"] = df_with_date["timestamp"].dt.date
    merged = df_with_date[["device_id", "date"]].join(
        day_device.set_index(["device_id", "date"]),
        on=["device_id", "date"]
    )
    feats["device_velocity_1h"] = merged["device_daily_tx"].values / 24.0  # Approximate hourly rate

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Velocity Features
# ═══════════════════════════════════════════════════════════════════════════

def _temporal_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """5m / 30m / 1h / 24h velocity windows per customer and per device."""
    feats = pd.DataFrame(index=df.index)

    # Sort by timestamp for rolling computations
    df_sorted = df.sort_values("timestamp")

    # Per-customer velocity (using pandas rolling on sorted data)
    for window_name, minutes in [("5m", 5), ("30m", 30), ("1h", 60), ("24h", 1440)]:
        # Customer velocity
        cust_vel = (
            df_sorted.groupby("customer_id")["timestamp"]
            .transform(lambda ts: _count_in_window(ts, minutes))
        )
        feats[f"velocity_{window_name}"] = cust_vel.reindex(df.index)

    # Per-device velocity for 1h window
    dev_vel_1h = (
        df_sorted.groupby("device_id")["timestamp"]
        .transform(lambda ts: _count_in_window(ts, 60))
    )
    feats["device_velocity_1h_exact"] = dev_vel_1h.reindex(df.index)

    # Per-merchant velocity for 1h window
    merch_vel_1h = (
        df_sorted.groupby("merchant_id")["timestamp"]
        .transform(lambda ts: _count_in_window(ts, 60))
    )
    feats["merchant_velocity_1h"] = merch_vel_1h.reindex(df.index)

    return feats


def _count_in_window(timestamps: pd.Series, window_minutes: int) -> pd.Series:
    """Count how many timestamps fall within [t - window, t] for each t."""
    ts = timestamps.values.astype("datetime64[s]").astype(np.int64)
    window_secs = window_minutes * 60
    counts = np.zeros(len(ts), dtype=int)

    for i in range(len(ts)):
        # Count how many previous timestamps are within the window
        window_start = ts[i] - window_secs
        counts[i] = np.sum((ts[:i+1] >= window_start) & (ts[:i+1] <= ts[i]))

    return pd.Series(counts, index=timestamps.index)


# ═══════════════════════════════════════════════════════════════════════════
# Merchant Baseline Features
# ═══════════════════════════════════════════════════════════════════════════

def _merchant_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-merchant rolling stats: volume, risk rate, failure rate, refund rate."""
    feats = pd.DataFrame(index=df.index)

    merch_stats = df.groupby("merchant_id").agg(
        merch_volume=("amount", "count"),
        merch_total_amount=("amount", "sum"),
        merch_avg_amount=("amount", "mean"),
        merch_fraud_rate=("is_fraudulent", "mean"),
        merch_failure_rate=("status", lambda x: (x == "failed").mean()),
        merch_refund_rate=("refund_id", lambda x: x.notna().mean()),
    )

    feats = feats.join(
        df[["merchant_id"]].join(merch_stats, on="merchant_id").drop("merchant_id", axis=1)
    )

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Relationship Features
# ═══════════════════════════════════════════════════════════════════════════

def _relationship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shared device count, shared IP count, cluster size."""
    feats = pd.DataFrame(index=df.index)

    # Shared device count: for each customer, how many OTHER customers
    # have used the same device(s)?
    customer_devices = df.groupby("customer_id")["device_id"].apply(set)
    device_customers = df.groupby("device_id")["customer_id"].apply(set)

    shared_device_counts = {}
    for cust_id, devices in customer_devices.items():
        other_customers = set()
        for dev_id in devices:
            if dev_id in device_customers.index:
                other_customers.update(device_customers[dev_id])
        other_customers.discard(cust_id)
        shared_device_counts[cust_id] = len(other_customers)

    feats["shared_device_count"] = df["customer_id"].map(shared_device_counts).fillna(0)

    # Shared IP count (if ip_address column exists)
    if "ip_address" in df.columns:
        customer_ips = df.groupby("customer_id")["ip_address"].apply(set)
        ip_customers = df.groupby("ip_address")["customer_id"].apply(set)

        shared_ip_counts = {}
        for cust_id, ips in customer_ips.items():
            other_customers = set()
            for ip in ips:
                if ip in ip_customers.index:
                    other_customers.update(ip_customers[ip])
            other_customers.discard(cust_id)
            shared_ip_counts[cust_id] = len(other_customers)

        feats["shared_ip_count"] = df["customer_id"].map(shared_ip_counts).fillna(0)
    else:
        feats["shared_ip_count"] = 0

    # Cluster size: connected component size in the customer-device bipartite graph
    # (approximated by shared_device_count + 1)
    feats["cluster_size"] = feats["shared_device_count"] + 1

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def load_feature_matrix() -> Tuple[pd.DataFrame, List[str]]:
    """Load previously computed feature matrix from disk."""
    df = pd.read_csv(FEATURES_DIR / "feature_matrix.csv", index_col=0)
    exclude_cols = {"transaction_id", "merchant_id", "customer_id", "device_id",
                    "is_fraudulent", "scenario_type", "timestamp", "currency",
                    "ip_address", "order_id", "refund_id", "dispute_id"}
    feature_names = [c for c in df.columns if c not in exclude_cols]
    return df, feature_names
