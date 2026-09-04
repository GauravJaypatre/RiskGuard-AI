"""
RiskGuard AI — Transaction Generator

Generates normal-behavior payment transactions with linked orders, refunds,
and disputes. Individual scenario injectors are in scenarios.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DATASET_START_DATE, DATASET_DAYS, RANDOM_SEED
from src.data_generator.entities import CATEGORY_AMOUNT_PROFILES


# Payment method distribution
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
PAYMENT_METHOD_WEIGHTS = [0.40, 0.35, 0.15, 0.10]

# IP address pool (simulated)
IP_POOLS = {
    "residential": "192.168.{}.{}",
    "commercial": "10.0.{}.{}",
    "mobile": "172.16.{}.{}",
}


def generate_normal_transactions(
    merchants_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
    target_per_merchant: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Generate normal-behavior transactions for all merchants.

    Each merchant gets a volume proportional to their avg_monthly_volume.
    Transactions are spread across the 100-day dataset period with
    realistic hourly/daily patterns.

    Parameters
    ----------
    merchants_df : pd.DataFrame
        Merchant profiles.
    customers_df : pd.DataFrame
        Customer profiles.
    devices_df : pd.DataFrame
        Device profiles.
    rng : np.random.RandomState
        Seeded random state.
    target_per_merchant : Optional[Dict[str, int]]
        Override per-merchant tx counts.

    Returns
    -------
    pd.DataFrame
        Normal transactions with all required columns.
    """
    all_transactions = []
    start_date = datetime.fromisoformat(DATASET_START_DATE)
    device_ids = devices_df["device_id"].tolist()

    for _, merchant in merchants_df.iterrows():
        merchant_id = merchant["merchant_id"]
        category = merchant["category"]
        amount_profile = CATEGORY_AMOUNT_PROFILES[category]

        # Get this merchant's customers
        merchant_customers = customers_df[
            customers_df["merchant_id"] == merchant_id
        ]["customer_id"].tolist()

        if not merchant_customers:
            continue

        # Target transaction count
        if target_per_merchant and merchant_id in target_per_merchant:
            num_tx = target_per_merchant[merchant_id]
        else:
            # ~100 days of transactions
            daily_avg = merchant["avg_monthly_volume"] / 30.0
            num_tx = int(daily_avg * DATASET_DAYS * rng.uniform(0.8, 1.2))

        # Generate transactions
        for i in range(num_tx):
            # Timestamp: spread across the dataset period with hourly bias
            day_offset = rng.randint(0, DATASET_DAYS)
            hour = _sample_hour(rng, category)
            minute = rng.randint(0, 60)
            second = rng.randint(0, 60)
            timestamp = start_date + timedelta(
                days=day_offset, hours=hour, minutes=minute, seconds=second
            )

            # Customer: mostly regulars, some one-timers
            customer_id = rng.choice(merchant_customers)

            # Device: each customer tends to use 1-2 devices consistently
            device_id = _assign_device(customer_id, device_ids, rng)

            # Amount: category-appropriate
            amount = _sample_amount(amount_profile, rng)

            # Payment method
            method = rng.choice(PAYMENT_METHODS, p=PAYMENT_METHOD_WEIGHTS)

            # Status: most succeed, small fraction fail
            status = rng.choice(
                ["success", "failed"],
                p=[0.95, 0.05]
            )

            # IP address
            ip = _generate_ip(rng)

            all_transactions.append({
                "transaction_id": f"TXN-{merchant_id}-{i:06d}",
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "device_id": device_id,
                "amount": round(amount, 2),
                "currency": "INR",
                "payment_method": method,
                "status": status,
                "is_fraudulent": False,
                "scenario_type": "normal",
                "timestamp": timestamp,
                "ip_address": ip,
                "order_id": None,
                "refund_id": None,
                "dispute_id": None,
            })

    df = pd.DataFrame(all_transactions)
    return df


def generate_orders(transactions_df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate order records for successful transactions."""
    successful = transactions_df[transactions_df["status"] == "success"]
    orders = []

    for _, tx in successful.iterrows():
        orders.append({
            "order_id": f"ORD-{tx['transaction_id'][4:]}",
            "transaction_id": tx["transaction_id"],
            "merchant_id": tx["merchant_id"],
            "customer_id": tx["customer_id"],
            "amount": tx["amount"],
            "status": rng.choice(["paid", "fulfilled"], p=[0.3, 0.7]),
            "created_at": tx["timestamp"],
        })

    # Link back to transactions
    order_df = pd.DataFrame(orders)
    return order_df


def generate_refunds(
    transactions_df: pd.DataFrame,
    rng: np.random.RandomState,
    refund_rate: float = 0.03,
) -> pd.DataFrame:
    """Generate refund records for a fraction of successful transactions."""
    successful = transactions_df[transactions_df["status"] == "success"]
    num_refunds = int(len(successful) * refund_rate)
    refund_indices = rng.choice(successful.index, size=num_refunds, replace=False)

    refunds = []
    reasons = ["product_not_as_described", "duplicate_charge", "customer_request",
               "defective_product", "late_delivery"]

    for idx in refund_indices:
        tx = successful.loc[idx]
        refund_amount = tx["amount"] * rng.uniform(0.5, 1.0)  # Partial or full refund
        refund_delay = timedelta(days=rng.randint(1, 30))

        refunds.append({
            "refund_id": f"REF-{tx['transaction_id'][4:]}",
            "transaction_id": tx["transaction_id"],
            "merchant_id": tx["merchant_id"],
            "customer_id": tx["customer_id"],
            "amount": round(refund_amount, 2),
            "reason": rng.choice(reasons),
            "status": rng.choice(["processed", "initiated"], p=[0.9, 0.1]),
            "created_at": tx["timestamp"] + refund_delay,
        })

    return pd.DataFrame(refunds)


def generate_disputes(
    transactions_df: pd.DataFrame,
    rng: np.random.RandomState,
    dispute_rate: float = 0.005,
) -> pd.DataFrame:
    """Generate dispute / chargeback records."""
    successful = transactions_df[transactions_df["status"] == "success"]
    num_disputes = int(len(successful) * dispute_rate)
    dispute_indices = rng.choice(successful.index, size=num_disputes, replace=False)

    disputes = []
    reasons = ["unauthorized_transaction", "product_not_received", "duplicate_transaction",
               "not_as_described", "credit_not_processed"]

    for idx in dispute_indices:
        tx = successful.loc[idx]
        dispute_delay = timedelta(days=rng.randint(5, 60))

        disputes.append({
            "dispute_id": f"DIS-{tx['transaction_id'][4:]}",
            "transaction_id": tx["transaction_id"],
            "merchant_id": tx["merchant_id"],
            "customer_id": tx["customer_id"],
            "amount": tx["amount"],
            "reason": rng.choice(reasons),
            "status": rng.choice(["open", "won", "lost"], p=[0.3, 0.4, 0.3]),
            "created_at": tx["timestamp"] + dispute_delay,
        })

    return pd.DataFrame(disputes)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

# Customer → device assignment cache (for consistency)
_customer_device_cache: Dict[str, List[str]] = {}


def _assign_device(customer_id: str, device_ids: List[str], rng: np.random.RandomState) -> str:
    """Assign a device to a customer. Each customer uses 1-2 devices consistently."""
    if customer_id not in _customer_device_cache:
        num_devices = rng.choice([1, 2], p=[0.7, 0.3])
        assigned = rng.choice(device_ids, size=num_devices, replace=False).tolist()
        _customer_device_cache[customer_id] = assigned

    devices = _customer_device_cache[customer_id]
    return rng.choice(devices)


def reset_device_cache():
    """Reset the customer-device cache. Call between scenario runs."""
    global _customer_device_cache
    _customer_device_cache = {}


def _sample_hour(rng: np.random.RandomState, category: str) -> int:
    """Sample a transaction hour with business-appropriate distribution."""
    # Peak hours: 10-13 and 18-22
    # Off-peak: 0-6
    hours = list(range(24))
    weights = [
        1, 1, 0.5, 0.5, 0.5, 0.5,  # 0-5: very low
        2, 3, 4, 5,                   # 6-9: morning ramp
        8, 9, 9, 8,                   # 10-13: morning peak
        6, 6, 7, 7,                   # 14-17: afternoon
        9, 10, 10, 8,                 # 18-21: evening peak
        5, 3,                         # 22-23: winding down
    ]
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return int(rng.choice(hours, p=weights))


def _sample_amount(profile: Dict, rng: np.random.RandomState) -> float:
    """Sample a transaction amount from a category's profile."""
    amount = rng.lognormal(
        mean=np.log(profile["mean"]),
        sigma=0.5
    )
    return float(np.clip(amount, profile["min"], profile["max"]))


def _generate_ip(rng: np.random.RandomState) -> str:
    """Generate a realistic-looking IP address."""
    pool = rng.choice(["residential", "commercial", "mobile"], p=[0.6, 0.25, 0.15])
    template = IP_POOLS[pool]
    return template.format(rng.randint(1, 255), rng.randint(1, 255))
