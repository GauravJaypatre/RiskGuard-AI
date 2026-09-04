"""
RiskGuard AI — Scenario Injectors

Injects 7 distinct fraud/anomaly scenarios into the synthetic dataset:
  1. normal           — clean baseline behavior (already covered by transaction gen)
  2. fraud_spike      — obvious velocity spike (naive attacker)
  3. slow_ring        — threshold-aware fraud ring (adversarial)
  4. device_cluster   — device-sharing cluster
  5. amount_manipulation — amounts just below review thresholds
  6. flash_sale       — legitimate high-volume event (false-positive case)
  7. post_incident    — fraud spike then normalization (drift testing)

The slow_ring scenario is ONLY injected into the test period (days 81-100)
so it serves as a true adversarial held-out challenge.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DATASET_START_DATE,
    DATASET_DAYS,
    TRAIN_DAYS,
    VALIDATION_DAYS,
    TEST_DAYS,
    NAIVE_MAX_TX_PER_HOUR,
    RANDOM_SEED,
)
from src.data_generator.entities import CATEGORY_AMOUNT_PROFILES
from src.data_generator.transactions import (
    PAYMENT_METHODS,
    PAYMENT_METHOD_WEIGHTS,
    _generate_ip,
    _sample_amount,
)


def inject_all_scenarios(
    transactions_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    Inject all scenario types into the dataset based on merchant assignments.

    Returns the enriched transaction DataFrame with fraud scenarios added.
    """
    start_date = datetime.fromisoformat(DATASET_START_DATE)
    injected = []

    for scenario_type in ["fraud_spike", "slow_ring", "device_cluster",
                          "amount_manipulation", "flash_sale", "post_incident"]:
        scenario_merchants = merchants_df[
            merchants_df["scenario_type"] == scenario_type
        ]

        for _, merchant in scenario_merchants.iterrows():
            merchant_id = merchant["merchant_id"]
            category = merchant["category"]

            # Get customers for this merchant
            mcusts = customers_df[customers_df["merchant_id"] == merchant_id]
            if mcusts.empty:
                continue

            if scenario_type == "fraud_spike":
                fraud_txs = _inject_fraud_spike(
                    merchant_id, category, mcusts, devices_df, rng, start_date
                )
            elif scenario_type == "slow_ring":
                fraud_txs = _inject_slow_ring(
                    merchant_id, category, mcusts, devices_df, rng, start_date
                )
            elif scenario_type == "device_cluster":
                fraud_txs = _inject_device_cluster(
                    merchant_id, category, mcusts, devices_df, rng, start_date
                )
            elif scenario_type == "amount_manipulation":
                fraud_txs = _inject_amount_manipulation(
                    merchant_id, category, mcusts, devices_df, rng, start_date
                )
            elif scenario_type == "flash_sale":
                fraud_txs = _inject_flash_sale(
                    merchant_id, category, mcusts, devices_df, rng, start_date
                )
            elif scenario_type == "post_incident":
                fraud_txs = _inject_post_incident(
                    merchant_id, category, mcusts, devices_df, rng, start_date
                )
            else:
                continue

            injected.extend(fraud_txs)

    if not injected:
        return transactions_df

    injected_df = pd.DataFrame(injected)
    combined = pd.concat([transactions_df, injected_df], ignore_index=True)
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: FRAUD SPIKE (Naive Attacker)
# ═══════════════════════════════════════════════════════════════════════════

def _inject_fraud_spike(
    merchant_id: str,
    category: str,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
    start_date: datetime,
) -> List[Dict]:
    """
    Obvious fraud spike: a single customer makes many rapid transactions.
    Easily caught by naive velocity rules (> 10 tx/hour).

    Injected into BOTH train/validation AND test periods:
      - Train/val: so the model learns to detect this pattern
      - Test: so the adversarial comparison can show naive rules catching it
    """
    txns = []
    amount_profile = CATEGORY_AMOUNT_PROFILES[category]
    device_ids = devices_df["device_id"].tolist()

    # Pick 2-3 "fraudster" customers for TRAIN/VALIDATION
    fraud_custs_tv = customers_df.sample(min(3, len(customers_df)), random_state=rng)

    for _, cust in fraud_custs_tv.iterrows():
        customer_id = cust["customer_id"]
        device_id = rng.choice(device_ids)

        # Spike in train/validation period
        spike_day = rng.randint(TRAIN_DAYS[0], VALIDATION_DAYS[1])
        spike_hour = rng.randint(10, 20)
        num_rapid = rng.randint(20, 40)

        for j in range(num_rapid):
            minute_offset = j * 3  # One tx every ~3 minutes
            timestamp = start_date + timedelta(
                days=spike_day, hours=spike_hour,
                minutes=minute_offset % 60,
                seconds=rng.randint(0, 60)
            )
            if minute_offset >= 120:
                timestamp += timedelta(hours=minute_offset // 60)

            txns.append(_make_fraud_tx(
                merchant_id, customer_id, device_id, amount_profile,
                "fraud_spike", timestamp, rng, prefix=f"FS-TV-{j:04d}",
            ))

    # Pick 2-3 DIFFERENT "fraudster" customers for TEST period
    # Use a FORKED RNG to avoid disturbing the shared rng state that
    # downstream scenario injectors (slow_ring, etc.) depend on.
    # This keeps the original model training data identical.
    test_rng = np.random.RandomState(RANDOM_SEED + 1000)

    remaining_custs = customers_df[
        ~customers_df["customer_id"].isin(fraud_custs_tv["customer_id"])
    ]
    if remaining_custs.empty:
        remaining_custs = customers_df
    fraud_custs_test = remaining_custs.sample(
        min(3, len(remaining_custs)), random_state=test_rng
    )

    for _, cust in fraud_custs_test.iterrows():
        customer_id = cust["customer_id"]
        device_id = test_rng.choice(device_ids)

        # Spike in test period
        spike_day = test_rng.randint(TEST_DAYS[0], TEST_DAYS[1])
        spike_hour = test_rng.randint(10, 20)
        num_rapid = test_rng.randint(20, 40)

        for j in range(num_rapid):
            minute_offset = j * 3
            timestamp = start_date + timedelta(
                days=spike_day, hours=spike_hour,
                minutes=minute_offset % 60,
                seconds=test_rng.randint(0, 60)
            )
            if minute_offset >= 120:
                timestamp += timedelta(hours=minute_offset // 60)

            txns.append(_make_fraud_tx(
                merchant_id, customer_id, device_id, amount_profile,
                "fraud_spike", timestamp, test_rng, prefix=f"FS-TE-{j:04d}",
            ))

    return txns


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: SLOW RING (Threshold-Aware Attacker) — TEST ONLY
# ═══════════════════════════════════════════════════════════════════════════

def _inject_slow_ring(
    merchant_id: str,
    category: str,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
    start_date: datetime,
) -> List[Dict]:
    """
    Threshold-aware fraud ring: 4-6 customers sharing 2-3 devices, each
    staying UNDER the naive velocity limit individually. The fraud is only
    visible when you look at the shared device/customer structure.

    CRITICAL: Only injected into the TEST period (days 81-100) so it
    serves as a true adversarial held-out challenge.
    """
    txns = []
    amount_profile = CATEGORY_AMOUNT_PROFILES[category]

    # Create a ring: 4-6 customers share 2-3 devices
    ring_size = rng.randint(4, 7)  # 4-6 customers
    num_shared_devices = rng.randint(2, 4)  # 2-3 shared devices

    ring_customers = customers_df.sample(
        min(ring_size, len(customers_df)), random_state=rng
    )["customer_id"].tolist()

    # Pick shared devices
    ring_devices = devices_df.sample(
        min(num_shared_devices, len(devices_df)), random_state=rng
    )["device_id"].tolist()

    # Shared IP pool (ring members use overlapping IPs)
    shared_ips = [f"10.42.{rng.randint(1,10)}.{rng.randint(1,255)}" for _ in range(3)]

    # Each customer makes a few transactions per day, spread across TEST days
    # Staying UNDER the naive limit (< 10 tx/hour per customer)
    max_tx_per_hour = NAIVE_MAX_TX_PER_HOUR - 2  # Stay 2 below the limit

    for day_offset in range(TEST_DAYS[0] - 1, TEST_DAYS[1]):
        for customer_id in ring_customers:
            # 3-8 transactions spread across the day
            num_tx = rng.randint(3, min(max_tx_per_hour, 9))
            hours = sorted(rng.choice(range(8, 23), size=num_tx, replace=True))

            for j, hour in enumerate(hours):
                device_id = rng.choice(ring_devices)  # Random shared device
                ip = rng.choice(shared_ips)

                timestamp = start_date + timedelta(
                    days=day_offset, hours=int(hour),
                    minutes=rng.randint(0, 60), seconds=rng.randint(0, 60)
                )

                tx = _make_fraud_tx(
                    merchant_id, customer_id, device_id, amount_profile,
                    "slow_ring", timestamp, rng,
                    prefix=f"SR-D{day_offset}-{j:03d}",
                )
                tx["ip_address"] = ip  # Override with shared IP
                txns.append(tx)

    return txns


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: DEVICE CLUSTER
# ═══════════════════════════════════════════════════════════════════════════

def _inject_device_cluster(
    merchant_id: str,
    category: str,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
    start_date: datetime,
) -> List[Dict]:
    """
    Device-sharing cluster: 1 device used by many different customers.
    Suspicious because legitimate users rarely share devices.
    """
    txns = []
    amount_profile = CATEGORY_AMOUNT_PROFILES[category]

    # One device, 6-10 customers
    shared_device = devices_df.sample(1, random_state=rng)["device_id"].iloc[0]
    num_customers = rng.randint(6, 11)
    cluster_customers = customers_df.sample(
        min(num_customers, len(customers_df)), random_state=rng
    )["customer_id"].tolist()

    # Spread across train + validation periods
    for customer_id in cluster_customers:
        num_tx = rng.randint(5, 15)
        for j in range(num_tx):
            day = rng.randint(TRAIN_DAYS[0], VALIDATION_DAYS[1])
            hour = rng.randint(8, 22)
            timestamp = start_date + timedelta(
                days=day, hours=hour,
                minutes=rng.randint(0, 60), seconds=rng.randint(0, 60)
            )

            txns.append(_make_fraud_tx(
                merchant_id, customer_id, shared_device, amount_profile,
                "device_cluster", timestamp, rng,
                prefix=f"DC-{j:04d}",
            ))

    return txns


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: AMOUNT MANIPULATION
# ═══════════════════════════════════════════════════════════════════════════

def _inject_amount_manipulation(
    merchant_id: str,
    category: str,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
    start_date: datetime,
) -> List[Dict]:
    """
    Amount manipulation: transactions deliberately set just below review
    thresholds (e.g., ₹9,999 when the threshold is ₹10,000).
    """
    txns = []
    device_ids = devices_df["device_id"].tolist()

    # Common review thresholds
    thresholds = [10000, 5000, 25000, 50000]
    fraud_custs = customers_df.sample(min(4, len(customers_df)), random_state=rng)

    for _, cust in fraud_custs.iterrows():
        customer_id = cust["customer_id"]
        device_id = rng.choice(device_ids)
        target_threshold = rng.choice(thresholds)

        num_tx = rng.randint(10, 25)
        for j in range(num_tx):
            # Amount just below the threshold (within 1-5%)
            amount = target_threshold * rng.uniform(0.95, 0.999)

            day = rng.randint(TRAIN_DAYS[0], VALIDATION_DAYS[1])
            hour = rng.randint(8, 22)
            timestamp = start_date + timedelta(
                days=day, hours=hour,
                minutes=rng.randint(0, 60), seconds=rng.randint(0, 60)
            )

            txns.append(_make_fraud_tx(
                merchant_id, customer_id, device_id,
                {"mean": amount, "std": 0, "min": amount * 0.95, "max": amount * 1.01},
                "amount_manipulation", timestamp, rng,
                prefix=f"AM-{j:04d}",
                override_amount=round(amount, 2),
            ))

    return txns


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: FLASH SALE (Legitimate — False Positive Case)
# ═══════════════════════════════════════════════════════════════════════════

def _inject_flash_sale(
    merchant_id: str,
    category: str,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
    start_date: datetime,
) -> List[Dict]:
    """
    Legitimate flash-sale spike: high volume, all GENUINE transactions.
    No fraud — this tests for false positives.

    Key differences from fraud spike:
      - Each customer uses their own device (no sharing)
      - Customer profiles are legitimate (older accounts)
      - Amounts are consistent with merchant pricing
      - No failed payments preceding success
    """
    txns = []
    amount_profile = CATEGORY_AMOUNT_PROFILES[category]
    device_ids = devices_df["device_id"].tolist()

    # Flash sale spans 1-2 days in the test period
    sale_start_day = rng.randint(TEST_DAYS[0], TEST_DAYS[1] - 2)

    # 5x-10x normal volume during the sale
    num_sale_tx = rng.randint(200, 500)

    # Use many different customers (each with their own device)
    available_custs = customers_df.sample(
        min(num_sale_tx, len(customers_df)), random_state=rng, replace=True
    )

    for i, (_, cust) in enumerate(available_custs.iterrows()):
        customer_id = cust["customer_id"]
        # Each customer uses their own device (NOT shared)
        device_id = rng.choice(device_ids)

        day_offset = sale_start_day + (i % 2)  # Spread across 2 days
        hour = rng.randint(8, 23)

        timestamp = start_date + timedelta(
            days=day_offset, hours=hour,
            minutes=rng.randint(0, 60), seconds=rng.randint(0, 60)
        )

        # Flash sale: discounted amounts (30-70% of normal)
        discount = rng.uniform(0.3, 0.7)
        amount = _sample_amount(amount_profile, rng) * discount

        txns.append({
            "transaction_id": f"TXN-{merchant_id}-FL-{i:06d}",
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "device_id": device_id,
            "amount": round(amount, 2),
            "currency": "INR",
            "payment_method": rng.choice(PAYMENT_METHODS, p=PAYMENT_METHOD_WEIGHTS),
            "status": "success",  # Flash sales: almost all succeed
            "is_fraudulent": False,  # ← LEGITIMATE
            "scenario_type": "flash_sale",
            "timestamp": timestamp,
            "ip_address": _generate_ip(rng),
            "order_id": None,
            "refund_id": None,
            "dispute_id": None,
        })

    return txns


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: POST-INCIDENT NORMALIZATION (Drift Testing)
# ═══════════════════════════════════════════════════════════════════════════

def _inject_post_incident(
    merchant_id: str,
    category: str,
    customers_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    rng: np.random.RandomState,
    start_date: datetime,
) -> List[Dict]:
    """
    Post-incident normalization:
      - Days 30-50: fraud spike (incident)
      - Days 51-60: gradual cleanup
      - Days 61-80: back to normal

    Used for drift / re-baselining testing.
    """
    txns = []
    amount_profile = CATEGORY_AMOUNT_PROFILES[category]
    device_ids = devices_df["device_id"].tolist()

    fraud_custs = customers_df.sample(min(3, len(customers_df)), random_state=rng)

    # Phase 1: Fraud spike (days 30-50)
    for _, cust in fraud_custs.iterrows():
        customer_id = cust["customer_id"]
        device_id = rng.choice(device_ids)

        for day in range(30, 51):
            num_tx = rng.randint(5, 12)
            for j in range(num_tx):
                hour = rng.randint(8, 22)
                timestamp = start_date + timedelta(
                    days=day, hours=hour,
                    minutes=rng.randint(0, 60), seconds=rng.randint(0, 60)
                )
                txns.append(_make_fraud_tx(
                    merchant_id, customer_id, device_id, amount_profile,
                    "post_incident", timestamp, rng,
                    prefix=f"PI-{day}-{j:03d}",
                ))

    # Phase 2: Cleanup (days 51-60) — decreasing fraud rate
    for day in range(51, 61):
        decay = 1.0 - (day - 50) / 10.0  # Linear decay
        if rng.random() > decay:
            continue  # Skip this day (fraud stopping)

        customer_id = rng.choice(fraud_custs["customer_id"].tolist())
        device_id = rng.choice(device_ids)
        num_tx = rng.randint(1, 4)

        for j in range(num_tx):
            hour = rng.randint(8, 22)
            timestamp = start_date + timedelta(
                days=day, hours=hour,
                minutes=rng.randint(0, 60), seconds=rng.randint(0, 60)
            )
            txns.append(_make_fraud_tx(
                merchant_id, customer_id, device_id, amount_profile,
                "post_incident", timestamp, rng,
                prefix=f"PI-C-{day}-{j:03d}",
            ))

    return txns


# ═══════════════════════════════════════════════════════════════════════════
# Shared Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_fraud_tx(
    merchant_id: str,
    customer_id: str,
    device_id: str,
    amount_profile: Dict,
    scenario_type: str,
    timestamp: datetime,
    rng: np.random.RandomState,
    prefix: str = "F",
    override_amount: float = None,
) -> Dict:
    """Create a single fraudulent transaction record."""
    amount = override_amount if override_amount else _sample_amount(amount_profile, rng)

    # Fraud transactions have higher failure rate
    status = rng.choice(["success", "failed"], p=[0.7, 0.3])

    return {
        "transaction_id": f"TXN-{merchant_id}-{prefix}-{rng.randint(0, 999999):06d}",
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "device_id": device_id,
        "amount": round(amount, 2),
        "currency": "INR",
        "payment_method": rng.choice(PAYMENT_METHODS, p=PAYMENT_METHOD_WEIGHTS),
        "status": status,
        "is_fraudulent": True,
        "scenario_type": scenario_type,
        "timestamp": timestamp,
        "ip_address": _generate_ip(rng),
        "order_id": None,
        "refund_id": None,
        "dispute_id": None,
    }
