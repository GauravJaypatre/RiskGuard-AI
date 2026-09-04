"""
RiskGuard AI — Dataset Generation Orchestrator

Produces the full synthetic dataset: entities, transactions, scenarios,
relationships, and time-based train/val/test splits.

Run this module directly to generate the dataset:
    python -m src.data_generator.generate
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    DATASET_START_DATE,
    DATASET_DAYS,
    TRAIN_DAYS,
    VALIDATION_DAYS,
    TEST_DAYS,
    RAW_DATA_DIR,
    SPLITS_DIR,
    SCENARIOS_DIR,
)
from src.data_generator.entities import (
    generate_merchants,
    generate_customers,
    generate_devices,
)
from src.data_generator.transactions import (
    generate_normal_transactions,
    generate_orders,
    generate_refunds,
    generate_disputes,
    reset_device_cache,
)
from src.data_generator.scenarios import inject_all_scenarios
from src.data_generator.relationships import link_relationships, validate_relationships

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_full_dataset() -> dict:
    """
    Generate the complete synthetic dataset.

    Returns a dict with all DataFrames:
      {merchants, customers, devices, transactions, orders, refunds, disputes,
       train, validation, test}
    """
    rng = np.random.RandomState(RANDOM_SEED)
    reset_device_cache()

    start_date = datetime.fromisoformat(DATASET_START_DATE)

    # ── Step 1: Generate entities ──────────────────────────────────────
    logger.info("Generating merchants...")
    merchants_df = generate_merchants(rng)
    logger.info(f"  → {len(merchants_df)} merchants across {merchants_df['scenario_type'].nunique()} scenarios")

    logger.info("Generating customers...")
    customers_df = generate_customers(merchants_df, rng)
    logger.info(f"  → {len(customers_df)} customers")

    logger.info("Generating devices...")
    devices_df = generate_devices(rng)
    logger.info(f"  → {len(devices_df)} devices")

    # ── Step 2: Generate normal transactions ────────────────────────────
    logger.info("Generating normal transactions...")
    transactions_df = generate_normal_transactions(merchants_df, customers_df, devices_df, rng)
    logger.info(f"  → {len(transactions_df)} normal transactions")

    # ── Step 3: Inject fraud/anomaly scenarios ─────────────────────────
    logger.info("Injecting fraud scenarios...")
    transactions_df = inject_all_scenarios(
        transactions_df, merchants_df, customers_df, devices_df, rng
    )
    fraud_count = transactions_df["is_fraudulent"].sum()
    total_count = len(transactions_df)
    logger.info(f"  → {total_count} total transactions ({fraud_count} fraudulent, "
                f"{fraud_count/total_count:.1%} fraud rate)")

    # ── Step 4: Generate related records ────────────────────────────────
    logger.info("Generating orders, refunds, disputes...")
    orders_df = generate_orders(transactions_df, rng)
    refunds_df = generate_refunds(transactions_df, rng)
    disputes_df = generate_disputes(transactions_df, rng)
    logger.info(f"  → {len(orders_df)} orders, {len(refunds_df)} refunds, {len(disputes_df)} disputes")

    # ── Step 5: Link relationships ──────────────────────────────────────
    logger.info("Linking relationships...")
    transactions_df = link_relationships(transactions_df, orders_df, refunds_df, disputes_df)

    # ── Step 6: Validate ──────────────────────────────────────────────
    logger.info("Validating referential integrity...")
    validation = validate_relationships(
        transactions_df, orders_df, refunds_df, disputes_df, customers_df, merchants_df
    )
    if validation["all_valid"]:
        logger.info("  ✓ All relationships valid")
    else:
        logger.warning(f"  ✗ Validation issues: {validation}")

    # ── Step 7: Time-based splits ───────────────────────────────────────
    logger.info("Splitting by time period...")
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
    transactions_df["day_number"] = (
        (transactions_df["timestamp"] - pd.Timestamp(start_date)).dt.days + 1
    )

    train_df = transactions_df[
        (transactions_df["day_number"] >= TRAIN_DAYS[0]) &
        (transactions_df["day_number"] <= TRAIN_DAYS[1])
    ].copy()
    val_df = transactions_df[
        (transactions_df["day_number"] >= VALIDATION_DAYS[0]) &
        (transactions_df["day_number"] <= VALIDATION_DAYS[1])
    ].copy()
    test_df = transactions_df[
        (transactions_df["day_number"] >= TEST_DAYS[0]) &
        (transactions_df["day_number"] <= TEST_DAYS[1])
    ].copy()

    logger.info(f"  Train:      {len(train_df):>8,} transactions (days {TRAIN_DAYS[0]}-{TRAIN_DAYS[1]})")
    logger.info(f"  Validation: {len(val_df):>8,} transactions (days {VALIDATION_DAYS[0]}-{VALIDATION_DAYS[1]})")
    logger.info(f"  Test:       {len(test_df):>8,} transactions (days {TEST_DAYS[0]}-{TEST_DAYS[1]})")

    # Report scenario distribution in each split
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        scenarios = split_df["scenario_type"].value_counts().to_dict()
        logger.info(f"  {split_name} scenarios: {scenarios}")

    # Verify adversarial scenario is in test only
    slow_ring_train = len(train_df[train_df["scenario_type"] == "slow_ring"])
    slow_ring_test = len(test_df[test_df["scenario_type"] == "slow_ring"])
    if slow_ring_train == 0 and slow_ring_test > 0:
        logger.info(f"  ✓ Slow ring (adversarial): 0 in train, {slow_ring_test} in test")
    else:
        logger.warning(f"  ⚠ Slow ring distribution: {slow_ring_train} in train, {slow_ring_test} in test")

    # ── Step 8: Save to disk ────────────────────────────────────────────
    logger.info("Saving to disk...")
    _save_all(merchants_df, customers_df, devices_df, transactions_df,
              orders_df, refunds_df, disputes_df, train_df, val_df, test_df)

    logger.info("✓ Dataset generation complete!")

    return {
        "merchants": merchants_df,
        "customers": customers_df,
        "devices": devices_df,
        "transactions": transactions_df,
        "orders": orders_df,
        "refunds": refunds_df,
        "disputes": disputes_df,
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }


def _save_all(merchants, customers, devices, transactions,
              orders, refunds, disputes, train, val, test):
    """Save all DataFrames to CSV."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    merchants.to_csv(RAW_DATA_DIR / "merchants.csv", index=False)
    customers.to_csv(RAW_DATA_DIR / "customers.csv", index=False)
    devices.to_csv(RAW_DATA_DIR / "devices.csv", index=False)
    transactions.to_csv(RAW_DATA_DIR / "transactions.csv", index=False)
    orders.to_csv(RAW_DATA_DIR / "orders.csv", index=False)
    refunds.to_csv(RAW_DATA_DIR / "refunds.csv", index=False)
    disputes.to_csv(RAW_DATA_DIR / "disputes.csv", index=False)

    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    # Per-scenario subsets for analysis
    for scenario in transactions["scenario_type"].unique():
        scenario_df = transactions[transactions["scenario_type"] == scenario]
        scenario_df.to_csv(SCENARIOS_DIR / f"{scenario}.csv", index=False)

    logger.info(f"  Saved to {RAW_DATA_DIR}, {SPLITS_DIR}, {SCENARIOS_DIR}")


def load_dataset() -> dict:
    """Load the previously generated dataset from disk."""
    return {
        "merchants": pd.read_csv(RAW_DATA_DIR / "merchants.csv"),
        "customers": pd.read_csv(RAW_DATA_DIR / "customers.csv"),
        "devices": pd.read_csv(RAW_DATA_DIR / "devices.csv"),
        "transactions": pd.read_csv(RAW_DATA_DIR / "transactions.csv"),
        "orders": pd.read_csv(RAW_DATA_DIR / "orders.csv"),
        "refunds": pd.read_csv(RAW_DATA_DIR / "refunds.csv"),
        "disputes": pd.read_csv(RAW_DATA_DIR / "disputes.csv"),
        "train": pd.read_csv(SPLITS_DIR / "train.csv"),
        "validation": pd.read_csv(SPLITS_DIR / "validation.csv"),
        "test": pd.read_csv(SPLITS_DIR / "test.csv"),
    }


if __name__ == "__main__":
    generate_full_dataset()
