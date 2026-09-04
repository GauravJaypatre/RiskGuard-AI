"""
RiskGuard AI — Entity Generators

Generates Merchants, Customers, and Devices with realistic profiles.
All random state is seeded from config.RANDOM_SEED for reproducibility.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    NUM_MERCHANTS,
    NUM_CUSTOMERS,
    NUM_DEVICES,
    RANDOM_SEED,
    DATASET_START_DATE,
    DATASET_DAYS,
    SCENARIO_DISTRIBUTION,
)

# ═══════════════════════════════════════════════════════════════════════════
# Merchant Profiles
# ═══════════════════════════════════════════════════════════════════════════

MERCHANT_CATEGORIES = ["ecommerce", "subscription", "digital_services", "marketplace"]

MERCHANT_NAME_PREFIXES = {
    "ecommerce": ["ShopNow", "QuickMart", "StyleBazaar", "GadgetZone", "FreshDeal",
                   "TrendStore", "MegaMart", "BuyEasy", "PrimeGoods", "UrbanShop"],
    "subscription": ["StreamVault", "FitPlan", "CloudSync", "DataPro", "MusicHub",
                      "LearnPro", "NewsDaily", "BoxClub", "MealPrep", "HealthPlus"],
    "digital_services": ["CodeForge", "DesignLab", "CloudHost", "APIConnect", "DataViz",
                          "WebBuilder", "AppScale", "PixelWorks", "DevTools", "AIServe"],
    "marketplace": ["TradeHub", "SkillShare", "FreelanceX", "ArtMarket", "BookSwap",
                     "GigConnect", "RentAll", "CraftBay", "ServiceLink", "LocalFind"],
}

# Average transaction amounts by category (INR)
CATEGORY_AMOUNT_PROFILES = {
    "ecommerce":        {"mean": 2500, "std": 1500, "min": 100, "max": 25000},
    "subscription":     {"mean": 499,  "std": 200,  "min": 99,  "max": 2999},
    "digital_services": {"mean": 5000, "std": 3000, "min": 500, "max": 50000},
    "marketplace":      {"mean": 1500, "std": 1000, "min": 50,  "max": 15000},
}

# Monthly transaction volume by category
CATEGORY_VOLUME_PROFILES = {
    "ecommerce":        {"mean": 5000, "std": 2000},
    "subscription":     {"mean": 3000, "std": 1000},
    "digital_services": {"mean": 2000, "std": 800},
    "marketplace":      {"mean": 4000, "std": 1500},
}


def generate_merchants(rng: np.random.RandomState) -> pd.DataFrame:
    """Generate merchant profiles with assigned scenario types."""
    merchants = []
    start_date = datetime.fromisoformat(DATASET_START_DATE)

    # Assign scenarios to merchants
    scenario_assignments = _assign_scenarios(NUM_MERCHANTS, rng)

    for i in range(NUM_MERCHANTS):
        category = MERCHANT_CATEGORIES[i % len(MERCHANT_CATEGORIES)]
        names = MERCHANT_NAME_PREFIXES[category]
        name = f"{names[i % len(names)]}_{i:03d}"

        vol_profile = CATEGORY_VOLUME_PROFILES[category]
        avg_monthly_vol = max(500, rng.normal(vol_profile["mean"], vol_profile["std"]))

        reg_date = start_date - timedelta(days=rng.randint(180, 730))

        merchants.append({
            "merchant_id": f"MER-{i:04d}",
            "name": name,
            "category": category,
            "registration_date": reg_date,
            "avg_monthly_volume": round(avg_monthly_vol),
            "scenario_type": scenario_assignments[i],
        })

    return pd.DataFrame(merchants)


def generate_customers(
    merchants_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate customer profiles distributed across merchants."""
    customers = []
    start_date = datetime.fromisoformat(DATASET_START_DATE)
    merchant_ids = merchants_df["merchant_id"].tolist()

    for i in range(NUM_CUSTOMERS):
        # Assign to a merchant (weighted by merchant volume)
        volumes = merchants_df["avg_monthly_volume"].values.astype(float)
        probs = volumes / volumes.sum()
        merchant_idx = rng.choice(len(merchant_ids), p=probs)
        merchant_id = merchant_ids[merchant_idx]

        # Account creation: some are old, some are very new
        days_ago = rng.choice([
            rng.randint(30, 365),     # Established customer (70%)
            rng.randint(1, 14),       # New customer (20%)
            rng.randint(365, 730),    # Very old customer (10%)
        ], p=[0.7, 0.2, 0.1])

        account_created = start_date - timedelta(days=int(days_ago))

        # Some customers are synthetic fraud accounts (will be overridden by scenario)
        is_fraud = False  # Scenarios set this later

        customers.append({
            "customer_id": f"CUS-{i:05d}",
            "merchant_id": merchant_id,
            "email": f"user{i}@{'example' if rng.random() > 0.3 else 'test'}.com",
            "account_created": account_created,
            "is_synthetic_fraud": is_fraud,
        })

    return pd.DataFrame(customers)


def generate_devices(rng: np.random.RandomState) -> pd.DataFrame:
    """Generate device profiles."""
    devices = []

    device_types = ["mobile", "desktop", "tablet"]
    device_weights = [0.6, 0.3, 0.1]

    os_by_type = {
        "mobile": (["Android", "iOS"], [0.65, 0.35]),
        "desktop": (["Windows", "macOS", "Linux"], [0.7, 0.2, 0.1]),
        "tablet": (["iPadOS", "Android"], [0.5, 0.5]),
    }

    browsers = {
        "mobile": (["Chrome Mobile", "Safari", "Firefox Mobile"], [0.5, 0.35, 0.15]),
        "desktop": (["Chrome", "Firefox", "Safari", "Edge"], [0.55, 0.2, 0.15, 0.1]),
        "tablet": (["Safari", "Chrome", "Firefox"], [0.45, 0.4, 0.15]),
    }

    for i in range(NUM_DEVICES):
        dtype = rng.choice(device_types, p=device_weights)
        os_choices, os_probs = os_by_type[dtype]
        browser_choices, browser_probs = browsers[dtype]

        devices.append({
            "device_id": f"DEV-{i:05d}",
            "device_type": dtype,
            "os": rng.choice(os_choices, p=os_probs),
            "browser": rng.choice(browser_choices, p=browser_probs),
            "fingerprint": uuid.uuid4().hex[:16],
        })

    return pd.DataFrame(devices)


def _assign_scenarios(num_merchants: int, rng: np.random.RandomState) -> List[str]:
    """Assign scenario types to merchants based on configured distribution."""
    scenarios = []
    for scenario, fraction in SCENARIO_DISTRIBUTION.items():
        count = max(1, round(fraction * num_merchants))
        scenarios.extend([scenario] * count)

    # Trim or pad to exact count
    while len(scenarios) < num_merchants:
        scenarios.append("normal")
    scenarios = scenarios[:num_merchants]

    rng.shuffle(scenarios)
    return scenarios
