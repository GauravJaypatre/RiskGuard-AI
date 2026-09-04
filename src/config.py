"""
RiskGuard AI — Centralized Configuration

All seeds, constants, model pins, thresholds, and paths in one place.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file if present (before reading env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    load_dotenv(override=True)
except ImportError:
    pass
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
SPLITS_DIR = DATA_DIR / "splits"
SCENARIOS_DIR = DATA_DIR / "scenarios"
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "saved"
CASES_DIR = DATA_DIR / "cases"

# Ensure directories exist
for d in [RAW_DATA_DIR, FEATURES_DIR, SPLITS_DIR, SCENARIOS_DIR, MODELS_DIR, CASES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility — fixed random seed
# ---------------------------------------------------------------------------
RANDOM_SEED = 42  # Used by data generator, train/test splits, all stochastic models

# ---------------------------------------------------------------------------
# Dataset Scale
# ---------------------------------------------------------------------------
NUM_MERCHANTS = 50
NUM_CUSTOMERS = 5_000
NUM_DEVICES = 2_000
TARGET_TRANSACTIONS = 200_000

# Time period: 100 days
DATASET_START_DATE = "2025-01-01"
DATASET_DAYS = 100

# Time-based splits (day numbers, 1-indexed)
TRAIN_DAYS = (1, 60)        # Days 1–60
VALIDATION_DAYS = (61, 80)  # Days 61–80
TEST_DAYS = (81, 100)       # Days 81–100

# ---------------------------------------------------------------------------
# Scenario Distribution (fraction of merchants)
# ---------------------------------------------------------------------------
SCENARIO_DISTRIBUTION = {
    "normal":               0.60,   # 30 merchants — clean baseline behavior
    "fraud_spike":          0.08,   # 4 merchants — obvious velocity spike
    "slow_ring":            0.08,   # 4 merchants — threshold-aware fraud ring
    "device_cluster":       0.06,   # 3 merchants — device-sharing cluster
    "amount_manipulation":  0.06,   # 3 merchants — amounts just below threshold
    "flash_sale":           0.06,   # 3 merchants — legitimate spike (false-positive)
    "post_incident":        0.06,   # 3 merchants — fraud then normalization
}

# ---------------------------------------------------------------------------
# Detection Thresholds
# ---------------------------------------------------------------------------
# Merchant baseline deviation — z-score threshold to flag
BASELINE_DEVIATION_SIGMA = 2.0

# Relationship engine
MIN_CLUSTER_SIZE = 3                # Minimum connected-component size to flag
MAX_CUSTOMERS_PER_DEVICE = 3        # Flag devices shared by more customers than this
CROSS_MERCHANT_SHARING_FLAG = True   # Flag if device appears across merchants

# Naive velocity rule (for adversarial comparison)
NAIVE_MAX_TX_PER_HOUR = 10          # Simple rule: flag customer with > N tx/hour

# ---------------------------------------------------------------------------
# Model Thresholds
# ---------------------------------------------------------------------------
DEFAULT_RISK_THRESHOLD = 0.33       # F1-optimal decision boundary (F1=0.828, 520 FPs, loss ₹9.6L vs ₹16.9L at 0.50)
HIGH_CONFIDENCE_THRESHOLD = 0.8     # Higher-confidence subset for exposure calc
THRESHOLD_SWEEP_POINTS = 100        # Number of points in the threshold lab curve

# ---------------------------------------------------------------------------
# Gemini API — Runtime Model Pinning
# ---------------------------------------------------------------------------
# This is the Gemini model the investigation agent calls AT RUNTIME.
# It is separate from whatever model is being used to write this code.
# Pinned here as a named constant so it's easy to see and change.
AGENT_MODEL = "gemini-3.6-flash"  # runtime investigation agent model (verified active quota)

# API key loaded from environment / .env file
# Set GEMINI_API_KEY in .env or your shell environment.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Agent / Policy
# ---------------------------------------------------------------------------
# Risk level thresholds (mapped from model probability)
RISK_LEVEL_THRESHOLDS = {
    "LOW":      (0.0, 0.3),
    "MEDIUM":   (0.3, 0.6),
    "HIGH":     (0.6, 0.8),
    "CRITICAL": (0.8, 1.0),
}

# Policy-bounded actions per risk level
POLICY_ACTIONS = {
    "LOW": {
        "actions": ["continue_monitoring"],
        "recommendation": "Continue routine monitoring. No immediate action required.",
        "requires_human_authorization": False,
    },
    "MEDIUM": {
        "actions": ["enhanced_verification", "request_additional_documents"],
        "recommendation": "Enhanced verification recommended. Review flagged transactions and request additional identity verification from involved parties.",
        "requires_human_authorization": False,
    },
    "HIGH": {
        "actions": ["merchant_review", "temporary_hold_pending_review", "escalate_to_risk_team"],
        "recommendation": "Merchant review recommended. Temporarily hold suspicious transactions and escalate to risk team for detailed investigation.",
        "requires_human_authorization": False,
    },
    "CRITICAL": {
        "actions": ["restrict_new_transactions", "freeze_settlements", "escalate_to_senior_risk_officer"],
        "recommendation": "Restrictive action recommended — REQUIRES HUMAN AUTHORIZATION. Restrict new transactions, freeze pending settlements, and escalate to senior risk officer for immediate review.",
        "requires_human_authorization": True,
    },
}

# ---------------------------------------------------------------------------
# Mock Mode
# ---------------------------------------------------------------------------
MOCK_MODE_BANNER = "⚠️ MOCK MODE — This investigation was generated without an AI agent. Do not treat this as a real investigation."
MOCK_MODE_API_FIELD = "mock"  # Value of agent_mode when in mock mode
LIVE_MODE_API_FIELD = "live"  # Value of agent_mode when using real Gemini API

# ---------------------------------------------------------------------------
# Drift / Re-baselining
# ---------------------------------------------------------------------------
DRIFT_CRITICAL_DAYS = 5       # Days to keep baseline frozen during active incident
DRIFT_MONITOR_DAYS = 5        # Days in monitoring/validation window
DRIFT_REBASELINE_DAYS = 5     # Days to gradually update baseline
DRIFT_VALIDATION_WINDOW = 5   # Days of clean data required before re-baselining
DRIFT_TOTAL_DAYS = 20         # Total simulation days

# ---------------------------------------------------------------------------
# Financial Parameters (for cost-of-inaction benchmark)
# ---------------------------------------------------------------------------
FP_COST_PER_TRANSACTION = 50.0      # Cost of a false positive (review cost, merchant friction)
FN_COST_MULTIPLIER = 1.0            # FN cost = missed fraud amount * multiplier
CHARGEBACK_FEE = 25.0               # Additional fee per chargeback

# ---------------------------------------------------------------------------
# Latency Benchmark
# ---------------------------------------------------------------------------
LATENCY_BATCH_SIZE = 1000            # Transactions per batch for throughput test

# ---------------------------------------------------------------------------
# Model Version Tracking
# ---------------------------------------------------------------------------
MODEL_VERSION = "riskguard-v1.0"
