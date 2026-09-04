"""
RiskGuard AI — Cost-of-Inaction Benchmark

Compares expected financial loss under three conditions:
  1. No system:         all fraud goes undetected
  2. Static rule engine: naive velocity rules catch obvious fraud
  3. RiskGuard AI:       full system catches both obvious and subtle fraud

Output: bar-chart-ready JSON with loss figures computed from actual synthetic data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    FP_COST_PER_TRANSACTION,
    FN_COST_MULTIPLIER,
    CHARGEBACK_FEE,
    NAIVE_MAX_TX_PER_HOUR,
    MODELS_DIR,
)
from src.schemas import CostCondition

logger = logging.getLogger(__name__)


def compute_cost_of_inaction(
    test_df: pd.DataFrame,
    naive_flags: Dict[str, bool],
    riskguard_flags: Dict[str, bool],
) -> List[CostCondition]:
    """
    Compute financial loss under three conditions.

    Parameters
    ----------
    test_df : pd.DataFrame
        Held-out test transactions with is_fraudulent and amount columns.
    naive_flags : Dict[str, bool]
        {transaction_id: flagged} from naive velocity rule.
    riskguard_flags : Dict[str, bool]
        {transaction_id: flagged} from RiskGuard.

    Returns
    -------
    List[CostCondition]
        Three conditions with computed loss figures.
    """
    fraud_df = test_df[test_df["is_fraudulent"]]
    legit_df = test_df[~test_df["is_fraudulent"]]

    total_fraud_amount = float(fraud_df["amount"].sum())
    total_fraud_count = len(fraud_df)

    conditions = []

    # ── Condition 1: No system ─────────────────────────────────────────
    # All fraud goes undetected. Loss = total fraud amount + chargebacks.
    no_system_loss = total_fraud_amount + total_fraud_count * CHARGEBACK_FEE
    conditions.append(CostCondition(
        name="No System",
        total_loss=round(no_system_loss, 2),
        fp_cost=0.0,
        fn_cost=round(no_system_loss, 2),
        detected_fraud_pct=0.0,
    ))

    # ── Condition 2: Static Rules ──────────────────────────────────────
    naive_tp = sum(1 for tid in fraud_df["transaction_id"] if naive_flags.get(tid, False))
    naive_fn = total_fraud_count - naive_tp
    naive_fp = sum(1 for tid in legit_df["transaction_id"] if naive_flags.get(tid, False))

    naive_fn_amount = float(
        fraud_df[~fraud_df["transaction_id"].isin(
            [tid for tid, flagged in naive_flags.items() if flagged]
        )]["amount"].sum()
    )

    naive_fn_cost = naive_fn_amount * FN_COST_MULTIPLIER + naive_fn * CHARGEBACK_FEE
    naive_fp_cost = naive_fp * FP_COST_PER_TRANSACTION
    naive_total = naive_fn_cost + naive_fp_cost

    conditions.append(CostCondition(
        name="Static Rules",
        total_loss=round(naive_total, 2),
        fp_cost=round(naive_fp_cost, 2),
        fn_cost=round(naive_fn_cost, 2),
        detected_fraud_pct=round(naive_tp / total_fraud_count * 100 if total_fraud_count else 0, 1),
    ))

    # ── Condition 3: RiskGuard AI ──────────────────────────────────────
    rg_tp = sum(1 for tid in fraud_df["transaction_id"] if riskguard_flags.get(tid, False))
    rg_fn = total_fraud_count - rg_tp
    rg_fp = sum(1 for tid in legit_df["transaction_id"] if riskguard_flags.get(tid, False))

    rg_fn_amount = float(
        fraud_df[~fraud_df["transaction_id"].isin(
            [tid for tid, flagged in riskguard_flags.items() if flagged]
        )]["amount"].sum()
    )

    rg_fn_cost = rg_fn_amount * FN_COST_MULTIPLIER + rg_fn * CHARGEBACK_FEE
    rg_fp_cost = rg_fp * FP_COST_PER_TRANSACTION
    rg_total = rg_fn_cost + rg_fp_cost

    conditions.append(CostCondition(
        name="RiskGuard AI",
        total_loss=round(rg_total, 2),
        fp_cost=round(rg_fp_cost, 2),
        fn_cost=round(rg_fn_cost, 2),
        detected_fraud_pct=round(rg_tp / total_fraud_count * 100 if total_fraud_count else 0, 1),
    ))

    # Log the comparison
    logger.info("Cost-of-Inaction Comparison:")
    logger.info(f"{'Condition':<20} {'Total Loss':>15} {'FP Cost':>12} {'FN Cost':>12} {'Detected':>10}")
    logger.info("-" * 70)
    for c in conditions:
        logger.info(
            f"{c.name:<20} ₹{c.total_loss:>13,.2f} ₹{c.fp_cost:>10,.2f} "
            f"₹{c.fn_cost:>10,.2f} {c.detected_fraud_pct:>9.1f}%"
        )

    return conditions


def save_cost_of_inaction(conditions: List[CostCondition]) -> None:
    """Save cost comparison to JSON for dashboard."""
    data = [
        {
            "name": c.name,
            "total_loss": c.total_loss,
            "fp_cost": c.fp_cost,
            "fn_cost": c.fn_cost,
            "detected_fraud_pct": c.detected_fraud_pct,
        }
        for c in conditions
    ]
    filepath = MODELS_DIR / "cost_of_inaction.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Cost-of-inaction saved to {filepath}")


def load_cost_of_inaction() -> List[Dict]:
    """Load cost comparison from disk."""
    with open(MODELS_DIR / "cost_of_inaction.json") as f:
        return json.load(f)
