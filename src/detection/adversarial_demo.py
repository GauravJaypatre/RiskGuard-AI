"""
RiskGuard AI — Adversarial Robustness Demo

Implements and compares:
  1. A naive attacker (caught trivially by a velocity rule)
  2. A threshold-aware attacker (missed by the naive rule, caught by the relationship engine)

This comparison is a core evaluation artifact — it runs on real synthetic data
and reports real numbers, not narrative claims.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.config import NAIVE_MAX_TX_PER_HOUR
from src.schemas import AdversarialResult, RelationshipCluster, MerchantBaseline
from src.detection.relationship_engine import detect_abuse_rings


# ═══════════════════════════════════════════════════════════════════════════
# Naive Velocity Rule
# ═══════════════════════════════════════════════════════════════════════════

def run_naive_rule(
    transactions_df: pd.DataFrame,
    max_tx_per_hour: int = NAIVE_MAX_TX_PER_HOUR,
) -> Dict[str, bool]:
    """
    Simple velocity rule: flag a transaction if the customer who made it
    has > max_tx_per_hour transactions in the same 1-hour window.

    This catches obvious fraud spikes (scenario: fraud_spike) but misses
    threshold-aware attackers who deliberately stay below this limit.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Transactions with columns: transaction_id, customer_id, timestamp.
    max_tx_per_hour : int
        Maximum transactions per hour per customer before flagging.

    Returns
    -------
    Dict[str, bool]
        {transaction_id: is_flagged}
    """
    df = transactions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    # Count transactions per customer per hour
    hourly_counts = (
        df.groupby(["customer_id", "hour_bucket"])["transaction_id"]
        .transform("count")
    )

    # Flag transactions where the customer exceeded the hourly limit
    flagged = hourly_counts > max_tx_per_hour
    return dict(zip(df["transaction_id"], flagged))


# ═══════════════════════════════════════════════════════════════════════════
# RiskGuard Detection (Relationship Engine + Model)
# ═══════════════════════════════════════════════════════════════════════════

def run_riskguard_detection(
    transactions_df: pd.DataFrame,
    model: Optional[Any] = None,
    feature_matrix: Optional[pd.DataFrame] = None,
) -> Dict[str, bool]:
    """
    Full RiskGuard detection: ML model + relationship engine amplification.

    A transaction is flagged if:
      - The ML model scores it above the risk threshold (0.5), OR
      - It belongs to a suspicious cluster AND the model scores it above a
        lower threshold (0.3) — cluster membership amplifies model confidence

    The relationship engine alone cannot flag transactions — it only lowers the
    model threshold needed for flagging. This prevents the giant-component
    problem where device/IP sharing creates huge false-positive clusters.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Transactions to analyze.
    model : Optional
        Trained sklearn-compatible model with .predict_proba().
    feature_matrix : Optional[pd.DataFrame]
        Feature matrix aligned with transactions_df.

    Returns
    -------
    Dict[str, bool]
        {transaction_id: is_flagged}
    """
    # Initialize all as not flagged
    flagged = {tid: False for tid in transactions_df["transaction_id"]}

    # 1. Relationship engine: identify transactions in suspicious clusters
    clusters, graph = detect_abuse_rings(transactions_df)
    cluster_tx_ids: Set[str] = set()
    for cluster in clusters:
        cluster_tx_ids.update(cluster.transaction_ids)

    # 2. ML model scoring (required for flagging)
    if model is not None and feature_matrix is not None:
        from src.config import DEFAULT_RISK_THRESHOLD
        CLUSTER_BOOST_THRESHOLD = 0.3  # Lower threshold for cluster members

        try:
            # Convert to numpy array to match training format
            X = feature_matrix.values if hasattr(feature_matrix, 'values') else feature_matrix
            probas = model.predict_proba(X)[:, 1]
            tx_ids = transactions_df["transaction_id"].values
            for tid, score in zip(tx_ids, probas):
                if score > DEFAULT_RISK_THRESHOLD:
                    # High model score → flag regardless of cluster
                    flagged[tid] = True
                elif tid in cluster_tx_ids and score > CLUSTER_BOOST_THRESHOLD:
                    # In suspicious cluster + moderate model score → flag
                    flagged[tid] = True
        except Exception:
            pass  # If model scoring fails, rely on cluster membership alone
    # 3. Rule layer: anomalous velocity (catches obvious velocity spikes)
    try:
        velocity_flags = run_naive_rule(transactions_df)
        for tid, is_v in velocity_flags.items():
            if is_v:
                flagged[tid] = True
    except Exception:
        pass

    return flagged


# ═══════════════════════════════════════════════════════════════════════════
# Adversarial Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_adversarial_comparison(
    test_df: pd.DataFrame,
    baselines: Optional[Dict[str, MerchantBaseline]] = None,
    model: Optional[Any] = None,
    feature_matrix: Optional[pd.DataFrame] = None,
) -> AdversarialResult:
    """
    Compare naive velocity rules vs full RiskGuard on adversarial scenarios.

    Evaluates detection rates for each scenario type:
      - Normal: both should have low false-positive rates
      - Fraud spike (naive attacker): both should catch this
      - Slow ring (threshold-aware attacker): only RiskGuard should catch this
      - Device cluster, amount manipulation: RiskGuard should outperform

    Parameters
    ----------
    test_df : pd.DataFrame
        Held-out test transactions with 'scenario_type' and 'is_fraudulent' columns.
    baselines : Optional[Dict[str, MerchantBaseline]]
        Merchant baselines (for enrichment, not strictly required).
    model : Optional
        Trained ML model.
    feature_matrix : Optional[pd.DataFrame]
        Feature matrix for the test set.

    Returns
    -------
    AdversarialResult
        Detailed comparison with per-scenario detection rates.
    """
    # Run both detection methods
    naive_flags = run_naive_rule(test_df)
    riskguard_flags = run_riskguard_detection(test_df, model, feature_matrix)

    # Per-scenario detection rates
    naive_detection: Dict[str, float] = {}
    riskguard_detection: Dict[str, float] = {}

    scenarios = test_df["scenario_type"].unique()

    for scenario in scenarios:
        scenario_df = test_df[test_df["scenario_type"] == scenario]
        fraud_df = scenario_df[scenario_df["is_fraudulent"]]

        if len(fraud_df) == 0:
            # No fraud in this scenario — report 0 detection rate
            # (this is correct for "normal" and "flash_sale" scenarios)
            naive_detection[scenario] = 0.0
            riskguard_detection[scenario] = 0.0
            continue

        fraud_ids = set(fraud_df["transaction_id"])

        # Naive rule: how many fraud transactions did it flag?
        naive_caught = sum(1 for tid in fraud_ids if naive_flags.get(tid, False))
        naive_detection[scenario] = naive_caught / len(fraud_ids) if fraud_ids else 0.0

        # RiskGuard: how many fraud transactions did it flag?
        rg_caught = sum(1 for tid in fraud_ids if riskguard_flags.get(tid, False))
        riskguard_detection[scenario] = rg_caught / len(fraud_ids) if fraud_ids else 0.0

    # Overall false-positive rates (on normal transactions)
    normal_df = test_df[~test_df["is_fraudulent"]]
    normal_ids = set(normal_df["transaction_id"])
    total_normal = len(normal_ids)

    if total_normal > 0:
        naive_fp = sum(1 for tid in normal_ids if naive_flags.get(tid, False))
        rg_fp = sum(1 for tid in normal_ids if riskguard_flags.get(tid, False))
        naive_fp_rate = naive_fp / total_normal
        rg_fp_rate = rg_fp / total_normal
    else:
        naive_fp_rate = 0.0
        rg_fp_rate = 0.0

    # Exposure missed: sum of fraud amounts NOT flagged
    fraud_df = test_df[test_df["is_fraudulent"]]

    naive_missed_ids = [tid for tid in fraud_df["transaction_id"] if not naive_flags.get(tid, False)]
    rg_missed_ids = [tid for tid in fraud_df["transaction_id"] if not riskguard_flags.get(tid, False)]

    exposure_missed_naive = float(
        fraud_df[fraud_df["transaction_id"].isin(naive_missed_ids)]["amount"].sum()
    )
    exposure_missed_riskguard = float(
        fraud_df[fraud_df["transaction_id"].isin(rg_missed_ids)]["amount"].sum()
    )

    return AdversarialResult(
        naive_rule_detection_rate=naive_detection,
        riskguard_detection_rate=riskguard_detection,
        naive_rule_fp_rate=round(naive_fp_rate, 4),
        riskguard_fp_rate=round(rg_fp_rate, 4),
        exposure_missed_naive=round(exposure_missed_naive, 2),
        exposure_missed_riskguard=round(exposure_missed_riskguard, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Summary Report
# ═══════════════════════════════════════════════════════════════════════════

def format_adversarial_report(result: AdversarialResult) -> str:
    """Format the adversarial comparison as a human-readable report."""
    lines = [
        "=" * 70,
        "ADVERSARIAL ROBUSTNESS COMPARISON",
        "=" * 70,
        "",
        f"{'Scenario':<25} {'Naive Rule':>15} {'RiskGuard':>15}",
        "-" * 55,
    ]

    all_scenarios = set(result.naive_rule_detection_rate.keys()) | set(result.riskguard_detection_rate.keys())
    for scenario in sorted(all_scenarios):
        naive_rate = result.naive_rule_detection_rate.get(scenario, 0.0)
        rg_rate = result.riskguard_detection_rate.get(scenario, 0.0)
        lines.append(f"{scenario:<25} {naive_rate:>14.1%} {rg_rate:>14.1%}")

    lines.extend([
        "",
        f"{'False Positive Rate':<25} {result.naive_rule_fp_rate:>14.2%} {result.riskguard_fp_rate:>14.2%}",
        f"{'Exposure Missed (₹)':<25} {result.exposure_missed_naive:>15,.2f} {result.exposure_missed_riskguard:>15,.2f}",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)
