"""
RiskGuard AI — Risk Spike Detector

Computes deviation between current and baseline merchant risk rates.
Raises incidents with risk-increase %, affected payment count,
estimated exposure, and confidence.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_DEVIATION_SIGMA,
    DEFAULT_RISK_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    RISK_LEVEL_THRESHOLDS,
)
from src.schemas import Incident, MerchantBaseline
from src.detection.merchant_baseline import check_deviation


# ═══════════════════════════════════════════════════════════════════════════
# Spike Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_spikes(
    transactions_df: pd.DataFrame,
    baselines: Dict[str, MerchantBaseline],
    threshold_sigma: float = BASELINE_DEVIATION_SIGMA,
    risk_scores: Optional[pd.Series] = None,
) -> List[Incident]:
    """
    Scan all merchants for material deviations from their baselines.

    For each merchant, checks:
      - volume spike (z > threshold)
      - risk-rate spike (z > threshold)
      - failure-rate spike (z > threshold)
      - refund-rate spike (z > threshold)

    If any metric is significantly elevated, creates an Incident.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Current-window transactions (e.g., the last 24h or last day).
    baselines : Dict[str, MerchantBaseline]
        Per-merchant baselines from training period.
    threshold_sigma : float
        Z-score threshold above which a deviation is flagged.
    risk_scores : Optional[pd.Series]
        ML model risk scores indexed by transaction_id.
        If provided, used for exposure estimation. Otherwise falls back
        to is_fraudulent ground truth.

    Returns
    -------
    List[Incident]
        One incident per merchant that has a significant deviation.
    """
    incidents: List[Incident] = []

    for merchant_id, baseline in baselines.items():
        mdf = transactions_df[transactions_df["merchant_id"] == merchant_id]

        if mdf.empty:
            continue

        # Compute z-scores for all metrics
        deviations = check_deviation(merchant_id, transactions_df, baseline)

        # Determine the trigger — pick the highest z-score
        trigger_metric = max(deviations, key=lambda k: deviations[k])
        max_z = deviations[trigger_metric]

        if max_z < threshold_sigma:
            continue  # No significant deviation

        # Determine trigger type
        trigger_type = _z_to_trigger(trigger_metric)

        # Compute risk increase percentage
        risk_increase_pct = _compute_risk_increase(
            deviations.get("risk_rate_z", 0.0),
            baseline.normal_risk_rate,
            baseline.risk_rate_std,
        )

        # Identify flagged transactions
        if risk_scores is not None:
            # Use model scores to identify flagged transactions
            merchant_scores = risk_scores.reindex(mdf["transaction_id"], fill_value=0.0)
            flagged_mask = merchant_scores > DEFAULT_RISK_THRESHOLD
            high_conf_mask = merchant_scores > HIGH_CONFIDENCE_THRESHOLD
            flagged_ids = merchant_scores[flagged_mask].index.tolist()
            high_conf_ids = merchant_scores[high_conf_mask].index.tolist()
        else:
            # Fall back to ground truth labels
            flagged_ids = mdf[mdf["is_fraudulent"]]["transaction_id"].tolist()
            high_conf_ids = flagged_ids  # Without scores, all fraud is "high confidence"

        # Compute exposure
        flagged_amounts = mdf[mdf["transaction_id"].isin(flagged_ids)]["amount"]
        high_conf_amounts = mdf[mdf["transaction_id"].isin(high_conf_ids)]["amount"]
        estimated_exposure = float(flagged_amounts.sum()) if len(flagged_amounts) > 0 else 0.0
        high_confidence_exposure = float(high_conf_amounts.sum()) if len(high_conf_amounts) > 0 else 0.0

        # Confidence: based on sample size and deviation magnitude
        confidence = _compute_confidence(len(mdf), max_z)

        # Risk level
        risk_level = _determine_risk_level(max_z, confidence, risk_increase_pct)

        # Related entities — customers and devices involved in flagged transactions
        related_entities = set()
        if flagged_ids:
            flagged_df = mdf[mdf["transaction_id"].isin(flagged_ids)]
            related_entities.update(flagged_df["customer_id"].unique())
            related_entities.update(flagged_df["device_id"].unique())

        incidents.append(Incident(
            incident_id=f"INC-{uuid.uuid4().hex[:8].upper()}",
            merchant_id=merchant_id,
            trigger_type=trigger_type,
            risk_increase_pct=risk_increase_pct,
            affected_payment_count=len(flagged_ids),
            estimated_exposure=estimated_exposure,
            high_confidence_exposure=high_confidence_exposure,
            confidence=confidence,
            risk_level=risk_level,
            flagged_transaction_ids=flagged_ids,
            related_entity_ids=list(related_entities),
            detected_at=datetime.now(),
            status="active",
        ))

    return incidents


# ═══════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _z_to_trigger(metric_key: str) -> str:
    """Map z-score metric key to incident trigger type."""
    mapping = {
        "volume_z": "spike",
        "risk_rate_z": "spike",
        "failure_rate_z": "spike",
        "refund_rate_z": "spike",
    }
    return mapping.get(metric_key, "spike")


def _compute_risk_increase(risk_z: float, baseline_rate: float, baseline_std: float) -> float:
    """
    Compute the percentage increase in risk rate over baseline.
    Returns 0 if baseline is zero (can't compute a meaningful percentage).
    """
    if baseline_rate <= 0 or np.isnan(baseline_rate):
        return 0.0

    current_rate = baseline_rate + risk_z * baseline_std
    increase_pct = ((current_rate - baseline_rate) / baseline_rate) * 100.0
    return max(0.0, float(increase_pct))


def _compute_confidence(sample_size: int, max_z: float) -> float:
    """
    Compute confidence score (0–1) based on:
      - sample_size: more transactions → more confident
      - max_z: larger deviation → more confident (less likely random)

    Simple sigmoid-style combination.
    """
    # Sample-size component: diminishing returns above 100
    size_factor = min(1.0, sample_size / 100.0)

    # Z-score component: higher z → more confident
    z_factor = min(1.0, max(0.0, (max_z - 1.0) / 4.0))  # Linear from z=1 to z=5

    # Combine with weights
    confidence = 0.4 * size_factor + 0.6 * z_factor
    return round(min(1.0, max(0.0, confidence)), 3)


def _determine_risk_level(max_z: float, confidence: float, risk_increase_pct: float) -> str:
    """
    Determine risk level from deviation severity and confidence.

    Uses a composite score that accounts for:
      - How extreme the deviation is (z-score)
      - How confident we are
      - How much the risk rate has actually increased
    """
    # Composite score: weighted combination
    composite = 0.4 * min(max_z / 5.0, 1.0) + 0.3 * confidence + 0.3 * min(risk_increase_pct / 500.0, 1.0)

    for level, (low, high) in RISK_LEVEL_THRESHOLDS.items():
        if low <= composite < high:
            return level

    return "CRITICAL"  # If composite >= 1.0


def detect_merchant_incidents(
    current_df: pd.DataFrame,
    baselines: Dict[str, MerchantBaseline],
    risk_scores: Optional[pd.Series] = None,
) -> List[Incident]:
    """
    Convenience wrapper: detect spikes across all merchants in a window.
    This is the primary entry point for the spike detector.

    Parameters
    ----------
    current_df : pd.DataFrame
        Current monitoring window transactions.
    baselines : Dict[str, MerchantBaseline]
        Per-merchant baselines.
    risk_scores : Optional[pd.Series]
        ML risk scores keyed by transaction_id.

    Returns
    -------
    List[Incident]
        All detected incidents.
    """
    return detect_spikes(current_df, baselines, risk_scores=risk_scores)
