"""
RiskGuard AI — Merchant Baseline Engine

Learns each merchant's normal volume / risk-rate / failure-rate / refund-rate
from training-period transactions. Provides z-score deviation checks to separate
"one bad transaction" from "an emerging merchant-level incident."
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from src.schemas import MerchantBaseline


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_baselines(transactions_df: pd.DataFrame) -> Dict[str, MerchantBaseline]:
    """
    Compute per-merchant baselines from training-period transactions.

    For each merchant, calculates rolling daily statistics:
      - normal_volume:       mean daily transaction count
      - normal_risk_rate:    mean daily (fraud_tx / total_tx)
      - normal_failure_rate: mean daily (failed_tx / total_tx)
      - normal_refund_rate:  mean daily (refund_tx / total_tx)
    Plus standard deviations for each, used by the spike detector.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Must contain columns: merchant_id, timestamp, is_fraudulent, status,
        refund_id (nullable).

    Returns
    -------
    Dict[str, MerchantBaseline]
        Keyed by merchant_id.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    baselines: Dict[str, MerchantBaseline] = {}

    for merchant_id, mdf in df.groupby("merchant_id"):
        daily = mdf.groupby("date").apply(_daily_stats, include_groups=False).reset_index()

        # Guard against merchants with 0 transactions on some days
        if daily.empty:
            continue

        period_start = mdf["timestamp"].min()
        period_end = mdf["timestamp"].max()

        # Ensure period_start/period_end are datetime objects
        if isinstance(period_start, pd.Timestamp):
            period_start = period_start.to_pydatetime()
        if isinstance(period_end, pd.Timestamp):
            period_end = period_end.to_pydatetime()

        baselines[merchant_id] = MerchantBaseline(
            merchant_id=merchant_id,
            period_start=period_start,
            period_end=period_end,
            normal_volume=float(daily["volume"].mean()),
            normal_risk_rate=float(daily["risk_rate"].mean()),
            normal_failure_rate=float(daily["failure_rate"].mean()),
            normal_refund_rate=float(daily["refund_rate"].mean()),
            volume_std=float(daily["volume"].std(ddof=1)) if len(daily) > 1 else 0.0,
            risk_rate_std=float(daily["risk_rate"].std(ddof=1)) if len(daily) > 1 else 0.0,
            failure_rate_std=float(daily["failure_rate"].std(ddof=1)) if len(daily) > 1 else 0.0,
            refund_rate_std=float(daily["refund_rate"].std(ddof=1)) if len(daily) > 1 else 0.0,
        )

    return baselines


def _daily_stats(day_df: pd.DataFrame) -> pd.Series:
    """Compute single-day statistics for one merchant."""
    total = len(day_df)
    fraud = day_df["is_fraudulent"].sum() if "is_fraudulent" in day_df.columns else 0
    failed = (day_df["status"] == "failed").sum() if "status" in day_df.columns else 0
    refunds = day_df["refund_id"].notna().sum() if "refund_id" in day_df.columns else 0

    return pd.Series({
        "volume": total,
        "risk_rate": fraud / total if total > 0 else 0.0,
        "failure_rate": failed / total if total > 0 else 0.0,
        "refund_rate": refunds / total if total > 0 else 0.0,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Deviation Checking
# ═══════════════════════════════════════════════════════════════════════════

def check_deviation(
    merchant_id: str,
    current_window_df: pd.DataFrame,
    baseline: MerchantBaseline,
) -> Dict[str, float]:
    """
    Compare a current window of transactions against the merchant's baseline.

    Returns z-scores for volume, risk_rate, failure_rate, and refund_rate.
    A positive z-score means the current window is ABOVE the baseline.

    Parameters
    ----------
    merchant_id : str
        Merchant to check.
    current_window_df : pd.DataFrame
        Transactions in the current monitoring window for this merchant.
    baseline : MerchantBaseline
        The merchant's learned baseline.

    Returns
    -------
    Dict[str, float]
        {"volume_z": ..., "risk_rate_z": ..., "failure_rate_z": ..., "refund_rate_z": ...}
    """
    mdf = current_window_df[current_window_df["merchant_id"] == merchant_id]

    if mdf.empty:
        return {"volume_z": 0.0, "risk_rate_z": 0.0, "failure_rate_z": 0.0, "refund_rate_z": 0.0}

    # Compute current-window daily stats
    mdf = mdf.copy()
    mdf["date"] = pd.to_datetime(mdf["timestamp"]).dt.date
    daily = mdf.groupby("date").apply(_daily_stats, include_groups=False).reset_index()
    num_days = len(daily)

    if num_days == 0:
        return {"volume_z": 0.0, "risk_rate_z": 0.0, "failure_rate_z": 0.0, "refund_rate_z": 0.0}

    current_volume = float(daily["volume"].mean())
    current_risk = float(daily["risk_rate"].mean())
    current_failure = float(daily["failure_rate"].mean())
    current_refund = float(daily["refund_rate"].mean())

    return {
        "volume_z": _z_score(current_volume, baseline.normal_volume, baseline.volume_std),
        "risk_rate_z": _z_score(current_risk, baseline.normal_risk_rate, baseline.risk_rate_std),
        "failure_rate_z": _z_score(current_failure, baseline.normal_failure_rate, baseline.failure_rate_std),
        "refund_rate_z": _z_score(current_refund, baseline.normal_refund_rate, baseline.refund_rate_std),
    }


def _z_score(current: float, mean: float, std: float) -> float:
    """Compute z-score, handling zero std gracefully."""
    if std == 0.0 or np.isnan(std):
        # If baseline has zero variance, any deviation is infinite — cap it
        if abs(current - mean) < 1e-9:
            return 0.0
        return 10.0 if current > mean else -10.0  # Capped sentinel
    return (current - mean) / std


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Update (for drift re-baselining)
# ═══════════════════════════════════════════════════════════════════════════

def update_baseline(
    old_baseline: MerchantBaseline,
    new_window_df: pd.DataFrame,
    blend_factor: float = 0.3,
) -> MerchantBaseline:
    """
    Gradually update a baseline by blending old baseline with new window stats.
    Used during post-incident re-baselining (NOT blind online learning).

    blend_factor controls how much weight the new window gets:
      new_value = (1 - blend_factor) * old_value + blend_factor * new_value

    Only call this AFTER validation confirms the new window is clean.

    Parameters
    ----------
    old_baseline : MerchantBaseline
        Existing baseline to update.
    new_window_df : pd.DataFrame
        Clean post-incident transactions.
    blend_factor : float
        Weight for new data. Default 0.3 (conservative update).

    Returns
    -------
    MerchantBaseline
        Updated baseline.
    """
    mdf = new_window_df[new_window_df["merchant_id"] == old_baseline.merchant_id].copy()

    if mdf.empty:
        return old_baseline

    mdf["date"] = pd.to_datetime(mdf["timestamp"]).dt.date
    daily = mdf.groupby("date").apply(_daily_stats, include_groups=False).reset_index()

    if daily.empty:
        return old_baseline

    def _blend(old_val: float, new_val: float) -> float:
        return (1 - blend_factor) * old_val + blend_factor * new_val

    new_period_end = mdf["timestamp"].max()
    if isinstance(new_period_end, pd.Timestamp):
        new_period_end = new_period_end.to_pydatetime()

    return MerchantBaseline(
        merchant_id=old_baseline.merchant_id,
        period_start=old_baseline.period_start,
        period_end=new_period_end,
        normal_volume=_blend(old_baseline.normal_volume, float(daily["volume"].mean())),
        normal_risk_rate=_blend(old_baseline.normal_risk_rate, float(daily["risk_rate"].mean())),
        normal_failure_rate=_blend(old_baseline.normal_failure_rate, float(daily["failure_rate"].mean())),
        normal_refund_rate=_blend(old_baseline.normal_refund_rate, float(daily["refund_rate"].mean())),
        volume_std=_blend(old_baseline.volume_std, float(daily["volume"].std(ddof=1)) if len(daily) > 1 else old_baseline.volume_std),
        risk_rate_std=_blend(old_baseline.risk_rate_std, float(daily["risk_rate"].std(ddof=1)) if len(daily) > 1 else old_baseline.risk_rate_std),
        failure_rate_std=_blend(old_baseline.failure_rate_std, float(daily["failure_rate"].std(ddof=1)) if len(daily) > 1 else old_baseline.failure_rate_std),
        refund_rate_std=_blend(old_baseline.refund_rate_std, float(daily["refund_rate"].std(ddof=1)) if len(daily) > 1 else old_baseline.refund_rate_std),
    )
