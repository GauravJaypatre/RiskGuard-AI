"""
RiskGuard AI — Model Drift / Re-baselining Simulation

After a detected incident subsides, simulates gradual re-baselining
ONLY after a validation window confirms the new state is clean.
Not blind online learning on possibly-contaminated data.

Shows day-by-day baseline movement:
  CRITICAL → MONITOR → NEW NORMAL

Timeline (default 20 days):
  Days 1–5:   CRITICAL — incident active, baseline frozen
  Days 6–10:  MONITOR — fraud rate dropping, validation window begins
  Days 11–15: MONITOR → NEW NORMAL (if validation passes, gradual update)
  Days 16–20: NEW NORMAL — baseline fully updated
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    DRIFT_CRITICAL_DAYS,
    DRIFT_MONITOR_DAYS,
    DRIFT_REBASELINE_DAYS,
    DRIFT_VALIDATION_WINDOW,
    DRIFT_TOTAL_DAYS,
    RANDOM_SEED,
)
from src.schemas import DriftDay, MerchantBaseline
from src.detection.merchant_baseline import update_baseline


def simulate_drift_rebaseline(
    merchant_id: str,
    incident_transactions: pd.DataFrame,
    post_incident_transactions: pd.DataFrame,
    original_baseline: MerchantBaseline,
    validation_window_days: int = DRIFT_VALIDATION_WINDOW,
) -> List[DriftDay]:
    """
    Simulate a 20-day post-incident re-baselining process.

    The simulation generates synthetic day-by-day data showing:
    1. CRITICAL period: incident is active, baseline is frozen
    2. MONITOR period: incident subsiding, validation window in progress
    3. NEW NORMAL: validation passed, baseline gradually updated

    Parameters
    ----------
    merchant_id : str
        The merchant being re-baselined.
    incident_transactions : pd.DataFrame
        Transactions during the incident period (elevated fraud).
    post_incident_transactions : pd.DataFrame
        Transactions after the incident (should be cleaner).
    original_baseline : MerchantBaseline
        The merchant's pre-incident baseline.
    validation_window_days : int
        Number of clean days required before re-baselining starts.

    Returns
    -------
    List[DriftDay]
        Day-by-day timeline of status transitions.
    """
    rng = np.random.RandomState(RANDOM_SEED + hash(merchant_id) % 10000)

    days: List[DriftDay] = []
    start_date = datetime.now() - timedelta(days=DRIFT_TOTAL_DAYS)

    # Compute incident-period risk rate (elevated)
    incident_risk_rate = _compute_risk_rate(incident_transactions, merchant_id)

    # Compute post-incident risk rate (should be low/normal)
    post_risk_rate = _compute_risk_rate(post_incident_transactions, merchant_id)

    # Current baseline is frozen at pre-incident values
    current_baseline_risk = original_baseline.normal_risk_rate

    # Simulate day-by-day
    consecutive_clean_days = 0
    validation_passed = False
    rebaseline_progress = 0.0

    for day_num in range(1, DRIFT_TOTAL_DAYS + 1):
        date = start_date + timedelta(days=day_num)

        # Determine the "actual" current risk rate for this day
        if day_num <= DRIFT_CRITICAL_DAYS:
            # CRITICAL: incident is still active, elevated risk
            status = "CRITICAL"
            # Risk rate decays from incident peak toward post-incident
            decay_factor = day_num / DRIFT_CRITICAL_DAYS
            current_risk = incident_risk_rate * (1 - 0.5 * decay_factor) + post_risk_rate * 0.5 * decay_factor
            current_risk += rng.normal(0, 0.01)  # Add noise
            current_risk = max(0.0, current_risk)
            validation_result = None

        elif day_num <= DRIFT_CRITICAL_DAYS + DRIFT_MONITOR_DAYS:
            # MONITOR: incident subsiding, validation window
            status = "MONITOR"
            # Risk rate should be approaching normal
            monitor_day = day_num - DRIFT_CRITICAL_DAYS
            approach_factor = monitor_day / DRIFT_MONITOR_DAYS
            current_risk = post_risk_rate * (0.5 + 0.5 * approach_factor) + original_baseline.normal_risk_rate * (0.5 - 0.5 * approach_factor)
            current_risk += rng.normal(0, 0.005)
            current_risk = max(0.0, current_risk)

            # Check if this day is "clean"
            clean_threshold = original_baseline.normal_risk_rate + 2 * original_baseline.risk_rate_std
            if current_risk <= max(clean_threshold, 0.05):
                consecutive_clean_days += 1
            else:
                consecutive_clean_days = 0

            validation_result = consecutive_clean_days >= validation_window_days
            if validation_result:
                validation_passed = True

        else:
            # RE-BASELINING / NEW NORMAL
            if validation_passed:
                status = "NEW_NORMAL"
                # Gradually update baseline
                rebaseline_day = day_num - DRIFT_CRITICAL_DAYS - DRIFT_MONITOR_DAYS
                rebaseline_progress = min(1.0, rebaseline_day / DRIFT_REBASELINE_DAYS)

                # Blend baseline toward post-incident normal
                current_baseline_risk = (
                    (1 - rebaseline_progress) * original_baseline.normal_risk_rate
                    + rebaseline_progress * post_risk_rate
                )

                current_risk = post_risk_rate + rng.normal(0, 0.003)
                current_risk = max(0.0, current_risk)
                validation_result = True
            else:
                status = "MONITOR"
                # Still monitoring — validation hasn't passed yet
                current_risk = post_risk_rate + rng.normal(0, 0.005)
                current_risk = max(0.0, current_risk)

                clean_threshold = original_baseline.normal_risk_rate + 2 * original_baseline.risk_rate_std
                if current_risk <= max(clean_threshold, 0.05):
                    consecutive_clean_days += 1
                else:
                    consecutive_clean_days = 0
                validation_result = consecutive_clean_days >= validation_window_days

        days.append(DriftDay(
            day=day_num,
            date=date,
            status=status,
            baseline_risk_rate=round(current_baseline_risk, 4),
            current_risk_rate=round(current_risk, 4),
            validation_passed=validation_result,
        ))

    return days


def _compute_risk_rate(df: pd.DataFrame, merchant_id: str) -> float:
    """Compute fraud rate for a merchant in a DataFrame."""
    mdf = df[df["merchant_id"] == merchant_id] if "merchant_id" in df.columns else df

    if mdf.empty:
        return 0.0

    if "is_fraudulent" not in mdf.columns:
        return 0.0

    return float(mdf["is_fraudulent"].sum() / len(mdf))


def format_drift_timeline(days: List[DriftDay]) -> str:
    """Format drift simulation as a human-readable timeline."""
    lines = [
        "=" * 75,
        "DRIFT / RE-BASELINING TIMELINE",
        "=" * 75,
        "",
        f"{'Day':>4} {'Date':>12} {'Status':<12} {'Baseline':>10} {'Current':>10} {'Valid':>8}",
        "-" * 65,
    ]

    for d in days:
        valid_str = {True: "✓ PASS", False: "✗ FAIL", None: "—"}.get(d.validation_passed, "—")
        lines.append(
            f"{d.day:>4} {str(d.date.date()):>12} {d.status:<12} "
            f"{d.baseline_risk_rate:>10.4f} {d.current_risk_rate:>10.4f} {valid_str:>8}"
        )

    lines.extend(["", "=" * 75])
    return "\n".join(lines)
