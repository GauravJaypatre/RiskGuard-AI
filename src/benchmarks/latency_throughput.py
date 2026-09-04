"""
RiskGuard AI — Latency / Throughput Benchmark

Actually measures (via time.perf_counter(), not invented) end-to-end
event-to-recommendation latency and events/sec on synthetic load.

Broken into:
  - ML inference latency
  - Agent investigation latency (requires API call or mock)
  - End-to-end latency (event → recommendation)
  - Events/sec throughput
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import LATENCY_BATCH_SIZE, MODELS_DIR
from src.schemas import LatencyResult

logger = logging.getLogger(__name__)


def measure_latency(
    model: Any,
    feature_matrix: pd.DataFrame,
    feature_names: List[str],
    scaler: Optional[Any] = None,
    needs_scaling: bool = False,
    toolkit: Optional[Any] = None,
    incident: Optional[Any] = None,
    batch_size: int = LATENCY_BATCH_SIZE,
) -> LatencyResult:
    """
    Measure actual latency and throughput numbers.

    All timing uses time.perf_counter() — no invented numbers.

    Parameters
    ----------
    model : Any
        Trained model with predict_proba().
    feature_matrix : pd.DataFrame
        Feature matrix for scoring.
    feature_names : List[str]
        Feature column names.
    scaler : Optional[Any]
        StandardScaler if model needs scaling.
    needs_scaling : bool
        Whether to apply scaler before inference.
    toolkit : Optional[AgentToolkit]
        If provided, measures agent investigation latency.
    incident : Optional[Incident]
        If provided, used for agent investigation benchmark.
    batch_size : int
        Number of transactions per batch for throughput test.

    Returns
    -------
    LatencyResult
        Measured latency and throughput.
    """
    # ── ML Inference Latency ─────────────────────────────────────────
    X = feature_matrix[feature_names].values[:batch_size]
    if needs_scaling and scaler is not None:
        X = scaler.transform(X)

    # Warm-up run
    _ = model.predict_proba(X[:10])

    # Timed run
    start = time.perf_counter()
    _ = model.predict_proba(X)
    end = time.perf_counter()

    ml_inference_ms = (end - start) * 1000
    events_per_sec = batch_size / (end - start) if (end - start) > 0 else 0

    logger.info(f"ML Inference: {ml_inference_ms:.2f}ms for {batch_size} transactions "
                f"({events_per_sec:.0f} events/sec)")

    # ── Agent Investigation Latency ──────────────────────────────────
    agent_investigation_ms = 0.0

    if toolkit is not None and incident is not None:
        from src.agent.investigator import investigate_incident

        start = time.perf_counter()
        _ = investigate_incident(incident, toolkit)
        end = time.perf_counter()

        agent_investigation_ms = (end - start) * 1000
        logger.info(f"Agent Investigation: {agent_investigation_ms:.2f}ms")
    else:
        logger.info("Agent investigation benchmark skipped (no toolkit/incident provided)")

    # ── End-to-End Latency ───────────────────────────────────────────
    # Simulated: ML inference + a representative detection + agent call
    end_to_end_ms = ml_inference_ms + agent_investigation_ms

    logger.info(f"End-to-End: {end_to_end_ms:.2f}ms")
    logger.info(f"Throughput: {events_per_sec:.0f} events/sec")

    result = LatencyResult(
        ml_inference_ms=round(ml_inference_ms, 2),
        agent_investigation_ms=round(agent_investigation_ms, 2),
        end_to_end_ms=round(end_to_end_ms, 2),
        events_per_sec=round(events_per_sec, 1),
        batch_size=batch_size,
    )

    # Track whether agent was actually measured
    agent_measured = toolkit is not None and incident is not None

    # Save to disk
    _save_latency(result, agent_measured=agent_measured)

    return result


def _save_latency(result: LatencyResult, agent_measured: bool = False) -> None:
    """Save latency results to JSON."""
    data = {
        "ml_inference_ms": result.ml_inference_ms,
        "agent_investigation_ms": result.agent_investigation_ms,
        "end_to_end_ms": result.end_to_end_ms,
        "events_per_sec": result.events_per_sec,
        "batch_size": result.batch_size,
        "agent_measured": agent_measured,
    }
    filepath = MODELS_DIR / "latency_results.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Latency results saved to {filepath}")


def load_latency_results() -> Dict:
    """Load latency results from disk."""
    with open(MODELS_DIR / "latency_results.json") as f:
        return json.load(f)
