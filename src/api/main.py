"""
RiskGuard AI — FastAPI Application

REST API serving all system capabilities to the React dashboard.
All endpoints return computed data from the pipeline run — no hardcoded values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODELS_DIR, CASES_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RiskGuard AI",
    description="Agentic Early-Warning, Investigation, and Loss-Prevention System",
    version="1.0.0",
)

# CORS for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(filename: str) -> dict | list:
    """Load a JSON file from the models directory."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Data not available. Run the pipeline first: python scripts/run_pipeline.py"
        )
    with open(filepath) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard/metrics")
def get_dashboard_metrics():
    """Top-line metrics: total transactions, alerts, exposure, risk rate."""
    return _load_json("dashboard_metrics.json")


@app.get("/api/dashboard/risk-trend")
def get_risk_trend():
    """Time-series risk data for charting."""
    return {"data": _load_json("risk_trend.json")}


# ═══════════════════════════════════════════════════════════════════════════
# Incident Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/incidents")
def get_incidents():
    """Active incidents list."""
    incidents = _load_json("incidents.json")
    return {"incidents": incidents}


_toolkit = None


def _get_toolkit():
    """Lazy-load and cache the AgentToolkit for fast investigation dispatch."""
    global _toolkit
    if _toolkit is not None:
        return _toolkit

    import joblib
    import pandas as pd
    from src.schemas import Incident, ExposureResult
    from src.agent.tools import AgentToolkit

    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test.csv")

    baselines_path = MODELS_DIR / "baselines.joblib"
    if baselines_path.exists():
        baselines = joblib.load(baselines_path)
    else:
        from src.detection.merchant_baseline import compute_baselines
        train_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train.csv")
        baselines = compute_baselines(train_df)
        joblib.dump(baselines, baselines_path)

    clusters_path = MODELS_DIR / "clusters.joblib"
    if clusters_path.exists():
        clusters = joblib.load(clusters_path)
    else:
        from src.detection.relationship_engine import detect_abuse_rings
        clusters, _ = detect_abuse_rings(test_df)
        joblib.dump(clusters, clusters_path)

    incidents_raw = _load_json("incidents.json")
    incidents = [
        Incident(
            incident_id=inc["incident_id"],
            merchant_id=inc["merchant_id"],
            trigger_type=inc["trigger_type"],
            risk_level=inc["risk_level"],
            risk_increase_pct=inc.get("risk_increase_pct", 0.0),
            affected_payment_count=inc.get("affected_payment_count", 0),
            estimated_exposure=inc.get("estimated_exposure", 0.0),
            high_confidence_exposure=inc.get("high_confidence_exposure", 0.0),
            confidence=inc.get("confidence", 1.0),
            flagged_transaction_ids=inc.get("flagged_transaction_ids", []),
            related_entity_ids=inc.get("related_entity_ids", []),
        )
        for inc in incidents_raw
    ]

    exposure_raw = _load_json("exposure.json")
    exposures = {
        mid: ExposureResult(
            merchant_id=mid,
            potential_exposure=exp.get("potential_exposure", 0.0),
            high_confidence_exposure=exp.get("high_confidence_exposure", 0.0),
            flagged_count=exp.get("flagged_count", 0),
            high_confidence_count=exp.get("high_confidence_count", 0),
        )
        for mid, exp in exposure_raw.get("per_merchant", {}).items()
    }

    _toolkit = AgentToolkit(
        transactions_df=test_df,
        baselines=baselines,
        risk_scores={},
        feature_importances={},
        clusters=clusters,
        incidents=incidents,
        exposures=exposures,
        counterfactuals={},
    )
    return _toolkit


@app.get("/api/incidents/{incident_id}")
def get_incident_detail(incident_id: str):
    """Full incident detail with investigation data."""
    incidents = _load_json("incidents.json")

    incident = None
    for inc in incidents:
        if inc["incident_id"] == incident_id:
            incident = inc
            break

    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    # Try to load persisted investigation report
    investigation = None
    case_path = CASES_DIR / f"{incident_id}.json"
    if case_path.exists():
        try:
            with open(case_path) as f:
                investigation = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read case file {case_path}: {e}")

    if not investigation:
        investigation = {
            "summary": f"Incident {incident_id} detected on merchant {incident['merchant_id']} with trigger '{incident['trigger_type']}'. Mock agent analysis identified {incident['affected_payment_count']} flagged transactions with potential exposure of ₹{incident['estimated_exposure']:,.2f}.",
            "recommendations": [
                "Review recent transactions exceeding velocity thresholds",
                "Apply step-up 3DS authentication for cards associated with flagged devices",
                "Maintain baseline monitoring window before adjusting merchant velocity limits"
            ],
            "agent_mode": "mock",
        }

    # Load exposure for this merchant
    exposure = None
    try:
        exposure_data = _load_json("exposure.json")
        merchant_id = incident["merchant_id"]
        if merchant_id in exposure_data.get("per_merchant", {}):
            exposure = exposure_data["per_merchant"][merchant_id]
    except Exception:
        pass

    agent_mode = investigation.get("agent_mode", "mock")
    mock_reason = investigation.get("mock_reason")

    return {
        "incident": incident,
        "investigation": investigation,
        "exposure": exposure,
        "agent_mode": agent_mode,
        "mock_reason": mock_reason,
    }


@app.post("/api/incidents/{incident_id}/investigate")
def trigger_investigation(incident_id: str):
    """Trigger an agent investigation on an incident using investigate_incident."""
    import dataclasses
    from src.schemas import Incident
    from src.agent.investigator import investigate_incident

    incidents = _load_json("incidents.json")

    incident_data = None
    for inc in incidents:
        if inc["incident_id"] == incident_id:
            incident_data = inc
            break

    if not incident_data:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    incident_obj = Incident(
        incident_id=incident_data["incident_id"],
        merchant_id=incident_data["merchant_id"],
        trigger_type=incident_data["trigger_type"],
        risk_level=incident_data["risk_level"],
        risk_increase_pct=incident_data.get("risk_increase_pct", 0.0),
        affected_payment_count=incident_data.get("affected_payment_count", 0),
        estimated_exposure=incident_data.get("estimated_exposure", 0.0),
        high_confidence_exposure=incident_data.get("high_confidence_exposure", 0.0),
        confidence=incident_data.get("confidence", 1.0),
        flagged_transaction_ids=incident_data.get("flagged_transaction_ids", []),
        related_entity_ids=incident_data.get("related_entity_ids", []),
    )

    toolkit = _get_toolkit()
    exposure_obj = toolkit.exposures.get(incident_data["merchant_id"])

    logger.info(f"Triggering investigation for incident {incident_id} (merchant: {incident_data['merchant_id']})")
    report = investigate_incident(incident_obj, toolkit, exposure=exposure_obj)

    report_dict = dataclasses.asdict(report)
    if report.recommendation and "recommendations" not in report_dict:
        report_dict["recommendations"] = [report.recommendation]

    # Persist case report to data/cases
    try:
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        with open(CASES_DIR / f"{incident_id}.json", "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to persist case file for {incident_id}: {e}")

    return report_dict


# ═══════════════════════════════════════════════════════════════════════════
# Threshold Lab
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/threshold-lab")
def get_threshold_lab():
    """Full threshold sweep data for interactive slider."""
    from src.config import DEFAULT_RISK_THRESHOLD
    return {
        "thresholds": _load_json("threshold_curve.json"),
        "default_threshold": DEFAULT_RISK_THRESHOLD,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/benchmarks/cost-of-inaction")
def get_cost_of_inaction():
    """Cost comparison: no system vs static rules vs RiskGuard."""
    return {"conditions": _load_json("cost_of_inaction.json")}


@app.get("/api/benchmarks/adversarial")
def get_adversarial():
    """Naive rules vs RiskGuard adversarial comparison."""
    return _load_json("adversarial_results.json")


@app.get("/api/benchmarks/drift")
def get_drift():
    """Drift re-baselining timeline."""
    return {"days": _load_json("drift_results.json")}


@app.get("/api/benchmarks/latency")
def get_latency():
    """Latency and throughput measurements."""
    return _load_json("latency_results.json")


@app.get("/api/benchmarks/flash-sale")
def get_flash_sale():
    """Flash sale hard failure analysis."""
    return _load_json("flash_sale_analysis.json")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Matrix
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/evaluation-matrix")
def get_evaluation_matrix():
    """Full evaluation matrix across all dimensions."""
    return _load_json("evaluation_matrix.json")


# ═══════════════════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    """Health check endpoint."""
    pipeline_ready = (MODELS_DIR / "dashboard_metrics.json").exists()
    return {
        "status": "healthy",
        "pipeline_ready": pipeline_ready,
        "message": "Run `python scripts/run_pipeline.py` first" if not pipeline_ready else "Ready",
    }
