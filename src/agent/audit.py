"""
RiskGuard AI — Audit Trail

Every incident gets a complete case record:
  case_id, timestamp, merchant, trigger, risk_score, model_version,
  evidence, related entities, exposure, counterfactual, agent recommendation,
  policy result, human decision, final outcome.

Case records are persisted to JSON for auditability.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import CASES_DIR, MODEL_VERSION
from src.schemas import (
    CaseRecord,
    CounterfactualResult,
    ExposureResult,
    Incident,
    InvestigationReport,
)


def create_case(
    incident: Incident,
    investigation: InvestigationReport,
    exposure: Optional[ExposureResult] = None,
    counterfactual: Optional[CounterfactualResult] = None,
) -> CaseRecord:
    """
    Build a complete audit trail record from an investigated incident.

    Parameters
    ----------
    incident : Incident
        The triggering incident.
    investigation : InvestigationReport
        The agent's investigation report.
    exposure : Optional[ExposureResult]
        Financial exposure estimate.
    counterfactual : Optional[CounterfactualResult]
        Counterfactual explanation.

    Returns
    -------
    CaseRecord
        Complete case record with all fields populated.
    """
    case = CaseRecord(
        case_id=f"CASE-{uuid.uuid4().hex[:8].upper()}",
        timestamp=datetime.now(),
        merchant_id=incident.merchant_id,
        incident_id=incident.incident_id,
        trigger_type=incident.trigger_type,
        risk_score=incident.confidence,  # Incident-level confidence as risk score
        model_version=MODEL_VERSION,
        evidence_summary=investigation.summary[:500] if investigation.summary else "",
        related_entities=incident.related_entity_ids,
        exposure_estimate=exposure or investigation.exposure,
        counterfactual_summary=counterfactual or investigation.counterfactuals,
        agent_recommendation=investigation.recommendation,
        agent_mode=investigation.agent_mode,  # "live" or "mock" — always recorded
        policy_result=(
            investigation.policy_result.recommendation
            if investigation.policy_result else ""
        ),
        human_decision=None,   # Pending human action
        final_outcome=None,    # Pending resolution
    )

    # Persist to disk
    _persist_case(case)

    return case


def create_case_from_dict(case_data: Dict[str, Any]) -> CaseRecord:
    """
    Create a case record from a dictionary (used by the agent tool).

    Parameters
    ----------
    case_data : Dict
        Partial case data from the agent. Missing fields are filled with defaults.

    Returns
    -------
    CaseRecord
        Complete case record.
    """
    case = CaseRecord(
        case_id=case_data.get("case_id", f"CASE-{uuid.uuid4().hex[:8].upper()}"),
        timestamp=datetime.now(),
        merchant_id=case_data.get("merchant_id", "unknown"),
        incident_id=case_data.get("incident_id", "unknown"),
        trigger_type=case_data.get("trigger_type", "agent_created"),
        risk_score=float(case_data.get("risk_score", 0.0)),
        model_version=MODEL_VERSION,
        evidence_summary=case_data.get("evidence_summary", ""),
        related_entities=case_data.get("related_entities", []),
        agent_recommendation=case_data.get("recommendation", ""),
        agent_mode=case_data.get("agent_mode", "unknown"),
        policy_result=case_data.get("policy_result", ""),
        human_decision=None,
        final_outcome=None,
    )

    _persist_case(case)
    return case


def _persist_case(case: CaseRecord) -> None:
    """Save case record to JSON file."""
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CASES_DIR / f"{case.case_id}.json"

    # Convert to dict, handling nested dataclasses
    case_dict = _to_serializable(asdict(case))

    with open(filepath, "w") as f:
        json.dump(case_dict, f, indent=2, default=str)


def load_case(case_id: str) -> Optional[CaseRecord]:
    """Load a case record from disk."""
    filepath = CASES_DIR / f"{case_id}.json"
    if not filepath.exists():
        return None

    with open(filepath) as f:
        data = json.load(f)

    return CaseRecord(**data)


def list_cases() -> List[Dict[str, Any]]:
    """List all case records (summary view)."""
    cases = []
    if not CASES_DIR.exists():
        return cases

    for filepath in sorted(CASES_DIR.glob("CASE-*.json"), reverse=True):
        try:
            with open(filepath) as f:
                data = json.load(f)
            cases.append({
                "case_id": data.get("case_id"),
                "timestamp": data.get("timestamp"),
                "merchant_id": data.get("merchant_id"),
                "trigger_type": data.get("trigger_type"),
                "risk_score": data.get("risk_score"),
                "agent_mode": data.get("agent_mode"),
                "human_decision": data.get("human_decision"),
                "final_outcome": data.get("final_outcome"),
            })
        except Exception:
            continue

    return cases


def update_case_decision(case_id: str, human_decision: str, final_outcome: str) -> bool:
    """
    Update a case with human decision and final outcome.

    Returns True if the case was found and updated.
    """
    filepath = CASES_DIR / f"{case_id}.json"
    if not filepath.exists():
        return False

    with open(filepath) as f:
        data = json.load(f)

    data["human_decision"] = human_decision
    data["final_outcome"] = final_outcome

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return True


def _to_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return _to_serializable(obj.__dict__)
    else:
        return obj
