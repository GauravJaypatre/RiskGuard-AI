"""
RiskGuard AI — Investigation Agent

Gemini-powered agent that investigates triggered incidents using the 10 tools.
Produces a structured InvestigationReport.

The agent does NOT compute fraud scores — it reasons over evidence the
ML/rules layers already produced.

MOCK MODE: When GEMINI_API_KEY is unavailable, falls back to a mock agent
that produces structured reports. Mock mode is ALWAYS surfaced:
  - InvestigationReport.agent_mode = "mock"
  - A visible banner in the report summary
  - The API response includes agent_mode field
  - The dashboard shows a "⚠️ MOCK MODE" banner
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.config import (
    GEMINI_API_KEY,
    AGENT_MODEL,
    MOCK_MODE_BANNER,
    MOCK_MODE_API_FIELD,
    LIVE_MODE_API_FIELD,
)
from src.schemas import (
    CounterfactualResult,
    ExposureResult,
    Incident,
    InvestigationReport,
    PolicyResult,
)
from src.agent.tools import AgentToolkit, _build_gemini_tool_declarations
from src.agent.policy import evaluate_policy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Agent System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are RiskGuard AI's Investigation Agent. You investigate risk incidents triggered by the detection layer.

Your role:
1. INVESTIGATE the incident by querying evidence using your tools
2. SYNTHESIZE findings into a structured risk assessment
3. RECOMMEND policy-bounded actions (you do NOT take autonomous actions)

You have access to these tools:
- get_transaction: Get transaction details + risk score
- get_customer_history: Get customer's history, refunds, disputes
- get_device_history: Get device usage and unique customer count
- get_merchant_baseline: Get merchant's normal behavior baseline
- find_related_activity: Find clusters of related entities
- calculate_velocity: Check transaction velocity in time windows
- calculate_exposure: Get financial exposure estimates
- get_risk_score: Get ML risk score + top features
- get_policy: Get allowed actions for a risk level
- create_case: Create audit record

Investigation workflow:
1. Start by examining the flagged transactions
2. Check customer and device histories for patterns
3. Look for related activity (shared devices, fraud rings)
4. Calculate velocity across windows
5. Review merchant baseline for deviations
6. Get risk scores and exposure estimates
7. Look up the policy for the determined risk level
8. Synthesize everything into a structured report

CONSTRAINTS:
- You are defense-only. Never suggest how to evade detection.
- You do NOT compute fraud scores. Refer to the ML model's outputs.
- Frame all findings as evidence, not certainties.
- Financial numbers must come from the tools, not from your own computation.
- For CRITICAL risk levels, always note that human authorization is required.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Live Agent (Gemini API)
# ═══════════════════════════════════════════════════════════════════════════

def investigate_incident(
    incident: Incident,
    toolkit: AgentToolkit,
    exposure: Optional[ExposureResult] = None,
    counterfactual: Optional[CounterfactualResult] = None,
) -> InvestigationReport:
    """
    Investigate an incident using Gemini or mock agent.

    Automatically detects whether the Gemini API is available.
    agent_mode is ALWAYS set to "live" or "mock".

    Parameters
    ----------
    incident : Incident
        The incident to investigate.
    toolkit : AgentToolkit
        Pre-initialized toolkit with all data.
    exposure : Optional[ExposureResult]
        Pre-computed exposure for the merchant.
    counterfactual : Optional[CounterfactualResult]
        Pre-computed counterfactual for a representative flagged transaction.

    Returns
    -------
    InvestigationReport
        Structured report with agent_mode ALWAYS set.
    """
    api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    if api_key:
        try:
            return _live_investigation(incident, toolkit, exposure, counterfactual)
        except Exception as e:
            err_code = str(getattr(e, "code", "") or getattr(e, "status_code", "") or "")
            err_str = f"{err_code} {str(e)} {repr(e)}".lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str or "rate_limit" in err_str or "ratelimit" in err_str or "too many requests" in err_str:
                mock_reason = "rate_limited"
            elif "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                mock_reason = "rate_limited"
            elif "400" in err_str or "api_key_invalid" in err_str or "invalid_argument" in err_str or "permission_denied" in err_str or "forbidden" in err_str or "401" in err_str or "403" in err_str:
                mock_reason = "invalid_key"
            elif "connection" in err_str or "timeout" in err_str or "timed out" in err_str or "network" in err_str:
                mock_reason = "network_error"
            else:
                mock_reason = "api_error"

            logger.warning(f"Live agent failed ({mock_reason}), falling back to mock: {e}")
            return _mock_investigation(incident, toolkit, exposure, counterfactual, mock_reason=mock_reason)
    else:
        logger.info("No GEMINI_API_KEY — using mock agent")
        return _mock_investigation(incident, toolkit, exposure, counterfactual, mock_reason="no_api_key")


def _live_investigation(
    incident: Incident,
    toolkit: AgentToolkit,
    exposure: Optional[ExposureResult] = None,
    counterfactual: Optional[CounterfactualResult] = None,
) -> InvestigationReport:
    """Run actual Gemini investigation with function calling."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    client = genai.Client(api_key=api_key)

    # Build Gemini function declarations for tool definitions
    tool_declarations = _build_gemini_tool_declarations()

    # Build the initial user message describing the incident
    user_message = _build_incident_prompt(incident)

    # Conversation history as Gemini Content objects
    contents: List[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    ]

    # Agentic loop: keep calling Gemini until it stops requesting tools
    max_iterations = 15  # Safety limit
    all_evidence: List[Dict[str, Any]] = []
    all_related: List[Dict[str, Any]] = []

    for iteration in range(max_iterations):
        logger.info(f"Calling Gemini API with model: {AGENT_MODEL} (iteration {iteration + 1})")

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=AGENT_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=[types.Tool(function_declarations=tool_declarations)],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True,
                        ),
                        temperature=0.2,
                    ),
                )
                break
            except Exception as call_err:
                if ("503" in str(call_err) or "UNAVAILABLE" in str(call_err) or "429" in str(call_err)) and attempt < 2:
                    logger.warning(f"Transient Gemini API issue ({call_err}), retrying in 3s (attempt {attempt + 1}/3)...")
                    import time
                    time.sleep(3)
                else:
                    raise call_err

        # Check if Gemini wants to call functions
        function_calls = response.function_calls
        if function_calls:
            # Add the model's response (with function call parts) to history
            contents.append(response.candidates[0].content)

            # Process each function call and build function response parts
            function_response_parts: List[types.Part] = []
            case_created_data = None

            for fc in function_calls:
                tool_name = fc.name
                tool_input = dict(fc.args) if fc.args else {}

                logger.info(f"  Tool call: {tool_name}({tool_input})")

                # Execute the tool
                result = _execute_tool(toolkit, tool_name, tool_input)

                # Track evidence
                all_evidence.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result,
                })

                # Track related entities
                if tool_name == "find_related_activity":
                    all_related.append(result)

                # Check if case was created (terminal action)
                if tool_name == "create_case":
                    case_created_data = tool_input.get("case_data", {})

                # Build FunctionResponse part
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                )

            # If Gemini created a case, the investigation is successfully concluded!
            if case_created_data:
                logger.info(f"Gemini agent concluded investigation via create_case at iteration {iteration + 1}")
                policy = evaluate_policy(incident.risk_level)
                summary_text = case_created_data.get("evidence_summary") or case_created_data.get("summary") or "Investigation concluded."
                recommendation_text = case_created_data.get("recommendation") or policy.recommendation
                return InvestigationReport(
                    incident_id=incident.incident_id,
                    agent_mode=LIVE_MODE_API_FIELD,
                    summary=summary_text,
                    evidence=all_evidence,
                    related_entities=all_related,
                    exposure=exposure,
                    counterfactuals=counterfactual,
                    risk_assessment=f"Risk Level: {incident.risk_level} (confidence: {incident.confidence:.1%})",
                    recommendation=recommendation_text,
                    policy_result=policy,
                    confidence_notes=f"Synthesized by Gemini ({AGENT_MODEL}) using {len(all_evidence)} evidence points.",
                )

            # Add all function responses as a user turn
            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

        else:
            # Gemini finished — extract the final text
            final_text = response.text or ""

            logger.info(f"Gemini agent completed after {iteration + 1} iterations, "
                        f"{len(all_evidence)} evidence points collected")

            # Parse the final text into a structured report
            policy = evaluate_policy(incident.risk_level)

            return InvestigationReport(
                incident_id=incident.incident_id,
                agent_mode=LIVE_MODE_API_FIELD,
                summary=final_text,
                evidence=all_evidence,
                related_entities=all_related,
                exposure=exposure,
                counterfactuals=counterfactual,
                risk_assessment=f"Risk Level: {incident.risk_level} (confidence: {incident.confidence:.1%})",
                recommendation=policy.recommendation,
                policy_result=policy,
                confidence_notes=f"Based on {len(all_evidence)} evidence points collected during investigation.",
            )

    # Safety: hit max iterations
    logger.warning("Gemini agent hit max iterations limit")
    policy = evaluate_policy(incident.risk_level)
    return InvestigationReport(
        incident_id=incident.incident_id,
        agent_mode=LIVE_MODE_API_FIELD,
        summary="Investigation reached maximum iteration limit. Partial findings are included.",
        evidence=all_evidence,
        related_entities=all_related,
        exposure=exposure,
        counterfactuals=counterfactual,
        risk_assessment=f"Risk Level: {incident.risk_level}",
        recommendation=policy.recommendation,
        policy_result=policy,
        confidence_notes="Investigation was truncated at max iterations.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Mock Agent (No API Key)
# ═══════════════════════════════════════════════════════════════════════════

def _mock_investigation(
    incident: Incident,
    toolkit: AgentToolkit,
    exposure: Optional[ExposureResult] = None,
    counterfactual: Optional[CounterfactualResult] = None,
    mock_reason: Optional[str] = "no_api_key",
) -> InvestigationReport:
    """
    Generate a structured mock investigation report.

    This DOES call the real tools to gather real data — it just doesn't
    use an LLM to synthesize. The mock agent follows a fixed investigation
    pattern and assembles findings.

    Mock mode is ALWAYS visibly indicated.
    """
    evidence: List[Dict[str, Any]] = []
    related_entities: List[Dict[str, Any]] = []

    # Step 1: Examine flagged transactions
    sample_tx_ids = incident.flagged_transaction_ids[:5]
    tx_details = []
    for tid in sample_tx_ids:
        result = toolkit.get_transaction(tid)
        evidence.append({"tool": "get_transaction", "input": {"transaction_id": tid}, "result": result})
        tx_details.append(result)

    # Step 2: Check customer histories
    customer_ids = set()
    for tx in tx_details:
        cid = tx.get("customer_id", "")
        if cid and cid not in customer_ids:
            customer_ids.add(cid)
            result = toolkit.get_customer_history(cid)
            evidence.append({"tool": "get_customer_history", "input": {"customer_id": cid}, "result": result})

    # Step 3: Check device histories
    device_ids = set()
    for tx in tx_details:
        did = tx.get("device_id", "")
        if did and did not in device_ids:
            device_ids.add(did)
            result = toolkit.get_device_history(did)
            evidence.append({"tool": "get_device_history", "input": {"device_id": did}, "result": result})

    # Step 4: Find related activity
    for entity_id in list(customer_ids)[:3]:
        result = toolkit.find_related_activity(entity_id)
        evidence.append({"tool": "find_related_activity", "input": {"entity_id": entity_id}, "result": result})
        if result.get("related_clusters"):
            related_entities.append(result)

    # Step 5: Get merchant baseline
    baseline = toolkit.get_merchant_baseline(incident.merchant_id)
    evidence.append({"tool": "get_merchant_baseline", "input": {"merchant_id": incident.merchant_id}, "result": baseline})

    # Step 6: Get exposure
    exp_data = toolkit.calculate_exposure(incident.merchant_id)
    evidence.append({"tool": "calculate_exposure", "input": {"merchant_id": incident.merchant_id}, "result": exp_data})

    # Step 7: Get policy
    policy = evaluate_policy(incident.risk_level)

    # Step 8: Synthesize summary
    summary = _build_mock_summary(incident, tx_details, evidence, exposure, policy)

    if mock_reason == "rate_limited":
        banner = "⚠️ MOCK MODE — Live agent rate-limited (free-tier quota). Try again in a minute, or this is expected if multiple investigations were run in quick succession."
    elif mock_reason == "invalid_key":
        banner = "⚠️ MOCK MODE — Invalid API key or permission denied."
    elif mock_reason == "api_error":
        banner = "⚠️ MOCK MODE — Live agent encountered an unexpected API error. Fallback activated."
    else:
        banner = MOCK_MODE_BANNER

    return InvestigationReport(
        incident_id=incident.incident_id,
        agent_mode=MOCK_MODE_API_FIELD,  # ← ALWAYS "mock"
        mock_reason=mock_reason,
        summary=f"{banner}\n\n{summary}",
        evidence=evidence,
        related_entities=related_entities,
        exposure=exposure,
        counterfactuals=counterfactual,
        risk_assessment=f"Risk Level: {incident.risk_level} (confidence: {incident.confidence:.1%})",
        recommendation=policy.recommendation,
        policy_result=policy,
        confidence_notes=(
            f"{banner}\n"
            f"This report was generated using a structured template with real data from "
            f"{len(evidence)} tool calls. An AI agent was not used for synthesis."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_incident_prompt(incident: Incident) -> str:
    """Build the initial prompt describing the incident for the LLM."""
    return f"""Investigate this risk incident:

Incident ID: {incident.incident_id}
Merchant: {incident.merchant_id}
Trigger: {incident.trigger_type}
Risk Level: {incident.risk_level}
Risk Increase: {incident.risk_increase_pct:.1f}%
Affected Payments: {incident.affected_payment_count}
Estimated Exposure: ₹{incident.estimated_exposure:,.2f}
Confidence: {incident.confidence:.1%}

Flagged Transaction IDs (sample): {incident.flagged_transaction_ids[:5]}
Related Entity IDs: {incident.related_entity_ids[:10]}

Please investigate this incident thoroughly:
1. Examine the flagged transactions
2. Check customer and device histories
3. Look for related activity and fraud rings
4. Check velocity patterns
5. Review the merchant baseline
6. Get exposure estimates
7. Determine the appropriate policy response
8. Create an audit case record

Produce a detailed investigation report with your findings, risk assessment, and recommendations."""


def _execute_tool(toolkit: AgentToolkit, tool_name: str, tool_input: Dict) -> Dict:
    """Dispatch a tool call to the toolkit."""
    method = getattr(toolkit, tool_name, None)
    if method is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        # Handle the case where tool_input might have nested keys
        if tool_name == "create_case":
            return method(tool_input.get("case_data", tool_input))
        elif len(tool_input) == 1:
            # Single argument — pass the value directly
            key = list(tool_input.keys())[0]
            return method(tool_input[key])
        else:
            return method(**tool_input)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


def _build_mock_summary(
    incident: Incident,
    tx_details: List[Dict],
    evidence: List[Dict],
    exposure: Optional[ExposureResult],
    policy: PolicyResult,
) -> str:
    """Build a structured mock investigation summary from real data."""
    lines = [
        f"## Investigation Report — {incident.incident_id}",
        "",
        f"**Merchant**: {incident.merchant_id}",
        f"**Trigger**: {incident.trigger_type} detected",
        f"**Risk Level**: {incident.risk_level}",
        "",
        "### Findings",
        "",
        f"- **{incident.affected_payment_count}** transactions flagged "
        f"with a risk increase of **{incident.risk_increase_pct:.1f}%** over baseline.",
    ]

    # Add exposure info
    if exposure:
        lines.extend([
            f"- Potential exposure identified: **₹{exposure.potential_exposure:,.2f}** "
            f"({exposure.flagged_count} transactions)",
            f"- High-confidence exposure: **₹{exposure.high_confidence_exposure:,.2f}** "
            f"({exposure.high_confidence_count} transactions)",
        ])

    # Summarize transaction patterns
    if tx_details:
        amounts = [tx.get("amount", 0) for tx in tx_details]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        lines.append(f"- Average flagged transaction amount: **₹{avg_amount:,.2f}**")

    # Evidence count
    lines.extend([
        "",
        f"### Evidence",
        f"This investigation examined {len(evidence)} data points across "
        f"transactions, customer histories, device histories, and relationship clusters.",
        "",
        "### Recommendation",
        f"{policy.recommendation}",
    ])

    if policy.requires_human_authorization:
        lines.append("\n**⚠️ CRITICAL: This recommendation requires human authorization before any action is taken.**")

    return "\n".join(lines)
