"""
Project Health Scoring Module
==============================
Computes an overall 0-100 health score with dimension-wise breakdown
from the structured outputs of the scope, blocker, and risk agents.

Dimensions:
  - Scope clarity   (0-100): how complete and well-structured the scope data is
  - Timeline risk    (0-100): inverse of delivery risk (100 = no risk, 0 = severely delayed)
  - Blocker severity (0-100): inverse of blocker impact (100 = no blockers, 0 = many open)
  - Risk exposure    (0-100): inverse of risk severity weighted count

Overall = weighted average of dimensions.

Public API
----------
    from agents.health_score import compute
    score = compute(scope_data, blocker_data, risk_data)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

EMPTY_SCORE: dict[str, Any] = {
    "overall":            0,
    "scope_clarity":      0,
    "timeline_risk":      0,
    "blocker_severity":   0,
    "risk_exposure":      0,
    "grade":             "F",
    "summary":           "Insufficient data to compute health score.",
    "dimensions":        [],
}


# ---------------------------------------------------------------------------
# Dimension weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "scope_clarity":    0.25,
    "timeline_risk":    0.30,
    "blocker_severity": 0.25,
    "risk_exposure":    0.20,
}

# Severity weights for risk exposure (higher severity = more exposure)
_SEVERITY_WEIGHTS = {
    "Critical": 1.0,
    "High":     0.75,
    "Medium":   0.50,
    "Low":      0.25,
}

# Threshold for "too many blockers" — at this count, blocker score is 0
_MAX_BLOCKERS = 10

# Threshold for "too many risks" — at this count of high-severity risks, risk score is 0
_MAX_HIGH_RISKS = 8


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute(
    scope_data: dict | None = None,
    blocker_data: dict | None = None,
    risk_data: dict | None = None,
) -> dict[str, Any]:
    """Compute the project health score from agent outputs.

    All inputs are the raw dicts produced by scope/blocker/risk agents.
    Never raises — returns EMPTY_SCORE on any error.
    """
    try:
        scope_data   = scope_data   or {}
        blocker_data = blocker_data or {}
        risk_data    = risk_data    or {}

        sc, sc_details = _scope_clarity_score(scope_data)
        tr, tr_details = _timeline_risk_score(risk_data)
        bs, bs_details = _blocker_severity_score(blocker_data)
        re, re_details = _risk_exposure_score(risk_data)

        overall = round(
            sc * _WEIGHTS["scope_clarity"]
            + tr * _WEIGHTS["timeline_risk"]
            + bs * _WEIGHTS["blocker_severity"]
            + re * _WEIGHTS["risk_exposure"]
        )

        grade = _grade(overall)
        summary = _summary_text(overall, sc, tr, bs, re)

        dimensions = [
            {"name": "Scope Clarity",   "score": sc, "weight": _WEIGHTS["scope_clarity"],   "details": sc_details},
            {"name": "Timeline Risk",   "score": tr, "weight": _WEIGHTS["timeline_risk"],   "details": tr_details},
            {"name": "Blocker Impact",  "score": bs, "weight": _WEIGHTS["blocker_severity"],  "details": bs_details},
            {"name": "Risk Exposure",   "score": re, "weight": _WEIGHTS["risk_exposure"],   "details": re_details},
        ]

        return {
            "overall":           overall,
            "scope_clarity":     sc,
            "timeline_risk":      tr,
            "blocker_severity":   bs,
            "risk_exposure":      re,
            "grade":             grade,
            "summary":           summary,
            "dimensions":        dimensions,
        }

    except Exception as exc:
        logger.exception("health_score compute error: %s", exc)
        return {**EMPTY_SCORE}


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

def _scope_clarity_score(scope_data: dict) -> tuple[int, str]:
    """Score 0-100 based on how complete the scope extraction is.

    Checks for: project_goals, scope items, out_of_scope, deliverables, milestones.
    Also checks sub-field completeness within deliverables and flags sparse extraction
    when all categories are present but each contains only a single item — a strong
    signal of context starvation rather than a genuinely minimal project.

    Returns (score, details_string) where details explains the calculation.
    """
    if not scope_data:
        return 0, "No scope data — score 0."

    score = 100
    details_parts = []

    # ── Required presence checks ──────────────────────────────────────────
    goals = scope_data.get("project_goals", [])
    if not goals:
        score -= 25
        details_parts.append("project_goals: missing (-25)")

    scope_items = scope_data.get("scope", [])
    if not scope_items:
        score -= 20
        details_parts.append("scope items: missing (-20)")

    # out_of_scope is worth a smaller deduction — not every project documents it,
    # but its absence is worth flagging when other scope data exists.
    out_of_scope = scope_data.get("out_of_scope", [])
    if not out_of_scope and (goals or scope_items):
        score -= 10
        details_parts.append("out_of_scope: not documented (-10)")

    deliverables = scope_data.get("deliverables", [])
    if not deliverables:
        score -= 25
        details_parts.append("deliverables: missing (-25)")
    else:
        # Sub-field completeness within extracted deliverables
        missing_desc     = sum(1 for d in deliverables if isinstance(d, dict) and not d.get("description"))
        missing_deadline = sum(1 for d in deliverables if isinstance(d, dict) and not d.get("deadline"))
        missing_owner    = sum(1 for d in deliverables if isinstance(d, dict) and not d.get("owner"))
        total_d = len(deliverables)

        if missing_desc > 0:
            penalty = min(missing_desc * 5, 15)
            score -= penalty
            details_parts.append(f"deliverables: {missing_desc}/{total_d} missing description (-{penalty})")
        if missing_deadline > 0:
            penalty = min(missing_deadline * 3, 10)
            score -= penalty
            details_parts.append(f"deliverables: {missing_deadline}/{total_d} missing deadline (-{penalty})")
        if missing_owner > 0:
            penalty = min(missing_owner * 2, 10)
            score -= penalty
            details_parts.append(f"deliverables: {missing_owner}/{total_d} missing owner (-{penalty})")

    milestones = scope_data.get("milestones", [])
    if not milestones:
        score -= 20
        details_parts.append("milestones: missing (-20)")

    # ── Sparse-extraction heuristic ───────────────────────────────────────
    # If all four main lists are present but each has only 1 item, the
    # extraction is suspiciously thin — likely caused by limited context
    # reaching the LLM rather than a genuinely minimal project.
    # Only apply when the LLM did NOT flag any missing_information
    # (otherwise the gap penalty below already covers it).
    missing_info = scope_data.get("missing_information", [])
    gaps         = scope_data.get("extraction_gaps", [])
    all_present  = goals and scope_items and deliverables and milestones
    all_sparse   = (
        all_present
        and len(goals) <= 1
        and len(scope_items) <= 1
        and len(deliverables) <= 1
        and len(milestones) <= 1
        and not missing_info
        and not gaps
    )
    if all_sparse:
        score -= 15
        details_parts.append(
            "sparse extraction: all categories have ≤1 item — "
            "likely incomplete retrieval (-15)"
        )

    # ── LLM-flagged gaps ─────────────────────────────────────────────────
    flagged = len(missing_info) + len(gaps)
    if flagged > 0:
        penalty = min(flagged * 3, 20)
        score -= penalty
        details_parts.append(f"{flagged} documented gaps (-{penalty})")

    if not details_parts:
        details_parts.append("All 5 categories present with sub-fields — score 100")

    return max(score, 0), "; ".join(details_parts)


def _timeline_risk_score(risk_data: dict) -> tuple[int, str]:
    """Score 0-100 (100 = on track, 0 = severely delayed).

    Uses the delivery_forecast status as the primary signal, then cross-checks
    against confirmed Critical/High risks — if those exist, caps the score so
    an "On Track" label with confirmed critical risks doesn't produce a fake-high score.

    Returns (score, details_string).
    """
    if not risk_data:
        return 50, "No risk data — neutral score 50."

    forecast = risk_data.get("delivery_forecast", {})
    if not isinstance(forecast, dict):
        return 50, "No delivery forecast — neutral score 50."

    status = (forecast.get("status") or "").lower().strip()

    base_scores = {
        "on track":        90,
        "at risk":         60,
        "delayed":         25,
        "insufficient data": 50,
    }
    score = base_scores.get(status, 50)

    # Cross-check: confirmed Critical or High risks cap the score
    # regardless of what the LLM labelled the forecast
    risks = risk_data.get("risks", [])
    confirmed_critical = sum(
        1 for r in risks
        if isinstance(r, dict)
        and r.get("status", "").lower() == "confirmed"
        and r.get("severity", "").lower() in ("critical", "high")
    )
    potential_critical = sum(
        1 for r in risks
        if isinstance(r, dict)
        and r.get("status", "").lower() != "confirmed"
        and r.get("severity", "").lower() in ("critical", "high")
    )

    cap = 100
    cap_reasons = []
    if confirmed_critical >= 2:
        cap = min(cap, 30)
        cap_reasons.append(f"{confirmed_critical} confirmed Critical/High risks → cap 30")
    elif confirmed_critical == 1:
        cap = min(cap, 55)
        cap_reasons.append(f"1 confirmed Critical/High risk → cap 55")
    if potential_critical >= 3:
        cap = min(cap, 65)
        cap_reasons.append(f"{potential_critical} potential Critical/High risks → cap 65")

    score = min(score, cap)
    detail = f"forecast='{status}' → base {base_scores.get(status, 50)}"
    if cap_reasons:
        detail += "; " + "; ".join(cap_reasons) + f" → final {score}"
    return score, detail


def _blocker_severity_score(blocker_data: dict) -> tuple[int, str]:
    """Score 0-100 (100 = no blockers, 0 = many open blockers).

    Penalizes open blockers more than resolved ones.
    Returns (score, details_string).
    """
    if not blocker_data:
        return 100, "No blocker data — score 100."

    blockers = blocker_data.get("blockers", [])
    if not blockers:
        return 100, "No blockers found — score 100."

    open_count = sum(
        1 for b in blockers
        if isinstance(b, dict) and b.get("status", "").lower() == "open"
    )
    resolved_count = len(blockers) - open_count

    # Resolved blockers are much less impactful than open ones
    effective = open_count + (resolved_count * 0.2)

    if effective <= 0:
        return 100, f"All {len(blockers)} blockers resolved — score 100."
    if effective >= _MAX_BLOCKERS:
        return 0, f"{open_count} open + {resolved_count} resolved (effective {effective:.1f}) — score 0."

    score = max(round(100 - (effective / _MAX_BLOCKERS) * 100), 0)
    return score, f"{open_count} open + {resolved_count} resolved (effective {effective:.1f}, max {_MAX_BLOCKERS}) — score {score}"


def _risk_exposure_score(risk_data: dict) -> tuple[int, str]:
    """Score 0-100 (100 = no risk exposure, 0 = critical risks everywhere).

    Weights risks by severity. Confirmed risks penalize more than potential ones.
    Returns (score, details_string).
    """
    if not risk_data:
        return 100, "No risk data — score 100."

    risks = risk_data.get("risks", [])
    if not risks:
        return 100, "No risks found — score 100."

    total_exposure = 0.0
    breakdown = []
    for r in risks:
        if not isinstance(r, dict):
            continue
        sev = (r.get("severity") or "Medium").strip().capitalize()
        weight = _SEVERITY_WEIGHTS.get(sev, 0.50)

        # Confirmed risks count double
        confirmed = (r.get("status") or "").lower() == "confirmed"
        if confirmed:
            weight *= 1.5

        total_exposure += weight
        breakdown.append(f"{r.get('risk_id','?')}({sev},{'C' if confirmed else 'P'})={weight:.1f}")

    if total_exposure <= 0:
        return 100, "No risk exposure — score 100."
    if total_exposure >= _MAX_HIGH_RISKS:
        return 0, f"Total exposure {total_exposure:.1f} ≥ {_MAX_HIGH_RISKS} — score 0."

    score = max(round(100 - (total_exposure / _MAX_HIGH_RISKS) * 100), 0)
    return score, f"Exposure: {' + '.join(breakdown)} = {total_exposure:.1f} (max {_MAX_HIGH_RISKS}) — score {score}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grade(score: int) -> str:
    """Convert 0-100 score to letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _summary_text(overall: int, sc: int, tr: int, bs: int, re: int) -> str:
    """Generate a one-line human-readable summary of the health score."""
    parts = []

    if overall >= 80:
        parts.append("Project is in good health.")
    elif overall >= 60:
        parts.append("Project has moderate concerns.")
    elif overall >= 40:
        parts.append("Project is at risk — several issues need attention.")
    else:
        parts.append("Project is in critical condition — immediate action required.")

    # Highlight the weakest dimension
    dims = {
        "scope clarity": sc,
        "timeline":      tr,
        "blockers":       bs,
        "risk exposure":  re,
    }
    weakest = min(dims, key=dims.get)
    if dims[weakest] < 50:
        parts.append(f"Primary concern: {weakest} (score {dims[weakest]}/100).")

    return " ".join(parts)
