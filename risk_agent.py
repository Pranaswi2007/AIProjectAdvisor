"""
Agent 3 — Risk Detection & Delivery Forecasting Agent
======================================================
Identifies project risks and evaluates potential delivery delays using
scope, milestones, dependencies, and progress information from documents.

Public API
----------
    from agents.risk_agent import run as run_risk_agent
    result = run_risk_agent(project_id, scope_data, blocker_data)
"""

import logging
import math
import re
from datetime import date, datetime, timedelta
from typing import Any

from agents.llm_client import (
    LLMError, build_context, call_llm_json, tpm_context_cap,
    DEFAULT_MODEL, get_cached_llm_result, set_cached_llm_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

EMPTY_OUTPUT: dict[str, Any] = {
    "risks":             [],
    "delivery_forecast": {
        "status":                   "Insufficient data",
        "estimated_completion_date": None,
        "delayed_milestones":        [],
        "assumptions":               [],
        "confidence":                "Low",
    },
    "dependency_gaps":    [],
    "missing_information": [],
    "error":              None,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior project risk analyst. Your task is to
identify project risks and assess delivery forecasts using the document
excerpts and structured data provided.

Rules you MUST follow:
1. Base every risk on evidence in the provided documents or structured data.
2. Never invent completion percentages, durations, or predicted dates.
3. If information is insufficient, return "Insufficient data" in the
   relevant fields and explain what is missing.
4. Extract EVERY risk mentioned in the documents — do not omit any.
5. For each risk, extract: probability, impact, severity rating, owner,
   mitigation, trigger condition, and documented status when available.
6. Cite the source filename for every risk.
7. Severity levels: Critical, High, Medium, Low.
8. Status: "Confirmed" (delay already happened) or "Potential" (may happen).
   Do not invent statuses — use "Unknown" if not documented.
9. Confidence: "High" only if dates+progress are both documented;
   "Medium" if one is present; "Low" otherwise.

Reply with a single valid JSON object — no markdown, no extra text:
{
  "risks": [
    {"risk_id": "R001", "title": "string", "description": "string", "severity": "Critical | High | Medium | Low", "probability": "High | Medium | Low | Unknown", "impact": "string or null", "affected_deliverable": "string or null", "potential_impact": "string", "mitigation": "string", "trigger": "string or null", "owner": "string or null", "status": "Confirmed | Potential | Unknown", "source": "filename"}
  ],
  "delivery_forecast": {
    "status": "On Track | At Risk | Delayed | Insufficient data",
    "estimated_completion_date": "YYYY-MM-DD or null",
    "delayed_milestones": ["string"],
    "assumptions": ["string"],
    "confidence": "High | Medium | Low",
    "forecast_method": "rule-based | qualitative | hybrid | none"
  },
  "dependency_gaps": [
    {"gap_id": "G001", "description": "string", "affected_task": "string", "source": "filename"}
  ],
  "missing_information": ["string"]
}"""

# Maximum characters for the structured summary sent to the Risk Agent LLM.
# Developer plan: 250K TPM — no per-agent token rationing needed.
# max_tokens=16000 because openai/gpt-oss-20b is a reasoning model: it
# consumes internal reasoning tokens before emitting visible output. A small
# max_tokens budget is exhausted by reasoning before any JSON is written,
# producing finish_reason=length with empty content.
_MAX_RISK_SUMMARY_CHARS = 400
_MAX_RISK_OUTPUT_TOKENS = 1200   # non-reasoning model (qwen3.8-27b): all tokens go to output, no hidden thinking

RETRIEVAL_QUERIES = [
    "risk issue delay deadline missed overdue schedule conflict",
    "dependency blocked incomplete missing owner unassigned",
    "delivery forecast completion estimate progress status",
    "technical constraint resource budget testing failure",
]

# ---------------------------------------------------------------------------
# Rule-based forecasting helpers
# ---------------------------------------------------------------------------

def _parse_date(val: Any) -> date | None:
    """Try to parse a date string (YYYY-MM-DD) into a date object."""
    if not val or not isinstance(val, str):
        return None
    try:
        return datetime.strptime(val[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _rule_based_forecast(scope_data: dict) -> dict:
    """Compute a delivery forecast from milestone and deliverable dates.

    Classification rules:
    - A milestone/deliverable is OVERDUE only if its own date has passed AND
      it is not marked as completed.
    - A milestone is AT RISK if it is upcoming but other incomplete items are
      already overdue (overdue dependencies threaten it).
    - "Kickoff", "start", or already-completed milestones are never overdue.
    - We never mark a future milestone as "Delayed" just because dependencies
      are behind — that is "At Risk", not "Delayed".
    - Returns "Insufficient data" only when no dates are available.
    - Never fabricates dates, percentages, or completion status.
    """
    today = date.today()

    # ── Collect milestones with (name, date, is_complete) ──────────────────
    milestones = scope_data.get("milestones", [])
    ms_items: list[tuple[str, date, bool]] = []
    for m in milestones:
        d = _parse_date(m.get("date"))
        if d is None:
            continue
        name = m.get("name", "?")
        desc = m.get("description", "")
        complete = bool(re.search(
            r"\b(?:kickoff|kick.off|started|completed|done|approved|launched)\b",
            name + " " + desc, re.IGNORECASE,
        ))
        ms_items.append((name, d, complete))

    # ── Fall back to deliverable deadlines when no milestones have dates ───
    dated_items: list[tuple[str, date, bool]] = list(ms_items)
    if not dated_items:
        for deliv in scope_data.get("deliverables", []):
            d = _parse_date(deliv.get("deadline"))
            if d:
                # Check deliverable completion status the same way as milestones
                name = deliv.get("name", "Deliverable")
                desc = deliv.get("description", "")
                done = bool(re.search(
                    r"\b(?:completed|done|approved|delivered|launched)\b",
                    name + " " + desc, re.IGNORECASE,
                ))
                dated_items.append((name, d, done))

    if not dated_items:
        return {
            "status":                    "Insufficient data",
            "estimated_completion_date": None,
            "delayed_milestones":        [],
            "assumptions":               [
                "No milestone or deliverable dates found in documents.",
                "Upload a project proposal or SRS with explicit deadlines.",
            ],
            "confidence":      "Low",
            "forecast_method": "none",
        }

    # ── Classify each item ─────────────────────────────────────────────────
    # confirmed_overdue: past date, incomplete
    confirmed_overdue = [
        name for name, d, done in dated_items
        if not done and d < today
    ]
    # upcoming: future date, incomplete
    upcoming = [(name, d) for name, d, done in dated_items if not done and d >= today]
    latest   = max(d for _, d, _ in dated_items)

    # Status rules:
    # - Delayed  → at least one item has already passed its own deadline
    #              AND there are no remaining upcoming milestones to recover to
    # - At Risk  → some items are overdue BUT future milestones still exist
    #              OR the next upcoming milestone is within 7 days
    # - On Track → all items are upcoming and none are imminently due (≤7 days)
    if confirmed_overdue:
        if upcoming:
            # There are overdue items but work still lies ahead —
            # the project is at risk, not yet confirmed delayed end-to-end.
            status = "At Risk"
        else:
            # Everything that should be done by now is still pending — delayed.
            status = "Delayed"
    elif upcoming:
        soon = [d for _, d in upcoming if (d - today).days <= 7]
        status = "At Risk" if soon else "On Track"
    else:
        status = "Insufficient data"

    confidence  = "High" if len(dated_items) >= 2 else "Medium"
    used_source = "milestones" if ms_items else "deliverable deadlines"

    delay_reasons = [
        f"{name} (due {d.isoformat()}, not yet complete)"
        for name, d, done in dated_items
        if not done and d < today
    ]

    # Build a note about at-risk upcoming items
    at_risk_notes: list[str] = []
    if confirmed_overdue and upcoming:
        at_risk_notes.append(
            f"Upcoming items at risk due to overdue dependencies: "
            + ", ".join(n for n, _ in upcoming[:3])
        )

    return {
        "status":                    status,
        "estimated_completion_date": latest.isoformat(),
        "delayed_milestones":        confirmed_overdue,
        "assumptions": [
            f"Forecast based on {used_source} extracted from documents.",
            f"Reference date: {today.isoformat()}.",
            "Milestones/deliverables labelled as kickoff or completed are excluded from overdue checks.",
            "A future milestone is marked At Risk (not Delayed) when its dependencies are behind.",
        ]
        + ([f"Overdue items: {'; '.join(delay_reasons)}"] if delay_reasons else [])
        + at_risk_notes,
        "confidence":      confidence,
        "forecast_method": "rule-based",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    project_id: str,
    scope_data: dict | None = None,
    blocker_data: dict | None = None,
    context_k: int = 4,
) -> dict[str, Any]:
    """Detect risks and forecast delivery for *project_id*.

    Accepts structured outputs from scope_agent and blocker_agent to
    enrich the risk analysis without re-querying the vector store.
    Never raises — errors captured in result["error"].
    """
    result = {**EMPTY_OUTPUT, "error": None}
    result["delivery_forecast"] = {**EMPTY_OUTPUT["delivery_forecast"]}

    scope_data   = scope_data   or {}
    blocker_data = blocker_data or {}

    try:
        # 1. Rule-based forecast (deterministic, no LLM)
        rb_forecast = _rule_based_forecast(scope_data)
        result["delivery_forecast"] = rb_forecast

        # 2. Retrieve context — size computed from remaining TPM budget after
        # system prompt + summary + output tokens are accounted for.
        # summary_tokens ≈ len(summary)//4; we reserve that plus framing.
        _ctx_framing = 70 + len(_summarise_structured(scope_data, blocker_data)[:_MAX_RISK_SUMMARY_CHARS]) // 4
        _risk_ctx_chars = tpm_context_cap(SYSTEM_PROMPT, _MAX_RISK_OUTPUT_TOKENS, framing_tokens=_ctx_framing)
        context = build_context(project_id, RETRIEVAL_QUERIES, k=context_k,
                                max_chars=_risk_ctx_chars)
        if not context:
            result["missing_information"] = [
                "No documents indexed — risk analysis limited to structured data only."
            ]
            return result

        # 3. Summarise structured inputs for the LLM (capped to avoid overflow)
        structured_summary = _summarise_structured(scope_data, blocker_data)
        if len(structured_summary) > _MAX_RISK_SUMMARY_CHARS:
            structured_summary = structured_summary[:_MAX_RISK_SUMMARY_CHARS] + "\n...[truncated]"

        user_prompt = (
            f"Project ID: {project_id}\n\n"
            "=== STRUCTURED DATA FROM PREVIOUS AGENTS ===\n"
            f"{structured_summary}\n\n"
            "=== DOCUMENT EXCERPTS ===\n"
            f"{context}\n\n"
            "Produce a risk assessment and delivery forecast in the required JSON schema."
        )

        try:
            # Check LLM result cache first — avoids re-calling Groq for unchanged inputs
            _AGENT = "risk"
            cached = get_cached_llm_result(
                _AGENT, project_id, SYSTEM_PROMPT,
                user_prompt, _MAX_RISK_OUTPUT_TOKENS, DEFAULT_MODEL,
            )
            if cached is not None:
                logger.info("risk_agent: using cached LLM result for '%s'", project_id)
                result = _merge(result, cached, rb_forecast)
                return result

            raw = call_llm_json(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                fallback=None,
                max_tokens=_MAX_RISK_OUTPUT_TOKENS,
                timeout=30,
                json_mode=False,
            )
            set_cached_llm_result(
                _AGENT, project_id, SYSTEM_PROMPT,
                user_prompt, _MAX_RISK_OUTPUT_TOKENS, DEFAULT_MODEL, raw,
            )
        except LLMError as exc:
            # LLM failed (truncation, rate limit, etc.) — but the rule-based
            # forecast was already computed above and is in result["delivery_forecast"].
            # Don't mark the whole agent as "failed" — the forecast is still useful.
            # Log the issue in missing_information so the UI shows it as a warning,
            # not a fatal error.
            logger.warning("Risk agent LLM failed: %s — rule-based forecast retained", exc)
            result["error"] = None  # don't crash the report
            result["missing_information"] = [
                f"LLM qualitative risk analysis unavailable ({exc}). "
                "Rule-based forecast and extracted facts are still shown."
            ]
            return result

        result = _merge(result, raw, rb_forecast)

    except Exception as exc:
        logger.exception("risk_agent unexpected error: %s", exc)
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_structured(scope_data: dict, blocker_data: dict) -> str:
    """Format scope + blocker data as a readable text block for the LLM."""
    lines = []

    goals = scope_data.get("project_goals", [])
    if goals:
        lines.append("Project goals: " + "; ".join(g.get("goal", "") for g in goals[:5]))

    deliverables = scope_data.get("deliverables", [])
    if deliverables:
        lines.append("Deliverables:")
        for d in deliverables[:8]:
            name     = d.get("name", "Unnamed")
            deadline = d.get("deadline") or "no deadline"
            owner    = d.get("owner") or "no owner"
            lines.append(f"  - {name} | deadline: {deadline} | owner: {owner}")

    milestones = scope_data.get("milestones", [])
    if milestones:
        lines.append("Milestones:")
        for m in milestones[:8]:
            lines.append(f"  - {m.get('name','?')} | {m.get('date','no date')}")

    conflicts = scope_data.get("conflicts", [])
    if conflicts:
        lines.append("Conflicts detected in scope data:")
        for c in conflicts:
            lines.append(f"  - {c.get('description','')}")

    blockers = blocker_data.get("blockers", [])
    if blockers:
        lines.append("Open blockers:")
        for b in blockers[:8]:
            lines.append(f"  - [{b.get('blocker_id','?')}] {b.get('description','')} (owner: {b.get('owner','unassigned')})")

    action_items = blocker_data.get("action_items", [])
    overdue = [a for a in action_items if a.get("status") == "Overdue"]
    if overdue:
        lines.append("Overdue action items:")
        for a in overdue[:5]:
            lines.append(f"  - [{a.get('action_id','?')}] {a.get('description','')} (due: {a.get('due_date','?')})")

    return "\n".join(lines) if lines else "No structured data available."


def _merge(base: dict, raw: dict, rb_forecast: dict) -> dict:
    """Merge LLM output into base, keeping rule-based forecast intact.

    Normalizes compact LLM output to the full template-expected format
    so every field the UI references always exists.
    """
    out = {**base}

    # --- Risks ---
    val = raw.get("risks")
    if isinstance(val, list):
        norm = []
        for r in val:
            if not isinstance(r, dict):
                continue
            norm.append({
                "risk_id":              r.get("risk_id", r.get("id", "")),
                "title":                r.get("title", ""),
                "description":          r.get("description", ""),
                "severity":             r.get("severity", "Medium"),
                "probability":          r.get("probability", "Unknown"),
                "impact":               r.get("impact"),
                "affected_deliverable": r.get("affected_deliverable"),
                "evidence":             r.get("evidence", ""),
                "source":               r.get("source", ""),
                "potential_impact":     r.get("potential_impact", ""),
                "mitigation":           r.get("mitigation", ""),
                "trigger":              r.get("trigger"),
                "owner":                r.get("owner"),
                "status":               r.get("status", "Unknown"),
            })
        out["risks"] = norm

    # --- Dependency Gaps ---
    val = raw.get("dependency_gaps")
    if isinstance(val, list):
        norm = []
        for g in val:
            if not isinstance(g, dict):
                continue
            norm.append({
                "gap_id":        g.get("gap_id", g.get("id", "")),
                "description":   g.get("description", ""),
                "affected_task": g.get("affected_task", ""),
                "source":        g.get("source", ""),
            })
        out["dependency_gaps"] = norm

    # --- Missing Information ---
    val = raw.get("missing_information")
    if isinstance(val, list):
        out["missing_information"] = val

    # Merge forecasts: rule-based dates are authoritative; LLM fills qualitative gaps
    llm_fc = raw.get("delivery_forecast")
    if isinstance(llm_fc, dict) and llm_fc:  # non-empty dict only
        merged_fc = {**rb_forecast}
        if llm_fc.get("status") and llm_fc["status"] != "Insufficient data":
            merged_fc["status"] = llm_fc["status"]
        # If rule-based has no date (no milestones/deliverables), use LLM date
        if not merged_fc.get("estimated_completion_date") and llm_fc.get("estimated_completion_date"):
            merged_fc["estimated_completion_date"] = llm_fc["estimated_completion_date"]
        if llm_fc.get("delayed_milestones"):
            existing = set(merged_fc.get("delayed_milestones", []))
            for m in llm_fc["delayed_milestones"]:
                existing.add(m)
            merged_fc["delayed_milestones"] = list(existing)
        if llm_fc.get("assumptions"):
            merged_fc["assumptions"] = (
                merged_fc.get("assumptions", []) + llm_fc["assumptions"]
            )
        merged_fc["forecast_method"] = "hybrid"
        if llm_fc.get("confidence") == "High":
            merged_fc["confidence"] = "High"
        out["delivery_forecast"] = merged_fc
    else:
        out["delivery_forecast"] = rb_forecast

    out["error"] = None
    return out
