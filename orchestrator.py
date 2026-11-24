"""
Orchestrator — runs all agents and merges their outputs
=======================================================
Execution plan:

  Step 1 — ONE LLM CALL: combined_agent extracts scope + blockers + risks
            in a single request.  On the free tier (8 K TPM) this is the
            only viable approach: three parallel calls each waste ~1 400
            tokens on system-prompt overhead, leaving almost nothing for
            document context.  One combined call gets ~6 300 tokens of
            context instead.
  Step 2 — INSTANT: Docs  (deterministic, no LLM call)
  Step 3 — INSTANT: Health score (pure computation)

Wall-clock time ≈ one API call latency ≈ 5–10 seconds.

Public API
----------
    from agents.orchestrator import run as run_analysis
    report = run_analysis(project_id)   # returns AnalysisReport dict
"""

import logging
import time
from datetime import datetime
from typing import Any

import agents.combined_agent as _combined_agent_mod
import agents.doc_agent      as _doc_agent_mod
import agents.health_score   as _health_score_mod
import agents.risk_agent     as _risk_agent_mod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------

def empty_report(project_id: str) -> dict[str, Any]:
    return {
        "project_id":       project_id,
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "status":           "error",
        "scope":            {},
        "blockers":         {},
        "risks":            {},
        "docs":             {},
        "health_score":     {},
        "errors":           [],
        "warnings":         [],
        "agent_status":     {
            "scope":    "unknown",
            "blockers": "unknown",
            "risks":    "unknown",
            "docs":     "unknown",
        },
        "duration_seconds": 0.0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(project_id: str) -> dict[str, Any]:
    """Run all agents and return a merged report.

    Step 1 (single LLM call): combined_agent → scope + blockers + risks
    Step 1b (instant fallback): rule-based delivery forecast if LLM returned
                                "Insufficient data" but scope has dated milestones
    Step 2 (instant):         docs agent — deterministic, no LLM
    Step 3 (instant):         health score — pure computation

    Never raises — all agent errors are captured in report["errors"].
    """
    report = empty_report(project_id)
    t0 = time.time()

    # ── Step 1: single LLM call for all three data sections ───────────────
    logger.info("[orchestrator] Step 1 — combined LLM call for '%s'", project_id)
    try:
        scope_result, blocker_result, risk_result = _combined_agent_mod.run(project_id)
    except Exception as exc:
        logger.exception("combined_agent crashed: %s", exc)
        from agents.combined_agent import _EMPTY_SCOPE, _EMPTY_BLOCKERS, _EMPTY_RISKS
        scope_result   = {**_EMPTY_SCOPE,   "error": str(exc)}
        blocker_result = {**_EMPTY_BLOCKERS, "error": str(exc)}
        risk_result    = {**_EMPTY_RISKS,    "error": str(exc)}
    logger.info("[orchestrator] Step 1 done in %.1fs", time.time() - t0)

    # ── Step 1b: Rule-based forecast enhancement ──────────────────────────
    # The combined agent's LLM may return a qualitative forecast but miss
    # specific milestone dates that are in the extracted scope data.
    # Always run the rule-based forecast and merge its concrete date
    # information with whatever the LLM produced.  This costs zero API
    # calls and enriches the forecast with real dates.
    try:
        rb_fc = _risk_agent_mod._rule_based_forecast(scope_result)
        llm_fc = risk_result.get("delivery_forecast", {})
        if not isinstance(llm_fc, dict):
            llm_fc = {}

        # If LLM said "Insufficient data" but rule-based found dates → replace
        if rb_fc.get("status") != "Insufficient data":
            if llm_fc.get("status") == "Insufficient data" or not llm_fc:
                risk_result["delivery_forecast"] = rb_fc
                logger.info("[orchestrator] rule-based forecast applied: %s", rb_fc["status"])
            else:
                # LLM has a forecast — enrich it with rule-based dates
                # Keep the LLM's status but add concrete date info
                if rb_fc.get("estimated_completion_date") and not llm_fc.get("estimated_completion_date"):
                    llm_fc["estimated_completion_date"] = rb_fc["estimated_completion_date"]
                if rb_fc.get("delayed_milestones") and not llm_fc.get("delayed_milestones"):
                    llm_fc["delayed_milestones"] = rb_fc["delayed_milestones"]
                if rb_fc.get("assumptions"):
                    existing = llm_fc.get("assumptions", [])
                    for a in rb_fc["assumptions"]:
                        if a not in existing:
                            existing.append(a)
                    llm_fc["assumptions"] = existing
                llm_fc["forecast_method"] = "hybrid"
                risk_result["delivery_forecast"] = llm_fc
                logger.info("[orchestrator] forecast enriched with rule-based dates (hybrid)")
    except Exception as exc:
        logger.warning("[orchestrator] rule-based forecast failed: %s", exc)

    report["scope"]    = scope_result
    report["blockers"] = blocker_result
    report["risks"]    = risk_result

    if scope_result.get("error"):
        report["errors"].append(f"Scope agent: {scope_result['error']}")
        report["agent_status"]["scope"] = "failed"
    else:
        _has = any(
            isinstance(scope_result.get(k), list) and scope_result[k]
            for k in ("project_goals", "scope", "deliverables", "milestones")
        )
        report["agent_status"]["scope"] = "ok" if _has else "partial"

    if blocker_result.get("error"):
        report["errors"].append(f"Blocker agent: {blocker_result['error']}")
        report["agent_status"]["blockers"] = "failed"
    else:
        _has = any(
            isinstance(blocker_result.get(k), list) and blocker_result[k]
            for k in ("blockers", "action_items", "pending_decisions", "unresolved_issues")
        )
        report["agent_status"]["blockers"] = "ok" if _has else "partial"

    if risk_result.get("error"):
        report["errors"].append(f"Risk agent: {risk_result['error']}")
        report["agent_status"]["risks"] = "failed"
    else:
        _has = (
            isinstance(risk_result.get("risks"), list) and risk_result["risks"]
        ) or (
            isinstance(risk_result.get("delivery_forecast"), dict)
            and risk_result["delivery_forecast"].get("status") != "Insufficient data"
        )
        report["agent_status"]["risks"] = "ok" if _has else "partial"

    # ── Step 2: Docs — instant, no LLM ───────────────────────────────────
    docs_result = _safe_run(_doc_agent_mod.run, project_id,
                            scope_data=scope_result,
                            blocker_data=blocker_result,
                            risk_data=risk_result)
    report["docs"] = docs_result
    if docs_result.get("error"):
        report["errors"].append(f"Doc agent: {docs_result['error']}")
        report["agent_status"]["docs"] = "failed"
    else:
        _has = any(
            isinstance(docs_result.get(k), list) and docs_result[k]
            for k in ("user_stories", "risk_register", "action_item_list")
        )
        report["agent_status"]["docs"] = "ok" if _has else "partial"

    # ── Step 3: Health score — instant, no LLM ───────────────────────────
    report["health_score"] = _health_score_mod.compute(
        scope_data=scope_result,
        blocker_data=blocker_result,
        risk_data=risk_result,
    )

    # ── Warnings ─────────────────────────────────────────────────────────
    for agent_key in ("scope", "blockers", "risks", "docs"):
        for item in report[agent_key].get("missing_information", []):
            report["warnings"].append(f"[{agent_key}] {item}")

    # ── Overall status ────────────────────────────────────────────────────
    if not report["errors"]:
        report["status"] = "ok"
    elif len(report["errors"]) < 4:
        report["status"] = "partial"
    else:
        report["status"] = "error"

    report["duration_seconds"] = round(time.time() - t0, 2)
    logger.info(
        "[orchestrator] Done for '%s' — status=%s, %.1fs",
        project_id, report["status"], report["duration_seconds"],
    )
    return report


def _safe_run(fn, *args, **kwargs) -> dict:
    """Call fn(*args, **kwargs) and catch any unexpected exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.exception("Agent %s crashed: %s", fn.__name__, exc)
        return {"error": str(exc)}
