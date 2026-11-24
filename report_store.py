"""
Report Store — persist and load analysis reports
=================================================
Saves the last analysis report for each project to disk as a compact JSON
summary so the dashboard can display health scores, risk counts and other
insights without re-running analysis.

Storage layout:
    reports/<project_id>.json   — full report (overwritten on each run)

The module is intentionally simple: no database, no migrations.  A missing
or corrupt file is silently ignored — the dashboard just shows no data for
that project.

Public API
----------
    from report_store import save_report, load_summary, load_report, list_summaries
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_report(project_id: str, report: dict) -> None:
    """Persist *report* to disk for *project_id*.

    Saves the full report plus a lightweight ``summary`` sub-dict that the
    dashboard reads without deserialising the whole file.
    """
    if not project_id or not isinstance(report, dict):
        return
    try:
        # Augment with a compact summary for fast dashboard reads
        report_with_meta = {
            **report,
            "summary": _build_summary(report),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        path = REPORTS_DIR / f"{project_id}.json"
        path.write_text(
            json.dumps(report_with_meta, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("report_store: saved report for '%s'", project_id)
    except Exception as exc:
        logger.warning("report_store: could not save report for '%s': %s", project_id, exc)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_report(project_id: str) -> dict | None:
    """Return the full saved report for *project_id*, or ``None``."""
    path = REPORTS_DIR / f"{project_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("report_store: could not read report for '%s': %s", project_id, exc)
        return None


def load_summary(project_id: str) -> dict | None:
    """Return just the ``summary`` sub-dict for *project_id*, or ``None``.

    Much cheaper than loading the full report when only the dashboard
    overview fields are needed.
    """
    report = load_report(project_id)
    if report is None:
        return None
    return report.get("summary")


def list_summaries() -> dict[str, dict]:
    """Return ``{project_id: summary_dict}`` for every persisted report."""
    result: dict[str, dict] = {}
    for path in REPORTS_DIR.glob("*.json"):
        project_id = path.stem
        summary = load_summary(project_id)
        if summary:
            result[project_id] = summary
    return result


def delete_report(project_id: str) -> None:
    """Remove the persisted report for *project_id* (called on project delete)."""
    path = REPORTS_DIR / f"{project_id}.json"
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("report_store: could not delete report for '%s': %s", project_id, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_summary(report: dict) -> dict[str, Any]:
    """Extract the dashboard-relevant fields into a compact dict."""
    hs = report.get("health_score") or {}
    scope = report.get("scope") or {}
    blockers = report.get("blockers") or {}
    risks = report.get("risks") or {}
    docs = report.get("docs") or {}

    # Count open blockers
    open_blockers = sum(
        1 for b in blockers.get("blockers", [])
        if isinstance(b, dict) and b.get("status", "").lower() == "open"
    )

    # Count risks by severity
    risk_list = risks.get("risks", [])
    critical_risks = sum(1 for r in risk_list if isinstance(r, dict) and r.get("severity") == "Critical")
    high_risks     = sum(1 for r in risk_list if isinstance(r, dict) and r.get("severity") == "High")
    total_risks    = len(risk_list)

    # Forecast status
    fc = risks.get("delivery_forecast") or {}
    forecast_status = fc.get("status", "Insufficient data")

    # Counts
    goal_count        = len(scope.get("project_goals", []))
    deliverable_count = len(scope.get("deliverables", []))
    action_item_count = len(blockers.get("action_items", []))
    user_story_count  = len(docs.get("user_stories", []))

    return {
        "project_id":       report.get("project_id", ""),
        "generated_at":     report.get("generated_at", ""),
        "saved_at":         datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": report.get("duration_seconds", 0),
        "status":           report.get("status", "unknown"),
        "agent_status":     report.get("agent_status", {}),
        # Health
        "overall":          hs.get("overall", 0),
        "grade":            hs.get("grade", "F"),
        "scope_clarity":    hs.get("scope_clarity", 0),
        "timeline_risk":    hs.get("timeline_risk", 0),
        "blocker_severity": hs.get("blocker_severity", 0),
        "risk_exposure":    hs.get("risk_exposure", 0),
        "summary_text":     hs.get("summary", ""),
        # Forecast
        "forecast_status":  forecast_status,
        # Counts
        "open_blockers":    open_blockers,
        "total_risks":      total_risks,
        "critical_risks":   critical_risks,
        "high_risks":       high_risks,
        "goal_count":       goal_count,
        "deliverable_count": deliverable_count,
        "action_item_count": action_item_count,
        "user_story_count": user_story_count,
    }
