"""
Agent 4 — Documentation Generation Agent
=========================================
Produces user stories, a consolidated risk register, and structured action
item lists from the outputs of the scope, blocker, and risk agents.

This agent is DETERMINISTIC — it derives its output directly from the
structured data already extracted by the other three agents, with no LLM
call. This keeps it instant and avoids burning extra API tokens.

Public API
----------
    from agents.doc_agent import run as run_doc_agent
    result = run_doc_agent(project_id, scope_data, blocker_data, risk_data)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

EMPTY_OUTPUT: dict[str, Any] = {
    "user_stories":         [],
    "risk_register":        [],
    "action_item_list":     [],
    "missing_information":  [],
    "error":                None,
}

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    project_id: str,
    scope_data: dict | None = None,
    blocker_data: dict | None = None,
    risk_data: dict | None = None,
) -> dict[str, Any]:
    """Generate user stories, risk register, and action items deterministically.

    Derives output entirely from the structured data produced by the other
    agents — no LLM call, runs in milliseconds.
    Never raises — errors captured in result["error"].
    """
    result = {**EMPTY_OUTPUT}

    scope_data   = scope_data   or {}
    blocker_data = blocker_data or {}
    risk_data    = risk_data    or {}

    try:
        result["user_stories"]    = _generate_user_stories(scope_data)
        result["risk_register"]   = _build_risk_register(risk_data, blocker_data)
        result["action_item_list"] = _build_action_list(blocker_data)
        result["error"] = None

    except Exception as exc:
        logger.exception("doc_agent unexpected error: %s", exc)
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Deterministic generators
# ---------------------------------------------------------------------------

def _generate_user_stories(scope_data: dict) -> list[dict]:
    """Generate user stories from scope items and deliverables.

    Format: "As a [role], I want [feature], so that [benefit]."
    Role is inferred from common project roles. Benefits are derived from
    the deliverable description when available, not generic filler.
    """
    stories = []
    us_id = 1

    # From deliverables — use description for benefit when available
    for d in scope_data.get("deliverables", [])[:15]:
        if not isinstance(d, dict):
            continue
        name = d.get("name", "")
        if not name:
            continue
        desc = d.get("description", "")
        role = _infer_role(name + " " + desc)
        # Derive a meaningful benefit from the description
        if desc and len(desc) > 10:
            benefit = desc[:120]
        else:
            benefit = "I can accomplish my project-related tasks efficiently"
        stories.append({
            "id":      f"US{us_id:03d}",
            "role":    role,
            "want":    name,
            "benefit": benefit,
            "source":  d.get("source", ""),
        })
        us_id += 1

    # From in-scope features — use the item text as the "want"
    for s in scope_data.get("scope", [])[:10]:
        if not isinstance(s, dict):
            continue
        item = s.get("item", "")
        if not item:
            continue
        role = _infer_role(item)
        stories.append({
            "id":      f"US{us_id:03d}",
            "role":    role,
            "want":    item.rstrip("."),
            "benefit": "I can complete my workflow without manual workarounds",
            "source":  s.get("source", ""),
        })
        us_id += 1

    # From project goals — each goal becomes a user story
    for g in scope_data.get("project_goals", [])[:5]:
        if not isinstance(g, dict):
            continue
        goal = g.get("goal", "")
        if not goal:
            continue
        stories.append({
            "id":      f"US{us_id:03d}",
            "role":    "stakeholder",
            "want":    goal.rstrip("."),
            "benefit": "the project delivers its intended value",
            "source":  g.get("source", ""),
        })
        us_id += 1

    return stories


def _infer_role(text: str) -> str:
    """Infer a user role from feature/deliverable text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ("admin", "manager", "dashboard")):
        return "admin"
    if any(w in text_lower for w in ("staff", "employee", "operator")):
        return "staff member"
    if any(w in text_lower for w in ("student", "learner", "attendee")):
        return "student"
    if any(w in text_lower for w in ("report", "analytics", "data")):
        return "project manager"
    return "user"


def _build_risk_register(risk_data: dict, blocker_data: dict) -> list[dict]:
    """Consolidate risks from the risk agent into a risk register.

    Includes all risk fields: severity, impact, mitigation, owner, status,
    probability, and affected deliverable.
    """
    register = []

    for r in risk_data.get("risks", []):
        if not isinstance(r, dict):
            continue
        register.append({
            "risk_id":    r.get("risk_id", ""),
            "title":      r.get("title", ""),
            "severity":   r.get("severity", "Medium"),
            "impact":     r.get("potential_impact") or r.get("description", ""),
            "mitigation": r.get("mitigation", ""),
            "owner":      r.get("owner"),
            "status":     r.get("status", "Potential"),
            "source":     r.get("source", ""),
        })

    # Promote critical open blockers as risks too
    b_id = len(register) + 1
    for b in blocker_data.get("blockers", []):
        if not isinstance(b, dict):
            continue
        if b.get("status", "").lower() == "open":
            register.append({
                "risk_id":    f"R-B{b_id:03d}",
                "title":      f"Blocker: {b.get('description', '')[:60]}",
                "severity":   "High",
                "impact":     b.get("impact", ""),
                "mitigation": b.get("recommended_resolution", ""),
                "owner":      b.get("owner"),
                "status":     "Confirmed",
            })
            b_id += 1

    return register


def _build_action_list(blocker_data: dict) -> list[dict]:
    """Build a structured action item list from the blocker agent output."""
    actions = []

    for a in blocker_data.get("action_items", []):
        if not isinstance(a, dict):
            continue
        actions.append({
            "action_id":   a.get("action_id", ""),
            "description": a.get("description", ""),
            "owner":       a.get("owner"),
            "due_date":    a.get("due_date"),
            "priority":    a.get("priority", "Unspecified"),
            "status":      a.get("status", "Unknown"),
            "source":      a.get("source", ""),
        })

    return actions
