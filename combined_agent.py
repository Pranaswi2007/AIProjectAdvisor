"""
Combined Agent — TWO sequential LLM calls for reliable extraction
==================================================================
Problem with one call: 13+ JSON lists in a single response causes
truncation, skipped items, and inconsistent results.

Solution: TWO sequential calls:
  Call 1: Scope only (goals, scope, OOS, deliverables, milestones)
  Call 2: Blockers + Risks (blockers, actions, decisions, risks, forecast)

Each call gets a simpler JSON schema → less truncation.
Each call gets more output tokens for its topic → more complete.
Each call gets focused context → better retrieval.
Total tokens: ~7500 — under the 8K TPM free-tier limit.
Wall-clock: ~8s + ~9s = ~17 seconds.

Public API
----------
    from agents.combined_agent import run as run_combined
    scope_data, blocker_data, risk_data = run_combined(project_id)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agents.llm_client import (
    LLMError, build_context, call_llm_json,
    DEFAULT_MODEL, get_cached_llm_result, set_cached_llm_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token budget — 2 calls with higher output to prevent truncation
#
#   Call 1 (Scope):     150 sys + 60 framing + 1000 ctx + 3000 output = 4210
#   Call 2 (Blk+Risk):  250 sys + 60 framing + 1200 ctx + 3000 output = 4510
#   Total:              ~8720 — may slightly exceed 8K TPM; retry logic handles 429
#   The extra output tokens prevent finish_reason=length truncation.
#
#   Each call: 2 retrieval passes × 2000 chars = 4000 chars ≈ 1000 tokens
#   Output per call: 3000 tokens (was 2500 — increase fixes truncation)
#   k=10 per query (was 8 — more chunks available for retrieval)
# ---------------------------------------------------------------------------
_SCOPE_OUTPUT_TOKENS   = 6144   # max output — prevents finish_reason=length truncation
_BLK_RISK_OUTPUT_TOKENS = 8192   # max output — full blockers+risks JSON
_CTX_PER_PASS_CHARS    = 2000   # per retrieval pass (vector search fallback)
_K_CHUNKS              = 15     # high k for better coverage in fallback path
_FULL_CONTEXT_THRESHOLD = 50000 # if total text < this, bypass vector search entirely

# ---------------------------------------------------------------------------
# Retrieval queries — targeted keywords for structured project documents
# ---------------------------------------------------------------------------
_SCOPE_QUERIES = [
    "project objectives goals in scope deliverables milestones dependencies functional requirements acceptance criteria",
    "out of scope excluded not included deliverable D- milestone M- owner deadline charter proposal SRS",
]
_OOS_QUERIES = [
    "out of scope excluded not included constraints assumptions limitations non-functional requirements boundary",
    "not part of project excluded functionality assumptions constraints business rules",
]
_BLOCKER_QUERIES = [
    "confirmed blocker BL- action item ACT- open decision OD- task tracker EE- assigned owner due date priority status",
    "blocked pending waiting unresolved issue decision DEC- follow up overdue behind schedule task ID dependency",
]
_RISK_QUERIES = [
    "risk R- mitigation trigger probability impact rating owner status at risk schedule delay deadline",
    "dependency gap constraint resource budget technical failure project status pilot launch kickoff",
]

# ---------------------------------------------------------------------------
# System prompts — one per call, each focused and simple
# ---------------------------------------------------------------------------
SCOPE_PROMPT = """/no_think
You are a project scope analyst. Extract ALL scope information from the document text into JSON.

Return ONLY this JSON — no markdown fences, no extra text:
{
  "project_goals": [{"goal": "string", "source": "filename"}],
  "scope": [{"item": "string", "source": "filename"}],
  "out_of_scope": [{"item": "string", "source": "filename"}],
  "deliverables": [{"name": "string", "description": "string", "deadline": "YYYY-MM-DD or null", "owner": "string or null", "source": "filename"}],
  "milestones": [{"name": "string", "date": "YYYY-MM-DD or null", "source": "filename"}],
  "missing_information": ["string"]
}

UNIVERSAL EXTRACTION RULES:
- Exhaustiveness: Extract EVERY goal, scope item, out-of-scope item, deliverable, milestone, requirement, and acceptance criterion across ALL files. Never stop after the first few rows of a table. Read every row.
- Exact Fidelity: Preserve exact IDs (D-01, M-01, FR-01, etc.), names, dates, statuses, owners, and dependencies verbatim from the text. Never invent, assume, or overwrite values.
- Table Awareness: When you see a Markdown table (| col | col |), extract EVERY data row as a separate item — do not skip rows.
- Classification: Deliverables = named work packages with owners/deadlines. Milestones = dated checkpoints. Scope = features/capabilities in the project. Out-of-scope = explicitly excluded features.
- Do NOT report items as "missing" or "not documented" if they appear ANYWHERE in the provided text.
- Cite the source filename for every extracted item.
- Empty lists ONLY when that information is genuinely absent from ALL text."""

BLK_RISK_PROMPT = """/no_think
You are a project risk and blocker analyst. Extract ALL blockers, actions, decisions, and risks from the document text into JSON.

Return ONLY this JSON — no markdown fences, no extra text:
{
  "blockers": [{"blocker_id": "string", "description": "string", "status": "Open|Resolved|Unknown", "impact": "string", "owner": "string or null", "evidence": "string", "recommended_resolution": "string", "source": "filename"}],
  "action_items": [{"action_id": "string", "description": "string", "owner": "string or null", "due_date": "YYYY-MM-DD or null", "priority": "High|Medium|Low|Unspecified", "status": "Pending|In Progress|Completed|Overdue|Unknown", "source": "filename"}],
  "pending_decisions": [{"decision_id": "string", "description": "string", "owner": "string or null", "due_date": "YYYY-MM-DD or null", "source": "filename"}],
  "unresolved_issues": [{"issue_id": "string", "description": "string", "impact": "string", "source": "filename"}],
  "risks": [{"risk_id": "string", "title": "string", "description": "string", "severity": "Critical|High|Medium|Low", "probability": "High|Medium|Low|Unknown", "potential_impact": "string", "affected_deliverable": "string or null", "mitigation": "string", "owner": "string or null", "status": "Confirmed|Potential|Unknown", "source": "filename"}],
  "delivery_forecast": {"status": "On Track|At Risk|Delayed|Insufficient data", "estimated_completion_date": "YYYY-MM-DD or null", "delayed_milestones": ["string"], "confidence": "High|Medium|Low", "forecast_method": "qualitative"},
  "dependency_gaps": [{"gap_id": "string", "description": "string", "affected_task": "string", "source": "filename"}],
  "missing_information": ["string"]
}

UNIVERSAL EXTRACTION RULES:
- Exhaustiveness: Extract EVERY blocker, action item, decision, risk, and task tracker row across ALL files. Never stop after the first few rows of a table. Read every row.
- Exact Fidelity: Preserve exact IDs (BL-01, ACT-01, OD-01, R-01, EE-01, DEC-01, etc.), names, dates, statuses, owners, and priorities verbatim. Never invent, assume, or overwrite undecided items.
- Table Awareness: When you see a Markdown table (| col | col |), extract EVERY data row as a separate item — do not skip rows.
- Strict Classification Boundaries:
  * Blockers = things actively blocking progress NOW (status Open). Never classify out-of-scope features, section headings, or document instructions as blockers.
  * Action items = assigned tasks with an explicit owner or due date. Do NOT infer ownership.
  * Pending decisions = UNDECIDED items. Do NOT invent a resolution. Keep confirmed decisions (DEC-xx) separate from open decisions (OD-xx).
  * Risks = things that MAY cause problems (status Confirmed or Potential). Keep a defect (BL-xx) distinct from the action to fix it (ACT-xx).
- Forecast Rules: Aggregate ALL extracted milestones, deliverable deadlines, and task statuses to compute the forecast. If open high-priority blockers exist OR high-rated risks are confirmed, status must be "At Risk" or "Delayed" — never "On Track" when blockers are open.
- Do NOT report items as "missing" if they appear ANYWHERE in the provided text.
- Cite source filename for every extracted item.
- Empty lists ONLY when that information is genuinely absent from ALL text."""

# ---------------------------------------------------------------------------
# Empty output shapes
# ---------------------------------------------------------------------------
_EMPTY_SCOPE: dict[str, Any] = {
    "project_goals": [], "scope": [], "out_of_scope": [],
    "deliverables": [], "milestones": [],
    "missing_information": [], "conflicts": [], "extraction_gaps": [],
    "error": None,
}
_EMPTY_BLOCKERS: dict[str, Any] = {
    "blockers": [], "action_items": [], "pending_decisions": [],
    "unresolved_issues": [], "missing_information": [],
    "extraction_source": "llm", "error": None,
}
_EMPTY_RISKS: dict[str, Any] = {
    "risks": [],
    "delivery_forecast": {
        "status": "Insufficient data",
        "estimated_completion_date": None,
        "delayed_milestones": [], "assumptions": [],
        "confidence": "Low", "forecast_method": "none",
    },
    "dependency_gaps": [], "missing_information": [], "error": None,
}

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(project_id: str) -> tuple[dict, dict, dict]:
    """Run two sequential LLM calls and return (scope_data, blocker_data, risk_data).

    Call 1: Scope extraction (goals, scope, OOS, deliverables, milestones)
    Call 2: Blockers + Risks extraction (blockers, actions, risks, forecast)

    Never raises — errors are captured in result["error"].
    """
    scope_out   = {**_EMPTY_SCOPE}
    blocker_out = {**_EMPTY_BLOCKERS}
    risk_out    = {**_EMPTY_RISKS}

    try:
        # ── Full-context routing: if total project text is small enough,
        #    pass the FULL unabridged text instead of top-k vector search.
        #    This eliminates chunk blindness — no section or table row is missed.
        # ──────────────────────────────────────────────────────────────
        from rag import get_all_text as _get_all_text
        full_text = _get_all_text(project_id)
        total_chars = len(full_text)

        USE_FULL_CONTEXT = total_chars > 0 and total_chars <= _FULL_CONTEXT_THRESHOLD
        logger.info("combined_agent: project '%s' has %d chars → %s",
                     project_id, total_chars,
                     "FULL CONTEXT" if USE_FULL_CONTEXT else "vector search")

        if USE_FULL_CONTEXT:
            # ── FULL CONTEXT PATH: pass all text to both calls ──────────
            # Each call gets the full text delimited by filename.
            scope_context = full_text
            br_context = full_text
        else:
            # ── VECTOR SEARCH PATH: top-k retrieval with targeted queries ──
            scope_ctx   = build_context(project_id, _SCOPE_QUERIES,
                                        k=_K_CHUNKS, max_chars=_CTX_PER_PASS_CHARS)
            oos_ctx     = build_context(project_id, _OOS_QUERIES,
                                        k=_K_CHUNKS, max_chars=_CTX_PER_PASS_CHARS)
            blocker_ctx = build_context(project_id, _BLOCKER_QUERIES,
                                        k=_K_CHUNKS, max_chars=_CTX_PER_PASS_CHARS)
            risk_ctx    = build_context(project_id, _RISK_QUERIES,
                                        k=_K_CHUNKS, max_chars=_CTX_PER_PASS_CHARS)

            if not scope_ctx and not blocker_ctx and not risk_ctx and not oos_ctx:
                msg = ("No documents are indexed for this project. "
                       "Upload documents before running analysis.")
                for out in (scope_out, blocker_out, risk_out):
                    out["missing_information"] = [msg]
                    out["error"] = "No indexed content found."
                return scope_out, blocker_out, risk_out

            # Combine into labelled sections
            scope_sections = []
            if scope_ctx:
                scope_sections.append(f"=== SCOPE / GOALS / DELIVERABLES / MILESTONES ===\n{scope_ctx}")
            if oos_ctx:
                scope_sections.append(f"=== OUT-OF-SCOPE / EXCLUDED / CONSTRAINTS (=OOS=) ===\n{oos_ctx}")
            scope_context = "\n\n".join(scope_sections)

            br_sections = []
            if blocker_ctx:
                br_sections.append(f"=== BLOCKERS / ACTION ITEMS / DECISIONS ===\n{blocker_ctx}")
            if risk_ctx:
                br_sections.append(f"=== RISKS / FORECAST / DEPENDENCIES ===\n{risk_ctx}")
            br_context = "\n\n".join(br_sections)

        scope_user = (
            f"Project ID: {project_id}\n\n"
            "Extract ALL scope information from these excerpts. "
            "The =OOS= section = out_of_scope items.\n\n"
            f"{scope_context}"
        )

        # ── Execution strategy ────────────────────────────────────────────
        # Full-context path: br_context IS the full text, so it already
        # contains all milestones and goals — scope_summary from Call 1
        # adds zero information. Run both calls in parallel.
        #
        # Vector-search path: br_context is targeted excerpts; scope_summary
        # (extracted milestone dates from Call 1) meaningfully improves the
        # delivery forecast in Call 2. Keep sequential there.
        # ──────────────────────────────────────────────────────────────────

        if USE_FULL_CONTEXT:
            # ── PARALLEL path (full-context only) ─────────────────────────
            br_user_parallel = (
                f"Project ID: {project_id}\n\n"
                "Extract ALL blockers, action items, and risks from these excerpts.\n"
                "Use the extracted milestones and deliverable deadlines to determine "
                "the delivery forecast.\n\n"
                f"{br_context}"
            )

            def _call_scope():
                cached = get_cached_llm_result(
                    "scope_v2", project_id, SCOPE_PROMPT,
                    scope_context, _SCOPE_OUTPUT_TOKENS, DEFAULT_MODEL,
                )
                if cached is not None:
                    logger.info("combined_agent: scope cache hit for '%s'", project_id)
                    return cached
                logger.info("combined_agent: Call 1 (scope, parallel) for '%s'", project_id)
                raw = call_llm_json(
                    system=SCOPE_PROMPT,
                    user=scope_user,
                    fallback=None,
                    max_tokens=_SCOPE_OUTPUT_TOKENS,
                    timeout=50,
                    json_mode=False,
                )
                set_cached_llm_result(
                    "scope_v2", project_id, SCOPE_PROMPT,
                    scope_context, _SCOPE_OUTPUT_TOKENS, DEFAULT_MODEL, raw,
                )
                return raw

            def _call_br():
                cached = get_cached_llm_result(
                    "blkrisk_v2", project_id, BLK_RISK_PROMPT,
                    br_context, _BLK_RISK_OUTPUT_TOKENS, DEFAULT_MODEL,
                )
                if cached is not None:
                    logger.info("combined_agent: blockers+risks cache hit for '%s'", project_id)
                    return cached
                logger.info("combined_agent: Call 2 (blk+risk, parallel) for '%s'", project_id)
                raw = call_llm_json(
                    system=BLK_RISK_PROMPT,
                    user=br_user_parallel,
                    fallback=None,
                    max_tokens=_BLK_RISK_OUTPUT_TOKENS,
                    timeout=50,
                    json_mode=False,
                )
                set_cached_llm_result(
                    "blkrisk_v2", project_id, BLK_RISK_PROMPT,
                    br_context, _BLK_RISK_OUTPUT_TOKENS, DEFAULT_MODEL, raw,
                )
                return raw

            scope_raw = br_raw = None
            scope_exc = br_exc = None

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_scope = pool.submit(_call_scope)
                fut_br    = pool.submit(_call_br)
                for fut in as_completed([fut_scope, fut_br]):
                    try:
                        result = fut.result()
                        if fut is fut_scope:
                            scope_raw = result
                        else:
                            br_raw = result
                    except LLMError as exc:
                        if fut is fut_scope:
                            scope_exc = exc
                        else:
                            br_exc = exc

            if scope_exc:
                logger.warning("combined_agent: scope call failed: %s", scope_exc)
                scope_out["missing_information"] = [
                    f"Scope extraction issue ({scope_exc}). The analysis may be incomplete."
                ]
            elif scope_raw:
                _fill_scope(scope_out, scope_raw)

            if br_exc:
                logger.warning("combined_agent: blockers+risks call failed: %s", br_exc)
                blocker_out["missing_information"] = [
                    f"Blocker/risk extraction issue ({br_exc}). The analysis may be incomplete."
                ]
                risk_out["missing_information"] = [
                    f"Risk extraction issue ({br_exc}). The analysis may be incomplete."
                ]
            elif br_raw:
                _fill_blockers(blocker_out, br_raw)
                _fill_risks(risk_out, br_raw)

        else:
            # ── SEQUENTIAL path (vector-search) ───────────────────────────
            # scope_summary (milestone dates from Call 1) improves Call 2 forecast.

            # Call 1: Scope
            scope_cached = get_cached_llm_result(
                "scope_v2", project_id, SCOPE_PROMPT,
                scope_context, _SCOPE_OUTPUT_TOKENS, DEFAULT_MODEL,
            )
            if scope_cached is not None:
                logger.info("combined_agent: scope cache hit for '%s'", project_id)
                _fill_scope(scope_out, scope_cached)
            else:
                logger.info("combined_agent: Call 1 (scope) for '%s'", project_id)
                try:
                    scope_raw = call_llm_json(
                        system=SCOPE_PROMPT,
                        user=scope_user,
                        fallback=None,
                        max_tokens=_SCOPE_OUTPUT_TOKENS,
                        timeout=50,
                        json_mode=False,
                    )
                    set_cached_llm_result(
                        "scope_v2", project_id, SCOPE_PROMPT,
                        scope_context, _SCOPE_OUTPUT_TOKENS, DEFAULT_MODEL, scope_raw,
                    )
                    _fill_scope(scope_out, scope_raw)
                except LLMError as exc:
                    logger.warning("combined_agent: scope call failed: %s", exc)
                    scope_out["missing_information"] = [
                        f"Scope extraction issue ({exc}). The analysis may be incomplete."
                    ]

            # Build scope_summary from Call 1 results for the forecast
            scope_summary = ""
            if scope_out.get("milestones"):
                ms_names = [f"{m.get('name','')} ({m.get('date','?')})" for m in scope_out["milestones"] if isinstance(m, dict)]
                if ms_names:
                    scope_summary = "\n=== EXTRACTED MILESTONES (for forecast) ===\n" + "; ".join(ms_names)
            if scope_out.get("deliverables"):
                dl_names = [f"{d.get('name','')} due {d.get('deadline','?')}" for d in scope_out["deliverables"] if isinstance(d, dict) and d.get("deadline")]
                if dl_names:
                    scope_summary += "\n=== EXTRACTED DELIVERABLE DEADLINES ===\n" + "; ".join(dl_names)
            if scope_out.get("project_goals"):
                goals = [g.get("goal", "") for g in scope_out["project_goals"] if isinstance(g, dict)]
                if goals:
                    scope_summary += "\n=== PROJECT GOALS ===\n" + "; ".join(goals[:5])

            # Call 2: Blockers + Risks
            br_user = (
                f"Project ID: {project_id}\n\n"
                "Extract ALL blockers, action items, and risks from these excerpts.\n"
                "Use the extracted milestones and deliverable deadlines to determine the delivery forecast.\n\n"
                f"{br_context}{scope_summary}"
            )
            br_cached = get_cached_llm_result(
                "blkrisk_v2", project_id, BLK_RISK_PROMPT,
                br_context + scope_summary, _BLK_RISK_OUTPUT_TOKENS, DEFAULT_MODEL,
            )
            if br_cached is not None:
                logger.info("combined_agent: blockers+risks cache hit for '%s'", project_id)
                _fill_blockers(blocker_out, br_cached)
                _fill_risks(risk_out, br_cached)
            else:
                logger.info("combined_agent: Call 2 (blockers+risks) for '%s'", project_id)
                try:
                    br_raw = call_llm_json(
                        system=BLK_RISK_PROMPT,
                        user=br_user,
                        fallback=None,
                        max_tokens=_BLK_RISK_OUTPUT_TOKENS,
                        timeout=50,
                        json_mode=False,
                    )
                    set_cached_llm_result(
                        "blkrisk_v2", project_id, BLK_RISK_PROMPT,
                        br_context + scope_summary, _BLK_RISK_OUTPUT_TOKENS, DEFAULT_MODEL, br_raw,
                    )
                    _fill_blockers(blocker_out, br_raw)
                    _fill_risks(risk_out, br_raw)
                except LLMError as exc:
                    logger.warning("combined_agent: blockers+risks call failed: %s", exc)
                    blocker_out["missing_information"] = [
                        f"Blocker/risk extraction issue ({exc}). The analysis may be incomplete."
                    ]
                    risk_out["missing_information"] = [
                        f"Risk extraction issue ({exc}). The analysis may be incomplete."
                    ]

        return scope_out, blocker_out, risk_out

    except Exception as exc:
        logger.exception("combined_agent unexpected error for '%s': %s", project_id, exc)
        for out in (scope_out, blocker_out, risk_out):
            out["error"] = str(exc)
        return scope_out, blocker_out, risk_out


# ---------------------------------------------------------------------------
# Helpers — same fill functions as before
# ---------------------------------------------------------------------------

def _fill_scope(out: dict, s: dict) -> None:
    if not isinstance(s, dict):
        return

    for key in ("project_goals", "scope", "out_of_scope", "milestones", "missing_information"):
        val = s.get(key)
        if isinstance(val, list):
            out[key] = val

    gaps = []
    deliverables = []
    for d in s.get("deliverables", []):
        if not isinstance(d, dict):
            continue
        name = d.get("name", "")
        entry = {
            "name":                name,
            "description":         d.get("description", ""),
            "deadline":            d.get("deadline"),
            "owner":               d.get("owner"),
            "dependencies":        d.get("dependencies", []),
            "acceptance_criteria": d.get("acceptance_criteria", []),
            "source":              d.get("source", ""),
            "inferred":            False,
        }
        if not entry["description"]:
            gaps.append(f"Deliverable '{name}' missing description")
        if not entry["deadline"]:
            gaps.append(f"Deliverable '{name}' missing deadline")
        if not entry["owner"]:
            gaps.append(f"Deliverable '{name}' missing owner")
        deliverables.append(entry)

    out["deliverables"]    = deliverables
    out["extraction_gaps"] = gaps
    out["error"]           = None


def _fill_blockers(out: dict, b: dict) -> None:
    if not isinstance(b, dict):
        return

    val = b.get("blockers")
    if isinstance(val, list):
        norm = []
        for item in val:
            if not isinstance(item, dict):
                continue
            norm.append({
                "blocker_id":             item.get("blocker_id", item.get("id", "")),
                "description":            item.get("description", ""),
                "status":                 item.get("status", "Unknown"),
                "impact":                 item.get("impact", ""),
                "owner":                  item.get("owner"),
                "evidence":               item.get("evidence", ""),
                "recommended_resolution": item.get("recommended_resolution", ""),
                "source":                 item.get("source", ""),
                "inferred":               False,
            })
        out["blockers"] = norm

    val = b.get("action_items")
    if isinstance(val, list):
        norm = []
        for item in val:
            if not isinstance(item, dict):
                continue
            norm.append({
                "action_id":       item.get("action_id", item.get("id", "")),
                "description":     item.get("description", ""),
                "owner":           item.get("owner"),
                "due_date":        item.get("due_date"),
                "priority":        item.get("priority", "Unspecified"),
                "status":          item.get("status", "Unknown"),
                "related_blocker": item.get("related_blocker"),
                "source":          item.get("source", ""),
                "inferred":        False,
            })
        out["action_items"] = norm

    val = b.get("pending_decisions")
    if isinstance(val, list):
        norm = []
        for item in val:
            if not isinstance(item, dict):
                continue
            norm.append({
                "decision_id": item.get("decision_id", item.get("id", "")),
                "description": item.get("description", ""),
                "owner":       item.get("owner"),
                "due_date":    item.get("due_date"),
                "source":      item.get("source", ""),
            })
        out["pending_decisions"] = norm

    val = b.get("unresolved_issues")
    if isinstance(val, list):
        norm = []
        for item in val:
            if not isinstance(item, dict):
                continue
            norm.append({
                "issue_id":    item.get("issue_id", item.get("id", "")),
                "description": item.get("description", ""),
                "impact":      item.get("impact", ""),
                "source":      item.get("source", ""),
            })
        out["unresolved_issues"] = norm

    val = b.get("missing_information")
    if isinstance(val, list):
        out["missing_information"] = val

    out["extraction_source"] = "llm"
    out["error"] = None


def _fill_risks(out: dict, r: dict) -> None:
    if not isinstance(r, dict):
        return

    val = r.get("risks")
    if isinstance(val, list):
        norm = []
        for item in val:
            if not isinstance(item, dict):
                continue
            norm.append({
                "risk_id":              item.get("risk_id", ""),
                "title":                item.get("title", ""),
                "description":          item.get("description", item.get("potential_impact", "")),
                "severity":             item.get("severity", "Medium"),
                "probability":          item.get("probability", "Unknown"),
                "affected_deliverable": item.get("affected_deliverable"),
                "evidence":             item.get("evidence", ""),
                "potential_impact":     item.get("potential_impact", ""),
                "mitigation":           item.get("mitigation", ""),
                "owner":                item.get("owner"),
                "status":               item.get("status", "Unknown"),
                "source":               item.get("source", ""),
            })
        out["risks"] = norm

    fc = r.get("delivery_forecast")
    if isinstance(fc, dict):
        base = out["delivery_forecast"]
        out["delivery_forecast"] = {
            "status":                    fc.get("status", base["status"]),
            "estimated_completion_date": fc.get("estimated_completion_date"),
            "delayed_milestones":        fc.get("delayed_milestones", []),
            "assumptions":               fc.get("assumptions", base.get("assumptions", [])),
            "confidence":                fc.get("confidence", base["confidence"]),
            "forecast_method":           fc.get("forecast_method", "qualitative"),
        }

    val = r.get("dependency_gaps")
    if isinstance(val, list):
        out["dependency_gaps"] = val

    val = r.get("missing_information")
    if isinstance(val, list):
        out["missing_information"] = val

    out["error"] = None
