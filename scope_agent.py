"""
Agent 1 — Scope & Deliverable Extraction Agent
================================================
Retrieves project goals, scope, deliverables, milestones, owners,
dependencies, and acceptance criteria from the indexed knowledge base.

Public API
----------
    from agents.scope_agent import run as run_scope_agent
    result = run_scope_agent(project_id)   # returns dict
"""

import json
import logging
from typing import Any

from agents.llm_client import (
    LLMError, build_context, call_llm_json, tpm_context_cap,
    DEFAULT_MODEL, get_cached_llm_result, set_cached_llm_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema (empty/safe default returned when data is insufficient)
# ---------------------------------------------------------------------------

EMPTY_OUTPUT: dict[str, Any] = {
    "project_goals":       [],
    "scope":               [],
    "out_of_scope":        [],
    "deliverables":        [],
    "milestones":          [],
    "missing_information": [],
    "conflicts":           [],
    "extraction_gaps":     [],
    "error":               None,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a project analyst. Extract structured information from the document excerpts.

Rules:
1. Extract ALL information present — do not omit partial mentions.
2. Use empty list only when information is genuinely absent from ALL excerpts.
3. Deduplicate synonymous entries.
4. Cite the source filename for every extracted item.
5. Do NOT confuse kickoff dates or status report dates with actual project milestones.
   Kickoff = project start event. Milestones = named checkpoints/deliverable deadlines.
6. If a deliverable has dependencies or acceptance criteria mentioned, include them.

Reply with ONLY this JSON — no markdown, no extra text:
{
  "project_goals": [{"goal": "string", "source": "filename"}],
  "scope": [{"item": "string", "source": "filename"}],
  "out_of_scope": [{"item": "string", "source": "filename"}],
  "deliverables": [{"name": "string", "description": "string", "deadline": "YYYY-MM-DD or null", "owner": "string or null", "dependencies": ["string"], "acceptance_criteria": ["string"], "source": "filename"}],
  "milestones": [{"name": "string", "date": "YYYY-MM-DD or null", "description": "string", "source": "filename"}],
  "missing_information": ["string with source reference where possible"]
}"""

# ---------------------------------------------------------------------------
# Retrieval queries — cast a wide net across common project document language
# ---------------------------------------------------------------------------

RETRIEVAL_QUERIES = [
    # Out-of-scope runs FIRST so it gets context budget before scope/goals fill it
    "out of scope excluded not included constraints assumptions",
    "project objectives goals deliverables scope",
    "milestones deadlines timeline start end date",
    "task owner responsible dependencies acceptance criteria",
]

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(project_id: str, context_k: int = 8) -> dict[str, Any]:
    """Extract scope and deliverables for *project_id*.

    context_k=8 so every file in a 7-file project can contribute ≥1 chunk
    per retrieval query (max_per_source = ceil(8/7) = 2).
    Never raises — errors are captured in result["error"].
    """
    result = {**EMPTY_OUTPUT, "error": None}

    try:
        _AGENT     = "scope"
        _MAX_TOK   = 1200   # non-reasoning model (qwen3.8-27b): all tokens go to output, no hidden thinking
        # Compute how many context chars fit within the TPM budget
        _CTX_CHARS = tpm_context_cap(SYSTEM_PROMPT, _MAX_TOK)
        context = build_context(project_id, RETRIEVAL_QUERIES, k=context_k, max_chars=_CTX_CHARS)
        if not context:
            result["missing_information"] = [
                "No documents are indexed for this project. "
                "Upload documents before running analysis."
            ]
            result["error"] = "No indexed content found."
            return result

        # 2. Build user prompt
        user_prompt = (
            f"Project ID: {project_id}\n\n"
            "Below are excerpts from the project documents. "
            "Extract all scope and deliverable information according to the schema.\n\n"
            f"{context}"
        )

        # 3. Call LLM (check result cache first)
        cached = get_cached_llm_result(_AGENT, project_id, SYSTEM_PROMPT,
                                       context, _MAX_TOK, DEFAULT_MODEL)
        if cached is not None:
            # Only use the cache if it contains actual extracted content.
            # An empty result means a previous LLM call returned nothing useful —
            # try again rather than repeatedly serving stale empty data.
            has_content = any(
                isinstance(cached.get(k), list) and len(cached.get(k, [])) > 0
                for k in ("project_goals", "scope", "deliverables", "milestones")
            )
            if has_content:
                logger.info("scope_agent: using cached LLM result for '%s'", project_id)
                return _merge(result, cached)
            else:
                logger.info("scope_agent: cached result is empty — re-running LLM for '%s'", project_id)

        # max_tokens=1200, timeout=30: qwen3.8-27b is non-reasoning — no hidden thinking overhead.
        # Context chars computed dynamically by tpm_context_cap() above.
        try:
            raw = call_llm_json(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                fallback=None,
                max_tokens=_MAX_TOK,
                timeout=30,
                json_mode=False,   # json_object mode returns 400 on reasoning model
            )
        except LLMError as exc:
            # LLM failed (rate limit, JSON validation, truncation, etc.)
            # Don't mark the agent as "failed" — return empty results with
            # a clear note so the UI shows a warning, not a red error badge.
            logger.warning("Scope agent LLM failed: %s — returning empty scope", exc)
            result["error"] = None
            result["missing_information"] = [
                f"LLM-based scope extraction unavailable ({exc}). "
                "Upload more documents or try again later."
            ]
            return result

        # 4. Validate, cache, and merge into result
        set_cached_llm_result(_AGENT, project_id, SYSTEM_PROMPT,
                               context, _MAX_TOK, DEFAULT_MODEL, raw)
        result = _merge(result, raw)

    except Exception as exc:
        logger.exception("scope_agent unexpected error: %s", exc)
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge(base: dict, raw: dict) -> dict:
    """Safely merge LLM output into the base schema.

    Handles both the simplified schema (flat strings for goals/scope) and
    the original rich schema (objects with source/inferred).  Missing fields
    are filled with safe defaults so the template never crashes.
    Also computes extraction_completeness to track what was extracted vs missed.
    """
    out = {**base}
    _gaps = []  # track genuinely missing information

    for key in (
        "project_goals", "scope", "out_of_scope",
        "deliverables", "milestones",
        "missing_information",
    ):
        val = raw.get(key)
        if not isinstance(val, list):
            continue

        if key == "project_goals":
            norm = []
            for v in val:
                if isinstance(v, str):
                    norm.append({"goal": v, "source": "", "inferred": False})
                elif isinstance(v, dict):
                    norm.append({
                        "goal":     v.get("goal", v.get("description", "")),
                        "source":   v.get("source", ""),
                        "inferred": v.get("inferred", False),
                    })
            out[key] = norm

        elif key in ("scope", "out_of_scope"):
            norm = []
            for v in val:
                if isinstance(v, str):
                    norm.append({"item": v, "source": "", "inferred": False})
                elif isinstance(v, dict):
                    norm.append({
                        "item":    v.get("item", v.get("description", "")),
                        "source":  v.get("source", ""),
                        "inferred": v.get("inferred", False),
                    })
            out[key] = norm

        elif key == "deliverables":
            norm = []
            for v in val:
                if isinstance(v, dict):
                    d = {
                        "name":                v.get("name", ""),
                        "description":         v.get("description", ""),
                        "deadline":            v.get("deadline"),
                        "owner":               v.get("owner"),
                        "dependencies":        v.get("dependencies", []),
                        "acceptance_criteria": v.get("acceptance_criteria", []),
                        "source":              v.get("source", ""),
                        "inferred":            v.get("inferred", False),
                    }
                    # Track sub-field gaps for health scoring
                    if not d["description"]:
                        _gaps.append(f"Deliverable '{d['name']}' missing description")
                    if not d["deadline"]:
                        _gaps.append(f"Deliverable '{d['name']}' missing deadline")
                    if not d["owner"]:
                        _gaps.append(f"Deliverable '{d['name']}' missing owner")
                    norm.append(d)
            out[key] = norm

        elif key == "milestones":
            norm = []
            for v in val:
                if isinstance(v, dict):
                    norm.append({
                        "name":        v.get("name", ""),
                        "date":        v.get("date"),
                        "description": v.get("description", ""),
                        "source":      v.get("source", ""),
                    })
            out[key] = norm

        else:
            out[key] = val

    # conflicts: keep whatever the LLM returns or default to []
    conflicts = raw.get("conflicts")
    if isinstance(conflicts, list):
        out["conflicts"] = conflicts

    # Track extraction completeness for health scoring and UI
    out["extraction_gaps"] = _gaps

    out["error"] = None
    return out
