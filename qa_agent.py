"""
Conversational Project Intelligence Assistant
===============================================
RAG-powered Q&A that answers project questions grounded in uploaded
documents.  Every answer cites the source chunks used, and if no
relevant chunks are found, the assistant says so rather than fabricating.

Example questions:
  "Are we on track?"
  "What are our biggest risks?"
  "What blockers are open?"

Public API
----------
    from agents.qa_agent import answer
    result = answer(project_id, question)
    # → {"answer": "...", "sources": [...], "confidence": "High|Medium|Low|None"}
"""

import json
import logging
from typing import Any

from agents.llm_client import (
    LLMError, build_context, call_llm_json, tpm_context_cap,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

EMPTY_ANSWER: dict[str, Any] = {
    "answer":     "",
    "sources":    [],
    "confidence": "None",
    "error":      None,
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a project intelligence assistant. Answer the user's
question using ONLY the document excerpts provided below.

Rules:
1. Answer based ONLY on the provided excerpts — never use outside knowledge.
2. If the excerpts do not contain relevant information, say:
   "I could not find information about that in the uploaded documents."
3. Keep answers concise (2-4 sentences) and directly address the question.
4. Reference sources naturally, e.g. "According to [filename], ..."
5. If asked about status, timeline, risks, or blockers, cite specific evidence.

Reply with ONLY this JSON — no markdown, no extra text:
{
  "answer": "string",
  "confidence": "High | Medium | Low",
  "sources_cited": ["filename or source label"]
}"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer(project_id: str, question: str) -> dict[str, Any]:
    """Answer a project question using RAG retrieval + LLM generation.

    Returns a dict with:
      - answer:     the generated answer text
      - sources:    list of source chunks used (filename + chunk index)
      - confidence: High | Medium | Low | None
      - error:      None on success, error message on failure

    Never raises — errors captured in result["error"].
    """
    result = {**EMPTY_ANSWER}

    if not question or not question.strip():
        result["error"] = "No question provided."
        return result

    question = question.strip()

    try:
        # 1. Retrieve relevant context via RAG
        _MAX_TOK = 2000
        _CTX_CHARS = tpm_context_cap(SYSTEM_PROMPT, _MAX_TOK, framing_tokens=40)
        context = build_context(
            project_id,
            [question, question + " status", question + " risk blocker"],
            k=8,
            max_chars=_CTX_CHARS,
        )

        # 1b. Load saved analysis report and inject key results as context
        analysis_context = _build_analysis_context(project_id)
        if analysis_context:
            context = (analysis_context + "\n\n" + context) if context else analysis_context

        if not context:
            result["answer"] = (
                "No documents are indexed for this project. "
                "Upload documents first, then ask your question."
            )
            result["confidence"] = "None"
            return result

        # 2. Parse source labels from the context for citation
        sources = _extract_sources(context)

        # 3. Build the user prompt
        user_prompt = (
            f"Project: {project_id}\n\n"
            f"Question: {question}\n\n"
            "=== DOCUMENT EXCERPTS ===\n"
            f"{context}\n\n"
            "Answer the question using only the excerpts above."
        )

        # 4. Call LLM
        try:
            raw = call_llm_json(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                fallback=None,
                max_tokens=_MAX_TOK,
                timeout=60,
                json_mode=False,
            )
        except LLMError as exc:
            logger.warning("QA agent LLM failed: %s", exc)
            # Fall back to returning the raw context as a best-effort answer
            result["answer"] = (
                "I found relevant document excerpts but could not generate "
                f"a summary at this time ({exc}). Please try again later."
            )
            result["sources"] = sources
            result["confidence"] = "Low"
            return result

        # 5. Merge LLM output
        answer_text = raw.get("answer", "")
        confidence  = raw.get("confidence", "Low")
        cited       = raw.get("sources_cited", [])

        result["answer"]     = answer_text or "No answer could be generated."
        result["confidence"] = confidence if confidence in ("High", "Medium", "Low") else "Low"
        # Use the parsed sources from context, filtered by what the LLM cited
        if cited:
            result["sources"] = [s for s in sources if any(
                c.lower() in s["source"].lower() or s["source"].lower() in c.lower()
                for c in cited
            )] or sources
        else:
            result["sources"] = sources

    except Exception as exc:
        logger.exception("qa_agent unexpected error: %s", exc)
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sources(context: str) -> list[dict[str, Any]]:
    """Parse [Source: filename | Chunk #N] headers from the context string."""
    import re
    sources = []
    seen = set()
    for m in re.finditer(r"\[Source:\s*(.+?)\s*\|\s*Chunk #(\d+)\]", context):
        source_name = m.group(1)
        chunk_idx = int(m.group(2))
        # Deduplicate by filename only — don't show the same file twice
        if source_name not in seen:
            seen.add(source_name)
            sources.append({"source": source_name, "chunk": chunk_idx})
    return sources


def _build_analysis_context(project_id: str) -> str:
    """Load the saved analysis report and format key results as context.

    This lets the Q&A agent answer questions about analysis outputs
    (missing information, risks, blockers, health score, forecast)
    that are not present in the raw document chunks.
    """
    try:
        from report_store import load_report
        report = load_report(project_id)
        if not report:
            return ""

        parts: list[str] = []

        # Health score
        hs = report.get("health_score") or {}
        if hs:
            parts.append(
                f"[Source: Analysis Report | Chunk #0]\n"
                f"=== ANALYSIS: HEALTH SCORE ===\n"
                f"Overall: {hs.get('overall', '?')}/100 (Grade {hs.get('grade', '?')})\n"
                f"Scope Clarity: {hs.get('scope_clarity', '?')}/100\n"
                f"Timeline Risk: {hs.get('timeline_risk', '?')}/100\n"
                f"Blocker Severity: {hs.get('blocker_severity', '?')}/100\n"
                f"Risk Exposure: {hs.get('risk_exposure', '?')}/100\n"
                f"Summary: {hs.get('summary', '')}"
            )

        # Scope — missing information
        scope = report.get("scope") or {}
        missing = scope.get("missing_information") or []
        if missing:
            parts.append(
                f"[Source: Analysis Report | Chunk #1]\n"
                f"=== ANALYSIS: MISSING INFORMATION ===\n"
                + "\n".join(f"- {m}" for m in missing)
            )

        # Blockers
        blockers = report.get("blockers") or {}
        blk_list = blockers.get("blockers") or []
        if blk_list:
            lines = []
            for b in blk_list:
                if isinstance(b, dict):
                    lines.append(
                        f"- {b.get('blocker_id', '?')}: {b.get('description', '')} "
                        f"(Status: {b.get('status', '?')}, Owner: {b.get('owner', '?')})"
                    )
            if lines:
                parts.append(
                    f"[Source: Analysis Report | Chunk #2]\n"
                    f"=== ANALYSIS: BLOCKERS ===\n" + "\n".join(lines)
                )

        # Risks
        risks = report.get("risks") or {}
        risk_list = risks.get("risks") or []
        if risk_list:
            lines = []
            for r in risk_list:
                if isinstance(r, dict):
                    lines.append(
                        f"- {r.get('risk_id', '?')}: {r.get('title', r.get('description', ''))} "
                        f"(Severity: {r.get('severity', '?')}, Probability: {r.get('probability', '?')})"
                    )
            if lines:
                parts.append(
                    f"[Source: Analysis Report | Chunk #3]\n"
                    f"=== ANALYSIS: RISKS ===\n" + "\n".join(lines)
                )

        # Delivery forecast
        fc = risks.get("delivery_forecast") or {}
        if fc and fc.get("status"):
            parts.append(
                f"[Source: Analysis Report | Chunk #4]\n"
                f"=== ANALYSIS: DELIVERY FORECAST ===\n"
                f"Status: {fc.get('status', '?')}\n"
                f"Estimated completion: {fc.get('estimated_completion_date', 'N/A')}\n"
                f"Confidence: {fc.get('confidence', '?')}\n"
                f"Delayed milestones: {', '.join(fc.get('delayed_milestones', [])) or 'None'}"
            )

        # Blockers missing information
        blk_missing = blockers.get("missing_information") or []
        if blk_missing:
            parts.append(
                f"[Source: Analysis Report | Chunk #5]\n"
                f"=== ANALYSIS: BLOCKERS/risks MISSING INFORMATION ===\n"
                + "\n".join(f"- {m}" for m in blk_missing)
            )

        # Deliverables
        deliverables = scope.get("deliverables") or []
        if deliverables:
            lines = []
            for d in deliverables:
                if isinstance(d, dict):
                    lines.append(
                        f"- {d.get('name', '?')}: {d.get('description', '')} "
                        f"(Deadline: {d.get('deadline', '?')}, Owner: {d.get('owner', '?')})"
                    )
            if lines:
                parts.append(
                    f"[Source: Analysis Report | Chunk #6]\n"
                    f"=== ANALYSIS: DELIVERABLES ===\n" + "\n".join(lines)
                )

        # Milestones
        milestones = scope.get("milestones") or []
        if milestones:
            lines = []
            for m in milestones:
                if isinstance(m, dict):
                    lines.append(f"- {m.get('name', '?')}: {m.get('date', '?')}")
            if lines:
                parts.append(
                    f"[Source: Analysis Report | Chunk #7]\n"
                    f"=== ANALYSIS: MILESTONES ===\n" + "\n".join(lines)
                )

        return "\n\n".join(parts) if parts else ""

    except Exception as exc:
        logger.warning("qa_agent: could not load analysis report: %s", exc)
        return ""
