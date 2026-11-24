"""
Agent 2 — Blocker & Action Item Identification Agent
=====================================================
Extracts unresolved issues, blockers, action items, pending decisions,
and follow-up actions from meeting notes and progress updates.

Strategy:
  1. Try LLM extraction via Groq (with JSON mode + robust parsing).
  2. If LLM fails or returns empty, fall back to a deterministic
     rule-based extractor that scans document text for blocker/action patterns.
  3. Report the extraction source ("llm", "fallback", or "combined").

Public API
----------
    from agents.blocker_agent import run as run_blocker_agent
    result = run_blocker_agent(project_id)   # returns dict
"""

import logging
import re
from pathlib import Path
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
    "blockers":            [],
    "action_items":        [],
    "pending_decisions":   [],
    "unresolved_issues":   [],
    "missing_information":  [],
    "error":               None,
    "extraction_source":   "none",   # "llm" | "fallback" | "combined" | "none"
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a project coordinator analyst. Your task is to
extract blockers, action items, pending decisions, and unresolved issues
from the project document excerpts provided.

Rules you MUST follow:
1. Only use information that appears in the provided excerpts.
2. Do not assume a person is responsible unless the document explicitly
   assigns them.
3. Use null for missing owners and null for missing due dates.
4. Do not create duplicate action items — if the same item appears in
   multiple sources, merge them and cite all source references.
5. Preserve the original meaning of meeting notes; do not paraphrase
   in a way that changes intent.
6. Cite the source filename for every extracted item.
7. SEPARATE confirmed decisions from pending decisions:
   - If a decision is documented as made, agreed, or committed, it is NOT
     a pending decision — do not include it.
   - If a decision is documented as "not yet", "pending", "needs agreement",
     or similar, include it as a pending decision.
8. Do NOT duplicate task-tracker entries (T-numbers) as formal action items
   unless they appear in meeting minutes or action-item sections with an
   assigned owner or due date.

Reply with a single valid JSON object — no markdown, no extra text:
{
  "blockers": [
    {"id": "B001", "description": "string", "status": "Open | Resolved | Unknown", "owner": "string or null", "source": "filename"}
  ],
  "action_items": [
    {"id": "A001", "description": "string", "owner": "string or null", "due_date": "YYYY-MM-DD or null", "priority": "High | Medium | Low | Unspecified", "status": "Pending | In Progress | Completed | Overdue | Unknown", "source": "filename"}
  ],
  "pending_decisions": [
    {"id": "D001", "description": "string", "owner": "string or null", "due_date": "YYYY-MM-DD or null", "source": "filename"}
  ],
  "unresolved_issues": [
    {"id": "I001", "description": "string", "source": "filename"}
  ],
  "missing_information": ["string"]
}"""

RETRIEVAL_QUERIES = [
    "blocker blocked pending waiting dependency not started",
    "action item task assigned owner due date follow up",
    "pending decision unresolved issue not working",
    "delayed overdue behind schedule at risk meeting notes",
]

# ---------------------------------------------------------------------------
# Rule-based fallback patterns
# ---------------------------------------------------------------------------

# Keywords that signal a blocker — require specific blocker evidence,
# not generic project-management language that applies to deliverables too.
# Deliberately narrowed to avoid false-positives from scope/goal sentences.
_BLOCKER_KEYWORDS = re.compile(
    r"(?:"
    r"blocker|"                                             # explicit word "blocker"
    r"blocked\s+(?:by|until|on|pending)|"                  # "blocked by/until/on/pending"
    r"cannot\s+proceed|"
    r"waiting\s+for\s+\w|"                                 # "waiting for <something>"
    r"not\s+yet\s+(?:committed|provided|agreed|confirmed|granted|resolved)|"
    r"unresolved\s+\w|"                                    # "unresolved <issue>"
    r"not\s+working\b|"
    r"has\s+not\s+been\s+(?:provided|resolved|confirmed|granted|approved)|"
    r"have\s+not\s+been\s+(?:provided|resolved|confirmed|granted)|"
    r"overdue\b|"
    r"behind\s+schedule\b"
    r")",
    re.IGNORECASE,
)

# Keywords that signal an action item — require explicit assignment or deadline,
# not general verbs that also describe scope items.
_ACTION_KEYWORDS = re.compile(
    r"(?:"
    r"due\s+\d|"                         # "due 22"
    r"by\s+\d{1,2}\s+\w|"               # "by 22 October"
    r"deadline\b|"
    r"action\s+item|"                    # explicit "action item" phrase
    r"follow[- ]up\b|"
    r"assigned\s+to\b|"                  # "assigned to X"
    r"owner\s*[:]\s*\w|"                 # "owner: X"
    r"resolve\s+(?:by|before|the)\b|"   # "resolve by/before"
    r"confirm\s+(?:by|before|the)\b|"
    r"obtain\s+\w|"                      # "obtain X"
    r"escalate\b"
    r")",
    re.IGNORECASE,
)

# Pattern for task lines: "T1 — Rohan: ..." or "T1 - Rohan: ..."
_TASK_LINE = re.compile(
    r"(?:T\d+|Task\s+\d+|Action\s*(?:Item)?\s*\d*)\s*[—\-:]\s*"
    r"(?:([A-Z][a-z]+)\s*[—:]\s*)?(.+?)(?:\.\s*Due\s+(.+?)\.)?(?:\.\s*Status:\s*(.+?))?(?:\.\s*(?:Blocker|Dependency):\s*(.+?))?$",
    re.IGNORECASE,
)

# Date patterns: "22 October", "October 22", "22 Oct", "29 October 2026"
_DATE_PATTERN = re.compile(
    r"(?:by\s+|due\s+|deadline[:\s]+)?"
    r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)(?:\s+\d{4})?)",
    re.IGNORECASE,
)

# Pending decision patterns
_DECISION_KEYWORDS = re.compile(
    r"(?:not\s+yet\s+committed|needs?\s+agreement|needs?\s+to\s+agree|"
    r"pending\s+decision|not\s+agreed|has\s+not\s+been\s+confirmed|"
    r"not\s+yet\s+(?:reserved|committed|agreed|confirmed))",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Section-heading / label filters — lines that should never become blockers
# or action items (they are structural document artefacts).
# ---------------------------------------------------------------------------

# A line is a heading if it matches any of these patterns:
#   - Short all-caps word(s): "BLOCKED", "ACTION", "RISKS"
#   - Numbered section:  "3. Open Tasks and Blockers"
#   - Lettered section:  "A. Meeting Notes"
#   - Pure status label: "Status: Not started"
#   - Risk-register prefix: "R1 —", "R2:"
#   - Forecast/narrative openers (no owner, no date)
_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"\d+\.\s+[A-Z]|"                        # "3. Open Tasks …"
    r"[A-Z]\.\s+[A-Z]|"                      # "A. Meeting …"
    r"[A-Z]{2,}(?:\s+[A-Z]+)*\s*$|"         # "BLOCKED", "OPEN TASKS"
    r"STATUS\s*(?:DATE|REPORT)\b|"            # "Status Date", "Status report date"
    r"(?:R|B|T|D|I)\d+\s*[—:\-]\s*(?:Schedule|Access|Interface|Testing)\s+risk"  # "R1 — Schedule risk"
    r")",
    re.IGNORECASE,
)

# Lines that are purely dates/metadata with no task content
_META_LINE_PATTERN = re.compile(
    r"^(?:"
    r"Status\s+report\s+date|"
    r"Project\s+kickoff|"
    r"Planned\s+demo|"
    r"Overall\s+status|"
    r"Reporting\s+period|"
    r"Prepared\s+by|"
    # README / instruction file patterns — these are test guidance, not project facts
    r"(?:test\s+project|sample\s+(?:test\s+)?document|upload\s+all|coverage\s+for|"
    r"expected\s+validation|do\s+not\s+classify|files?\s*[:]\s*\d|"
    r"cross.document\s+check|this\s+is\s+a\s+(?:test|sample)|libraryseat\s+test)|"
    r"\d+\.\s+(?:Status\s+Date|Completed\s+Work|Open\s+Tasks|Meeting\s+Notes|Risks|"
    r"Team|Schedule|Dependencies|Assumptions|Success)"
    r")",
    re.IGNORECASE,
)

# Risk-register lines (Rx — ... risk:) should go to risks, not blockers
_RISK_REGISTER_LINE = re.compile(
    r"^\s*R\d+\s*[—:\-]",
    re.IGNORECASE,
)

# Lines that describe scope/out-of-scope items — NOT blockers or actions
_SCOPE_LINE = re.compile(
    r"^(?:"
    r"(?:in|out of)\s+scope\b|"
    r"excluded?[:\s]|"
    r"not\s+in\s+scope\b|"
    r"(?:will|shall)\s+(?:not|include)|"
    r"(?:D|Deliverable)\d+\s*[—:\-]|"    # deliverable labels like "D1 —"
    r"(?:goal|objective)[s]?\s*[:]\s*|"  # goal/objective headers
    r"this\s+project\s+(?:will|shall|aims)|"
    r"the\s+system\s+(?:will|shall|must)|"
    # README / instruction content
    r"(?:files?\s*[:]\s*\d|coverage\s+for\s+the|expected\s+validation|"
    r"libraryseat\s+test\s+project|upload\s+all\s+(?:four|five|six|seven)|"
    r"(?:staging\s+credentials\s+remain|integration\s+tests\s+are)\b)"
    r")",
    re.IGNORECASE,
)

# Lines that are purely general narrative/context with no specific task assignment
# (catches lines that start with "The project", "Users can", "This feature", etc.)
_GENERAL_FACT_LINE = re.compile(
    r"^(?:"
    r"the\s+(?:project|team|system|application|app|solution|product)\s+(?:will|shall|is|has|does|can|must|aims|should|provides?|allows?|supports?|enables?)\b|"
    r"users?\s+(?:will|shall|can|are|should)\s+\b|"
    r"this\s+(?:feature|module|component|service|api|endpoint)\s+\b|"
    r"all\s+(?:users?|data|requests?|transactions?)\s+\b"
    r")",
    re.IGNORECASE,
)

# Explicit status extraction from task lines like "Status: Blocked"
_STATUS_FIELD = re.compile(
    r"\bStatus\s*[:]\s*"
    r"(Blocked|In\s+progress|Not\s+started|Completed?|Done|Pending|Overdue)",
    re.IGNORECASE,
)

# Extended owner pattern — also matches "Whole team", "Dev", multi-word "Rohan and Anika"
_OWNER_PATTERN_EXT = re.compile(
    r"(?:"
    r"(?:^|—\s*)([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?|Whole\s+team)\s*[—:]|"  # "Rohan —", "Whole team:"
    r"\bAssigned\s+to\s*:\s*([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?|Whole\s+team)|"
    r"\bOwner\s*:\s*([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?|Whole\s+team)"
    r")",
    re.IGNORECASE,
)

# Dates that are reference/history (not deadlines): "Status date: 21 October",
# "Kickoff: 1 October", "Meeting Notes — 20 October"
_HISTORICAL_DATE_CONTEXT = re.compile(
    r"(?:status\s+(?:date|report)|kickoff|kick.off|meeting\s+notes?|reporting\s+period|"
    r"planned\s+demo|requested\s+(?:access|on)|followed\s+up\s+on|"
    r"(?:contacted|emailed)\s+\w+\s+on)\b.*?"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)",
    re.IGNORECASE,
)

# Months for date normalization
_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _normalize_date(date_str: str) -> str | None:
    """Convert '22 October' or '22 October 2026' to 'YYYY-MM-DD' or None."""
    if not date_str:
        return None
    date_str = date_str.strip()
    m = re.match(r"(\d{1,2})\s+(\w+)\s*(\d{4})?", date_str, re.IGNORECASE)
    if m:
        day = m.group(1).zfill(2)
        month = _MONTHS.get(m.group(2).lower())
        year = m.group(3) or "2026"
        if month:
            return f"{year}-{month}-{day}"
    return None


def _split_into_lines(text: str) -> list[tuple[str, int]]:
    """Split text into non-empty lines with their line numbers."""
    lines = []
    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if stripped and len(stripped) > 5:
            lines.append((stripped, i))
    return lines


def _extract_owner(line: str) -> str | None:
    """Extract an owner name or role from a line.

    Recognises single names ("Rohan"), "Whole team", and "Name and Name" pairs.
    Filters common false positives.
    """
    # Try the extended pattern first
    m = _OWNER_PATTERN_EXT.search(line)
    if m:
        name = next((g for g in m.groups() if g), None)
        if name:
            return name.strip()
    # Fall back to original simple pattern for plain "Rohan:" / "Anika —"
    m2 = re.search(r"(?:^|—\s*|by\s+)([A-Z][a-z]+)\s*[—:]", line)
    if m2:
        name = m2.group(1)
        if name.lower() not in ("the", "this", "that", "team", "status", "due",
                                "risk", "note", "update", "action"):
            return name
    return None


def _extract_due_date(line: str) -> str | None:
    """Try to extract a due date from a line."""
    m = _DATE_PATTERN.search(line)
    if m:
        return _normalize_date(m.group(1))
    return None


def _rule_based_fallback(project_id: str, context: str = "") -> dict[str, Any]:
    """Deterministic fallback: scan document text for blocker/action patterns.

    Reads raw documents from the uploads folder and also uses any
    ChromaDB context provided. Returns structured output matching
    the agent's schema. All items are marked with evidence and source.
    """
    result = {**EMPTY_OUTPUT, "extraction_source": "fallback"}

    # Collect text from raw documents on disk
    # Only read from the project's own uploads folder — never another project's.
    texts: list[tuple[str, str]] = []  # (source_label, text)

    uploads_dir = Path("uploads") / project_id
    try:
        from ingestion import ingest_file
        if uploads_dir.exists():
            # Deduplicate: if both .pdf and .docx exist for the same basename,
            # prefer .docx (richer text extraction) and skip the .pdf.
            seen_stems: set[str] = set()
            candidates: list[Path] = sorted(uploads_dir.iterdir())
            # Sort so .docx comes after .pdf alphabetically; we'll reverse-priority below
            docx_stems = {
                f.stem.lower() for f in candidates
                if f.suffix.lower() in (".docx", ".doc")
            }
            for f in candidates:
                if f.suffix.lower() not in (".pdf", ".docx", ".doc", ".csv", ".txt", ".md"):
                    continue
                stem = f.stem.lower()
                # Skip .pdf if a .docx/doc version of the same file exists
                if f.suffix.lower() == ".pdf" and stem in docx_stems:
                    logger.debug(
                        "Skipping '%s' — .docx version exists for project '%s'",
                        f.name, project_id,
                    )
                    continue
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                try:
                    doc = ingest_file(str(f), project_id)
                    texts.append((f.name, doc.text))
                    logger.debug("Fallback loaded '%s' for project '%s'", f.name, project_id)
                except Exception as exc:
                    logger.warning("Fallback: could not read '%s': %s", f.name, exc)
    except Exception as exc:
        logger.warning("Fallback: could not read uploads for '%s': %s", project_id, exc)

    # Also parse ChromaDB context passed in (only chunks from this project)
    if context:
        # Parse source labels from context format: [Source: filename | Chunk #N]
        for block in context.split("\n\n---\n\n"):
            src_match = re.search(r"\[Source:\s*(.+?)\s*\|", block)
            src_name = src_match.group(1) if src_match else "unknown"
            # Remove the source header line
            body = re.sub(r"^\[Source:.*?\]\n", "", block, flags=re.DOTALL)
            texts.append((src_name, body))

    if not texts:
        result["missing_information"] = [
            "No documents available for rule-based fallback extraction."
        ]
        return result

    blockers: list[dict] = []
    action_items: list[dict] = []
    pending_decisions: list[dict] = []
    unresolved_issues: list[dict] = []

    b_id = 1
    a_id = 1
    d_id = 1
    i_id = 1

    # Use normalised first-100-chars as dedup key across ALL source texts
    seen_blocker_descs: set[str] = set()
    seen_action_descs: set[str] = set()
    seen_issue_descs: set[str] = set()

    for source_name, text in texts:
        lines = _split_into_lines(text)

        for line_text, line_num in lines:
            # --- Pre-filter: skip headings, metadata, risk-register,
            #     scope/out-of-scope statements, and general project facts ---
            if _HEADING_PATTERN.search(line_text) or _META_LINE_PATTERN.search(line_text):
                continue
            if _SCOPE_LINE.search(line_text) or _GENERAL_FACT_LINE.search(line_text):
                continue
            if _RISK_REGISTER_LINE.search(line_text):
                # Risk-register entries go to unresolved_issues, not blockers
                desc = _clean_line(line_text)
                desc_key = desc.lower()[:100]
                if len(desc) >= 15 and desc_key not in seen_issue_descs:
                    seen_issue_descs.add(desc_key)
                    unresolved_issues.append({
                        "issue_id": f"I{i_id:03d}",
                        "description": desc,
                        "impact": "",
                        "source": f"{source_name}#L{line_num}",
                    })
                    i_id += 1
                continue

            # --- Extract explicit status from "Status: X" field in the line ---
            doc_status_match = _STATUS_FIELD.search(line_text)
            doc_status_raw   = doc_status_match.group(1).strip() if doc_status_match else None

            # Normalise doc_status to our schema values
            _STATUS_MAP = {
                "blocked":     "Pending",
                "in progress": "In Progress",
                "not started": "Unknown",
                "completed":   "Completed",
                "complete":    "Completed",
                "done":        "Completed",
                "pending":     "Pending",
                "overdue":     "Overdue",
            }
            doc_status = _STATUS_MAP.get(
                doc_status_raw.lower() if doc_status_raw else "", None
            )

            # --- Check for blockers ---
            if _BLOCKER_KEYWORDS.search(line_text):
                desc = _clean_line(line_text)
                # Require a meaningful description (at least 25 chars)
                if len(desc) >= 25:
                    desc_key = desc.lower()[:100]
                    if desc_key not in seen_blocker_descs:
                        seen_blocker_descs.add(desc_key)
                        owner = _extract_owner(line_text)

                        # Blocker status: default Open; resolved if evidence says so
                        if re.search(r"\b(?:completed|resolved|done|fixed|closed)\b", line_text, re.IGNORECASE):
                            b_status = "Resolved"
                        else:
                            b_status = "Open"

                        blockers.append({
                            "blocker_id":             f"B{b_id:03d}",
                            "description":            desc,
                            "status":                 b_status,
                            "impact":                 "",
                            "owner":                  owner,           # null when absent
                            "source":                 f"{source_name}#L{line_num}",
                            "evidence":               desc[:200],
                            "recommended_resolution": "",
                            "inferred":               False,
                        })
                        b_id += 1

            # --- Check for action items ---
            # Require explicit due-date signal (excluding historical dates) OR
            # explicit assignment + action keyword.  This prevents scope sentences
            # from becoming action items.
            has_due_date = False
            due_date_val = None
            if not _HISTORICAL_DATE_CONTEXT.search(line_text):
                raw_due = _extract_due_date(line_text)
                if raw_due:
                    has_due_date = True
                    due_date_val = raw_due

            has_action_kw = bool(_ACTION_KEYWORDS.search(line_text))
            owner = _extract_owner(line_text)

            if has_due_date or (has_action_kw and owner):
                desc = _clean_line(line_text)
                if len(desc) >= 20:
                    desc_key = desc.lower()[:100]
                    if desc_key not in seen_action_descs:
                        seen_action_descs.add(desc_key)

                        # Priority
                        priority = "Unspecified"
                        if re.search(r"\b(?:critical|urgent|high\s+priority)\b", line_text, re.IGNORECASE):
                            priority = "High"
                        elif re.search(r"\bmedium\b", line_text, re.IGNORECASE):
                            priority = "Medium"
                        elif re.search(r"\b(?:low|minor)\b", line_text, re.IGNORECASE):
                            priority = "Low"

                        # Status: use explicit field first, then infer, default Unknown
                        if doc_status:
                            a_status = doc_status
                        elif re.search(r"\b(?:in\s+progress|started|working\s+on)\b", line_text, re.IGNORECASE):
                            a_status = "In Progress"
                        elif re.search(r"\b(?:completed|done|finished)\b", line_text, re.IGNORECASE):
                            a_status = "Completed"
                        elif re.search(r"\boverdue\b", line_text, re.IGNORECASE):
                            a_status = "Overdue"
                        elif re.search(r"\bnot\s+started\b", line_text, re.IGNORECASE):
                            a_status = "Unknown"
                        elif re.search(r"\bblocked\b", line_text, re.IGNORECASE):
                            a_status = "Pending"
                        else:
                            a_status = "Pending"

                        action_items.append({
                            "action_id":      f"A{a_id:03d}",
                            "description":    desc,
                            "owner":          owner,       # null when absent
                            "due_date":       due_date_val,
                            "priority":       priority,
                            "status":         a_status,
                            "related_blocker": None,
                            "source":         f"{source_name}#L{line_num}",
                            "inferred":       False,
                        })
                        a_id += 1

            # --- Check for pending decisions ---
            if _DECISION_KEYWORDS.search(line_text):
                desc = _clean_line(line_text)
                if len(desc) >= 20:
                    pending_decisions.append({
                        "decision_id": f"D{d_id:03d}",
                        "description": desc,
                        "owner":       _extract_owner(line_text),
                        "due_date":    None if _HISTORICAL_DATE_CONTEXT.search(line_text)
                                       else _extract_due_date(line_text),
                        "source":      f"{source_name}#L{line_num}",
                    })
                    d_id += 1

            # --- Check for unresolved issues ---
            if re.search(
                r"\b(?:unresolved|not\s+working|not\s+yet\s+(?:implemented|integrated))\b",
                line_text, re.IGNORECASE,
            ):
                desc = _clean_line(line_text)
                desc_key = desc.lower()[:100]
                if len(desc) >= 15 and desc_key not in seen_issue_descs:
                    seen_issue_descs.add(desc_key)
                    unresolved_issues.append({
                        "issue_id":   f"I{i_id:03d}",
                        "description": desc,
                        "impact":     "",
                        "source":     f"{source_name}#L{line_num}",
                    })
                    i_id += 1

    result["blockers"]           = blockers
    result["action_items"]       = action_items
    result["pending_decisions"]  = pending_decisions
    result["unresolved_issues"]  = unresolved_issues
    result["error"]              = None

    if not blockers and not action_items:
        result["missing_information"] = [
            "No blockers or action items found in the documents using rule-based extraction."
        ]

    # --- Validate output schema before returning ---
    result = _validate_fallback_output(result)
    return result


def _validate_fallback_output(result: dict) -> dict:
    """Validate and sanitise fallback output against the expected schema.

    - Ensures every required field is present in each record.
    - Replaces missing string fields with sensible defaults (empty string or None).
    - Removes records whose description is empty after cleaning.
    - Does NOT invent owners, dates, or evidence — uses None / "" / "Unknown" only.
    """
    _VALID_BLOCKER_STATUSES  = {"Open", "Resolved", "Unknown"}
    _VALID_ACTION_STATUSES   = {"Pending", "In Progress", "Completed", "Overdue", "Unknown"}
    _VALID_PRIORITIES        = {"High", "Medium", "Low", "Unspecified"}

    # --- Blockers ---
    clean_blockers = []
    for b in result.get("blockers", []):
        if not isinstance(b, dict):
            continue
        desc = str(b.get("description", "")).strip()
        if not desc:
            continue
        clean_blockers.append({
            "blocker_id":             str(b.get("blocker_id", f"B{len(clean_blockers)+1:03d}")),
            "description":            desc,
            "status":                 b.get("status") if b.get("status") in _VALID_BLOCKER_STATUSES else "Unknown",
            "impact":                 str(b.get("impact", "")),
            "owner":                  b.get("owner") or None,
            "source":                 str(b.get("source", "unknown")),
            "evidence":               str(b.get("evidence", desc[:200])),
            "recommended_resolution": str(b.get("recommended_resolution", "")),
            "inferred":               bool(b.get("inferred", False)),
        })

    # --- Action items ---
    clean_actions = []
    for a in result.get("action_items", []):
        if not isinstance(a, dict):
            continue
        desc = str(a.get("description", "")).strip()
        if not desc:
            continue
        clean_actions.append({
            "action_id":      str(a.get("action_id", f"A{len(clean_actions)+1:03d}")),
            "description":    desc,
            "owner":          a.get("owner") or None,
            "due_date":       a.get("due_date") or None,
            "priority":       a.get("priority") if a.get("priority") in _VALID_PRIORITIES else "Unspecified",
            "status":         a.get("status") if a.get("status") in _VALID_ACTION_STATUSES else "Unknown",
            "related_blocker": a.get("related_blocker") or None,
            "source":         str(a.get("source", "unknown")),
            "inferred":       bool(a.get("inferred", False)),
        })

    # --- Pending decisions ---
    clean_decisions = []
    for d in result.get("pending_decisions", []):
        if not isinstance(d, dict):
            continue
        desc = str(d.get("description", "")).strip()
        if not desc:
            continue
        clean_decisions.append({
            "decision_id": str(d.get("decision_id", f"D{len(clean_decisions)+1:03d}")),
            "description": desc,
            "owner":       d.get("owner") or None,
            "due_date":    d.get("due_date") or None,
            "source":      str(d.get("source", "unknown")),
        })

    # --- Unresolved issues ---
    clean_issues = []
    for i in result.get("unresolved_issues", []):
        if not isinstance(i, dict):
            continue
        desc = str(i.get("description", "")).strip()
        if not desc:
            continue
        clean_issues.append({
            "issue_id":    str(i.get("issue_id", f"I{len(clean_issues)+1:03d}")),
            "description": desc,
            "impact":      str(i.get("impact", "")),
            "source":      str(i.get("source", "unknown")),
        })

    result["blockers"]          = clean_blockers
    result["action_items"]      = clean_actions
    result["pending_decisions"] = clean_decisions
    result["unresolved_issues"] = clean_issues
    return result


def _clean_line(line: str) -> str:
    """Clean up a line for use as a description: remove bullet chars, extra whitespace."""
    line = re.sub(r"^[•·○●▪‣›\-\*]+\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    # Remove (cid:NN) artifacts from PDF extraction
    line = re.sub(r"\(cid:\d+\)", "", line).strip()
    return line[:300]  # cap length


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(project_id: str, context_k: int = 4) -> dict[str, Any]:
    """Identify blockers and action items for *project_id*.

    Strategy:
      1. Try LLM extraction.
      2. If LLM fails, run rule-based fallback.
      3. Report extraction source.

    Never raises — errors captured in result["error"].
    """
    result = {**EMPTY_OUTPUT, "error": None}

    try:
        _AGENT     = "blocker"
        _MAX_TOK   = 1200   # non-reasoning model (qwen3.8-27b): all tokens go to output, no hidden thinking
        _CTX_CHARS = tpm_context_cap(SYSTEM_PROMPT, _MAX_TOK)
        context = build_context(project_id, RETRIEVAL_QUERIES, k=context_k, max_chars=_CTX_CHARS)
        if not context:
            # No ChromaDB context — but raw docs might still exist
            logger.info("No ChromaDB context for '%s', trying fallback with raw docs", project_id)
            fallback = _rule_based_fallback(project_id, context="")
            if fallback["blockers"] or fallback["action_items"]:
                return fallback
            result["missing_information"] = [
                "No documents are indexed for this project."
            ]
            result["error"] = "No indexed content found."
            return result

        user_prompt = (
            f"Project ID: {project_id}\n\n"
            "Below are excerpts from project meeting notes and progress updates. "
            "Extract all blockers, action items, pending decisions, "
            "and unresolved issues.\n\n"
            f"{context}"
        )

        llm_result: dict | None = None
        llm_error: str | None = None

        cached = get_cached_llm_result(_AGENT, project_id, SYSTEM_PROMPT,
                                       context, _MAX_TOK, DEFAULT_MODEL)
        if cached is not None:
            logger.info("blocker_agent: using cached LLM result for '%s'", project_id)
            result = _merge(result, cached)
            result["extraction_source"] = "llm"
            result["error"] = None
            return result

        try:
            llm_result = call_llm_json(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                fallback=None,
                max_tokens=_MAX_TOK,
                timeout=30,
                json_mode=False,
            )
        except LLMError as exc:
            llm_error = str(exc)
            logger.warning("Blocker agent LLM failed: %s — falling back to rule-based", exc)

        # --- Process LLM result if we got one ---
        if llm_result and isinstance(llm_result, dict):
            set_cached_llm_result(_AGENT, project_id, SYSTEM_PROMPT,
                                   context, _MAX_TOK, DEFAULT_MODEL, llm_result)
            result = _merge(result, llm_result)
            result["extraction_source"] = "llm"
            result["error"] = None
            return result

        # --- LLM failed or returned empty — run rule-based fallback ---
        logger.info("Running rule-based fallback for blocker agent (project '%s')", project_id)
        fallback = _rule_based_fallback(project_id, context=context)

        if fallback["blockers"] or fallback["action_items"]:
            # Fallback found results
            fallback["extraction_source"] = "fallback"
            if llm_error:
                fallback["error"] = None  # don't report error if fallback worked
                # Note the LLM issue in missing_information without blocking
                fallback["missing_information"] = (
                    fallback.get("missing_information", [])
                    + [f"LLM extraction failed ({llm_error}); rule-based fallback used."]
                )
            return fallback
        else:
            # Fallback also found nothing
            result["error"] = llm_error or "No blockers found in documents."
            result["extraction_source"] = "none"
            result["missing_information"] = [
                "No blockers or action items found in the uploaded documents."
            ]
            if llm_error:
                result["missing_information"].append(
                    f"LLM extraction also failed: {llm_error}"
                )
            return result

    except Exception as exc:
        logger.exception("blocker_agent unexpected error: %s", exc)
        result["error"] = str(exc)
        result["extraction_source"] = "none"
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge(base: dict, raw: dict) -> dict:
    """Safely merge LLM output into the base schema.

    Handles the compact schema (uses "id" instead of "blocker_id" etc.)
    and fills all fields the template expects with safe defaults.
    """
    out = {**base}

    # --- Blockers ---
    val = raw.get("blockers")
    if isinstance(val, list):
        norm = []
        for b in val:
            if not isinstance(b, dict):
                continue
            norm.append({
                "blocker_id":             b.get("blocker_id", b.get("id", "")),
                "description":            b.get("description", ""),
                "status":                 b.get("status", "Unknown"),
                "impact":                 b.get("impact", ""),
                "owner":                  b.get("owner"),
                "source":                 b.get("source", ""),
                "evidence":               b.get("evidence", ""),
                "recommended_resolution": b.get("recommended_resolution", ""),
                "inferred":               b.get("inferred", False),
            })
        out["blockers"] = norm

    # --- Action Items ---
    val = raw.get("action_items")
    if isinstance(val, list):
        norm = []
        for a in val:
            if not isinstance(a, dict):
                continue
            norm.append({
                "action_id":       a.get("action_id", a.get("id", "")),
                "description":     a.get("description", ""),
                "owner":           a.get("owner"),
                "due_date":        a.get("due_date"),
                "priority":        a.get("priority", "Unspecified"),
                "status":          a.get("status", "Unknown"),
                "related_blocker": a.get("related_blocker"),
                "source":          a.get("source", ""),
                "inferred":        a.get("inferred", False),
            })
        out["action_items"] = norm

    # --- Pending Decisions ---
    val = raw.get("pending_decisions")
    if isinstance(val, list):
        norm = []
        for d in val:
            if not isinstance(d, dict):
                continue
            norm.append({
                "decision_id": d.get("decision_id", d.get("id", "")),
                "description": d.get("description", ""),
                "owner":       d.get("owner"),
                "due_date":    d.get("due_date"),
                "source":      d.get("source", ""),
            })
        out["pending_decisions"] = norm

    # --- Unresolved Issues ---
    val = raw.get("unresolved_issues")
    if isinstance(val, list):
        norm = []
        for i in val:
            if not isinstance(i, dict):
                continue
            norm.append({
                "issue_id":    i.get("issue_id", i.get("id", "")),
                "description": i.get("description", ""),
                "impact":      i.get("impact", ""),
                "source":      i.get("source", ""),
            })
        out["unresolved_issues"] = norm

    # --- Missing Information ---
    val = raw.get("missing_information")
    if isinstance(val, list):
        out["missing_information"] = val

    out["error"] = None
    return out
