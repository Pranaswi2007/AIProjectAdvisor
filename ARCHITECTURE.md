# Architecture — AI Project Intelligence & Risk Advisor

## Overview

The system is a Flask web application with four logical layers:

```
Browser
  │
  ▼
Flask (app.py)          ← HTTP routes, file I/O, flash messages
  │
  ▼
Agent Pipeline          ← LLM calls, deterministic processing
  │  combined_agent → orchestrator → doc_agent + health_score
  │
  ▼
RAG / Vector Store      ← ChromaDB + sentence-transformers
  │
  ▼
Document Ingestion      ← pdfplumber, python-docx, pandas, plain text
```

---

## Component Map

### `app.py` — Flask Application

Ten routes covering the full user journey:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard — project list with last-analysis summaries |
| `/upload` | GET/POST | Upload documents; supports incremental updates |
| `/knowledge-base` | GET | Browse indexed files per project |
| `/knowledge-base/delete` | POST | Remove one file from ChromaDB + disk |
| `/knowledge-base/delete-project` | POST | Remove entire project |
| `/search` | GET/POST | Semantic RAG search |
| `/analyse` | GET/POST | Run analysis pipeline; save report |
| `/qa` | GET | Q&A chat page |
| `/ask` | POST | Answer a question via RAG + LLM (JSON) |

**Key behaviours:**
- `get_all_projects()` reads every folder under `uploads/`, merges ChromaDB stats with the last saved report summary, and computes a `stale` flag when any file on disk is newer than `saved_at` on the report.
- Every successful analysis calls `report_store.save_report()` immediately after `run_analysis()`.
- Both LLM result cache and context cache are cleared before every analysis run to prevent stale data from a previous project bleeding through.

---

### `ingestion.py` — Document Loaders

One loader function per file type, all returning plain text:

| Extension | Loader | Library |
|---|---|---|
| `.pdf` | `load_pdf` | `pdfplumber` |
| `.docx` | `load_docx` | `python-docx` (paragraphs + tables) |
| `.csv` | `load_csv` | `pandas` — each row becomes `col: value, col: value` |
| `.txt` / `.md` | `load_txt` | built-in `Path.read_text` |

After loading, `clean_text()` normalises whitespace, removes null bytes, and collapses multiple blank lines. Each document gets a stable MD5-based ID from `project_id + filename` so re-uploading the same file replaces rather than duplicates it.

---

### `rag.py` — Vector Store

Built on **ChromaDB** with **sentence-transformers** (all-MiniLM-L6-v2 by default).

**Indexing path:**
1. `ingest_file()` → returns a `Document` with `.text`
2. `index_document(doc)` → LangChain `RecursiveCharacterTextSplitter` splits text into ~500-char chunks with 50-char overlap
3. Each chunk is embedded by the local sentence-transformer model
4. Stored in ChromaDB collection `project-<sanitised_id>`; metadata includes `source` (filename) and `chunk_index`

**Retrieval path:**
1. `search(project_id, query, k)` embeds the query with the same model
2. ChromaDB cosine similarity search returns top-k chunks
3. Returns `SearchResult` objects with `.text`, `.source`, `.chunk_index`, `.score`

**Collection naming:** `project-` + first 57 chars of a sanitised project ID (keeps total under ChromaDB's 63-char limit).

---

### `agents/llm_client.py` — LLM Client

Wraps the Groq API with:

- **`call_llm(system, user, ...)`** — raw text response with retry on 429 (rate limit), immediate failure on 400/401/404/422, 3 attempts by default
- **`call_llm_json(system, user, ...)`** — calls `call_llm` then parses JSON through a four-step pipeline: (1) direct `json.loads`, (2) strip markdown fences, (3) extract first `{...}` block, (4) `_repair_json` closes unclosed brackets
- **`reasoning_effort="none"`** injected automatically for Qwen3 models to disable thinking mode and prevent token budget from being consumed before any output is written
- **`tpm_context_cap()`** — computes safe context character limit given the 8 000 TPM free-tier ceiling: `available = TPM_LIMIT - system_tokens - framing - max_output_tokens`
- **LLM result cache** — keyed on `(agent, project_id, model, max_tokens, MD5(system+context))`, TTL 1 hour; cleared explicitly before every fresh analysis
- **Context cache** — caches `build_context()` output keyed on `(project_id, queries, k, max_chars)`, TTL 5 minutes

---

### `agents/combined_agent.py` — Single LLM Call

The core design decision for the free tier (8 000 TPM):

**Problem:** Three separate LLM calls each burn ~400 tokens on system prompt overhead, leaving only ~1 000 tokens for actual document context per agent.

**Solution:** One combined call uses ~300 tokens for the system prompt and ~3 500 tokens for output, leaving ~4 200 tokens for document context — over 4× more signal.

**Flow:**
1. Retrieve context via `build_context()` with 4 retrieval queries (scope, blockers, risks, general)
2. Single `call_llm_json()` returning `{"scope": {...}, "blockers": {...}, "risks": {...}}`
3. `_split()` delegates to `_fill_scope()`, `_fill_blockers()`, `_fill_risks()` which normalise field names to match what the templates expect

**Field normalisation matters:** The LLM uses compact field names (`blocker_id`, `action_id`, etc.) while earlier agents used `id`. The fill functions always read both: `item.get("blocker_id", item.get("id", ""))`.

---

### `agents/orchestrator.py` — Pipeline Coordinator

Three sequential steps, all in a single function call:

```
Step 1  combined_agent.run(project_id)
        → scope_result, blocker_result, risk_result
        (5–10 seconds — the only network call)

Step 2  doc_agent.run(...)    ← deterministic, instant
        → user_stories, risk_register, action_item_list

Step 3  health_score.compute(...)  ← pure arithmetic, instant
        → overall, grade, 4 dimension scores
```

The report dict has a stable schema so templates never crash on missing keys:

```python
{
  "project_id", "generated_at", "status",   # "ok" | "partial" | "error"
  "scope", "blockers", "risks", "docs",
  "health_score", "agent_status",            # per-agent ok/partial/failed
  "errors", "warnings", "duration_seconds"
}
```

---

### `agents/health_score.py` — Health Scoring

Four weighted dimensions, each 0–100:

| Dimension | Weight | Signal |
|---|---|---|
| Scope Clarity | 25% | Presence of goals, scope items, deliverables, milestones; sub-field completeness; LLM-flagged gaps |
| Timeline Risk | 30% | Delivery forecast status (On Track=90, At Risk=60, Delayed=25); **capped down** when confirmed Critical/High risks exist regardless of forecast label |
| Blocker Severity | 25% | Open blockers weighted 1.0, resolved 0.2; linear scale to 0 at 10 effective blockers |
| Risk Exposure | 20% | Severity-weighted sum (Critical=1.0, High=0.75, Medium=0.5, Low=0.25); confirmed risks ×1.5; linear scale to 0 at exposure=8 |

**Anti-gaming:** Timeline Risk is cross-checked against the confirmed risk list. If the LLM labels the forecast "On Track" but there are ≥2 confirmed High/Critical risks in the document, the timeline score is capped at 30 — preventing a fake-high overall score.

Overall = `round(sc×0.25 + tr×0.30 + bs×0.25 + re×0.20)`

Grades: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F < 60.

---

### `agents/doc_agent.py` — Deterministic Document Generation

No LLM call. Generates structured output from already-extracted data:

- **User stories** — one per deliverable (`As a user I want to <name> so that <description>`) plus one per scope item
- **Risk register** — normalises `risk_data.risks[]` into the template schema
- **Action item list** — normalises `blocker_data.action_items[]` and derives items from open blockers

This avoids a fourth LLM call and eliminates truncation errors that previously occurred when the LLM tried to generate lengthy document output within a tight token budget.

---

### `agents/qa_agent.py` — Conversational Q&A

RAG-grounded question answering:

1. `build_context(project_id, queries_from_question, k=8)` retrieves relevant chunks
2. Single `call_llm_json()` with the question + context → `{answer, confidence, sources_cited}`
3. Confidence is `"High"` only when the answer is directly supported by quoted text; `"Low"` for inferences

---

### `report_store.py` — Report Persistence

Saves the full report plus a compact `summary` sub-dict to `reports/<project_id>.json` after every analysis.

The summary contains only the fields the dashboard needs (health scores, counts, forecast status, timestamps) — typically ~800 bytes vs ~50 KB for the full report. `get_all_projects()` calls `list_summaries()` which reads only the summary sub-dict from each file, keeping dashboard load time under 50ms even with many projects.

**Stale detection logic:**
```python
saved_at = datetime.fromisoformat(summary["saved_at"])
for file in project_folder.iterdir():
    if datetime.fromtimestamp(file.stat().st_mtime) > saved_at:
        stale = True
        break
```

---

## Data Flow: Upload → Analyse → Dashboard

```
User uploads file
      │
      ▼
app.py /upload (POST)
  ingest_file()       → Document(text, word_count, project_id, filename)
  index_document()    → ChromaDB chunks (embedded)
  clear_llm_cache()   → invalidate stale LLM results
  clear_ctx_cache()   → invalidate stale context
      │
      ▼ (later)
app.py /analyse (POST)
  combined_agent.run()
    build_context()       → retrieves top-k chunks from ChromaDB
    call_llm_json()       → one Groq API call (~5–10s)
    _fill_scope/blockers/risks()
  doc_agent.run()         → instant
  health_score.compute()  → instant
  save_report()           → reports/<project_id>.json
      │
      ▼ (dashboard page load)
app.py /  (GET)
  get_all_projects()
    list_summaries()      → reads reports/*.json summary sub-dicts
    stale detection       → compares file mtime vs saved_at
  render_template("dashboard.html", projects=...)
```

---

## Token Budget (Free Tier — 8 000 TPM)

```
Combined agent call breakdown:

  System prompt     ~300 tokens
  Framing overhead   ~60 tokens
  Document context  ~875 tokens  (3 500 chars × ¼ token/char)
  LLM output       3 500 tokens
  ─────────────────────────────
  Total            ~4 735 tokens  (well within 8 000 TPM)

Three-agent approach (rejected):
  Per-call overhead: 400 + 60 + 1200 output = 1660 tokens
  Context per call: (8000 - 1660) / 3 ≈ 2113 tokens → ~850 chars (tiny)
  Total API calls: 3 × ~1660 = ~5000 tokens consumed just on overhead
```

---

## Test Strategy

| Layer | Test file | Approach |
|---|---|---|
| Unit — agents | `test_agents.py` | Mock LLM + mock ChromaDB |
| Unit — LLM client | `test_llm_reliability.py` | Mock HTTP responses; test retry/cache logic |
| Unit — extraction | `test_extraction_accuracy.py` | Test `_merge()` normalisation with varied JSON shapes |
| Unit — health score | `test_milestone3.py` | Test scoring edge cases; weighted-math verification |
| Integration — RAG | `test_integration.py` | Real ChromaDB + real embeddings + mock LLM |
| Integration — M4 | `test_milestone4.py` | Real PDF/DOCX generation via reportlab/python-docx; real ingestion; Flask client |
| Regression — blockers | `test_blocker_fix.py` | Specific field-name and fallback regressions |

All LLM calls in tests are mocked with `unittest.mock.patch`. The embedding model runs for real in integration tests (pre-warmed in `conftest.py`).

**Total: 232 tests (194 original + 38 Milestone 4), 5 skipped (embedding model unavailable), 0 failures.**
