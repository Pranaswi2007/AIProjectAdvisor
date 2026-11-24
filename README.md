# AI Project Intelligence & Risk Advisor

An AI-powered Flask web application that analyses project documents and produces structured intelligence reports: scope extraction, blocker identification, risk assessment, delivery forecasting, and auto-generated documentation.

Built as part of the Infosys Springboard Internship programme (Milestones 1–4).

---

## Features

| Feature | Description |
|---|---|
| **Document ingestion** | Upload PDF, DOCX, CSV, TXT, and Markdown files; text is extracted, chunked, embedded, and stored in ChromaDB |
| **RAG search** | Semantic search across any project's knowledge base |
| **Multi-agent analysis** | Single optimised LLM call extracts scope, blockers, and risks simultaneously |
| **Health score dashboard** | Per-project 0–100 score with four weighted dimensions and colour-coded grade |
| **Incremental upload** | Add meeting notes or sprint updates to an existing project; stale-KB detection flags when re-analysis is recommended |
| **Conversational Q&A** | Ask free-text questions about any project; answers are grounded in indexed documents |
| **Generated documents** | Auto-generates user stories, risk register, and action item list from extracted data |
| **Report persistence** | Analysis reports saved to disk; dashboard shows last-run insights without re-running analysis |

---

## Quick Start

### 1. Prerequisites

- Python 3.10 or later
- A [Groq API key](https://console.groq.com) (free tier is sufficient)

### 2. Clone and set up

```bash
git clone <repo-url>
cd "Infosys project"

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Run the application

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Project Structure

```
Infosys project/
├── app.py                  # Flask routes and application entry point
├── ingestion.py            # Document loaders (PDF, DOCX, CSV, TXT/MD)
├── rag.py                  # ChromaDB vector store: index, search, stats
├── models.py               # Pydantic/dataclass models (Document, SearchResult)
├── report_store.py         # Persist and load analysis reports to/from disk
├── requirements.txt        # Pinned dependencies
│
├── agents/
│   ├── combined_agent.py   # Single LLM call → scope + blockers + risks
│   ├── scope_agent.py      # Scope & deliverable extraction (standalone)
│   ├── blocker_agent.py    # Blocker & action item extraction (standalone)
│   ├── risk_agent.py       # Risk detection & delivery forecast (standalone)
│   ├── doc_agent.py        # Deterministic doc generation (no LLM)
│   ├── health_score.py     # 0–100 health score computation
│   ├── orchestrator.py     # Runs all agents, assembles final report
│   ├── qa_agent.py         # Conversational Q&A (RAG + LLM)
│   ├── llm_client.py       # Groq API wrapper with retry, caching, TPM budget
│   └── __init__.py
│
├── templates/
│   ├── base.html           # Layout shell, navigation, shared CSS
│   ├── dashboard.html      # Project insights dashboard
│   ├── upload.html         # Document upload with drag-and-drop
│   ├── upload_result.html  # Upload results with incremental-update banner
│   ├── knowledge_base.html # Browse and manage indexed documents
│   ├── search.html         # Semantic search interface
│   ├── analysis.html       # Full analysis report (tabbed)
│   └── qa.html             # Conversational Q&A chat
│
├── tests/
│   ├── conftest.py             # Pre-warm embedding model for all tests
│   ├── test_agents.py          # Unit tests: all agents, LLM client, orchestrator
│   ├── test_integration.py     # Integration: ingest → index → orchestrator (mocked LLM)
│   ├── test_milestone3.py      # Health score, doc agent, Q&A agent, partial status
│   ├── test_milestone4.py      # PDF/DOCX ingestion, mixed pipeline, report persistence
│   ├── test_regression.py      # Forecast fallback, context starvation, /ask route, health score
│   ├── test_extraction_accuracy.py  # _merge() normalisation regressions
│   ├── test_llm_reliability.py      # Retry, cache, rate-limit handling
│   ├── test_blocker_fix.py          # Blocker agent edge cases
│   └── sample_data/            # Fixture files for integration tests
│
├── uploads/                # Uploaded files, organised by project ID
├── chroma_db/              # ChromaDB vector store (auto-created)
└── reports/                # Persisted analysis reports as JSON (auto-created)
```

---

## Usage Guide

### Uploading documents

1. Navigate to **Upload** in the top navigation.
2. Enter a project name (letters, numbers, hyphens, underscores).
3. Select a **Document Type** (optional but recommended — helps the AI interpret context).
4. Drag and drop files or use the file picker. Multiple files are supported.
5. Click **Process & Index Files**.

To add more documents to an existing project later, use the same project name. The knowledge base updates automatically, and the dashboard will flag the analysis as stale.

Supported formats: **PDF**, **DOCX**, **CSV**, **TXT**, **Markdown**.

### Running analysis

1. Navigate to **Analyse**.
2. Select a project.
3. Click **Run Analysis**.

The analysis runs a single LLM call (5–10 seconds on the free tier) using four independent retrieval passes — one each for scope, out-of-scope, blockers, and risks — so no section starves another. It produces:

- **Scope & Deliverables** — project goals, in-scope/out-of-scope items, deliverable table with owners and deadlines, milestones
- **Blockers & Actions** — open blockers with impact and evidence, action items with owners and due dates, pending decisions, unresolved issues
- **Risks & Forecast** — risk register grouped by severity (Critical/High/Medium/Low), delivery forecast with confidence rating and concrete milestone dates (merged from rule-based analysis)
- **Generated Documents** — user stories, risk register, structured action item list (all deterministic, no LLM)
- **Health Score** — 0–100 score with four weighted dimensions: scope clarity (25%), timeline risk (30%), blocker severity (25%), risk exposure (20%)

### Searching the knowledge base

Use the **Search** page to run semantic queries against any project's indexed documents. Results show the source file, chunk index, relevance score, and extracted text.

### Conversational Q&A

Use the **Q&A** page to ask free-text questions about a project. Answers are grounded in the indexed documents and include a confidence rating (High / Medium / Low) and source citations.

---

## Running Tests

```bash
# Full test suite (278 tests: 194 original + 38 Milestone 4 + 23 + 23 regression)
venv\Scripts\python.exe -m pytest tests/ -v

# Milestone 4 tests only (PDF/DOCX ingestion, persistence, Flask routes)
venv\Scripts\python.exe run_m4_tests.py

# Regression tests (forecast fallback, context starvation, health score)
venv\Scripts\python.exe run_regression_tests.py

# Legacy full suite runner
venv\Scripts\python.exe run_tests.py
```

The test suite uses mocked LLM calls throughout — no Groq API key is needed to run tests.

**Test coverage by file:**

| Test file | Tests | Coverage |
|---|---|---|
| `test_agents.py` | 36 | All agents with mocked LLM, LLM client retry/cache, orchestrator |
| `test_integration.py` | 19 | Real ChromaDB + embeddings, mocked LLM, Flask routes |
| `test_milestone3.py` | 38 | Health score, doc agent, Q&A agent, partial status, extraction gaps |
| `test_milestone4.py` | 38 | PDF/DOCX ingestion, mixed pipeline, report persistence, stale detection, Flask upload routes |
| `test_extraction_accuracy.py` | 42 | `_merge()` normalisation regressions for all three agents |
| `test_llm_reliability.py` | 30 | Retry, cache, rate-limit handling, max_tokens enforcement |
| `test_blocker_fix.py` | 30 | Blocker agent edge cases, fallback, field normalisation |
| `test_regression.py` | 23 | /ask route, forecast fallback, context starvation fix, sparse heuristic, OOS penalty |

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq Cloud API key |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | LLM model ID |
| `GROQ_TIMEOUT` | `30` | Per-request timeout in seconds |
| `GROQ_TPM_LIMIT` | `8000` | Tokens-per-minute budget (free tier = 8 000) |
| `FLASK_SECRET` | `dev-secret-change-in-prod` | Flask session secret key |

Set these in a `.env` file or as system environment variables.

---

## Dependencies

See `requirements.txt` for pinned versions. Key packages:

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX extraction |
| `pandas` + `openpyxl` | CSV / XLSX ingestion |
| `langchain-text-splitters` | Document chunking |
| `sentence-transformers` | Local embedding model (all-MiniLM-L6-v2) |
| `chromadb` | Vector store |
| `groq` | Groq LLM API client |
| `python-dotenv` | `.env` file support |
| `reportlab` | PDF generation in tests |
| `pytest` | Test runner |

---

## Milestones

| Milestone | Deliverables | Status |
|---|---|---|
| **1** | Document ingestion pipeline (PDF, DOCX, CSV, TXT), ChromaDB RAG, semantic search | ✅ Complete |
| **2** | Three-agent analysis pipeline (scope, blockers, risks), health scoring, orchestrator | ✅ Complete |
| **3** | Conversational Q&A assistant, auto-generated documents (user stories, risk register, action items) | ✅ Complete |
| **4** | Insights dashboard with health scores, incremental upload with stale detection, report persistence, end-to-end tests (278 total), technical documentation | ✅ Complete |

---

## Architecture Highlights

### Single Combined LLM Call
On the Groq free tier (8K TPM), three separate LLM calls waste ~1,400 tokens each on system-prompt overhead. The combined agent makes **one call** with **four independent retrieval passes** (scope, out-of-scope, blockers, risks), each getting its own 1,500-char context budget so no section starves another. Total: ~5,400 tokens, well within 8K TPM.

### Rule-Based Delivery Forecast
The orchestrator **always** runs a deterministic rule-based forecast from extracted milestone/deliverable dates and merges it with the LLM's qualitative assessment:
- If the LLM returns "Insufficient data" but milestones have dates → rule-based forecast replaces it
- If the LLM has a forecast → rule-based dates (completion date, delayed milestones, assumptions) are merged in as "hybrid"
- Never invents dates — only uses what was extracted from documents

### Health Score Anti-Gaming
The scope clarity score penalises:
- Missing categories (goals -25, scope -20, deliverables -25, milestones -20)
- Missing out-of-scope when other scope exists (-10)
- Sparse extraction: all categories with ≤1 item each (-15, signals context starvation)
- LLM-flagged gaps and missing sub-fields on deliverables

The timeline risk score cross-checks confirmed Critical/High risks against the forecast label — a project with 2+ confirmed High risks scores ≤30 on timeline even if the LLM says "On Track".

### Qwen3 Instruct Mode
`reasoning_effort="none"` is injected automatically for Qwen models in `llm_client.py` to disable thinking mode. Without this, Qwen3 burns tokens on internal reasoning before writing any output — causing empty responses on the free tier.
