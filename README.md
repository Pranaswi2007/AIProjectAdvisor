# AI Project Intelligence & Risk Advisor

An AI-powered Flask web application that analyses project documents and produces structured intelligence reports: scope extraction, blocker identification, risk assessment, delivery forecasting, and auto-generated documentation.

Built as part of the Infosys Springboard Internship programme (Milestones 1–4).

---

## Features

| Feature | Description |
|---|---|
| **Single-page dark-themed UI** | Sidebar navigation, upload, analysis, health dashboard, chatbot, and report sections on one page |
| **Document ingestion** | Upload PDF, DOCX, CSV, TXT, and Markdown files; text is extracted, chunked, embedded, and stored in ChromaDB |
| **RAG search** | Semantic search across any project's knowledge base |
| **Two-call parallel analysis** | Scope and blocker/risk LLM calls run concurrently via ThreadPoolExecutor for faster results |
| **Disk-cached results** | Analysis outputs are persisted to disk — re-analysis is instant even after Flask restart |
| **Health score dashboard** | Per-project 0–100 score with four weighted dimensions and colour-coded grade |
| **Incremental upload** | Add meeting notes or sprint updates to an existing project; the KB updates automatically |
| **Conversational Q&A** | Ask free-text questions about any project; answers are grounded in indexed documents and saved analysis results |
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
GROQ_MODEL=openai/gpt-oss-120b
```

### 4. Run the application

```bash
python app.py
```

Open `http://localhost:5000/app` in your browser.

---

## Project Structure

```
Infosys project/
├── app.py                  # Flask routes and application entry point
├── ingestion.py            # Document loaders (PDF, DOCX, CSV, TXT/MD)
├── rag.py                  # ChromaDB vector store: index, search, get_all_text
├── models.py               # Dataclass models (Document, Chunk, SearchResult)
├── report_store.py         # Persist and load analysis reports to/from disk
├── requirements.txt         # Pinned dependencies
├── run_tests.py            # Test runner
├── run_full_tests.py       # Full test suite runner
│
├── agents/
│   ├── combined_agent.py   # Two parallel LLM calls → scope + blockers/risks
│   ├── scope_agent.py      # Scope extraction (standalone, used by tests)
│   ├── blocker_agent.py    # Blocker extraction (standalone, used by tests)
│   ├── risk_agent.py       # Risk detection & forecast (standalone, used by tests)
│   ├── doc_agent.py        # Deterministic doc generation (no LLM)
│   ├── health_score.py     # 0–100 health score computation
│   ├── orchestrator.py     # Runs all agents, assembles final report
│   ├── qa_agent.py         # Conversational Q&A (RAG + LLM + analysis context)
│   ├── llm_client.py       # Groq API wrapper with retry, disk cache, TPM budget
│   └── __init__.py
│
├── templates/
│   ├── app_v2.html         # Single-page app (dark theme, sidebar, all sections)
│   ├── base.html           # Layout shell for legacy pages
│   ├── analysis.html       # Standalone analysis report (legacy)
│   ├── upload.html         # Standalone upload page (legacy)
│   ├── upload_result.html  # Upload results page (legacy)
│   ├── knowledge_base.html # Browse indexed documents (legacy)
│   ├── search.html         # Semantic search interface (legacy)
│   └── qa.html             # Conversational Q&A chat (legacy)
│
├── tests/
│   ├── conftest.py             # Pre-warm embedding model for all tests
│   ├── test_agents.py          # Unit tests: all agents, LLM client, orchestrator
│   ├── test_integration.py     # Integration: ingest → index → orchestrator (mocked LLM)
│   ├── test_milestone3.py      # Health score, doc agent, Q&A agent, partial status
│   ├── test_milestone4.py      # PDF/DOCX ingestion, mixed pipeline, report persistence
│   ├── test_regression.py      # Forecast fallback, context starvation, /ask route
│   ├── test_extraction_accuracy.py  # _merge() normalisation regressions
│   ├── test_llm_reliability.py      # Retry, cache, rate-limit handling
│   ├── test_blocker_fix.py          # Blocker agent edge cases
│   └── sample_data/            # Fixture files for integration tests
│
├── docs/
│   └── ARCHITECTURE.md         # Technical architecture documentation
│
├── uploads/                # Uploaded files (auto-created, gitignored)
├── chroma_db/              # ChromaDB vector store (auto-created, gitignored)
├── llm_cache/              # LLM result cache on disk (auto-created, gitignored)
└── reports/                # Persisted analysis reports (auto-created, gitignored)
```

---

## Usage Guide

### Uploading documents

1. Open the app at `http://localhost:5000/app`
2. Enter a project name in the Upload section
3. Drag and drop files or click to select. Multiple files are supported
4. Files are parsed, chunked, embedded, and indexed automatically

Supported formats: **PDF**, **DOCX**, **CSV**, **TXT**, **Markdown**.

To add more documents to an existing project later, use the same project name. The knowledge base updates automatically.

### Running analysis

1. Navigate to **Project Analysis** in the sidebar
2. Click **Run Analysis**

The analysis runs **two LLM calls in parallel** (scope + blockers/risks) using a ThreadPoolExecutor. On the first run it takes ~8 seconds. Results are cached to disk (`llm_cache/`), so subsequent runs (even after Flask restart) return instantly in ~1 second.

It produces:

- **Scope & Deliverables** — project goals, in-scope/out-of-scope items, deliverable table with owners and deadlines, milestones
- **Blockers & Actions** — open blockers with impact and evidence, action items with owners and due dates, pending decisions, unresolved issues
- **Risks & Forecast** — risk register grouped by severity, delivery forecast with confidence rating and concrete milestone dates
- **Generated Documents** — user stories, risk register, structured action item list (all deterministic, no LLM)
- **Health Score** — 0–100 score with four weighted dimensions: scope clarity (25%), timeline risk (30%), blocker severity (25%), risk exposure (20%)

### Conversational Q&A

Use the **AI Assistant** section to ask free-text questions about a project. Answers are grounded in both the indexed documents and the saved analysis report, with source citations and confidence rating (High / Medium / Low).

### Report

The **Report** section lets you view the full report inline and download it as a PDF.

---

## Running Tests

```bash
# Full test suite
venv\Scripts\python.exe -m pytest tests/ -v

# Or use the runner script
venv\Scripts\python.exe run_full_tests.py
```

The test suite uses mocked LLM calls throughout — no Groq API key is needed to run tests. 255 tests pass, 5 skipped.

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq Cloud API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | LLM model ID |
| `GROQ_TIMEOUT` | `60` | Per-request timeout in seconds |
| `FLASK_SECRET` | `dev-secret-change-in-prod` | Flask session secret key |

Set these in a `.env` file or as system environment variables.

---

## Dependencies

See `requirements.txt` for pinned versions. Key packages:

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX extraction (paragraphs + tables in document order) |
| `pandas` | CSV ingestion |
| `langchain-text-splitters` | Document chunking (800-char chunks) |
| `sentence-transformers` | Local embedding model (all-MiniLM-L6-v2) |
| `chromadb` | Vector store |
| `groq` | Groq LLM API client |
| `python-dotenv` | `.env` file support |
| `reportlab` | PDF report generation |
| `pytest` | Test runner |

---

## Milestones

| Milestone | Deliverables | Status |
|---|---|---|
| **1** | Document ingestion pipeline (PDF, DOCX, CSV, TXT), ChromaDB RAG, semantic search | ✅ Complete |
| **2** | Three-agent analysis pipeline (scope, blockers, risks), health scoring, orchestrator | ✅ Complete |
| **3** | Conversational Q&A assistant, auto-generated documents (user stories, risk register, action items) | ✅ Complete |
| **4** | Single-page dark-themed UI, parallel LLM calls, disk-cached results, incremental upload, report persistence, PDF report export, Q&A with analysis context | ✅ Complete |

---

## Architecture Highlights

### Two Parallel LLM Calls with Disk Cache
The combined agent makes **two LLM calls concurrently** via `ThreadPoolExecutor(max_workers=2)`:
- Call 1: Scope extraction (goals, deliverables, milestones, out-of-scope)
- Call 2: Blockers + Risks extraction (blockers, actions, decisions, risks, forecast)

For projects under 50K characters, the **full document text** is passed directly to both calls (bypassing vector search entirely), so no section or table row is missed. Results are cached to disk at `llm_cache/<project_id>.json` with a 24-hour TTL. Subsequent analysis runs — even after Flask restart — read from the disk cache in ~1 second with zero API calls.

### Rule-Based Delivery Forecast
The orchestrator always runs a deterministic rule-based forecast from extracted milestone/deliverable dates and merges it with the LLM's qualitative assessment:
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

### GPT-OSS Reasoning Control
`reasoning_effort="low"` is injected via `extra_body` for GPT-OSS models in `llm_client.py` to limit internal reasoning tokens. This keeps the model fast (~8 seconds per analysis) while maintaining output quality.
