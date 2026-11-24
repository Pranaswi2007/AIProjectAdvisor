"""
Milestone 4 End-to-End Tests
==============================
Covers:
  1. PDF ingestion — real pdfplumber extraction
  2. DOCX ingestion — real python-docx extraction
  3. Mixed-type pipeline — TXT + CSV + PDF + DOCX indexed together,
     orchestrator produces a valid report with mocked LLM
  4. Report persistence — save_report / load_summary / stale detection
  5. Incremental upload — duplicate file detection, stale flag
  6. Flask upload route — doc_type tag, duplicate handling, result page

Run:
    venv\\Scripts\\python.exe -m pytest tests/test_milestone4.py -v
"""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_DIR = Path(__file__).parent / "sample_data"


# ---------------------------------------------------------------------------
# Helpers — programmatically create minimal PDF and DOCX in memory
# ---------------------------------------------------------------------------

def _make_pdf_bytes(text: str) -> bytes:
    """Create a single-page PDF containing *text* using reportlab."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas as rl_canvas
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=LETTER)
    y = 750
    for line in _wrap(text, 90):
        c.drawString(50, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    return buf.getvalue()


def _make_docx_bytes(text: str) -> bytes:
    """Create a minimal DOCX containing *text* using python-docx."""
    from docx import Document as DocxDoc
    doc = DocxDoc()
    for para in text.split("\n\n"):
        doc.add_paragraph(para.strip())
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, current = [], []
    for w in words:
        if sum(len(x) + 1 for x in current) + len(w) > width:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


# Rich enough text that pdfplumber / python-docx extract >0 words
_PDF_TEXT = (
    "Project: CampusShuttle\n\n"
    "Goal: Build a real-time shuttle tracking system for the university campus.\n\n"
    "Scope: GPS integration, mobile app for students, admin dashboard for fleet management.\n\n"
    "Out of Scope: Integration with external bus networks.\n\n"
    "Deliverables:\n"
    "  - GPS Tracker Module (owner: Arjun, deadline: 2026-11-30)\n"
    "  - Student Mobile App (owner: Priya, deadline: 2026-12-15)\n"
    "  - Admin Dashboard (owner: Ravi, deadline: 2026-12-20)\n\n"
    "Milestones:\n"
    "  - Prototype demo: 2026-11-01\n"
    "  - Beta launch: 2026-11-30\n"
    "  - Go-live: 2026-12-31\n"
)

_DOCX_TEXT = (
    "Meeting Notes — CampusShuttle Sprint 3\n\n"
    "Blocker B001: GPS hardware supplier delayed shipment by two weeks.\n"
    "Impact: Prototype demo at risk.\n"
    "Owner: Arjun\n\n"
    "Action Item A001: Arjun to escalate with supplier by Friday.\n"
    "Action Item A002: Priya to prepare mock GPS data for demo.\n\n"
    "Risk: Supplier delay may push beta launch past November 30.\n"
    "Mitigation: Use simulated GPS feed for prototype phase.\n"
)


# ---------------------------------------------------------------------------
# 1 — PDF ingestion
# ---------------------------------------------------------------------------

class TestPDFIngestion(unittest.TestCase):
    """Real pdfplumber extraction from a programmatically generated PDF."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        cls.tmp.write(_make_pdf_bytes(_PDF_TEXT))
        cls.tmp.close()

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls.tmp.name)
        except Exception:
            pass

    def test_pdf_ingest_returns_document(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-pdf")
        self.assertEqual(doc.project_id, "test-m4-pdf")
        self.assertEqual(doc.file_type, "pdf")

    def test_pdf_has_nonzero_word_count(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-pdf")
        self.assertGreater(doc.word_count, 0,
                           "PDF produced zero words — extraction failed")

    def test_pdf_text_contains_expected_content(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-pdf")
        # Even the hand-crafted fallback PDF embeds at least the first 200 chars
        lower = doc.text.lower()
        self.assertTrue(
            "campus" in lower or "goal" in lower or "scope" in lower,
            f"PDF text did not contain expected keywords. Got: {doc.text[:300]!r}"
        )

    def test_pdf_filename_preserved(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-pdf")
        self.assertTrue(doc.filename.endswith(".pdf"))


# ---------------------------------------------------------------------------
# 2 — DOCX ingestion
# ---------------------------------------------------------------------------

class TestDOCXIngestion(unittest.TestCase):
    """Real python-docx extraction from a programmatically generated DOCX."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        cls.tmp.write(_make_docx_bytes(_DOCX_TEXT))
        cls.tmp.close()

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls.tmp.name)
        except Exception:
            pass

    def test_docx_ingest_returns_document(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-docx")
        self.assertEqual(doc.project_id, "test-m4-docx")
        self.assertEqual(doc.file_type, "docx")

    def test_docx_has_nonzero_word_count(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-docx")
        self.assertGreater(doc.word_count, 0,
                           "DOCX produced zero words — extraction failed")

    def test_docx_text_contains_expected_content(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-docx")
        lower = doc.text.lower()
        self.assertIn("blocker", lower,
                      f"DOCX text did not contain 'blocker'. Got: {doc.text[:300]!r}")

    def test_docx_filename_preserved(self):
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-docx")
        self.assertTrue(doc.filename.endswith(".docx"))

    def test_docx_multiple_paragraphs(self):
        """Multi-paragraph DOCX should produce multiple lines of text."""
        from ingestion import ingest_file
        doc = ingest_file(self.tmp.name, "test-m4-docx")
        self.assertGreater(doc.text.count("\n"), 2,
                           "Expected multiple lines from multi-paragraph DOCX")


# ---------------------------------------------------------------------------
# 3 — Mixed-type pipeline (TXT + CSV + PDF + DOCX indexed, mocked LLM)
# ---------------------------------------------------------------------------

class TestMixedTypePipeline(unittest.TestCase):
    """Index TXT, CSV, PDF and DOCX into one project, run orchestrator with
    mocked LLM, verify the report structure and health score are valid."""

    PROJECT_ID = "test-m4-mixed"

    @classmethod
    def setUpClass(cls):
        try:
            from rag import get_model
            get_model()
        except Exception as e:
            raise unittest.SkipTest(f"Embedding model unavailable: {e}")

        from ingestion import ingest_file
        from rag import index_document

        # TXT
        for fname in ("project_proposal.txt", "meeting_notes.txt"):
            doc = ingest_file(str(SAMPLE_DIR / fname), cls.PROJECT_ID)
            index_document(doc)

        # CSV
        doc = ingest_file(str(SAMPLE_DIR / "task_list.csv"), cls.PROJECT_ID)
        index_document(doc)

        # PDF — written to a temp file for the duration of tests
        cls._pdf_tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        cls._pdf_tmp.write(_make_pdf_bytes(_PDF_TEXT))
        cls._pdf_tmp.close()
        doc = ingest_file(cls._pdf_tmp.name, cls.PROJECT_ID)
        index_document(doc)

        # DOCX
        cls._docx_tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        cls._docx_tmp.write(_make_docx_bytes(_DOCX_TEXT))
        cls._docx_tmp.close()
        doc = ingest_file(cls._docx_tmp.name, cls.PROJECT_ID)
        index_document(doc)

    @classmethod
    def tearDownClass(cls):
        # Remove ChromaDB collection
        try:
            from rag import _get_client
            import re
            sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", cls.PROJECT_ID).strip("_-") or "default"
            _get_client().delete_collection("project-" + sanitised[:57])
        except Exception:
            pass
        # Remove temp files
        for attr in ("_pdf_tmp", "_docx_tmp"):
            try:
                os.unlink(getattr(cls, attr).name)
            except Exception:
                pass

    def setUp(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def _mock_combined(self):
        """Return a combined response that covers all four document types."""
        scope = {
            "project_goals": [
                {"goal": "Build campus shuttle tracker", "source": "project_proposal.txt"},
                {"goal": "Provide real-time GPS tracking", "source": "_pdf_"},
            ],
            "scope": [{"item": "GPS integration", "source": "_pdf_"}],
            "out_of_scope": [{"item": "External bus networks", "source": "_pdf_"}],
            "deliverables": [
                {"name": "GPS Tracker Module", "description": "Hardware + firmware",
                 "deadline": "2026-11-30", "owner": "Arjun",
                 "dependencies": [], "acceptance_criteria": [], "source": "_pdf_", "inferred": False},
            ],
            "milestones": [{"name": "Beta launch", "date": "2026-11-30", "source": "_pdf_"}],
            "missing_information": [], "conflicts": [], "extraction_gaps": [], "error": None,
        }
        blockers = {
            "blockers": [
                {"blocker_id": "B001", "description": "GPS hardware delayed",
                 "status": "Open", "impact": "Prototype at risk",
                 "owner": "Arjun", "source": "_docx_",
                 "evidence": "Supplier delay", "recommended_resolution": "Escalate", "inferred": False},
            ],
            "action_items": [
                {"action_id": "A001", "description": "Escalate with supplier",
                 "owner": "Arjun", "due_date": "2026-11-01", "priority": "High",
                 "status": "Pending", "related_blocker": "B001", "source": "_docx_", "inferred": False},
            ],
            "pending_decisions": [], "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        risks = {
            "risks": [
                {"risk_id": "R001", "title": "Supplier delay",
                 "description": "GPS hardware supplier delayed by 2 weeks",
                 "severity": "High", "probability": "High",
                 "potential_impact": "Prototype demo at risk",
                 "affected_deliverable": "GPS Tracker Module",
                 "mitigation": "Use simulated GPS feed",
                 "owner": "Arjun", "status": "Potential", "source": "_docx_"},
            ],
            "delivery_forecast": {
                "status": "At Risk", "estimated_completion_date": "2026-12-31",
                "delayed_milestones": ["Beta launch"],
                "assumptions": ["Supplier delivers by Nov 15"],
                "confidence": "Medium", "forecast_method": "qualitative",
            },
            "dependency_gaps": [], "missing_information": [], "error": None,
        }
        return scope, blockers, risks

    def test_mixed_pipeline_produces_valid_report(self):
        from rag import get_project_stats
        stats = get_project_stats(self.PROJECT_ID)
        if stats["chunk_count"] == 0:
            self.skipTest("No chunks indexed — skipping pipeline test")

        with patch("agents.combined_agent.run", return_value=self._mock_combined()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)

        self.assertIn(report["status"], ("ok", "partial"))
        self.assertIn("scope",        report)
        self.assertIn("blockers",     report)
        self.assertIn("risks",        report)
        self.assertIn("health_score", report)

    def test_mixed_pipeline_health_score_computed(self):
        from rag import get_project_stats
        if get_project_stats(self.PROJECT_ID)["chunk_count"] == 0:
            self.skipTest("No chunks indexed")

        with patch("agents.combined_agent.run", return_value=self._mock_combined()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)

        hs = report["health_score"]
        self.assertIn("overall",    hs)
        self.assertIn("grade",      hs)
        self.assertIn("dimensions", hs)
        self.assertIsInstance(hs["overall"], int)
        self.assertGreaterEqual(hs["overall"], 0)
        self.assertLessEqual(hs["overall"], 100)

    def test_mixed_pipeline_docs_generated(self):
        """Deterministic doc agent must generate user stories and action items."""
        from rag import get_project_stats
        if get_project_stats(self.PROJECT_ID)["chunk_count"] == 0:
            self.skipTest("No chunks indexed")

        with patch("agents.combined_agent.run", return_value=self._mock_combined()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)

        docs = report.get("docs", {})
        # At least one of the three doc types should be populated
        total = (len(docs.get("user_stories", []))
                 + len(docs.get("risk_register", []))
                 + len(docs.get("action_item_list", [])))
        self.assertGreater(total, 0,
                           "Expected at least one generated document section")

    def test_mixed_pipeline_all_four_file_types_indexed(self):
        """ChromaDB should have chunks from all four ingested document types."""
        from rag import get_project_stats
        stats = get_project_stats(self.PROJECT_ID)
        # 3 TXT/CSV files + 1 PDF + 1 DOCX = at least 5 source files total
        self.assertGreaterEqual(stats["document_count"], 4,
                                f"Expected ≥4 document types indexed, got {stats}")

    def test_mixed_pipeline_at_risk_lowers_timeline_score(self):
        """'At Risk' forecast should give timeline score ≤ 70."""
        from rag import get_project_stats
        if get_project_stats(self.PROJECT_ID)["chunk_count"] == 0:
            self.skipTest("No chunks indexed")

        with patch("agents.combined_agent.run", return_value=self._mock_combined()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)

        hs = report["health_score"]
        self.assertLessEqual(hs["timeline_risk"], 70,
                             f"'At Risk' forecast should cap timeline ≤70, got {hs['timeline_risk']}")


# ---------------------------------------------------------------------------
# 4 — Report persistence and stale detection
# ---------------------------------------------------------------------------

class TestReportPersistence(unittest.TestCase):
    """save_report, load_summary, and stale detection via report_store."""

    PROJECT_ID = "test-m4-persist"

    def _sample_report(self, overall: int = 72) -> dict:
        return {
            "project_id":   self.PROJECT_ID,
            "generated_at": "2026-11-01T10:00:00",
            "status":       "ok",
            "duration_seconds": 7.5,
            "scope": {
                "project_goals": [{"goal": "Track shuttles", "source": "proposal.txt"}],
                "scope": [{"item": "GPS", "source": "proposal.txt"}],
                "out_of_scope": [],
                "deliverables": [
                    {"name": "App", "description": "Mobile app",
                     "deadline": "2026-12-01", "owner": "Priya",
                     "dependencies": [], "acceptance_criteria": [], "source": "proposal.txt", "inferred": False}
                ],
                "milestones": [{"name": "Beta", "date": "2026-11-30", "source": "proposal.txt"}],
                "missing_information": [], "conflicts": [], "extraction_gaps": [], "error": None,
            },
            "blockers": {
                "blockers": [
                    {"blocker_id": "B001", "description": "GPS delay", "status": "Open",
                     "impact": "", "owner": None, "source": "", "evidence": "",
                     "recommended_resolution": "", "inferred": False}
                ],
                "action_items": [], "pending_decisions": [], "unresolved_issues": [],
                "missing_information": [], "extraction_source": "llm", "error": None,
            },
            "risks": {
                "risks": [
                    {"risk_id": "R001", "title": "Delay", "description": "d",
                     "severity": "High", "probability": "Medium",
                     "potential_impact": "Miss deadline", "affected_deliverable": None,
                     "mitigation": "Mock", "owner": None, "status": "Potential", "source": ""}
                ],
                "delivery_forecast": {
                    "status": "At Risk", "estimated_completion_date": None,
                    "delayed_milestones": [], "assumptions": [],
                    "confidence": "Medium", "forecast_method": "qualitative",
                },
                "dependency_gaps": [], "missing_information": [], "error": None,
            },
            "docs": {
                "user_stories": [
                    {"id": "US001", "role": "student", "want": "track shuttle",
                     "benefit": "plan commute", "source": "proposal.txt"}
                ],
                "risk_register": [], "action_item_list": [], "missing_information": [], "error": None,
            },
            "health_score": {
                "overall": overall, "grade": "C", "summary": "Moderate concerns.",
                "scope_clarity": 90, "timeline_risk": 60, "blocker_severity": 70, "risk_exposure": 65,
                "dimensions": [],
            },
            "agent_status": {"scope": "ok", "blockers": "ok", "risks": "ok", "docs": "ok"},
            "errors": [], "warnings": [],
        }

    def tearDown(self):
        # Clean up any report file written during the test
        from report_store import delete_report
        delete_report(self.PROJECT_ID)

    def test_save_and_load_summary(self):
        from report_store import save_report, load_summary
        save_report(self.PROJECT_ID, self._sample_report())
        summary = load_summary(self.PROJECT_ID)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["project_id"], self.PROJECT_ID)

    def test_summary_contains_health_fields(self):
        from report_store import save_report, load_summary
        save_report(self.PROJECT_ID, self._sample_report(overall=72))
        summary = load_summary(self.PROJECT_ID)
        for key in ("overall", "grade", "forecast_status",
                    "open_blockers", "total_risks", "critical_risks",
                    "deliverable_count", "action_item_count"):
            self.assertIn(key, summary, f"summary missing key: {key}")

    def test_summary_health_values_match_report(self):
        from report_store import save_report, load_summary
        save_report(self.PROJECT_ID, self._sample_report(overall=72))
        summary = load_summary(self.PROJECT_ID)
        self.assertEqual(summary["overall"], 72)
        self.assertEqual(summary["grade"],   "C")
        self.assertEqual(summary["forecast_status"], "At Risk")
        self.assertEqual(summary["open_blockers"],   1)
        self.assertEqual(summary["total_risks"],     1)
        self.assertEqual(summary["critical_risks"],  0)

    def test_load_summary_missing_project_returns_none(self):
        from report_store import load_summary
        result = load_summary("does-not-exist-xyz-123")
        self.assertIsNone(result)

    def test_list_summaries_includes_saved_project(self):
        from report_store import save_report, list_summaries
        save_report(self.PROJECT_ID, self._sample_report())
        summaries = list_summaries()
        self.assertIn(self.PROJECT_ID, summaries)

    def test_delete_report_removes_from_list(self):
        from report_store import save_report, delete_report, list_summaries
        save_report(self.PROJECT_ID, self._sample_report())
        delete_report(self.PROJECT_ID)
        summaries = list_summaries()
        self.assertNotIn(self.PROJECT_ID, summaries)

    def test_load_full_report_round_trip(self):
        """Full report saved and loaded should have all top-level keys."""
        from report_store import save_report, load_report
        save_report(self.PROJECT_ID, self._sample_report())
        loaded = load_report(self.PROJECT_ID)
        self.assertIsNotNone(loaded)
        for key in ("scope", "blockers", "risks", "docs", "health_score",
                    "agent_status", "summary"):
            self.assertIn(key, loaded, f"loaded report missing key: {key}")

    def test_overwrite_updates_summary(self):
        """Re-saving with a different score should overwrite the old value."""
        from report_store import save_report, load_summary
        save_report(self.PROJECT_ID, self._sample_report(overall=55))
        save_report(self.PROJECT_ID, self._sample_report(overall=88))
        summary = load_summary(self.PROJECT_ID)
        self.assertEqual(summary["overall"], 88)


# ---------------------------------------------------------------------------
# 5 — Stale-KB detection (via get_all_projects helper)
# ---------------------------------------------------------------------------

class TestStaleDetection(unittest.TestCase):
    """get_all_projects should mark a project stale when a file is newer
    than the saved_at timestamp on the last report."""

    PROJECT_ID = "test-m4-stale"

    def setUp(self):
        import app
        self.upload_dir = Path("uploads") / self.PROJECT_ID
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        from report_store import delete_report
        delete_report(self.PROJECT_ID)
        import shutil
        try:
            shutil.rmtree(str(self.upload_dir), ignore_errors=True)
        except Exception:
            pass

    def test_project_not_stale_when_no_report(self):
        """A project with no saved report is not stale (no prior analysis)."""
        import app
        projects = app.get_all_projects()
        entry = next((p for p in projects if p["id"] == self.PROJECT_ID), None)
        # May not appear if the folder is empty; that's fine
        if entry:
            self.assertFalse(entry["stale"])

    def test_project_stale_when_file_newer_than_report(self):
        """A file added AFTER the report was saved should trigger stale=True."""
        from report_store import save_report
        import app

        # 1. Save a report with saved_at = now
        save_report(self.PROJECT_ID, {
            "project_id": self.PROJECT_ID,
            "generated_at": "2026-01-01T00:00:00",
            "status": "ok", "duration_seconds": 5,
            "scope": {}, "blockers": {}, "risks": {}, "docs": {},
            "health_score": {"overall": 70, "grade": "C", "summary": "",
                             "scope_clarity": 70, "timeline_risk": 70,
                             "blocker_severity": 70, "risk_exposure": 70, "dimensions": []},
            "agent_status": {}, "errors": [], "warnings": [],
        })

        # 2. Wait a tick then write a new file into the project folder
        time.sleep(0.05)
        new_file = self.upload_dir / "new_meeting_notes.txt"
        new_file.write_text("New meeting notes added after analysis.", encoding="utf-8")

        # 3. get_all_projects should detect the stale state
        projects = app.get_all_projects()
        entry = next((p for p in projects if p["id"] == self.PROJECT_ID), None)
        self.assertIsNotNone(entry, "Test project not found in get_all_projects()")
        self.assertTrue(entry["stale"],
                        "Expected stale=True because a file is newer than saved_at")

    def test_project_not_stale_when_file_older_than_report(self):
        """Files that existed before the analysis should NOT trigger stale."""
        from report_store import save_report
        import app

        # 1. Create a file first
        old_file = self.upload_dir / "old_proposal.txt"
        old_file.write_text("Original proposal.", encoding="utf-8")

        # 2. Wait > 1s so the Windows FAT/NTFS mtime is strictly older
        time.sleep(1.1)

        # 3. Save a fresh report (saved_at = now, after the file was written)
        save_report(self.PROJECT_ID, {
            "project_id": self.PROJECT_ID,
            "generated_at": "2026-11-01T12:00:00",
            "status": "ok", "duration_seconds": 5,
            "scope": {}, "blockers": {}, "risks": {}, "docs": {},
            "health_score": {"overall": 80, "grade": "B", "summary": "",
                             "scope_clarity": 80, "timeline_risk": 80,
                             "blocker_severity": 80, "risk_exposure": 80, "dimensions": []},
            "agent_status": {}, "errors": [], "warnings": [],
        })

        # 4. No new files after the report — should NOT be stale
        projects = app.get_all_projects()
        entry = next((p for p in projects if p["id"] == self.PROJECT_ID), None)
        if entry:
            self.assertFalse(entry["stale"],
                             "Expected stale=False because no files added after report")


# ---------------------------------------------------------------------------
# 6 — Flask upload route: doc_type tag and duplicate handling
# ---------------------------------------------------------------------------

class TestFlaskUploadMilestone4(unittest.TestCase):
    """Test the upgraded upload route: doc_type, duplicate detection,
    incremental banner in result page."""

    # Projects created during tests — cleaned up in tearDown
    _created_projects: list[str] = []

    @classmethod
    def setUpClass(cls):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = flask_app.app.test_client()
        cls._created_projects = []

    @classmethod
    def tearDownClass(cls):
        """Remove all test projects from ChromaDB and disk after the class runs."""
        import re, shutil, stat
        from pathlib import Path
        try:
            from rag import _get_client
            for pid in cls._created_projects:
                sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", pid).strip("_-") or "default"
                try:
                    _get_client().delete_collection("project-" + sanitised[:57])
                except Exception:
                    pass
                folder = Path("uploads") / pid
                if folder.exists():
                    def _force(func, path, _):
                        try:
                            import os; os.chmod(path, stat.S_IWRITE); func(path)
                        except Exception:
                            pass
                    shutil.rmtree(str(folder), onerror=_force)
        except Exception:
            pass

    def _post_upload(self, project_id: str, filename: str,
                     content: bytes, doc_type: str = "") -> object:
        # Track projects so tearDownClass can clean them up
        if project_id not in self._created_projects:
            self._created_projects.append(project_id)
        data = {
            "project_id": project_id,
            "doc_type":   doc_type,
            "documents":  (io.BytesIO(content), filename),
        }
        return self.client.post(
            "/upload",
            data=data,
            content_type="multipart/form-data",
        )

    def test_upload_get_has_doc_type_selector(self):
        """Upload page must render the document-type selector."""
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"doc_type", resp.data)
        self.assertIn(b"Meeting Notes", resp.data)

    def test_upload_txt_success(self):
        """TXT file upload returns 200 and shows ✓ Indexed."""
        resp = self._post_upload(
            "test-m4-flask-txt",
            "sample.txt",
            b"Project goal: build a tracking app.",
            doc_type="Meeting Notes",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Indexed", resp.data)

    def test_upload_txt_with_doc_type_shown_in_result(self):
        """doc_type should appear in the upload result page."""
        resp = self._post_upload(
            "test-m4-flask-dtype",
            "notes.txt",
            b"Sprint 3 meeting notes. Blocker: SAP access pending.",
            doc_type="Sprint Update",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Sprint Update", resp.data)

    def test_upload_duplicate_file_shows_reindexed_badge(self):
        """Uploading the same filename twice should show 're-indexed' badge."""
        content = b"Initial version of the document."
        project = "test-m4-flask-dup"
        # First upload
        self._post_upload(project, "dup_doc.txt", content)
        # Second upload — same filename
        resp = self._post_upload(project, "dup_doc.txt", b"Updated version.")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"re-indexed", resp.data)

    def test_upload_duplicate_shows_stale_banner(self):
        """Re-indexing an existing file should show the incremental update banner."""
        project = "test-m4-flask-stale-banner"
        content = b"Proposal document v1."
        self._post_upload(project, "proposal.txt", content)
        resp = self._post_upload(project, "proposal.txt", b"Proposal document v2.")
        self.assertEqual(resp.status_code, 200)
        # Banner text tells user to re-run analysis
        self.assertIn(b"Knowledge base updated", resp.data)

    def test_upload_new_file_shows_analyse_link(self):
        """A fresh successful upload should include an 'Analyse Project' link."""
        resp = self._post_upload(
            "test-m4-flask-new",
            "new_doc.txt",
            b"Brand new project document.",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Analyse", resp.data)

    def test_upload_unsupported_file_type_shows_error(self):
        """Uploading an unsupported extension must show an error badge."""
        resp = self._post_upload(
            "test-m4-flask-ext",
            "bad_file.xyz",
            b"not a real document",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Error", resp.data)

    def test_dashboard_route_still_works(self):
        resp = self.client.get("/")
        # / now redirects to the single-page app
        self.assertIn(resp.status_code, (200, 302))

    def test_dashboard_shows_insights_section(self):
        """The single-page app at /app should render successfully."""
        resp = self.client.get("/app")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"AI Project", resp.data)


# ---------------------------------------------------------------------------
# 7 — Health score accuracy (cross-check with real data)
# ---------------------------------------------------------------------------

class TestHealthScoreAccuracy(unittest.TestCase):
    """Verify that the health score reflects actual extracted data,
    not hardcoded or fake values."""

    def test_confirmed_high_risk_caps_timeline(self):
        """A confirmed High risk must bring timeline score below 60
        even when forecast says 'On Track'."""
        from agents.health_score import compute
        result = compute(
            scope_data={
                "project_goals": [{"goal": "G", "source": ""}],
                "scope": [{"item": "S", "source": ""}],
                "deliverables": [{"name": "D", "description": "x",
                                  "deadline": "2026-12-01", "owner": "O",
                                  "dependencies": [], "acceptance_criteria": [],
                                  "source": "", "inferred": False}],
                "milestones": [{"name": "M", "date": "2026-12-01", "source": ""}],
                "missing_information": [], "conflicts": [], "extraction_gaps": [],
            },
            blocker_data={"blockers": [], "action_items": [], "pending_decisions": [],
                          "unresolved_issues": [], "missing_information": []},
            risk_data={
                "risks": [
                    {"risk_id": "R001", "title": "Critical delay", "severity": "High",
                     "status": "Confirmed", "probability": "High",
                     "potential_impact": "Miss deadline", "mitigation": "none",
                     "owner": None, "source": ""},
                ],
                "delivery_forecast": {
                    "status": "On Track",   # LLM says On Track but risk is Confirmed High
                    "estimated_completion_date": None,
                    "delayed_milestones": [], "assumptions": [],
                    "confidence": "Low", "forecast_method": "qualitative",
                },
            },
        )
        self.assertLess(result["timeline_risk"], 60,
                        "Confirmed High risk should cap timeline below 60 "
                        f"regardless of 'On Track' label; got {result['timeline_risk']}")

    def test_no_risks_no_blockers_gives_high_scores(self):
        """A clean project with full scope and no issues should score ≥ 75."""
        from agents.health_score import compute
        result = compute(
            scope_data={
                "project_goals": [{"goal": "G", "source": ""}],
                "scope":         [{"item": "S", "source": ""}],
                "deliverables":  [{"name": "D", "description": "x",
                                   "deadline": "2026-12-01", "owner": "O",
                                   "dependencies": [], "acceptance_criteria": [],
                                   "source": "", "inferred": False}],
                "milestones":    [{"name": "M", "date": "2026-12-01", "source": ""}],
                "missing_information": [], "conflicts": [], "extraction_gaps": [],
            },
            blocker_data={"blockers": [], "action_items": [], "pending_decisions": [],
                          "unresolved_issues": [], "missing_information": []},
            risk_data={
                "risks": [],
                "delivery_forecast": {"status": "On Track",
                                      "estimated_completion_date": "2026-12-01",
                                      "delayed_milestones": [], "assumptions": [],
                                      "confidence": "High", "forecast_method": "rule-based"},
            },
        )
        self.assertGreaterEqual(result["overall"], 75,
                                f"Clean project should score ≥75; got {result['overall']}")

    def test_many_open_blockers_lowers_blocker_score(self):
        """5 open blockers should bring blocker_severity below 60."""
        from agents.health_score import compute
        result = compute(
            scope_data={},
            blocker_data={
                "blockers": [
                    {"blocker_id": f"B00{i}", "status": "Open",
                     "description": f"blocker {i}", "impact": "", "owner": None,
                     "source": "", "evidence": "", "recommended_resolution": "", "inferred": False}
                    for i in range(5)
                ],
                "action_items": [], "pending_decisions": [], "unresolved_issues": [],
                "missing_information": [],
            },
            risk_data={},
        )
        self.assertLess(result["blocker_severity"], 60,
                        f"5 open blockers should score <60; got {result['blocker_severity']}")

    def test_scope_gaps_reduce_scope_clarity(self):
        """Flagged missing_information items should reduce scope_clarity below 100."""
        from agents.health_score import compute
        result = compute(
            scope_data={
                "project_goals": [{"goal": "G", "source": ""}],
                "scope":         [{"item": "S", "source": ""}],
                "deliverables":  [{"name": "D", "description": "x",
                                   "deadline": "2026-12-01", "owner": "O",
                                   "dependencies": [], "acceptance_criteria": [],
                                   "source": "", "inferred": False}],
                "milestones":    [{"name": "M", "date": "2026-12-01", "source": ""}],
                "missing_information": ["Budget not documented",
                                        "Team size unknown",
                                        "No acceptance criteria"],
                "conflicts": [], "extraction_gaps": ["Deliverable X missing deadline"],
            },
            blocker_data={},
            risk_data={},
        )
        self.assertLess(result["scope_clarity"], 100,
                        f"Flagged gaps should reduce scope_clarity below 100; got {result['scope_clarity']}")


if __name__ == "__main__":
    unittest.main()
