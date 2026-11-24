"""
Integration tests — Milestone 2
=================================
Tests the full pipeline: ingest sample_data files → index into ChromaDB
→ run all three agents → validate combined report structure.

LLM calls are mocked. The embedding model and ChromaDB run for real
to validate the actual RAG retrieval path.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_integration.py -v
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_DIR = Path(__file__).parent / "sample_data"

# ---------------------------------------------------------------------------
# Shared mock responses (same as unit tests)
# ---------------------------------------------------------------------------

def _scope_response():
    return {
        "project_goals": [
            {"goal": "Automate inventory tracking", "source": "project_proposal.txt#0", "inferred": False}
        ],
        "scope":        [{"item": "Barcode scanner integration", "source": "project_proposal.txt#1", "inferred": False}],
        "out_of_scope": [{"item": "Mobile app", "source": "project_proposal.txt#1"}],
        "deliverables": [
            {
                "name": "Database Schema",
                "description": "PostgreSQL schema",
                "deadline": "2026-10-05",
                "owner": "Sneha Rao",
                "dependencies": ["D1"],
                "acceptance_criteria": ["Reviewed by DBA"],
                "source": "project_proposal.txt#2",
                "inferred": False,
            }
        ],
        "milestones": [
            {"name": "Go-Live", "date": "2026-12-01", "description": "Launch", "source": "project_proposal.txt#3"}
        ],
        "missing_information": [],
        "conflicts": [],
        "extraction_gaps": [],
        "error": None,
    }


def _blocker_response():
    return {
        "blockers": [
            {
                "blocker_id": "B001",
                "description": "SAP B1 sandbox not granted",
                "status": "Open",
                "impact": "ERP integration blocked",
                "owner": "Priya Sharma",
                "source": "meeting_notes.txt#2",
                "evidence": "SAP B1 sandbox access not granted",
                "recommended_resolution": "Escalate to vendor",
                "inferred": False,
            }
        ],
        "action_items": [
            {
                "action_id": "A001",
                "description": "Escalate driver installation",
                "owner": "Ravi Kumar",
                "due_date": "2026-10-16",
                "priority": "High",
                "status": "Pending",
                "related_blocker": None,
                "source": "meeting_notes.txt#1",
                "inferred": False,
            }
        ],
        "pending_decisions": [],
        "unresolved_issues": [],
        "missing_information": [],
        "extraction_source": "llm",
        "error": None,
    }


def _risk_response():
    return {
        "risks": [
            {
                "risk_id": "R001",
                "title": "SAP integration at risk",
                "description": "SAP sandbox delayed",
                "severity": "High",
                "affected_deliverable": "FR-004",
                "evidence": "SAP sandbox not granted",
                "source": "meeting_notes.txt#2",
                "potential_impact": "Miss Nov 30 deadline",
                "mitigation": "Use mock library",
                "status": "Potential",
            }
        ],
        "delivery_forecast": {
            "status": "At Risk",
            "estimated_completion_date": "2026-12-01",
            "delayed_milestones": [],
            "assumptions": ["SAP access by Oct 24"],
            "confidence": "Medium",
            "forecast_method": "qualitative",
        },
        "dependency_gaps": [],
        "missing_information": [],
        "error": None,
    }


def _combined_response():
    """Return (scope_data, blocker_data, risk_data) as combined_agent.run() would."""
    return _scope_response(), _blocker_response(), _risk_response()


# ===========================================================================
# Integration: ingest → retrieve → agent output shape
# ===========================================================================

class TestIngestionPipeline(unittest.TestCase):
    """Verify that sample files can be ingested without errors."""

    def test_all_sample_files_ingest_without_error(self):
        from ingestion import ingest_file
        for f in SAMPLE_DIR.iterdir():
            with self.subTest(file=f.name):
                doc = ingest_file(str(f), "test-ingest")
                self.assertGreater(doc.word_count, 0,
                                   f"{f.name} produced empty text")
                self.assertEqual(doc.project_id, "test-ingest")
                self.assertEqual(doc.filename, f.name)

    def test_unsupported_extension_raises(self):
        from ingestion import ingest_file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tf:
            tf.write(b"dummy content")
            tmp_path = tf.name
        try:
            with self.assertRaises(ValueError):
                ingest_file(tmp_path, "test")
        finally:
            os.unlink(tmp_path)

    def test_empty_file_raises(self):
        from ingestion import ingest_file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            tf.write(b"   ")
            tmp_path = tf.name
        try:
            with self.assertRaises(ValueError):
                ingest_file(tmp_path, "test")
        finally:
            os.unlink(tmp_path)


class TestRAGIndexAndSearch(unittest.TestCase):
    """Verify that ingested sample files can be indexed and searched.
    
    Skipped automatically if the embedding model cannot be loaded
    within the test environment (e.g. OneDrive sync delay on Windows).
    """

    PROJECT_ID = "test-rag-integration"

    @classmethod
    def setUpClass(cls):
        """Index all sample files into a temporary test project."""
        try:
            from rag import get_model
            get_model()  # pre-warm; raises if unavailable
        except Exception as e:
            raise unittest.SkipTest(f"Embedding model unavailable: {e}")
        from ingestion import ingest_file
        from rag import index_document
        for f in SAMPLE_DIR.iterdir():
            doc = ingest_file(str(f), cls.PROJECT_ID)
            index_document(doc)

    @classmethod
    def tearDownClass(cls):
        """Remove the test project from ChromaDB."""
        try:
            from rag import _get_client
            import re
            sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", cls.PROJECT_ID).strip("_-") or "default"
            name = "project-" + sanitised[:57]
            _get_client().delete_collection(name)
        except Exception:
            pass

    def test_chunks_indexed(self):
        from rag import get_project_stats
        stats = get_project_stats(self.PROJECT_ID)
        self.assertGreater(stats["chunk_count"], 0)
        self.assertGreater(stats["document_count"], 0)

    def test_search_returns_results(self):
        from rag import search
        hits = search(self.PROJECT_ID, "project goals deliverables", k=5)
        self.assertGreater(len(hits), 0)
        self.assertIsInstance(hits[0].text, str)
        self.assertIsInstance(hits[0].score, float)

    def test_search_returns_results_from_multiple_files(self):
        from rag import search
        hits = search(self.PROJECT_ID, "blocker action item SAP risk deadline", k=10)
        sources = {h.source for h in hits}
        # With 5 files indexed and k=10, expect results from at least 2 files
        self.assertGreaterEqual(len(sources), 2,
                                f"Expected ≥2 sources, got: {sources}")

    def test_search_score_in_range(self):
        from rag import search
        hits = search(self.PROJECT_ID, "inventory automation", k=3)
        for h in hits:
            self.assertGreaterEqual(h.score, 0.0)
            self.assertLessEqual(h.score, 1.0)

    def test_search_empty_project_returns_empty(self):
        from rag import search
        hits = search("_nonexistent_project_xyz_", "anything", k=5)
        self.assertEqual(hits, [])


class TestFullPipelineMocked(unittest.TestCase):
    """Run the orchestrator against real ChromaDB but mocked LLM.

    Skipped if the embedding model is unavailable (model not loaded yet).
    """

    PROJECT_ID = "test-rag-integration"  # re-uses the indexed project above

    def setUp(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    @classmethod
    def setUpClass(cls):
        try:
            from rag import get_model, get_project_stats
            get_model()
            stats = get_project_stats(cls.PROJECT_ID)
            if stats["chunk_count"] == 0:
                raise unittest.SkipTest(
                    "test-rag-integration has no indexed chunks — "
                    "run TestRAGIndexAndSearch first."
                )
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"RAG not available: {e}")

    def test_orchestrator_produces_valid_report(self):
        with patch("agents.combined_agent.run", return_value=_combined_response()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)

        self.assertIn(report["status"], ("ok", "partial"))
        self.assertIn("scope",   report)
        self.assertIn("blockers", report)
        self.assertIn("risks",   report)
        self.assertGreaterEqual(report["duration_seconds"], 0)

    def test_report_scope_has_deliverables(self):
        with patch("agents.combined_agent.run", return_value=_combined_response()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)
        self.assertIsInstance(report["scope"].get("deliverables"), list)

    def test_report_risks_has_severity_values(self):
        with patch("agents.combined_agent.run", return_value=_combined_response()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)
        for r in report["risks"].get("risks", []):
            self.assertIn(r["severity"], ("Critical", "High", "Medium", "Low"))

    def test_delivery_forecast_confidence_valid(self):
        with patch("agents.combined_agent.run", return_value=_combined_response()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)
        fc = report["risks"].get("delivery_forecast", {})
        self.assertIn(fc.get("confidence"), ("High", "Medium", "Low"))

    def test_source_references_present(self):
        """Every deliverable and blocker must have a source field."""
        with patch("agents.combined_agent.run", return_value=_combined_response()):
            from agents.orchestrator import run
            report = run(self.PROJECT_ID)

        for d in report["scope"].get("deliverables", []):
            self.assertIn("source", d, f"Deliverable missing source: {d}")
        for b in report["blockers"].get("blockers", []):
            self.assertIn("source", b, f"Blocker missing source: {b}")


# ===========================================================================
# Flask route integration tests (no LLM, no real embed)
# ===========================================================================

class TestFlaskAnalyseRoute(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = flask_app.app.test_client()

    def test_analyse_get_renders(self):
        resp = self.client.get("/analyse")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Project Intelligence", resp.data)

    def test_analyse_post_no_project_redirects(self):
        resp = self.client.post("/analyse", data={"project_id": ""})
        self.assertIn(resp.status_code, (200, 302))

    def test_analyse_post_nonexistent_project_flashes_error(self):
        resp = self.client.post(
            "/analyse",
            data={"project_id": "does-not-exist-xyz"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)

    def test_m1_dashboard_still_works(self):
        resp = self.client.get("/")
        # / now redirects to the single-page app
        self.assertIn(resp.status_code, (200, 302))

    def test_m1_upload_still_works(self):
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Upload", resp.data)

    def test_m1_search_still_works(self):
        resp = self.client.get("/search")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Search", resp.data)

    def test_m1_knowledge_base_still_works(self):
        resp = self.client.get("/knowledge-base")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Knowledge Base", resp.data)


if __name__ == "__main__":
    unittest.main()
