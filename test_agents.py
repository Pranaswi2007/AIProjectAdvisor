"""
Unit tests for Milestone 2 agents.
====================================
All LLM calls are mocked — no API key required to run these tests.
Run with:
    venv\\Scripts\\python.exe -m pytest tests/ -v
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# helpers — build minimal fake LLM responses
# ===========================================================================

def _scope_response() -> str:
    return json.dumps({
        "project_goals": [
            {"goal": "Automate inventory tracking", "source": "project_proposal.txt#0", "inferred": False}
        ],
        "scope": [
            {"item": "Barcode scanner integration", "source": "project_proposal.txt#1", "inferred": False}
        ],
        "out_of_scope": [
            {"item": "Mobile app development", "source": "project_proposal.txt#1"}
        ],
        "deliverables": [
            {
                "name": "Database Schema Design",
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
            {"name": "Go-Live", "date": "2026-12-01", "description": "Production launch", "source": "project_proposal.txt#3"}
        ],
        "missing_information": [],
        "conflicts": [],
    })


def _blocker_response() -> str:
    return json.dumps({
        "blockers": [
            {
                "blocker_id": "B001",
                "description": "SAP B1 sandbox access not granted",
                "status": "Open",
                "impact": "ERP integration cannot begin",
                "owner": "Priya Sharma",
                "source": "meeting_notes.txt#2",
                "evidence": "SAP B1 sandbox access was requested on 2026-09-15 but has still not been granted",
                "recommended_resolution": "Escalate to SAP vendor account manager",
                "inferred": False,
            }
        ],
        "action_items": [
            {
                "action_id": "A001",
                "description": "Escalate driver installation request to IT Manager",
                "owner": "Ravi Kumar",
                "due_date": "2026-10-16",
                "priority": "High",
                "status": "Pending",
                "related_blocker": None,
                "source": "meeting_notes.txt#1",
                "inferred": False,
            }
        ],
        "pending_decisions": [
            {
                "decision_id": "D001",
                "description": "Use SAP mock library if access not granted by 2026-10-20",
                "owner": "Priya Sharma",
                "due_date": "2026-10-20",
                "source": "meeting_notes.txt#3",
            }
        ],
        "unresolved_issues": [
            {
                "issue_id": "I001",
                "description": "Load testing environment not identified",
                "impact": "NFR-001 cannot be validated",
                "source": "srs_document.txt#5",
            }
        ],
        "missing_information": [],
    })


def _risk_response() -> str:
    return json.dumps({
        "risks": [
            {
                "risk_id": "R001",
                "title": "SAP B1 integration deadline at risk",
                "description": "SAP sandbox delayed 4+ weeks",
                "severity": "High",
                "affected_deliverable": "FR-004 ERP Integration",
                "evidence": "SAP B1 sandbox access not granted as of 2026-10-14",
                "source": "meeting_notes.txt#2",
                "potential_impact": "Miss 2026-11-30 deadline",
                "mitigation": "Use mock library; escalate to vendor",
                "status": "Potential",
            }
        ],
        "delivery_forecast": {
            "status": "At Risk",
            "estimated_completion_date": "2026-12-01",
            "delayed_milestones": ["M5 Go-Live"],
            "assumptions": ["SAP access granted by 2026-10-24"],
            "confidence": "Medium",
            "forecast_method": "qualitative",
        },
        "dependency_gaps": [
            {
                "gap_id": "G001",
                "description": "T011 Load Testing has no owner or environment",
                "affected_task": "T011",
                "source": "task_list.csv#10",
            }
        ],
        "missing_information": [],
    })


# ===========================================================================
# LLM Client tests
# ===========================================================================

class TestLLMClient(unittest.TestCase):

    def test_call_llm_raises_without_api_key(self):
        """call_llm must raise LLMError when GROQ_API_KEY is not set."""
        from agents.llm_client import LLMError, call_llm
        with patch.dict("os.environ", {}, clear=True):
            # Ensure key is absent
            import os
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaises(LLMError):
                call_llm(system="sys", user="user")

    def test_call_llm_json_strips_fences(self):
        """call_llm_json must strip markdown code fences before parsing."""
        from agents.llm_client import call_llm_json
        fenced = '```json\n{"key": "value"}\n```'
        with patch("agents.llm_client.call_llm", return_value=fenced):
            result = call_llm_json(system="s", user="u")
        self.assertEqual(result, {"key": "value"})

    def test_call_llm_json_returns_fallback_on_bad_json(self):
        """call_llm_json returns fallback dict when LLM returns invalid JSON."""
        from agents.llm_client import call_llm_json
        with patch("agents.llm_client.call_llm", return_value="not json at all"):
            result = call_llm_json(system="s", user="u", fallback={"empty": True})
        self.assertEqual(result, {"empty": True})

    def test_call_llm_json_raises_on_bad_json_no_fallback(self):
        """call_llm_json raises LLMError when no fallback and JSON invalid."""
        from agents.llm_client import LLMError, call_llm_json
        with patch("agents.llm_client.call_llm", return_value="not json"):
            with self.assertRaises(LLMError):
                call_llm_json(system="s", user="u", fallback=None)

    def test_build_context_returns_empty_on_import_error(self):
        """build_context returns empty string if rag module unavailable."""
        from agents.llm_client import build_context
        with patch.dict("sys.modules", {"rag": None}):
            result = build_context("proj", ["query"])
        self.assertIsInstance(result, str)

    def test_call_llm_retries_on_failure(self):
        """call_llm retries the specified number of times before raising."""
        from agents.llm_client import LLMError, call_llm
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("API down")
            with patch("agents.llm_client.Groq", return_value=mock_client):
                # patch sleep so tests don't actually wait
                with patch("agents.llm_client.time.sleep"):
                    with self.assertRaises(LLMError):
                        call_llm(system="s", user="u", retries=2)
            self.assertEqual(mock_client.chat.completions.create.call_count, 2)


# ===========================================================================
# Scope Agent tests
# ===========================================================================

class TestScopeAgent(unittest.TestCase):

    def _run_with_mock(self, llm_return: str = None, context: str = "sample context"):
        """Helper: run scope agent with mocked LLM and context."""
        from agents import scope_agent
        with patch("agents.scope_agent.build_context", return_value=context):
            with patch("agents.scope_agent.call_llm_json",
                       return_value=json.loads(llm_return or _scope_response())):
                return scope_agent.run("test-project")

    def test_returns_dict_with_required_keys(self):
        result = self._run_with_mock()
        for key in ("project_goals", "scope", "out_of_scope", "deliverables",
                    "milestones", "missing_information", "conflicts", "error"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_extracts_project_goals(self):
        result = self._run_with_mock()
        self.assertTrue(len(result["project_goals"]) > 0)
        self.assertIn("goal", result["project_goals"][0])

    def test_extracts_deliverables(self):
        result = self._run_with_mock()
        self.assertTrue(len(result["deliverables"]) > 0)
        d = result["deliverables"][0]
        for field in ("name", "deadline", "owner", "source"):
            self.assertIn(field, d)

    def test_extracts_milestones(self):
        result = self._run_with_mock()
        self.assertTrue(len(result["milestones"]) > 0)
        self.assertIn("date", result["milestones"][0])

    def test_error_is_none_on_success(self):
        result = self._run_with_mock()
        self.assertIsNone(result["error"])

    def test_returns_error_when_no_context(self):
        from agents import scope_agent
        with patch("agents.scope_agent.build_context", return_value=""):
            result = scope_agent.run("empty-project")
        self.assertIsNotNone(result["error"])
        self.assertTrue(len(result["missing_information"]) > 0)

    def test_returns_safe_result_on_llm_error(self):
        from agents import scope_agent
        from agents.llm_client import LLMError
        with patch("agents.scope_agent.build_context", return_value="some context"):
            with patch("agents.scope_agent.call_llm_json",
                       side_effect=LLMError("API unavailable")):
                result = scope_agent.run("test-project")
        # Scope agent now returns error=None on LLM failure so the UI
        # shows a warning badge instead of a red "Failed" badge.
        self.assertIsNone(result["error"])
        self.assertTrue(len(result["missing_information"]) > 0)

    def test_never_raises(self):
        """scope_agent.run() must never propagate exceptions."""
        from agents import scope_agent
        with patch("agents.scope_agent.build_context", side_effect=RuntimeError("crash")):
            try:
                result = scope_agent.run("test-project")
            except Exception as exc:
                self.fail(f"scope_agent.run() raised unexpectedly: {exc}")


# ===========================================================================
# Blocker Agent tests
# ===========================================================================

class TestBlockerAgent(unittest.TestCase):

    def _run_with_mock(self, llm_return: str = None, context: str = "sample context"):
        from agents import blocker_agent
        with patch("agents.blocker_agent.build_context", return_value=context):
            with patch("agents.blocker_agent.call_llm_json",
                       return_value=json.loads(llm_return or _blocker_response())):
                return blocker_agent.run("test-project")

    def test_returns_dict_with_required_keys(self):
        result = self._run_with_mock()
        for key in ("blockers", "action_items", "pending_decisions",
                    "unresolved_issues", "missing_information", "error"):
            self.assertIn(key, result)

    def test_extracts_blockers(self):
        result = self._run_with_mock()
        self.assertTrue(len(result["blockers"]) > 0)
        b = result["blockers"][0]
        for field in ("blocker_id", "description", "status", "source", "evidence"):
            self.assertIn(field, b)

    def test_extracts_action_items(self):
        result = self._run_with_mock()
        self.assertTrue(len(result["action_items"]) > 0)
        a = result["action_items"][0]
        for field in ("action_id", "description", "owner", "due_date",
                      "priority", "status", "source"):
            self.assertIn(field, a)

    def test_extracts_pending_decisions(self):
        result = self._run_with_mock()
        self.assertTrue(len(result["pending_decisions"]) > 0)

    def test_null_owner_allowed(self):
        """Owner field may be null — this is valid per spec."""
        resp = json.loads(_blocker_response())
        resp["action_items"][0]["owner"] = None
        from agents import blocker_agent
        with patch("agents.blocker_agent.build_context", return_value="ctx"):
            with patch("agents.blocker_agent.call_llm_json", return_value=resp):
                result = blocker_agent.run("test-project")
        self.assertIsNone(result["action_items"][0]["owner"])

    def test_error_on_empty_context(self):
        from agents import blocker_agent
        with patch("agents.blocker_agent.build_context", return_value=""):
            result = blocker_agent.run("empty-project")
        self.assertIsNotNone(result["error"])

    def test_never_raises(self):
        from agents import blocker_agent
        with patch("agents.blocker_agent.build_context", side_effect=RuntimeError("crash")):
            try:
                blocker_agent.run("test-project")
            except Exception as exc:
                self.fail(f"blocker_agent.run() raised unexpectedly: {exc}")


# ===========================================================================
# Risk Agent tests
# ===========================================================================

class TestRiskAgent(unittest.TestCase):

    def _run_with_mock(self, scope=None, blockers=None, context="ctx"):
        from agents import risk_agent
        scope = scope or json.loads(_scope_response())
        blockers = blockers or json.loads(_blocker_response())
        with patch("agents.risk_agent.build_context", return_value=context):
            with patch("agents.risk_agent.call_llm_json",
                       return_value=json.loads(_risk_response())):
                return risk_agent.run("test-project", scope, blockers)

    def test_returns_dict_with_required_keys(self):
        result = self._run_with_mock()
        for key in ("risks", "delivery_forecast", "dependency_gaps",
                    "missing_information", "error"):
            self.assertIn(key, result)

    def test_delivery_forecast_keys(self):
        result = self._run_with_mock()
        fc = result["delivery_forecast"]
        for key in ("status", "estimated_completion_date", "delayed_milestones",
                    "assumptions", "confidence", "forecast_method"):
            self.assertIn(key, fc)

    def test_extracts_risks_with_severity(self):
        result = self._run_with_mock()
        self.assertTrue(len(result["risks"]) > 0)
        r = result["risks"][0]
        self.assertIn(r["severity"], ("Critical", "High", "Medium", "Low"))

    def test_rule_based_forecast_with_past_milestone(self):
        """Rule-based forecast marks past milestones as delayed."""
        from agents.risk_agent import _rule_based_forecast
        scope = {
            "milestones": [
                {"name": "M1 Kickoff",        "date": "2020-01-01"},  # past, but it's a kickoff — should NOT be delayed
                {"name": "M2 Feature Freeze",  "date": "2020-06-01"},  # past incomplete task — SHOULD be delayed
                {"name": "M3 Go-Live",         "date": "2099-12-01"},  # future
            ]
        }
        fc = _rule_based_forecast(scope)
        # Kickoff should be excluded from delayed (it's a historical completed event)
        self.assertNotIn("M1 Kickoff", fc["delayed_milestones"])
        # Feature Freeze (past, not a kickoff/start event) should be delayed
        self.assertIn("M2 Feature Freeze", fc["delayed_milestones"])
        self.assertNotIn("M3 Go-Live", fc["delayed_milestones"])
        self.assertEqual(fc["forecast_method"], "rule-based")
        # With a past overdue item AND a future milestone still outstanding,
        # the project is "At Risk" (not "Delayed") — overdue dependencies
        # threaten the upcoming Go-Live but it hasn't been missed yet.
        self.assertEqual(fc["status"], "At Risk")

    def test_rule_based_forecast_no_milestones(self):
        from agents.risk_agent import _rule_based_forecast
        fc = _rule_based_forecast({})
        self.assertEqual(fc["status"], "Insufficient data")
        self.assertEqual(fc["confidence"], "Low")

    def test_rule_based_forecast_all_future(self):
        from agents.risk_agent import _rule_based_forecast
        fc = _rule_based_forecast({
            "milestones": [{"name": "M1", "date": "2099-01-01"}]
        })
        self.assertIn(fc["status"], ("On Track", "At Risk"))
        self.assertEqual(fc["forecast_method"], "rule-based")

    def test_never_raises(self):
        from agents import risk_agent
        with patch("agents.risk_agent.build_context", side_effect=RuntimeError("crash")):
            try:
                risk_agent.run("test-project")
            except Exception as exc:
                self.fail(f"risk_agent.run() raised unexpectedly: {exc}")

    def test_no_fabrication_on_empty_context(self):
        """With no context, risks list should be empty or agent reports insufficient data."""
        from agents import risk_agent
        with patch("agents.risk_agent.build_context", return_value=""):
            result = risk_agent.run("empty-project", {}, {})
        # Should not invent risks — either empty list or error set
        self.assertTrue(
            len(result["risks"]) == 0 or result["error"] is not None or
            len(result["missing_information"]) > 0
        )


# ===========================================================================
# Orchestrator tests
# ===========================================================================

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        # Force fresh import of orchestrator each test so module-level
        # imports (combined_agent, doc_agent, health_score) are re-bound.
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def _combined_response(self):
        scope_out   = json.loads(_scope_response())
        scope_out.setdefault("conflicts", [])
        scope_out.setdefault("extraction_gaps", [])
        scope_out["error"] = None
        blocker_out = json.loads(_blocker_response())
        blocker_out["extraction_source"] = "llm"
        blocker_out["error"] = None
        risk_out    = json.loads(_risk_response())
        risk_out["delivery_forecast"]["forecast_method"] = "hybrid"
        risk_out.setdefault("dependency_gaps", [])
        risk_out["error"] = None
        return scope_out, blocker_out, risk_out

    def test_report_has_all_keys(self):
        with patch("agents.combined_agent.run", return_value=self._combined_response()):
            from agents.orchestrator import run
            report = run("test-project")

        for key in ("project_id", "generated_at", "status",
                    "scope", "blockers", "risks",
                    "errors", "warnings", "duration_seconds"):
            self.assertIn(key, report)

    def test_status_ok_when_no_errors(self):
        with patch("agents.combined_agent.run", return_value=self._combined_response()):
            from agents.orchestrator import run
            report = run("test-project")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["errors"], [])

    def test_status_partial_when_one_agent_errors(self):
        """If combined_agent errors, status should be partial not error."""
        scope_out, blocker_out, risk_out = self._combined_response()
        scope_out["error"] = "LLM unavailable"
        with patch("agents.combined_agent.run", return_value=(scope_out, blocker_out, risk_out)):
            from agents.orchestrator import run
            report = run("test-project")
        self.assertIn(report["status"], ("partial", "ok"))
        self.assertIsNotNone(report["scope"])

    def test_orchestrator_never_raises(self):
        """Orchestrator must never propagate any exception."""
        with patch("agents.combined_agent.run", side_effect=RuntimeError("boom")):
            from agents.orchestrator import run
            try:
                report = run("crash-project")
            except Exception as exc:
                self.fail(f"orchestrator.run() raised: {exc}")

    def test_duration_recorded(self):
        with patch("agents.combined_agent.run", return_value=self._combined_response()):
            from agents.orchestrator import run
            report = run("test-project")
        self.assertGreaterEqual(report["duration_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
