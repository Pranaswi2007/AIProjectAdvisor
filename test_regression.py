"""
Regression tests — targeted coverage for bugs fixed in the audit
================================================================
Covers:
  1. /ask Flask route (QA conversational assistant)
  2. Orchestrator rule-based delivery forecast fallback
  3. Context starvation fix — OOS and blockers each get their own retrieval
  4. Health score sparse-extraction heuristic
  5. Health score out_of_scope penalty

All LLM calls are mocked — no live Groq API quota is consumed.

Run:
    venv\\Scripts\\python.exe -m pytest tests/test_regression.py -v
"""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# 1. /ask Flask route — conversational Q&A
# ===========================================================================

class TestAskRoute(unittest.TestCase):
    """Flask /ask POST endpoint tests (no live LLM)."""

    @classmethod
    def setUpClass(cls):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def _post(self, project_id: str, question: str):
        return self.client.post(
            "/ask",
            data={"project_id": project_id, "question": question},
        )

    def test_missing_project_returns_400(self):
        resp = self._post("", "What are the risks?")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    def test_missing_question_returns_400(self):
        resp = self._post("test-proj", "")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    def test_nonexistent_project_returns_400(self):
        resp = self._post("does-not-exist-xyz-404", "What are the risks?")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    def test_valid_request_returns_answer_json(self):
        """With mocked RAG + LLM, /ask should return answer JSON."""
        qa_reply = json.dumps({
            "answer": "The project is on track with no open blockers.",
            "confidence": "High",
            "sources_cited": ["sprint_status.txt"],
        })
        with patch("agents.qa_agent.build_context",
                   return_value="[Source: sprint_status.txt | Chunk #0]\nProject on track."), \
             patch("agents.qa_agent.call_llm_json",
                   return_value=json.loads(qa_reply)), \
             patch("app.get_project_stats",
                   return_value={"chunk_count": 5, "document_count": 1, "sources": {}}):
            resp = self._post("test-ask-proj", "Are we on track?")

        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("on track", data["answer"].lower())

    def test_llm_failure_returns_fallback_not_500(self):
        """LLM errors should produce a graceful 200 with error field, not a 500."""
        from agents.llm_client import LLMError
        with patch("agents.qa_agent.build_context",
                   return_value="[Source: f.txt | Chunk #0]\nSome content."), \
             patch("agents.qa_agent.call_llm_json",
                   side_effect=LLMError("API down")), \
             patch("app.get_project_stats",
                   return_value={"chunk_count": 3, "document_count": 1, "sources": {}}):
            resp = self._post("test-ask-proj", "Are we on track?")

        # /ask catches LLMError internally and returns 200 with an answer
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("answer", data)
        # Should not be an empty answer
        self.assertGreater(len(data["answer"]), 10)

    def test_confidence_field_in_response(self):
        """Response must always contain a confidence field."""
        qa_reply = json.dumps({
            "answer": "Some answer.",
            "confidence": "Medium",
            "sources_cited": [],
        })
        with patch("agents.qa_agent.build_context",
                   return_value="[Source: f.txt | Chunk #0]\nContent."), \
             patch("agents.qa_agent.call_llm_json",
                   return_value=json.loads(qa_reply)), \
             patch("app.get_project_stats",
                   return_value={"chunk_count": 2, "document_count": 1, "sources": {}}):
            resp = self._post("test-proj", "What is the status?")

        data = json.loads(resp.data)
        self.assertIn("confidence", data)
        self.assertIn(data["confidence"], ("High", "Medium", "Low", "None"))

    def test_sources_field_in_response(self):
        """Response must always contain a sources field (list)."""
        qa_reply = json.dumps({
            "answer": "Some answer.",
            "confidence": "Low",
            "sources_cited": ["notes.txt"],
        })
        with patch("agents.qa_agent.build_context",
                   return_value="[Source: notes.txt | Chunk #0]\nMeeting notes."), \
             patch("agents.qa_agent.call_llm_json",
                   return_value=json.loads(qa_reply)), \
             patch("app.get_project_stats",
                   return_value={"chunk_count": 2, "document_count": 1, "sources": {}}):
            resp = self._post("test-proj", "What happened in the last meeting?")

        data = json.loads(resp.data)
        self.assertIn("sources", data)
        self.assertIsInstance(data["sources"], list)


# ===========================================================================
# 2. Orchestrator rule-based delivery forecast fallback
# ===========================================================================

class TestOrchestratorForecastFallback(unittest.TestCase):
    """Delivery forecast should not be 'Insufficient data' when scope has dated
    milestones, even if the combined LLM call returns no forecast."""

    def setUp(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def _make_scope(self, milestone_date: str = "2026-12-31") -> dict:
        return {
            "project_goals": [{"goal": "Build system", "source": "proposal.txt"}],
            "scope":         [{"item": "GPS module", "source": "proposal.txt"}],
            "out_of_scope":  [],
            "deliverables":  [{"name": "App", "description": "Mobile app",
                               "deadline": milestone_date, "owner": "Dev",
                               "dependencies": [], "acceptance_criteria": [],
                               "source": "proposal.txt", "inferred": False}],
            "milestones":    [{"name": "Go-Live", "date": milestone_date,
                               "source": "proposal.txt"}],
            "missing_information": [], "conflicts": [], "extraction_gaps": [],
            "error": None,
        }

    def _make_risk_insufficient(self) -> dict:
        return {
            "risks": [],
            "delivery_forecast": {
                "status": "Insufficient data",
                "estimated_completion_date": None,
                "delayed_milestones": [], "assumptions": [],
                "confidence": "Low", "forecast_method": "none",
            },
            "dependency_gaps": [], "missing_information": [], "error": None,
        }

    def test_fallback_fires_when_llm_returns_insufficient(self):
        """When combined agent returns Insufficient data but scope has future milestones,
        the orchestrator must replace the forecast using rule-based logic."""
        scope = self._make_scope("2099-12-31")   # far future → On Track
        blockers = {
            "blockers": [], "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        with patch("agents.combined_agent.run",
                   return_value=(scope, blockers, self._make_risk_insufficient())):
            from agents.orchestrator import run
            report = run("test-fallback-proj")

        fc = report["risks"]["delivery_forecast"]
        self.assertNotEqual(fc["status"], "Insufficient data",
                            "Forecast should not be Insufficient data when dated milestones exist")
        self.assertIn(fc["status"], ("On Track", "At Risk", "Delayed"))
        self.assertEqual(fc["forecast_method"], "rule-based")

    def test_fallback_does_not_fire_when_llm_has_forecast(self):
        """When the combined agent already has a valid forecast, the orchestrator
        enriches it with rule-based dates (hybrid) but keeps the LLM's status."""
        scope = self._make_scope("2099-12-31")
        blockers = {
            "blockers": [], "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        risk_with_forecast = {
            "risks": [],
            "delivery_forecast": {
                "status": "At Risk",
                "estimated_completion_date": "2099-12-31",
                "delayed_milestones": ["Sprint 3"],
                "assumptions": ["Based on LLM analysis"],
                "confidence": "Medium",
                "forecast_method": "qualitative",
            },
            "dependency_gaps": [], "missing_information": [], "error": None,
        }
        with patch("agents.combined_agent.run",
                   return_value=(scope, blockers, risk_with_forecast)):
            from agents.orchestrator import run
            report = run("test-no-fallback-proj")

        fc = report["risks"]["delivery_forecast"]
        self.assertEqual(fc["status"], "At Risk",
                         "LLM forecast 'At Risk' should be preserved, not overwritten")
        # With the always-merge logic, forecast_method becomes "hybrid"
        self.assertEqual(fc["forecast_method"], "hybrid")

    def test_fallback_delayed_when_all_milestones_overdue(self):
        """When all milestones have past dates (and are not marked complete),
        rule-based forecast should return Delayed (no upcoming milestones)."""
        scope = self._make_scope("2020-01-01")   # past date → overdue
        blockers = {
            "blockers": [], "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        with patch("agents.combined_agent.run",
                   return_value=(scope, blockers, self._make_risk_insufficient())):
            from agents.orchestrator import run
            report = run("test-delayed-proj")

        fc = report["risks"]["delivery_forecast"]
        self.assertIn(fc["status"], ("Delayed", "At Risk"),
                      "Past-due milestones should give Delayed or At Risk status")

    def test_no_dates_in_scope_keeps_insufficient(self):
        """When scope has no milestones or deliverable dates, the fallback
        should leave the forecast as Insufficient data."""
        scope = {
            "project_goals": [{"goal": "Build it", "source": ""}],
            "scope": [], "out_of_scope": [], "milestones": [],
            "deliverables": [{"name": "App", "description": "x",
                              "deadline": None, "owner": None,
                              "dependencies": [], "acceptance_criteria": [],
                              "source": "", "inferred": False}],
            "missing_information": [], "conflicts": [], "extraction_gaps": [],
            "error": None,
        }
        blockers = {
            "blockers": [], "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        with patch("agents.combined_agent.run",
                   return_value=(scope, blockers, self._make_risk_insufficient())):
            from agents.orchestrator import run
            report = run("test-no-dates-proj")

        fc = report["risks"]["delivery_forecast"]
        self.assertEqual(fc["status"], "Insufficient data",
                         "With no dates in scope, forecast should remain Insufficient data")


# ===========================================================================
# 3. Context starvation regression — OOS and blockers get independent budgets
# ===========================================================================

class TestContextStarvationFix(unittest.TestCase):
    """Verify that build_context is called separately for each section,
    so OOS and blocker queries cannot be starved by the scope query."""

    def test_four_independent_build_context_calls(self):
        """combined_agent.run() must call build_context exactly 4 times —
        once each for scope, OOS, blockers, and risks."""
        llm_response = {
            "scope": {
                "project_goals": [{"goal": "G", "source": "f.txt"}],
                "scope": [{"item": "S", "source": "f.txt"}],
                "out_of_scope": [{"item": "X", "source": "f.txt"}],
                "deliverables": [], "milestones": [], "missing_information": [],
            },
            "blockers": {
                "blockers": [], "action_items": [], "pending_decisions": [],
                "unresolved_issues": [], "missing_information": [],
            },
            "risks": {
                "risks": [],
                "delivery_forecast": {
                    "status": "On Track", "estimated_completion_date": None,
                    "delayed_milestones": [], "confidence": "Low",
                    "forecast_method": "qualitative",
                },
                "dependency_gaps": [], "missing_information": [],
            },
        }
        with patch("agents.combined_agent.build_context",
                   return_value="[Source: f.txt | Chunk #0]\nSome content.") as mock_ctx, \
             patch("agents.combined_agent.call_llm_json", return_value=llm_response), \
             patch("agents.combined_agent.get_cached_llm_result", return_value=None), \
             patch("agents.combined_agent.set_cached_llm_result"):
            from agents.combined_agent import run
            run("test-starvation-proj")

        # Must be called 4 times: scope, OOS, blockers, risks
        self.assertEqual(mock_ctx.call_count, 4,
                         f"Expected 4 build_context calls, got {mock_ctx.call_count}")

    def test_each_call_uses_independent_query_set(self):
        """Each of the 4 build_context calls must use a different query set
        so retrieval is semantically targeted per section."""
        from agents.combined_agent import _SCOPE_QUERIES, _OOS_QUERIES, \
            _BLOCKER_QUERIES, _RISK_QUERIES

        # All query sets must be distinct (no two are identical)
        all_query_sets = [
            tuple(_SCOPE_QUERIES),
            tuple(_OOS_QUERIES),
            tuple(_BLOCKER_QUERIES),
            tuple(_RISK_QUERIES),
        ]
        self.assertEqual(len(set(all_query_sets)), 4,
                         "All 4 query sets must be distinct")

    def test_oos_queries_target_exclusion_language(self):
        """The OOS query set must contain words that target out-of-scope language."""
        from agents.combined_agent import _OOS_QUERIES
        combined = " ".join(_OOS_QUERIES).lower()
        for keyword in ("scope", "exclud", "not"):
            self.assertIn(keyword, combined,
                          f"OOS queries should contain '{keyword}' — got: {combined}")

    def test_blocker_queries_target_blocker_language(self):
        """The blocker query set must contain words that target blocker language."""
        from agents.combined_agent import _BLOCKER_QUERIES
        combined = " ".join(_BLOCKER_QUERIES).lower()
        for keyword in ("blocker", "action", "owner"):
            self.assertIn(keyword, combined,
                          f"Blocker queries should contain '{keyword}' — got: {combined}")

    def test_oos_section_label_in_prompt(self):
        """The scope call's user prompt must label the OOS section distinctly
        so the model maps it to out_of_scope[] and not scope[]."""
        captured_prompts = []
        llm_response = {
            "scope": {"project_goals": [], "scope": [], "out_of_scope": [],
                      "deliverables": [], "milestones": [], "missing_information": []},
            "blockers": {"blockers": [], "action_items": [], "pending_decisions": [],
                         "unresolved_issues": [], "missing_information": []},
            "risks": {"risks": [],
                      "delivery_forecast": {"status": "Insufficient data",
                                            "estimated_completion_date": None,
                                            "delayed_milestones": [], "confidence": "Low",
                                            "forecast_method": "none"},
                      "dependency_gaps": [], "missing_information": []},
        }

        def capture_llm(system, user, **kwargs):
            captured_prompts.append(user)
            return llm_response

        with patch("agents.combined_agent.build_context",
                   return_value="[Source: f.txt | Chunk #0]\nSome content."), \
             patch("agents.combined_agent.call_llm_json", side_effect=capture_llm), \
             patch("agents.combined_agent.get_cached_llm_result", return_value=None), \
             patch("agents.combined_agent.set_cached_llm_result"):
            from agents.combined_agent import run
            run("test-label-proj")

        # With 2 calls, check that the FIRST call (scope) contains "OOS"
        self.assertGreater(len(captured_prompts), 0, "No LLM calls captured")
        scope_prompt = captured_prompts[0]  # first call = scope
        self.assertIn("OOS", scope_prompt,
                      "Scope call's user prompt must contain the =OOS= label for the out-of-scope section")


# ===========================================================================
# 4. Health score — sparse extraction heuristic
# ===========================================================================

class TestHealthScoreSparseHeuristic(unittest.TestCase):
    """Verify the sparse-extraction heuristic fires correctly."""

    def _compute(self, scope: dict) -> int:
        from agents.health_score import compute
        return compute(scope, {}, {"risks": [], "delivery_forecast": {"status": "On Track"}})["scope_clarity"]

    def test_single_item_per_category_no_oos_no_gaps_is_sparse(self):
        """All 4 categories with exactly 1 item each, no out_of_scope, no gaps
        → sparse heuristic fires → score < 100."""
        score = self._compute({
            "project_goals":  [{"goal": "G"}],
            "scope":          [{"item": "S"}],
            "out_of_scope":   [],
            "deliverables":   [{"name": "D", "description": "x",
                                "deadline": "2026-12-01", "owner": "O"}],
            "milestones":     [{"name": "M", "date": "2026-12-01"}],
            "missing_information": [],
            "extraction_gaps": [],
        })
        # OOS penalty (-10) + sparse penalty (-15) → score ≤ 75
        self.assertLessEqual(score, 75,
                             f"Single-item sparse extraction should score ≤75; got {score}")

    def test_two_items_per_category_does_not_trigger_sparse(self):
        """When any category has 2+ items, sparse heuristic must NOT fire."""
        score = self._compute({
            "project_goals":  [{"goal": "G1"}, {"goal": "G2"}],
            "scope":          [{"item": "S1"}, {"item": "S2"}],
            "out_of_scope":   [{"item": "X"}],
            "deliverables":   [{"name": "D", "description": "x",
                                "deadline": "2026-12-01", "owner": "O"}],
            "milestones":     [{"name": "M", "date": "2026-12-01"}],
            "missing_information": [],
            "extraction_gaps": [],
        })
        # goals has 2 items → sparse does not fire
        # out_of_scope present → no OOS penalty
        # deliverable has all subfields → no subfield penalty
        # → score should be 100
        self.assertEqual(score, 100,
                         f"Rich extraction with ≥2 items should score 100; got {score}")

    def test_sparse_does_not_fire_when_gaps_flagged(self):
        """If the LLM flagged missing_information, the gap penalty already covers it;
        the sparse heuristic must NOT also fire (avoid double-penalising)."""
        score = self._compute({
            "project_goals":  [{"goal": "G"}],
            "scope":          [{"item": "S"}],
            "out_of_scope":   [],
            "deliverables":   [{"name": "D", "description": "x",
                                "deadline": "2026-12-01", "owner": "O"}],
            "milestones":     [{"name": "M", "date": "2026-12-01"}],
            "missing_information": ["Budget not documented"],
            "extraction_gaps": [],
        })
        # OOS penalty: -10 (scope items exist)
        # Sparse does NOT fire (missing_information is non-empty)
        # Gap penalty: 1 item × 3 = -3
        # → 100 - 10 - 3 = 87
        self.assertEqual(score, 87,
                         f"When gaps are flagged, sparse heuristic should not fire; got {score}")

    def test_sparse_does_not_fire_when_category_empty(self):
        """Sparse heuristic requires ALL FOUR lists to be non-empty.
        If deliverables is empty, all_present is False → heuristic must not fire."""
        score = self._compute({
            "project_goals":  [{"goal": "G"}],
            "scope":          [{"item": "S"}],
            "out_of_scope":   [],
            "deliverables":   [],   # empty
            "milestones":     [{"name": "M", "date": "2026-12-01"}],
            "missing_information": [],
            "extraction_gaps": [],
        })
        # deliverables missing: -25
        # OOS penalty: -10 (goals + scope exist)
        # sparse does NOT fire (deliverables empty → all_present = False)
        # → 100 - 25 - 10 = 65
        self.assertEqual(score, 65,
                         f"Sparse heuristic should not fire with empty deliverables; got {score}")


# ===========================================================================
# 5. Health score — out_of_scope penalty
# ===========================================================================

class TestHealthScoreOOSPenalty(unittest.TestCase):
    """out_of_scope missing when other scope data exists should cost -10."""

    def _compute(self, scope: dict) -> dict:
        from agents.health_score import compute
        return compute(scope, {}, {"risks": [], "delivery_forecast": {"status": "On Track"}})

    def test_missing_oos_with_scope_data_costs_10(self):
        result = self._compute({
            "project_goals":  [{"goal": "G1"}, {"goal": "G2"}],
            "scope":          [{"item": "S1"}, {"item": "S2"}],
            "out_of_scope":   [],   # missing
            "deliverables":   [{"name": "D1", "description": "x",
                                "deadline": "2026-12-01", "owner": "O"},
                               {"name": "D2", "description": "y",
                                "deadline": "2026-11-01", "owner": "P"}],
            "milestones":     [{"name": "M1", "date": "2026-12-01"},
                               {"name": "M2", "date": "2026-11-01"}],
            "missing_information": [],
            "extraction_gaps": [],
        })
        # All present, 2+ items, no subfield gaps, no sparse → 100 - 10 (OOS) = 90
        self.assertEqual(result["scope_clarity"], 90,
                         f"Missing OOS with other scope present should score 90; got {result['scope_clarity']}")
        self.assertIn("out_of_scope", result["dimensions"][0]["details"])

    def test_oos_present_no_penalty(self):
        result = self._compute({
            "project_goals":  [{"goal": "G1"}, {"goal": "G2"}],
            "scope":          [{"item": "S1"}, {"item": "S2"}],
            "out_of_scope":   [{"item": "Payment processing"}],
            "deliverables":   [{"name": "D1", "description": "x",
                                "deadline": "2026-12-01", "owner": "O"},
                               {"name": "D2", "description": "y",
                                "deadline": "2026-11-01", "owner": "P"}],
            "milestones":     [{"name": "M1", "date": "2026-12-01"},
                               {"name": "M2", "date": "2026-11-01"}],
            "missing_information": [],
            "extraction_gaps": [],
        })
        # All present, 2+ items, OOS present → 100
        self.assertEqual(result["scope_clarity"], 100,
                         f"Full scope with OOS should score 100; got {result['scope_clarity']}")

    def test_no_oos_penalty_when_no_other_scope(self):
        """Out-of-scope penalty should NOT fire when goals and scope items are also empty,
        because there is no scope data to compare against."""
        result = self._compute({
            "project_goals":  [],
            "scope":          [],
            "out_of_scope":   [],
            "deliverables":   [],
            "milestones":     [],
            "missing_information": [],
            "extraction_gaps": [],
        })
        # goals -25, scope -20, deliverables -25, milestones -20 = -90 → score 10
        # OOS penalty: goals is [] and scope is [] → NOT applied
        self.assertEqual(result["scope_clarity"], 10,
                         f"Empty scope with no OOS should score 10 (not 0); got {result['scope_clarity']}")


if __name__ == "__main__":
    unittest.main()
