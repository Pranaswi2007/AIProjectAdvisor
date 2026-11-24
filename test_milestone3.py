"""
Regression tests for Milestone 3 components:
  - Documentation Generation Agent (doc_agent)
  - Project Health Scoring Module (health_score)
  - Conversational Q&A Assistant (qa_agent)
  - Orchestrator integration (docs + health_score in report)

All LLM calls are mocked — no live API calls are made.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# Health Scoring Module — pure computation, no LLM
# ===========================================================================

class TestHealthScore(unittest.TestCase):

    def _full_data(self):
        """Return agent data where all dimensions score high.

        Uses multiple items per category and includes out_of_scope to ensure
        the sparse-extraction heuristic and out_of_scope penalty do not fire.
        """
        return (
            {  # scope — rich extraction with 2+ items per category
                "project_goals": [
                    {"goal": "Build app"},
                    {"goal": "Automate workflow"},
                ],
                "scope": [
                    {"item": "Feature A"},
                    {"item": "Feature B"},
                ],
                "out_of_scope": [
                    {"item": "Mobile app"},
                ],
                "deliverables": [
                    {"name": "API", "description": "REST API",
                     "deadline": "2026-12-01", "owner": "Dev"},
                    {"name": "UI", "description": "Web frontend",
                     "deadline": "2026-12-15", "owner": "Designer"},
                ],
                "milestones": [
                    {"name": "Beta", "date": "2026-11-01"},
                    {"name": "Go-Live", "date": "2026-12-01"},
                ],
            },
            {  # blockers — none
                "blockers": [],
            },
            {  # risks — none, on track
                "risks": [],
                "delivery_forecast": {"status": "On Track"},
            },
        )

    def _empty_data(self):
        return ({}, {}, {})

    def _bad_data(self):
        return (
            {  # scope — nothing extracted
            },
            {  # blockers — many open
                "blockers": [{"status": "Open"} for _ in range(10)],
            },
            {  # risks — many critical confirmed
                "risks": [{"severity": "Critical", "status": "Confirmed"} for _ in range(8)],
                "delivery_forecast": {"status": "Delayed"},
            },
        )

    def test_full_data_scores_high(self):
        from agents.health_score import compute
        scope, blockers, risks = self._full_data()
        result = compute(scope, blockers, risks)
        self.assertGreaterEqual(result["overall"], 80)
        self.assertIn(result["grade"], ("A", "B"))
        # With 2+ items per category and out_of_scope present, scope should score
        # at or near 100 (all deliverables fully populated, no gaps flagged)
        self.assertGreaterEqual(result["scope_clarity"], 90)
        self.assertEqual(result["timeline_risk"], 90)
        self.assertEqual(result["blocker_severity"], 100)
        self.assertEqual(result["risk_exposure"], 100)

    def test_empty_data_neutral(self):
        from agents.health_score import compute
        result = compute(*self._empty_data())
        # No blockers/risks = high scores there, but no scope = 0
        # Weighted: 0*0.25 + 50*0.30 + 100*0.25 + 100*0.20 = 60
        self.assertEqual(result["scope_clarity"], 0)
        self.assertEqual(result["timeline_risk"], 50)  # neutral
        self.assertEqual(result["blocker_severity"], 100)  # no blockers = good
        self.assertEqual(result["risk_exposure"], 100)  # no risks = good
        self.assertEqual(result["overall"], 60)

    def test_bad_data_scores_low(self):
        from agents.health_score import compute
        result = compute(*self._bad_data())
        self.assertLess(result["overall"], 30)
        self.assertEqual(result["grade"], "F")
        self.assertEqual(result["scope_clarity"], 0)
        self.assertEqual(result["timeline_risk"], 25)  # delayed
        self.assertEqual(result["blocker_severity"], 0)  # 10 open blockers
        self.assertLess(result["risk_exposure"], 10)

    def test_dimensions_present(self):
        from agents.health_score import compute
        result = compute(*self._full_data())
        self.assertEqual(len(result["dimensions"]), 4)
        for dim in result["dimensions"]:
            self.assertIn("name", dim)
            self.assertIn("score", dim)
            self.assertIn("weight", dim)

    def test_summary_text_present(self):
        from agents.health_score import compute
        result = compute(*self._full_data())
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 10)

    def test_grade_thresholds(self):
        from agents.health_score import _grade
        self.assertEqual(_grade(95), "A")
        self.assertEqual(_grade(85), "B")
        self.assertEqual(_grade(75), "C")
        self.assertEqual(_grade(65), "D")
        self.assertEqual(_grade(55), "F")
        self.assertEqual(_grade(0), "F")

    def test_partial_scope_scores_medium(self):
        from agents.health_score import compute
        result = compute(
            {"project_goals": [{"goal": "X"}], "scope": [{"item": "Y"}],
             "deliverables": [], "milestones": []},
            {"blockers": []},
            {"risks": [], "delivery_forecast": {"status": "On Track"}},
        )
        # goals ✓, scope ✓, deliverables missing (-25), milestones missing (-20),
        # out_of_scope missing but other scope exists (-10) → 100 - 25 - 20 - 10 = 45
        self.assertEqual(result["scope_clarity"], 45)

    def test_resolved_blockers_less_severe(self):
        from agents.health_score import compute
        result = compute(
            {},
            {"blockers": [{"status": "Resolved"} for _ in range(10)]},
            {"risks": [], "delivery_forecast": {"status": "On Track"}},
        )
        # 10 resolved blockers = 10 * 0.2 = 2 effective → score should be high
        self.assertGreater(result["blocker_severity"], 70)

    def test_confirmed_risks_weighted_double(self):
        from agents.health_score import _risk_exposure_score
        # 2 confirmed Critical risks = 2 * 1.0 * 1.5 = 3.0 exposure
        score, _ = _risk_exposure_score({
            "risks": [
                {"severity": "Critical", "status": "Confirmed"},
                {"severity": "Critical", "status": "Confirmed"},
            ]
        })
        # Should be lower than 2 potential Critical risks
        score_potential, _ = _risk_exposure_score({
            "risks": [
                {"severity": "Critical", "status": "Potential"},
                {"severity": "Critical", "status": "Potential"},
            ]
        })
        self.assertLess(score, score_potential)

    def test_never_raises(self):
        from agents.health_score import compute
        try:
            compute(None, None, None)
            compute("bad", "bad", "bad")
        except Exception as exc:
            self.fail(f"compute() raised: {exc}")


# ===========================================================================
# Documentation Generation Agent
# ===========================================================================

class TestDocAgent(unittest.TestCase):
    """Tests for the deterministic doc_agent (no LLM — derived from structured data)."""

    # Structured inputs that exercise all three generators
    SCOPE_DATA = {
        "deliverables": [
            {"name": "Student Portal", "source": "02_Requirements.docx"},
        ],
        "scope": [
            {"item": "Admin dashboard for reporting", "source": "02_Requirements.docx"},
        ],
    }
    BLOCKER_DATA = {
        "blockers": [
            {"blocker_id": "B1", "description": "SAP sandbox not ready",
             "status": "Open", "impact": "Miss deadline",
             "recommended_resolution": "Use mock SAP", "owner": "Priya"},
        ],
        "action_items": [
            {"action_id": "A001", "description": "Escalate to IT", "owner": "Ravi",
             "due_date": "2026-10-16", "priority": "High", "status": "Pending",
             "source": "meeting_notes.txt"},
        ],
    }
    RISK_DATA = {
        "risks": [
            {"risk_id": "R001", "title": "SAP delay", "severity": "High",
             "potential_impact": "Miss deadline", "mitigation": "Use mock",
             "owner": "Priya", "status": "Potential"},
        ],
    }

    def _run(self, scope_data=None, blocker_data=None, risk_data=None):
        from agents.doc_agent import run
        return run(
            "test-proj",
            scope_data=scope_data  if scope_data  is not None else self.SCOPE_DATA,
            blocker_data=blocker_data if blocker_data is not None else self.BLOCKER_DATA,
            risk_data=risk_data   if risk_data   is not None else self.RISK_DATA,
        )

    def test_returns_required_keys(self):
        result = self._run()
        for key in ("user_stories", "risk_register", "action_item_list",
                    "missing_information", "error"):
            self.assertIn(key, result)

    def test_user_stories_normalized(self):
        """Each user story must have id, role, want, benefit, source."""
        result = self._run()
        self.assertGreater(len(result["user_stories"]), 0,
                           "Expected at least one user story from SCOPE_DATA")
        us = result["user_stories"][0]
        for field in ("id", "role", "want", "benefit", "source"):
            self.assertIn(field, us)

    def test_risk_register_normalized(self):
        """Each risk register entry must have all required fields."""
        result = self._run()
        self.assertGreater(len(result["risk_register"]), 0,
                           "Expected at least one risk from RISK_DATA")
        r = result["risk_register"][0]
        for field in ("risk_id", "title", "severity", "impact",
                      "mitigation", "owner", "status"):
            self.assertIn(field, r)
        self.assertEqual(r["severity"], "High")

    def test_action_items_normalized(self):
        """Each action item must have all required fields."""
        result = self._run()
        self.assertGreater(len(result["action_item_list"]), 0,
                           "Expected at least one action item from BLOCKER_DATA")
        a = result["action_item_list"][0]
        for field in ("action_id", "description", "owner", "due_date",
                      "priority", "status", "source"):
            self.assertIn(field, a)

    def test_error_none_on_success(self):
        result = self._run()
        self.assertIsNone(result["error"])

    def test_llm_failure_returns_missing_info(self):
        """doc_agent is deterministic — it never calls LLM, so never fails on LLM errors.
        Running with no input data should return empty lists but no error."""
        result = self._run(scope_data={}, blocker_data={}, risk_data={})
        self.assertIsNone(result["error"])

    def test_empty_context_still_runs(self):
        """Doc agent handles empty/None inputs gracefully."""
        result = self._run(scope_data={}, blocker_data={}, risk_data={})
        self.assertIsNotNone(result)
        self.assertIn("user_stories", result)

    def test_never_raises(self):
        from agents.doc_agent import run
        try:
            run("test-proj", scope_data=None, blocker_data=None, risk_data=None)
        except Exception as exc:
            self.fail(f"run() raised: {exc}")


# ===========================================================================
# Conversational Q&A Assistant
# ===========================================================================

class TestQAAgent(unittest.TestCase):

    QA_RESPONSE = json.dumps({
        "answer": "The project is on track according to the sprint status document.",
        "confidence": "High",
        "sources_cited": ["03_Sprint_Status.docx"],
    })

    def test_answer_with_context(self):
        from agents.qa_agent import answer
        context = "[Source: 03_Sprint_Status.docx | Chunk #0]\nSprint is on track. All tasks completed."
        with patch("agents.qa_agent.build_context", return_value=context):
            with patch("agents.qa_agent.call_llm_json",
                       return_value=json.loads(self.QA_RESPONSE)):
                result = answer("test-proj", "Are we on track?")
        self.assertIn("on track", result["answer"].lower())
        self.assertEqual(result["confidence"], "High")
        self.assertGreater(len(result["sources"]), 0)
        self.assertIsNone(result["error"])

    def test_empty_context_returns_message(self):
        from agents.qa_agent import answer
        with patch("agents.qa_agent.build_context", return_value=""):
            result = answer("test-proj", "What are our risks?")
        self.assertIn("No documents", result["answer"])
        self.assertEqual(result["confidence"], "None")

    def test_empty_question_returns_error(self):
        from agents.qa_agent import answer
        result = answer("test-proj", "")
        self.assertIsNotNone(result["error"])
        result2 = answer("test-proj", "   ")
        self.assertIsNotNone(result2["error"])

    def test_llm_failure_returns_fallback(self):
        from agents.qa_agent import answer
        from agents.llm_client import LLMError
        context = "[Source: test.txt | Chunk #0]\nSome content here."
        with patch("agents.qa_agent.build_context", return_value=context):
            with patch("agents.qa_agent.call_llm_json",
                       side_effect=LLMError("API down")):
                result = answer("test-proj", "Are we on track?")
        self.assertIn("could not generate", result["answer"].lower())
        self.assertEqual(result["confidence"], "Low")
        self.assertGreater(len(result["sources"]), 0)

    def test_sources_extracted_from_context(self):
        from agents.qa_agent import answer, _extract_sources
        context = (
            "[Source: fileA.docx | Chunk #0]\nText A\n\n---\n\n"
            "[Source: fileB.txt | Chunk #3]\nText B"
        )
        sources = _extract_sources(context)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["source"], "fileA.docx")
        self.assertEqual(sources[0]["chunk"], 0)
        self.assertEqual(sources[1]["source"], "fileB.txt")
        self.assertEqual(sources[1]["chunk"], 3)

    def test_confidence_validated(self):
        from agents.qa_agent import answer
        bad_conf = json.dumps({"answer": "test", "confidence": "Invalid", "sources_cited": []})
        with patch("agents.qa_agent.build_context", return_value="ctx"):
            with patch("agents.qa_agent.call_llm_json",
                       return_value=json.loads(bad_conf)):
                result = answer("test-proj", "question?")
        # Invalid confidence → defaults to Low
        self.assertEqual(result["confidence"], "Low")

    def test_never_raises(self):
        from agents.qa_agent import answer
        with patch("agents.qa_agent.build_context", side_effect=RuntimeError("crash")):
            try:
                answer("test-proj", "question?")
            except Exception as exc:
                self.fail(f"answer() raised: {exc}")


# ===========================================================================
# Orchestrator integration — docs + health_score in report
# ===========================================================================

class TestOrchestratorMilestone3(unittest.TestCase):

    def setUp(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def test_report_has_docs_and_health_score(self):
        # Patch combined_agent.run — the orchestrator now uses a single call
        scope_data = {
            "project_goals": [{"goal": "Goal A", "source": ""}],
            "scope": [{"item": "Scope A", "source": ""}],
            "out_of_scope": [],
            "deliverables": [{"name": "D1", "description": "desc", "deadline": "2026-12-01",
                              "owner": "X", "dependencies": [], "acceptance_criteria": [], "source": "", "inferred": False}],
            "milestones": [{"name": "M1", "date": "2026-12-01", "source": ""}],
            "missing_information": [], "conflicts": [], "extraction_gaps": [], "error": None,
        }
        blocker_data = {
            "blockers": [], "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        risk_data = {
            "risks": [],
            "delivery_forecast": {"status": "On Track", "estimated_completion_date": None,
                                  "delayed_milestones": [], "assumptions": [],
                                  "confidence": "High", "forecast_method": "none"},
            "dependency_gaps": [], "missing_information": [], "error": None,
        }
        with patch("agents.combined_agent.run", return_value=(scope_data, blocker_data, risk_data)):
            from agents.orchestrator import run
            report = run("test-m3-proj")

        self.assertIn("docs", report)
        self.assertIn("health_score", report)
        self.assertIn("overall", report["health_score"])
        self.assertIn("grade", report["health_score"])
        self.assertIn("dimensions", report["health_score"])
        self.assertIn("agent_status", report)
        self.assertIn("docs", report["agent_status"])
        self.assertEqual(report["agent_status"]["docs"], "ok")

    def test_health_score_computed_from_agent_outputs(self):
        """Health score must reflect actual agent output quality."""
        scope_data = {
            "project_goals": [], "scope": [], "out_of_scope": [],
            "deliverables": [], "milestones": [],
            "missing_information": [], "conflicts": [], "extraction_gaps": [], "error": None,
        }
        blocker_data = {
            "blockers": [{"blocker_id": "B1", "description": "x", "status": "Open"}],
            "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        risk_data = {
            "risks": [{"risk_id": "R1", "title": "bad", "severity": "Critical",
                       "status": "Confirmed", "potential_impact": "", "mitigation": "",
                       "owner": None, "source": ""}],
            "delivery_forecast": {"status": "Delayed", "estimated_completion_date": None,
                                  "delayed_milestones": [], "assumptions": [],
                                  "confidence": "Low", "forecast_method": "none"},
            "dependency_gaps": [], "missing_information": [], "error": None,
        }
        with patch("agents.combined_agent.run", return_value=(scope_data, blocker_data, risk_data)):
            from agents.orchestrator import run
            report = run("test-bad-proj")

        hs = report["health_score"]
        self.assertLess(hs["overall"], 50)
        self.assertEqual(hs["grade"], "F")
        self.assertLessEqual(hs["scope_clarity"], 10)  # no scope data → near-zero
        self.assertEqual(hs["timeline_risk"], 25)  # delayed

    def test_orchestrator_never_raises_with_new_agents(self):
        with patch("agents.combined_agent.run", side_effect=RuntimeError("boom")):
            from agents.orchestrator import run
            try:
                report = run("crash-m3-proj")
            except Exception as exc:
                self.fail(f"orchestrator.run() raised: {exc}")


# ===========================================================================
# Health Score — sub-field completeness & calculation evidence
# ===========================================================================

class TestHealthScoreSubFields(unittest.TestCase):

    def test_deliverables_missing_description_penalized(self):
        """Deliverables without description should reduce scope clarity below 100."""
        from agents.health_score import compute
        result = compute(
            {"project_goals": [{"goal": "G"}], "scope": [{"item": "S"}],
             "deliverables": [{"name": "API", "deadline": "2026-12-01", "owner": "Dev"}],
             "milestones": [{"name": "M1", "date": "2026-12-01"}]},
            {"blockers": []},
            {"risks": [], "delivery_forecast": {"status": "On Track"}},
        )
        self.assertLess(result["scope_clarity"], 100,
                        "Deliverable missing description should reduce score below 100")

    def test_no_100_when_subfields_missing(self):
        """Even with all 4 categories present, missing sub-fields must prevent 100."""
        from agents.health_score import compute
        result = compute(
            {"project_goals": [{"goal": "G"}], "scope": [{"item": "S"}],
             "deliverables": [{"name": "API"}],  # no desc, deadline, or owner
             "milestones": [{"name": "M1"}]},
            {},
            {"risks": [], "delivery_forecast": {"status": "On Track"}},
        )
        self.assertLess(result["scope_clarity"], 100)

    def test_dimensions_have_details(self):
        """Each dimension must include a 'details' string explaining the calculation."""
        from agents.health_score import compute
        result = compute(
            {"project_goals": [{"goal": "G"}], "scope": [{"item": "S"}],
             "deliverables": [{"name": "D", "description": "x", "deadline": "2026-12-01", "owner": "O"}],
             "milestones": [{"name": "M", "date": "2026-12-01"}]},
            {"blockers": []},
            {"risks": [], "delivery_forecast": {"status": "On Track"}},
        )
        for dim in result["dimensions"]:
            self.assertIn("details", dim, f"Dimension '{dim['name']}' missing details")
            self.assertIsInstance(dim["details"], str)
            self.assertGreater(len(dim["details"]), 5)

    def test_scope_clarity_zero_with_no_scope_data(self):
        from agents.health_score import compute
        result = compute({}, {}, {})
        self.assertEqual(result["scope_clarity"], 0)

    def test_weighted_math_correct(self):
        """Overall must equal the weighted sum of dimension scores."""
        from agents.health_score import compute, _WEIGHTS
        scope = {"project_goals": [{"goal": "G"}], "scope": [{"item": "S"}],
                 "deliverables": [{"name": "D", "description": "x", "deadline": "2026-12-01", "owner": "O"}],
                 "milestones": [{"name": "M", "date": "2026-12-01"}]}
        blockers = {"blockers": [{"status": "Open"}]}
        risks = {"risks": [{"severity": "High", "status": "Potential"}],
                 "delivery_forecast": {"status": "At Risk"}}
        result = compute(scope, blockers, risks)
        expected = round(
            result["scope_clarity"] * _WEIGHTS["scope_clarity"]
            + result["timeline_risk"] * _WEIGHTS["timeline_risk"]
            + result["blocker_severity"] * _WEIGHTS["blocker_severity"]
            + result["risk_exposure"] * _WEIGHTS["risk_exposure"]
        )
        self.assertEqual(result["overall"], expected,
                         f"Overall {result['overall']} != weighted sum {expected}")


# ===========================================================================
# Orchestrator — partial status when extraction is insufficient
# ===========================================================================

class TestOrchestratorPartialStatus(unittest.TestCase):

    def setUp(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        import sys
        for mod in ("agents.orchestrator", "agents.combined_agent"):
            sys.modules.pop(mod, None)

    def test_scope_empty_lists_gets_partial(self):
        """Scope agent returns empty arrays → agent_status should be 'partial'."""
        scope_data = {
            "project_goals": [], "scope": [], "out_of_scope": [],
            "deliverables": [], "milestones": [],
            "missing_information": [], "conflicts": [], "extraction_gaps": [], "error": None,
        }
        blocker_data = {
            "blockers": [{"blocker_id": "B1", "description": "x", "status": "Open"}],
            "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        risk_data = {
            "risks": [{"risk_id": "R1", "title": "x", "severity": "High", "status": "Potential",
                       "potential_impact": "", "mitigation": "", "owner": None, "source": ""}],
            "delivery_forecast": {"status": "At Risk", "estimated_completion_date": None,
                                  "delayed_milestones": [], "assumptions": [],
                                  "confidence": "Low", "forecast_method": "none"},
            "dependency_gaps": [], "missing_information": [], "error": None,
        }
        with patch("agents.combined_agent.run", return_value=(scope_data, blocker_data, risk_data)):
            from agents.orchestrator import run
            report = run("test-partial-proj")
        self.assertEqual(report["agent_status"]["scope"], "partial",
                         "Scope with all empty lists should be 'partial' not 'ok'")
        self.assertEqual(report["agent_status"]["blockers"], "ok")
        self.assertEqual(report["agent_status"]["risks"], "ok")

    def test_all_empty_gets_partial_for_scope_and_docs(self):
        # doc_agent is deterministic — empty inputs produce empty lists → "partial"
        scope_data = {
            "project_goals": [], "scope": [], "out_of_scope": [],
            "deliverables": [], "milestones": [],
            "missing_information": [], "conflicts": [], "extraction_gaps": [], "error": None,
        }
        blocker_data = {
            "blockers": [], "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
            "extraction_source": "llm", "error": None,
        }
        risk_data = {
            "risks": [],
            "delivery_forecast": {"status": "Insufficient data", "estimated_completion_date": None,
                                  "delayed_milestones": [], "assumptions": [],
                                  "confidence": "Low", "forecast_method": "none"},
            "dependency_gaps": [], "missing_information": [], "error": None,
        }
        with patch("agents.combined_agent.run", return_value=(scope_data, blocker_data, risk_data)):
            from agents.orchestrator import run
            report = run("test-all-empty")
        self.assertEqual(report["agent_status"]["scope"], "partial")
        self.assertEqual(report["agent_status"]["blockers"], "partial")
        self.assertEqual(report["agent_status"]["risks"], "partial")
        self.assertEqual(report["agent_status"]["docs"], "partial")


# ===========================================================================
# Scope Agent — extraction_gaps tracking
# ===========================================================================

class TestScopeExtractionGaps(unittest.TestCase):

    def test_extraction_gaps_populated_for_missing_subfields(self):
        """_merge should track deliverables missing description, deadline, owner."""
        from agents.scope_agent import _merge, EMPTY_OUTPUT
        raw = {"deliverables": [{"name": "API"}], "project_goals": ["goal"]}
        result = _merge(EMPTY_OUTPUT.copy(), raw)
        self.assertIn("extraction_gaps", result)
        gaps = result["extraction_gaps"]
        self.assertGreater(len(gaps), 0, "Should have gaps for missing description/deadline/owner")

    def test_no_gaps_when_all_fields_present(self):
        from agents.scope_agent import _merge, EMPTY_OUTPUT
        raw = {"deliverables": [{"name": "API", "description": "REST API",
                                  "deadline": "2026-12-01", "owner": "Dev"}]}
        result = _merge(EMPTY_OUTPUT.copy(), raw)
        self.assertEqual(result["extraction_gaps"], [])

    def test_extraction_gaps_in_empty_output(self):
        from agents.scope_agent import EMPTY_OUTPUT
        self.assertIn("extraction_gaps", EMPTY_OUTPUT)


# ===========================================================================
# Risk Agent — new fields (probability, impact, trigger, owner)
# ===========================================================================

class TestRiskNewFields(unittest.TestCase):

    def test_risk_merge_includes_new_fields(self):
        """_merge should normalize probability, impact, trigger, owner."""
        from agents.risk_agent import _merge, EMPTY_OUTPUT
        rb = {"status": "Insufficient data", "estimated_completion_date": None,
              "delayed_milestones": [], "assumptions": [],
              "confidence": "Low", "forecast_method": "none"}
        raw = {"risks": [{"risk_id": "R1", "title": "test", "probability": "High",
                          "impact": "Miss deadline", "trigger": "Sandbox not ready by Oct 20",
                          "owner": "Priya", "status": "Confirmed", "severity": "Critical"}]}
        result = _merge(EMPTY_OUTPUT.copy(), raw, rb)
        r = result["risks"][0]
        self.assertEqual(r["probability"], "High")
        self.assertEqual(r["impact"], "Miss deadline")
        self.assertEqual(r["trigger"], "Sandbox not ready by Oct 20")
        self.assertEqual(r["owner"], "Priya")

    def test_risk_unknown_status_default(self):
        """When LLM doesn't return a status, default should be 'Unknown' not 'Potential'."""
        from agents.risk_agent import _merge, EMPTY_OUTPUT
        rb = {"status": "Insufficient data", "estimated_completion_date": None,
              "delayed_milestones": [], "assumptions": [],
              "confidence": "Low", "forecast_method": "none"}
        raw = {"risks": [{"risk_id": "R1", "title": "test"}]}
        result = _merge(EMPTY_OUTPUT.copy(), raw, rb)
        self.assertEqual(result["risks"][0]["status"], "Unknown")


# ===========================================================================
# Blocker Agent — confirmed vs pending decision separation
# ===========================================================================

class TestBlockerDecisionSeparation(unittest.TestCase):

    CONFIRMED_RESPONSE = json.dumps({
        "blockers": [],
        "action_items": [],
        "pending_decisions": [{"id": "D001", "description": "Use mock library if SAP not ready by Oct 20",
                                "owner": "Priya", "due_date": "2026-10-20", "source": "meeting.txt"}],
        "unresolved_issues": [],
        "missing_information": [],
    })

    def test_pending_decision_extracted_with_source(self):
        from agents.blocker_agent import run
        from agents.llm_client import clear_llm_result_cache
        clear_llm_result_cache("test-proj")
        with patch("agents.blocker_agent.build_context", return_value="ctx"):
            with patch("agents.blocker_agent.call_llm_json",
                       return_value=json.loads(self.CONFIRMED_RESPONSE)):
                result = run("test-proj")
        self.assertEqual(len(result["pending_decisions"]), 1)
        d = result["pending_decisions"][0]
        self.assertEqual(d["decision_id"], "D001")
        self.assertEqual(d["owner"], "Priya")
        self.assertEqual(d["source"], "meeting.txt")

    def test_empty_blockers_with_actions_still_ok(self):
        """No blockers but action items exist → status should be 'ok'."""
        resp = json.dumps({
            "blockers": [],
            "action_items": [{"id": "A001", "description": "Fix schema", "owner": "Ravi",
                              "due_date": "2026-10-16", "priority": "High", "status": "Pending"}],
            "pending_decisions": [], "unresolved_issues": [], "missing_information": [],
        })
        from agents.blocker_agent import run
        with patch("agents.blocker_agent.build_context", return_value="ctx"):
            with patch("agents.blocker_agent.call_llm_json", return_value=json.loads(resp)):
                result = run("test-proj")
        self.assertEqual(len(result["action_items"]), 1)
        self.assertIsNone(result["error"])


# ===========================================================================
# Health score — incomplete data doesn't get 100
# ===========================================================================

class TestHealthScoreIncompleteData(unittest.TestCase):

    def test_deliverable_missing_owner_not_100(self):
        from agents.health_score import compute
        result = compute(
            {"project_goals": [{"goal": "G"}], "scope": [{"item": "S"}],
             "deliverables": [{"name": "API", "description": "x", "deadline": "2026-12-01"}],
             # no owner on deliverable
             "milestones": [{"name": "M", "date": "2026-12-01"}]},
            {"blockers": []},
            {"risks": [], "delivery_forecast": {"status": "On Track"}},
        )
        self.assertLess(result["scope_clarity"], 100,
                        "Missing owner should prevent 100 score")

    def test_deliverable_missing_deadline_not_100(self):
        from agents.health_score import compute
        result = compute(
            {"project_goals": [{"goal": "G"}], "scope": [{"item": "S"}],
             "deliverables": [{"name": "API", "description": "x", "owner": "Dev"}],
             "milestones": [{"name": "M", "date": "2026-12-01"}]},
            {"blockers": []},
            {"risks": [], "delivery_forecast": {"status": "On Track"}},
        )
        self.assertLess(result["scope_clarity"], 100)


if __name__ == "__main__":
    unittest.main()
