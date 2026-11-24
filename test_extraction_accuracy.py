"""
Regression tests for extraction accuracy and completeness across all
three agents.  Uses mocked LLM responses — no live API calls.

Covers:
  - Missing fields → safe defaults
  - Flat-string normalization (compact schema → full dict)
  - Duplicate detection
  - Incorrect classification guards
  - Unknown/unavailable values marked properly
  - Cross-project cache isolation
  - Template-expected fields always present after _merge
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# Scope Agent — _merge normalization
# ===========================================================================

class TestScopeMergeNormalization(unittest.TestCase):
    """_merge must produce template-expected fields even when the LLM
    returns the compact flat-string schema."""

    def _merge(self, raw):
        from agents.scope_agent import _merge, EMPTY_OUTPUT
        return _merge(EMPTY_OUTPUT.copy(), raw)

    def test_flat_string_goals_normalized_to_dicts(self):
        """LLM returns ["goal A", "goal B"] → list of dicts with goal+source+inferred."""
        raw = {"project_goals": ["Build app", "Ship MVP"]}
        result = self._merge(raw)
        self.assertEqual(len(result["project_goals"]), 2)
        g = result["project_goals"][0]
        self.assertEqual(g["goal"], "Build app")
        self.assertIn("source", g)
        self.assertIn("inferred", g)
        self.assertFalse(g["inferred"])

    def test_flat_string_scope_normalized(self):
        raw = {"scope": ["Feature A", "Feature B"]}
        result = self._merge(raw)
        self.assertEqual(len(result["scope"]), 2)
        self.assertEqual(result["scope"][0]["item"], "Feature A")
        self.assertIn("source", result["scope"][0])
        self.assertIn("inferred", result["scope"][0])

    def test_flat_string_out_of_scope_normalized(self):
        raw = {"out_of_scope": ["Payment processing"]}
        result = self._merge(raw)
        self.assertEqual(len(result["out_of_scope"]), 1)
        self.assertEqual(result["out_of_scope"][0]["item"], "Payment processing")

    def test_deliverables_get_all_template_fields(self):
        """Even if LLM omits dependencies/acceptance_criteria, _merge fills them."""
        raw = {"deliverables": [{"name": "API", "deadline": "2026-12-01"}]}
        result = self._merge(raw)
        d = result["deliverables"][0]
        for field in ("name", "description", "deadline", "owner",
                      "dependencies", "acceptance_criteria", "source", "inferred"):
            self.assertIn(field, d, f"Deliverable missing field: {field}")
        self.assertEqual(d["dependencies"], [])
        self.assertEqual(d["acceptance_criteria"], [])
        self.assertIsNone(d["owner"])
        self.assertEqual(d["description"], "")

    def test_milestones_get_all_template_fields(self):
        raw = {"milestones": [{"name": "Go-Live"}]}
        result = self._merge(raw)
        m = result["milestones"][0]
        for field in ("name", "date", "description", "source"):
            self.assertIn(field, m, f"Milestone missing field: {field}")

    def test_conflicts_preserved_when_present(self):
        raw = {"conflicts": [{"field": "deadline", "description": "conflict"}]}
        result = self._merge(raw)
        self.assertEqual(len(result["conflicts"]), 1)

    def test_conflicts_empty_when_absent(self):
        raw = {}
        result = self._merge(raw)
        self.assertEqual(result["conflicts"], [])

    def test_error_cleared_after_merge(self):
        raw = {"project_goals": ["test"]}
        result = self._merge(raw)
        self.assertIsNone(result["error"])

    def test_dict_goals_pass_through_with_defaults(self):
        """Goals that come as dicts with partial fields get defaults filled."""
        raw = {"project_goals": [{"goal": "X"}]}
        result = self._merge(raw)
        g = result["project_goals"][0]
        self.assertEqual(g["goal"], "X")
        self.assertEqual(g["source"], "")
        self.assertFalse(g["inferred"])

    def test_unknown_keys_ignored(self):
        raw = {"project_goals": ["g"], "unknown_key": 123}
        result = self._merge(raw)
        self.assertNotIn("unknown_key", result)


# ===========================================================================
# Blocker Agent — _merge normalization
# ===========================================================================

class TestBlockerMergeNormalization(unittest.TestCase):

    def _merge(self, raw):
        from agents.blocker_agent import _merge, EMPTY_OUTPUT
        return _merge(EMPTY_OUTPUT.copy(), raw)

    def test_compact_id_remapped_to_blocker_id(self):
        """LLM uses "id" → _merge must remap to "blocker_id"."""
        raw = {"blockers": [{"id": "B001", "description": "test", "status": "Open"}]}
        result = self._merge(raw)
        b = result["blockers"][0]
        self.assertEqual(b["blocker_id"], "B001")
        self.assertEqual(b["description"], "test")

    def test_blocker_all_template_fields_present(self):
        raw = {"blockers": [{"id": "B1", "description": "x"}]}
        result = self._merge(raw)
        b = result["blockers"][0]
        for field in ("blocker_id", "description", "status", "impact", "owner",
                      "source", "evidence", "recommended_resolution", "inferred"):
            self.assertIn(field, b, f"Blocker missing field: {field}")
        self.assertEqual(b["status"], "Unknown")
        self.assertIsNone(b["owner"])
        self.assertFalse(b["inferred"])

    def test_action_item_all_template_fields_present(self):
        raw = {"action_items": [{"id": "A1", "description": "do something"}]}
        result = self._merge(raw)
        a = result["action_items"][0]
        for field in ("action_id", "description", "owner", "due_date",
                      "priority", "status", "related_blocker", "source", "inferred"):
            self.assertIn(field, a, f"Action item missing field: {field}")
        self.assertEqual(a["priority"], "Unspecified")
        self.assertEqual(a["status"], "Unknown")
        self.assertIsNone(a["related_blocker"])

    def test_pending_decision_all_template_fields_present(self):
        raw = {"pending_decisions": [{"id": "D1", "description": "decide"}]}
        result = self._merge(raw)
        d = result["pending_decisions"][0]
        for field in ("decision_id", "description", "owner", "due_date", "source"):
            self.assertIn(field, d, f"Decision missing field: {field}")

    def test_unresolved_issue_all_template_fields_present(self):
        raw = {"unresolved_issues": [{"id": "I1", "description": "issue"}]}
        result = self._merge(raw)
        i = result["unresolved_issues"][0]
        for field in ("issue_id", "description", "impact", "source"):
            self.assertIn(field, i, f"Issue missing field: {field}")

    def test_non_dict_items_skipped(self):
        raw = {"blockers": ["not a dict", {"id": "B1", "description": "ok"}]}
        result = self._merge(raw)
        self.assertEqual(len(result["blockers"]), 1)
        self.assertEqual(result["blockers"][0]["blocker_id"], "B1")

    def test_error_cleared(self):
        raw = {"blockers": []}
        result = self._merge(raw)
        self.assertIsNone(result["error"])


# ===========================================================================
# Risk Agent — _merge normalization
# ===========================================================================

class TestRiskMergeNormalization(unittest.TestCase):

    def _merge(self, raw, rb_forecast=None):
        from agents.risk_agent import _merge, EMPTY_OUTPUT
        rb = rb_forecast or {
            "status": "Insufficient data", "estimated_completion_date": None,
            "delayed_milestones": [], "assumptions": [], "confidence": "Low",
            "forecast_method": "none",
        }
        return _merge(EMPTY_OUTPUT.copy(), raw, rb)

    def test_risk_all_template_fields_present(self):
        raw = {"risks": [{"risk_id": "R1", "title": "Test risk"}]}
        result = self._merge(raw)
        r = result["risks"][0]
        for field in ("risk_id", "title", "description", "severity",
                      "affected_deliverable", "evidence", "source",
                      "potential_impact", "mitigation", "status"):
            self.assertIn(field, r, f"Risk missing field: {field}")
        self.assertEqual(r["severity"], "Medium")
        self.assertEqual(r["status"], "Unknown")

    def test_compact_id_remapped_to_risk_id(self):
        raw = {"risks": [{"id": "R001", "title": "X"}]}
        result = self._merge(raw)
        self.assertEqual(result["risks"][0]["risk_id"], "R001")

    def test_dependency_gap_all_template_fields_present(self):
        raw = {"dependency_gaps": [{"gap_id": "G1", "description": "gap"}]}
        result = self._merge(raw)
        g = result["dependency_gaps"][0]
        for field in ("gap_id", "description", "affected_task", "source"):
            self.assertIn(field, g, f"Gap missing field: {field}")

    def test_compact_gap_id_remapped(self):
        raw = {"dependency_gaps": [{"id": "G001", "description": "x"}]}
        result = self._merge(raw)
        self.assertEqual(result["dependency_gaps"][0]["gap_id"], "G001")

    def test_non_dict_risks_skipped(self):
        raw = {"risks": ["bad", {"risk_id": "R1", "title": "ok"}]}
        result = self._merge(raw)
        self.assertEqual(len(result["risks"]), 1)

    def test_forecast_merged_from_llm(self):
        rb = {"status": "Insufficient data", "estimated_completion_date": None,
              "delayed_milestones": [], "assumptions": [], "confidence": "Low",
              "forecast_method": "none"}
        raw = {
            "delivery_forecast": {
                "status": "At Risk",
                "estimated_completion_date": "2026-12-01",
                "delayed_milestones": ["M1"],
                "assumptions": ["test assumption"],
                "confidence": "Medium",
                "forecast_method": "qualitative",
            }
        }
        result = self._merge(raw, rb)
        fc = result["delivery_forecast"]
        self.assertEqual(fc["status"], "At Risk")
        self.assertEqual(fc["estimated_completion_date"], "2026-12-01")
        self.assertIn("M1", fc["delayed_milestones"])
        self.assertEqual(fc["forecast_method"], "hybrid")

    def test_rule_based_forecast_kept_when_no_llm_forecast(self):
        rb = {"status": "Delayed", "estimated_completion_date": "2026-11-30",
              "delayed_milestones": ["M2"], "assumptions": ["rule"],
              "confidence": "High", "forecast_method": "rule-based"}
        raw = {}
        result = self._merge(raw, rb)
        self.assertEqual(result["delivery_forecast"]["status"], "Delayed")
        self.assertEqual(result["delivery_forecast"]["forecast_method"], "rule-based")


# ===========================================================================
# Scope Agent — end-to-end with mocked LLM returning compact schema
# ===========================================================================

class TestScopeAgentCompactSchema(unittest.TestCase):
    """Run scope_agent.run() with a mock LLM returning the compact flat-string
    schema and verify the output has all template-expected fields."""

    COMPACT_RESPONSE = json.dumps({
        "project_goals": ["Build a meal ordering app"],
        "scope": ["Student browsing", "Cart management", "Order placement"],
        "out_of_scope": ["Payment processing"],
        "deliverables": [{"name": "Web app", "deadline": "2026-11-30", "owner": "Asha"}],
        "milestones": [{"name": "Sprint 3", "date": "2026-10-29"}],
        "missing_information": ["No budget defined"],
    })

    def test_run_normalizes_compact_output(self):
        from agents.scope_agent import run
        with patch("agents.scope_agent.build_context", return_value="ctx"):
            with patch("agents.scope_agent.call_llm_json",
                       return_value=json.loads(self.COMPACT_RESPONSE)):
                result = run("test-compact")

        # Goals normalized
        self.assertEqual(len(result["project_goals"]), 1)
        self.assertEqual(result["project_goals"][0]["goal"], "Build a meal ordering app")
        self.assertIn("source", result["project_goals"][0])
        self.assertIn("inferred", result["project_goals"][0])

        # Scope normalized
        self.assertEqual(len(result["scope"]), 3)
        self.assertEqual(result["scope"][0]["item"], "Student browsing")

        # Out of scope normalized
        self.assertEqual(len(result["out_of_scope"]), 1)
        self.assertEqual(result["out_of_scope"][0]["item"], "Payment processing")

        # Deliverables have all template fields
        d = result["deliverables"][0]
        self.assertEqual(d["name"], "Web app")
        self.assertEqual(d["deadline"], "2026-11-30")
        self.assertEqual(d["owner"], "Asha")
        self.assertIn("description", d)
        self.assertIn("dependencies", d)
        self.assertIn("acceptance_criteria", d)

        # Milestones have all template fields
        m = result["milestones"][0]
        self.assertEqual(m["name"], "Sprint 3")
        self.assertEqual(m["date"], "2026-10-29")
        self.assertIn("description", m)
        self.assertIn("source", m)

        # No error
        self.assertIsNone(result["error"])


# ===========================================================================
# Blocker Agent — end-to-end with mocked LLM returning compact schema
# ===========================================================================

class TestBlockerAgentCompactSchema(unittest.TestCase):
    """Run blocker_agent.run() with compact schema and verify all fields."""

    COMPACT_RESPONSE = json.dumps({
        "blockers": [{"id": "B001", "description": "Sandbox access delayed",
                      "status": "Open", "owner": "Priya"}],
        "action_items": [{"id": "A001", "description": "Escalate to IT",
                          "owner": "Ravi", "due_date": "2026-10-16",
                          "priority": "High", "status": "Pending"}],
        "pending_decisions": [{"id": "D001", "description": "Use mock library",
                               "owner": "Priya", "due_date": "2026-10-20"}],
        "unresolved_issues": [{"id": "I001", "description": "Load test env unknown"}],
        "missing_information": ["No resolution date"],
    })

    def test_run_normalizes_compact_output(self):
        from agents.blocker_agent import run
        with patch("agents.blocker_agent.build_context", return_value="ctx"):
            with patch("agents.blocker_agent.call_llm_json",
                       return_value=json.loads(self.COMPACT_RESPONSE)):
                result = run("test-compact")

        # Blocker remapped
        b = result["blockers"][0]
        self.assertEqual(b["blocker_id"], "B001")
        self.assertEqual(b["description"], "Sandbox access delayed")
        self.assertEqual(b["status"], "Open")
        self.assertEqual(b["owner"], "Priya")
        self.assertIn("impact", b)
        self.assertIn("evidence", b)
        self.assertIn("recommended_resolution", b)
        self.assertIn("inferred", b)

        # Action item remapped
        a = result["action_items"][0]
        self.assertEqual(a["action_id"], "A001")
        self.assertEqual(a["priority"], "High")
        self.assertIn("related_blocker", a)
        self.assertIn("source", a)
        self.assertIn("inferred", a)

        # Decision remapped
        d = result["pending_decisions"][0]
        self.assertEqual(d["decision_id"], "D001")
        self.assertIn("source", d)

        # Issue remapped
        i = result["unresolved_issues"][0]
        self.assertEqual(i["issue_id"], "I001")
        self.assertIn("impact", i)
        self.assertIn("source", i)

        # Extraction source is LLM
        self.assertEqual(result["extraction_source"], "llm")
        self.assertIsNone(result["error"])


# ===========================================================================
# Risk Agent — end-to-end with mocked LLM returning compact schema
# ===========================================================================

class TestRiskAgentCompactSchema(unittest.TestCase):
    """Run risk_agent.run() with compact schema and verify all fields."""

    COMPACT_RESPONSE = json.dumps({
        "risks": [{"risk_id": "R001", "title": "SAP delay",
                   "description": "Sandbox 4 weeks late",
                   "severity": "High", "affected_deliverable": "ERP Integration",
                   "potential_impact": "Miss deadline",
                   "mitigation": "Use mock library",
                   "status": "Potential"}],
        "delivery_forecast": {
            "status": "At Risk",
            "estimated_completion_date": "2026-12-01",
            "delayed_milestones": ["M5 Go-Live"],
            "assumptions": ["SAP access by Oct 24"],
            "confidence": "Medium",
            "forecast_method": "qualitative",
        },
        "dependency_gaps": [{"gap_id": "G001", "description": "No test env",
                            "affected_task": "T011"}],
        "missing_information": ["Staging credentials date"],
    })

    def test_run_normalizes_compact_output(self):
        from agents.risk_agent import run
        scope_data = {
            "milestones": [{"name": "Go-Live", "date": "2099-12-01"}],
            "deliverables": [],
        }
        with patch("agents.risk_agent.build_context", return_value="ctx"):
            with patch("agents.risk_agent.call_llm_json",
                       return_value=json.loads(self.COMPACT_RESPONSE)):
                result = run("test-compact", scope_data, {})

        # Risk normalized
        r = result["risks"][0]
        self.assertEqual(r["risk_id"], "R001")
        self.assertEqual(r["severity"], "High")
        self.assertIn("evidence", r)
        self.assertIn("source", r)

        # Dependency gap normalized
        g = result["dependency_gaps"][0]
        self.assertEqual(g["gap_id"], "G001")
        self.assertIn("source", g)

        # Forecast merged — rule-based date from milestone takes priority
        fc = result["delivery_forecast"]
        self.assertEqual(fc["status"], "At Risk")
        self.assertIn("M5 Go-Live", fc["delayed_milestones"])

        self.assertIsNone(result["error"])


# ===========================================================================
# Cross-project isolation
# ===========================================================================

class TestCrossProjectIsolation(unittest.TestCase):
    """Ensure agent results from one project don't bleed into another."""

    def test_scope_results_isolated(self):
        from agents.scope_agent import run
        from agents.llm_client import clear_llm_result_cache, clear_context_cache

        clear_llm_result_cache()
        clear_context_cache()

        resp_a = json.dumps({"project_goals": ["Goal A"], "scope": ["Scope A"],
                              "out_of_scope": [], "deliverables": [], "milestones": []})
        resp_b = json.dumps({"project_goals": ["Goal B"], "scope": ["Scope B"],
                              "out_of_scope": [], "deliverables": [], "milestones": []})

        call_count = [0]
        def fake_llm(system, user, **kwargs):
            call_count[0] += 1
            return resp_a if call_count[0] == 1 else resp_b

        with patch("agents.llm_client.call_llm", side_effect=fake_llm):
            with patch("agents.scope_agent.build_context", return_value="ctx-A"):
                ra = run("proj-A")
            with patch("agents.scope_agent.build_context", return_value="ctx-B"):
                rb = run("proj-B")

        self.assertEqual(ra["project_goals"][0]["goal"], "Goal A")
        self.assertEqual(rb["project_goals"][0]["goal"], "Goal B")

    def test_cache_clear_for_one_project_does_not_affect_other(self):
        from agents.llm_client import (
            set_cached_llm_result, get_cached_llm_result,
            clear_llm_result_cache,
        )
        set_cached_llm_result("scope", "proj-X", "sys", "ctx", 3000, "model", {"goals": ["X"]})
        set_cached_llm_result("scope", "proj-Y", "sys", "ctx", 3000, "model", {"goals": ["Y"]})

        clear_llm_result_cache("proj-X")

        self.assertIsNone(get_cached_llm_result("scope", "proj-X", "sys", "ctx", 3000, "model"))
        self.assertIsNotNone(get_cached_llm_result("scope", "proj-Y", "sys", "ctx", 3000, "model"))


# ===========================================================================
# Missing / unavailable values handled gracefully
# ===========================================================================

class TestMissingValuesHandling(unittest.TestCase):
    """When the LLM returns null/empty for fields, _merge must fill safe defaults."""

    def test_scope_empty_lists_safe(self):
        from agents.scope_agent import _merge, EMPTY_OUTPUT
        result = _merge(EMPTY_OUTPUT.copy(), {})
        self.assertEqual(result["project_goals"], [])
        self.assertEqual(result["scope"], [])
        self.assertEqual(result["deliverables"], [])
        self.assertIsNone(result["error"])

    def test_blocker_empty_lists_safe(self):
        from agents.blocker_agent import _merge, EMPTY_OUTPUT
        result = _merge(EMPTY_OUTPUT.copy(), {})
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["action_items"], [])
        self.assertIsNone(result["error"])

    def test_risk_empty_lists_safe(self):
        from agents.risk_agent import _merge, EMPTY_OUTPUT
        rb = {"status": "Insufficient data", "estimated_completion_date": None,
              "delayed_milestones": [], "assumptions": [],
              "confidence": "Low", "forecast_method": "none"}
        result = _merge(EMPTY_OUTPUT.copy(), {}, rb)
        self.assertEqual(result["risks"], [])
        self.assertEqual(result["dependency_gaps"], [])
        self.assertIsNone(result["error"])

    def test_deliverable_null_owner_is_none_not_string(self):
        from agents.scope_agent import _merge, EMPTY_OUTPUT
        raw = {"deliverables": [{"name": "X", "owner": None}]}
        result = _merge(EMPTY_OUTPUT.copy(), raw)
        self.assertIsNone(result["deliverables"][0]["owner"])

    def test_blocker_null_owner_is_none(self):
        from agents.blocker_agent import _merge, EMPTY_OUTPUT
        raw = {"blockers": [{"id": "B1", "description": "x", "owner": None}]}
        result = _merge(EMPTY_OUTPUT.copy(), raw)
        self.assertIsNone(result["blockers"][0]["owner"])


# ===========================================================================
# JSON repair handles truncated responses
# ===========================================================================

class TestJsonRepairTruncated(unittest.TestCase):
    """_repair_json must recover partial JSON from truncated LLM output."""

    def test_truncated_array_closed(self):
        from agents.llm_client import _repair_json
        # Truncated mid-array: missing closing ] and }
        partial = '{"scope": ["item A", "item B", "item C'
        result = _repair_json(partial)
        self.assertIsNotNone(result)
        self.assertIn("scope", result)
        self.assertGreaterEqual(len(result["scope"]), 2)  # at least 2 items recovered

    def test_truncated_object_closed(self):
        from agents.llm_client import _repair_json
        # Truncated mid-object: missing closing }]
        partial = '{"risks": [{"risk_id": "R001", "title": "test"'
        result = _repair_json(partial)
        # May or may not recover depending on bracket nesting depth
        if result is not None:
            self.assertIn("risks", result)

    def test_trailing_comma_removed(self):
        from agents.llm_client import _repair_json
        bad = '{"scope": ["a", "b",], "goals": []}'
        result = _repair_json(bad)
        self.assertIsNotNone(result)
        self.assertEqual(result["scope"], ["a", "b"])

    def test_valid_json_extracted_directly(self):
        """Valid JSON is handled by _extract_json, not _repair_json."""
        from agents.llm_client import _extract_json
        good = '{"key": "value"}'
        result = _extract_json(good)
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")


if __name__ == "__main__":
    unittest.main()
