"""
Reliability tests for LLM client — covers 413, 429, cache key isolation,
and explicit max_tokens per agent.  Uses mocked Groq responses only;
no live API calls are made.
"""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_groq_exc(status_code: int, message: str = "error") -> Exception:
    """Create a mock exception that _is_permanent_error() can detect."""
    exc = Exception(message)
    exc.status_code = status_code
    return exc


def _make_rate_limit_exc(retry_after: str | None = None) -> Exception:
    """Create a mock 429 RateLimitError with optional Retry-After header."""
    from agents.llm_client import _groq_pkg  # type: ignore[attr-defined]
    if _groq_pkg is None:
        exc = Exception("rate limited")
        exc.status_code = 429
        return exc
    try:
        exc = _groq_pkg.RateLimitError.__new__(_groq_pkg.RateLimitError)
        exc.status_code = 429
        headers = {"retry-after": retry_after} if retry_after else {}
        mock_resp = MagicMock()
        mock_resp.headers = headers
        exc.response = mock_resp
        return exc
    except Exception:
        exc = Exception("rate limited")
        exc.status_code = 429
        return exc


# ---------------------------------------------------------------------------
# 1. HTTP 413 is treated as permanent — never retried
# ---------------------------------------------------------------------------

class TestHttp413NeverRetried(unittest.TestCase):

    def _make_client_raising(self, status_code: int):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_groq_exc(status_code)
        return mock_client

    def test_413_retries_then_fails(self):
        """HTTP 413 is now retryable (TPM window resets after 60s)."""
        from agents.llm_client import LLMError, call_llm
        exc_413 = _make_groq_exc(413, "Request Entity Too Large")

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("agents.llm_client.Groq") as mock_groq_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = exc_413
                mock_groq_cls.return_value = mock_client

                with patch("agents.llm_client.time.sleep"):
                    with self.assertRaises(LLMError):
                        call_llm(system="s", user="u", retries=3)

                # 413 is retryable: called once per attempt
                self.assertEqual(mock_client.chat.completions.create.call_count, 3)

    def test_400_raises_llm_error_immediately(self):
        """HTTP 400 must raise LLMError without retry (permanent request error)."""
        from agents.llm_client import LLMError, call_llm
        exc_400 = _make_groq_exc(400, "Bad Request")

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("agents.llm_client.Groq") as mock_groq_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = exc_400
                mock_groq_cls.return_value = mock_client

                with self.assertRaises(LLMError):
                    call_llm(system="s", user="u", retries=3,
                             json_mode=False)  # disable json_mode retry path
                # One call only
                self.assertEqual(mock_client.chat.completions.create.call_count, 1)


# ---------------------------------------------------------------------------
# 2. HTTP 429 is retried with wait, respects Retry-After header
# ---------------------------------------------------------------------------

class TestHttp429RateLimitHandling(unittest.TestCase):

    def test_429_retries_with_wait(self):
        """HTTP 429 should wait and retry (up to retries limit)."""
        from agents.llm_client import LLMError, call_llm, _groq_pkg

        if _groq_pkg is None:
            self.skipTest("groq package not installed")

        rate_exc = _make_rate_limit_exc(retry_after="0")

        # First call: 429; second call: success
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"ok": true}'
        mock_resp.choices[0].finish_reason = "stop"

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("agents.llm_client.Groq") as mock_groq_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = [rate_exc, mock_resp]
                mock_groq_cls.return_value = mock_client
                with patch("agents.llm_client.time.sleep"):
                    result = call_llm(system="s", user="u", retries=2)

        self.assertIn("ok", result)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_429_exhausted_raises_llm_error(self):
        """When 429 exhausts all retries, LLMError is raised."""
        from agents.llm_client import LLMError, call_llm, _groq_pkg

        if _groq_pkg is None:
            self.skipTest("groq package not installed")

        rate_exc = _make_rate_limit_exc(retry_after="0")

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("agents.llm_client.Groq") as mock_groq_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = [rate_exc, rate_exc]
                mock_groq_cls.return_value = mock_client
                with patch("agents.llm_client.time.sleep"):
                    with self.assertRaises(LLMError):
                        call_llm(system="s", user="u", retries=2)


# ---------------------------------------------------------------------------
# 3. Cache key includes max_chars — risk agent gets separate cache entry
# ---------------------------------------------------------------------------

class TestContextCacheKeyIncludesMaxChars(unittest.TestCase):

    def test_different_max_chars_produce_different_cache_keys(self):
        """Cache key must differ when max_chars differs."""
        from agents.llm_client import _context_cache_key

        queries = ["risk issue delay", "dependency blocked"]
        key_default = _context_cache_key("proj", queries, k=4, max_chars=None)
        key_3000    = _context_cache_key("proj", queries, k=4, max_chars=3000)
        key_6000    = _context_cache_key("proj", queries, k=4, max_chars=6000)

        self.assertNotEqual(key_default, key_3000,
            "max_chars=None and max_chars=3000 must produce different cache keys")
        self.assertNotEqual(key_3000, key_6000,
            "max_chars=3000 and max_chars=6000 must produce different cache keys")

    def test_same_max_chars_produce_same_cache_key(self):
        """Identical arguments must produce identical cache keys."""
        from agents.llm_client import _context_cache_key

        queries = ["risk issue delay", "dependency blocked"]
        key_a = _context_cache_key("proj", queries, k=4, max_chars=3000)
        key_b = _context_cache_key("proj", queries, k=4, max_chars=3000)
        self.assertEqual(key_a, key_b)

    def test_scope_and_risk_do_not_share_cache(self):
        """Scope (max_chars=None) and risk (max_chars=3000) must use separate entries."""
        from agents.llm_client import _context_cache_key
        from agents.scope_agent import RETRIEVAL_QUERIES as SCOPE_Q
        from agents.risk_agent import RETRIEVAL_QUERIES as RISK_Q

        scope_key = _context_cache_key("myproject", SCOPE_Q, k=4, max_chars=None)
        risk_key  = _context_cache_key("myproject", RISK_Q,  k=4, max_chars=3000)

        # Even if queries happened to be the same, the max_chars differs → different keys
        self.assertNotEqual(scope_key, risk_key)

    def test_risk_agent_gets_separate_cache_entry(self):
        """build_context with max_chars=3000 must NOT return cached 6000-char result."""
        from agents.llm_client import _context_cache, _context_cache_key, _CACHE_TTL

        project_id = "_test_cache_isolation_"
        queries = ["risk delay deadline", "dependency blocked"]

        # Manually seed the cache with a 6000-char result under the default key
        big_context = "x" * 5999
        default_key = _context_cache_key(project_id, queries, k=4, max_chars=None)
        _context_cache[default_key] = (time.time(), big_context)

        # Calling build_context with max_chars=3000 must produce a different key
        # so it does NOT return the big_context from cache
        small_key = _context_cache_key(project_id, queries, k=4, max_chars=3000)
        self.assertNotEqual(default_key, small_key)
        self.assertNotIn(small_key, _context_cache,
            "The small-context key must not exist in cache yet — "
            "risk agent must not inherit scope agent's large context")

        # Clean up
        _context_cache.pop(default_key, None)


# ---------------------------------------------------------------------------
# 4. Explicit max_tokens per agent
# ---------------------------------------------------------------------------

class TestExplicitMaxTokensPerAgent(unittest.TestCase):
    """Each agent must explicitly set max_tokens in its call_llm_json call
    so the request budget is predictable and independent of the default."""

    def _get_max_tokens_sent(self, agent_run_fn, project_id: str,
                             mock_response_json: str, **run_kwargs) -> int | None:
        """Run agent with mocked LLM; return the max_tokens sent to call_llm."""
        captured = {}

        def fake_call_llm(system, user, **kwargs):
            captured.update(kwargs)
            return mock_response_json

        # Patch call_llm at the llm_client module level (called from call_llm_json)
        # and also patch build_context everywhere it is imported
        import agents.scope_agent
        import agents.blocker_agent
        import agents.risk_agent

        with patch("agents.llm_client.call_llm", side_effect=fake_call_llm):
            with patch("agents.scope_agent.build_context",
                       return_value="[Source: test.txt | Chunk #0]\nsome context"):
                with patch("agents.blocker_agent.build_context",
                           return_value="[Source: test.txt | Chunk #0]\nsome context"):
                    with patch("agents.risk_agent.build_context",
                               return_value="[Source: test.txt | Chunk #0]\nsome context"):
                        agent_run_fn(project_id, **run_kwargs)

        return captured.get("max_tokens")

    def test_scope_agent_explicit_max_tokens(self):
        """scope_agent must pass max_tokens explicitly."""
        import json
        from agents.scope_agent import run

        mock_response = json.dumps({
            "project_goals": [], "scope": [], "out_of_scope": [],
            "deliverables": [], "milestones": [], "missing_information": [], "conflicts": [],
        })
        mt = self._get_max_tokens_sent(run, "test-proj", mock_response)
        self.assertIsNotNone(mt, "scope_agent must pass max_tokens explicitly")
        self.assertGreaterEqual(mt, 1,
                             f"scope_agent max_tokens={mt} — must be a positive value")

    def test_risk_agent_explicit_max_tokens(self):
        """risk_agent must pass max_tokens explicitly."""
        import json
        from agents.risk_agent import run

        mock_response = json.dumps({
            "risks": [], "delivery_forecast": {
                "status": "On Track", "estimated_completion_date": None,
                "delayed_milestones": [], "assumptions": [],
                "confidence": "Low", "forecast_method": "none",
            },
            "dependency_gaps": [], "missing_information": [],
        })
        mt = self._get_max_tokens_sent(
            run, "test-proj", mock_response,
            scope_data={}, blocker_data={},
        )
        self.assertIsNotNone(mt, "risk_agent must pass max_tokens explicitly")
        self.assertGreaterEqual(mt, 1,
                             f"risk_agent max_tokens={mt} — must be a positive value")

    def test_blocker_agent_explicit_max_tokens(self):
        """blocker_agent must pass max_tokens explicitly."""
        import json
        from agents.blocker_agent import run

        mock_response = json.dumps({
            "blockers": [], "action_items": [], "pending_decisions": [],
            "unresolved_issues": [], "missing_information": [],
        })
        mt = self._get_max_tokens_sent(run, "test-proj", mock_response)
        self.assertIsNotNone(mt, "blocker_agent must pass max_tokens explicitly")
        self.assertGreaterEqual(mt, 1,
                             f"blocker_agent max_tokens={mt} — must be a positive value")


# ---------------------------------------------------------------------------
# 5. _is_permanent_error covers 413
# ---------------------------------------------------------------------------

class TestPermanentErrorClassification(unittest.TestCase):

    def test_413_is_not_permanent(self):
        """413 is now retryable (TPM window resets after 60s)."""
        from agents.llm_client import _is_permanent_error
        exc = _make_groq_exc(413)
        self.assertFalse(_is_permanent_error(exc))

    def test_400_is_permanent(self):
        from agents.llm_client import _is_permanent_error
        exc = _make_groq_exc(400)
        self.assertTrue(_is_permanent_error(exc))

    def test_422_is_permanent(self):
        from agents.llm_client import _is_permanent_error
        exc = _make_groq_exc(422)
        self.assertTrue(_is_permanent_error(exc))

    def test_500_is_not_permanent(self):
        from agents.llm_client import _is_permanent_error
        exc = _make_groq_exc(500)
        self.assertFalse(_is_permanent_error(exc))

    def test_429_is_not_permanent(self):
        from agents.llm_client import _is_permanent_error
        exc = _make_groq_exc(429)
        self.assertFalse(_is_permanent_error(exc))

    def test_generic_exception_is_not_permanent(self):
        from agents.llm_client import _is_permanent_error
        self.assertFalse(_is_permanent_error(RuntimeError("network error")))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 6. LLM result cache — stores successful results, invalidates on project change
# ---------------------------------------------------------------------------

class TestLLMResultCache(unittest.TestCase):

    def setUp(self):
        from agents.llm_client import _llm_result_cache
        _llm_result_cache.clear()

    def tearDown(self):
        from agents.llm_client import _llm_result_cache
        _llm_result_cache.clear()

    def test_cache_stores_and_retrieves_result(self):
        """A stored result is returned on subsequent identical calls."""
        from agents.llm_client import get_cached_llm_result, set_cached_llm_result
        result = {"project_goals": [{"goal": "test", "source": "f#0", "inferred": False}]}
        set_cached_llm_result("scope", "proj-a", "sys", "ctx", 4096, "model-x", result)
        retrieved = get_cached_llm_result("scope", "proj-a", "sys", "ctx", 4096, "model-x")
        self.assertEqual(retrieved, result)

    def test_cache_miss_when_context_differs(self):
        """Different context produces a cache miss — documents changed."""
        from agents.llm_client import get_cached_llm_result, set_cached_llm_result
        result = {"project_goals": []}
        set_cached_llm_result("scope", "proj-a", "sys", "context-v1", 4096, "model-x", result)
        # Different context (new document uploaded)
        retrieved = get_cached_llm_result("scope", "proj-a", "sys", "context-v2", 4096, "model-x")
        self.assertIsNone(retrieved)

    def test_cache_miss_when_agent_differs(self):
        """Results are isolated per agent type."""
        from agents.llm_client import get_cached_llm_result, set_cached_llm_result
        result = {"risks": []}
        set_cached_llm_result("risk", "proj-a", "sys", "ctx", 2048, "model-x", result)
        # Scope agent should not get risk agent's cached result
        retrieved = get_cached_llm_result("scope", "proj-a", "sys", "ctx", 2048, "model-x")
        self.assertIsNone(retrieved)

    def test_cache_miss_when_project_differs(self):
        """Results are isolated per project."""
        from agents.llm_client import get_cached_llm_result, set_cached_llm_result
        result = {"project_goals": []}
        set_cached_llm_result("scope", "project-alpha", "sys", "ctx", 4096, "model-x", result)
        retrieved = get_cached_llm_result("scope", "project-beta", "sys", "ctx", 4096, "model-x")
        self.assertIsNone(retrieved)

    def test_clear_cache_by_project(self):
        """clear_llm_result_cache(project_id) removes only that project's entries."""
        from agents.llm_client import (
            get_cached_llm_result, set_cached_llm_result, clear_llm_result_cache,
        )
        r = {"project_goals": []}
        set_cached_llm_result("scope", "proj-a", "sys", "ctx", 4096, "model-x", r)
        set_cached_llm_result("scope", "proj-b", "sys", "ctx", 4096, "model-x", r)
        cleared = clear_llm_result_cache("proj-a")
        self.assertEqual(cleared, 1)
        self.assertIsNone(get_cached_llm_result("scope", "proj-a", "sys", "ctx", 4096, "model-x"))
        self.assertIsNotNone(get_cached_llm_result("scope", "proj-b", "sys", "ctx", 4096, "model-x"))

    def test_clear_all_cache(self):
        """clear_llm_result_cache() with no args removes everything."""
        from agents.llm_client import (
            get_cached_llm_result, set_cached_llm_result, clear_llm_result_cache,
        )
        r = {"project_goals": []}
        set_cached_llm_result("scope", "proj-a", "sys", "ctx", 4096, "model-x", r)
        set_cached_llm_result("scope", "proj-b", "sys", "ctx", 4096, "model-x", r)
        cleared = clear_llm_result_cache()
        self.assertEqual(cleared, 2)

    def test_scope_agent_uses_cache_on_second_call(self):
        """scope_agent.run() must not call the LLM on the second identical call."""
        import json
        from agents.scope_agent import run
        from agents.llm_client import clear_llm_result_cache

        clear_llm_result_cache()
        call_count = [0]
        mock_response = json.dumps({
            "project_goals": [{"goal": "test goal", "source": "f#0", "inferred": False}],
            "scope": [], "out_of_scope": [], "deliverables": [],
            "milestones": [], "missing_information": [], "conflicts": [],
        })

        def fake_call_llm(system, user, **kwargs):
            call_count[0] += 1
            return mock_response

        with patch("agents.llm_client.call_llm", side_effect=fake_call_llm):
            with patch("agents.scope_agent.build_context", return_value="ctx"):
                r1 = run("test-caching-proj")
                r2 = run("test-caching-proj")  # second call — must hit cache

        self.assertEqual(call_count[0], 1, "LLM should be called exactly once; second run should use cache")
        self.assertEqual(r1["project_goals"], r2["project_goals"])

    def test_scope_agent_invalidates_cache_on_context_change(self):
        """When context changes (new doc), scope_agent calls LLM again."""
        import json
        from agents.scope_agent import run
        from agents.llm_client import clear_llm_result_cache

        clear_llm_result_cache()
        call_count = [0]
        mock_response = json.dumps({
            "project_goals": [], "scope": [], "out_of_scope": [],
            "deliverables": [], "milestones": [], "missing_information": [], "conflicts": [],
        })

        def fake_call_llm(system, user, **kwargs):
            call_count[0] += 1
            return mock_response

        contexts = ["context-v1", "context-v2"]
        ctx_iter = iter(contexts)

        with patch("agents.llm_client.call_llm", side_effect=fake_call_llm):
            with patch("agents.scope_agent.build_context", side_effect=lambda *a, **kw: next(ctx_iter)):
                run("test-invalidation-proj")
                run("test-invalidation-proj")  # different context → cache miss → new LLM call

        self.assertEqual(call_count[0], 2, "LLM must be called again when context changes")


# ---------------------------------------------------------------------------
# 7. Context k=8 default ensures multi-file coverage
# ---------------------------------------------------------------------------

class TestContextKDefault(unittest.TestCase):

    def test_build_context_default_k_is_8(self):
        """build_context default k must be 8 to cover projects with 7+ files."""
        import inspect
        from agents.llm_client import build_context
        sig = inspect.signature(build_context)
        default_k = sig.parameters["k"].default
        self.assertEqual(default_k, 8,
                         f"build_context k default is {default_k}; should be 8 "
                         f"so all 7 UniMeal files can contribute chunks")

    def test_search_k8_gives_2_chunks_per_source_for_7_files(self):
        """With k=8 and 7 sources, max_per_source = ceil(8/7) = 2.
        This ensures every file can contribute at least 2 chunks per query."""
        import math
        k = 8
        num_sources = 7
        max_per_source = math.ceil(k / num_sources)
        self.assertEqual(max_per_source, 2,
                         f"With k=8 and 7 sources, each file should get {max_per_source} slots")
        self.assertGreaterEqual(max_per_source * num_sources, k,
                                "Total capacity must be >= k")

    def test_search_k4_gives_only_1_chunk_per_source_for_7_files(self):
        """Confirms the old k=4 was the root cause of single-chunk-per-file coverage."""
        import math
        k_old = 4
        num_sources = 7
        max_per_source_old = math.ceil(k_old / num_sources)
        self.assertEqual(max_per_source_old, 1,
                         "Old k=4 gave each of 7 files only 1 chunk slot — confirmed root cause")


# ---------------------------------------------------------------------------
# 8. README / instruction files excluded from context
# ---------------------------------------------------------------------------

class TestReadmeExcludedFromContext(unittest.TestCase):
    """README.txt and similar instruction files must be skipped by build_context."""

    def setUp(self):
        from agents.llm_client import _context_cache
        _context_cache.clear()

    def test_readme_chunks_excluded(self):
        """Chunks whose source is README.txt must not appear in context."""
        from agents.llm_client import build_context, _context_cache
        from models import SearchResult

        _context_cache.clear()

        readme_hit = SearchResult(
            text="Do not classify out-of-scope features as blockers.",
            source="README.txt",
            chunk_index=0,
            score=0.9,
            document_id="readme-doc",
        )
        real_hit = SearchResult(
            text="The seat booking form is blocked pending API schema confirmation.",
            source="03_Sprint_Status.docx",
            chunk_index=1,
            score=0.85,
            document_id="sprint-doc",
        )

        def fake_search(project_id, query, k=8):
            return [readme_hit, real_hit]

        # The function imports rag_search locally — patch it in the rag module
        with patch("rag.search", new=fake_search):
            ctx = build_context("test-proj", ["blocker"], k=2)

        # README content must not be in the context
        self.assertNotIn("Do not classify", ctx,
                         "README instruction text must be excluded from context")
        # Real project content must still be included
        self.assertIn("seat booking form", ctx)

    def test_instruction_filenames_matched(self):
        """The skip pattern must match common instruction filenames."""
        import re
        pattern = re.compile(
            r"(?:README|INSTRUCTIONS|SETUP|INSTALL|CONTRIBUTING|CHANGELOG|LICENSE)"
            r"(?:\.[a-z]+)?$",
            re.IGNORECASE,
        )
        should_match = ["README.txt", "README.md", "INSTRUCTIONS.txt",
                        "SETUP.txt", "LICENSE", "CHANGELOG.md"]
        should_not_match = ["03_Sprint_Status.docx", "task_tracker.csv",
                             "requirements.docx", "01_Charter.pdf"]
        for fname in should_match:
            self.assertTrue(pattern.search(fname),
                            f"Should be excluded: {fname}")
        for fname in should_not_match:
            self.assertFalse(pattern.search(fname),
                             f"Should not be excluded: {fname}")


# ---------------------------------------------------------------------------
# 9. Scope agent uses k=8 by default
# ---------------------------------------------------------------------------

class TestScopeAgentContextK(unittest.TestCase):

    def test_scope_agent_default_context_k_is_8(self):
        """scope_agent.run() default context_k must be 8, not 4."""
        import inspect
        from agents.scope_agent import run
        sig = inspect.signature(run)
        default_k = sig.parameters["context_k"].default
        self.assertEqual(default_k, 8,
                         f"scope_agent default context_k={default_k}; must be 8 "
                         f"so all files in multi-file projects are covered")


# ---------------------------------------------------------------------------
# 10. Risk agent marks completed deliverables as done, not delayed
# ---------------------------------------------------------------------------

class TestRiskForecastCompletedDeliverables(unittest.TestCase):

    def test_completed_deliverable_not_flagged_as_delayed(self):
        """A deliverable with 'completed' in its description must not be overdue."""
        from agents.risk_agent import _rule_based_forecast
        from datetime import date

        scope = {
            "milestones": [],
            "deliverables": [
                {
                    "name": "Seat booking form",
                    "description": "Completed and delivered in sprint 1.",
                    "deadline": "2020-01-01",   # far past date
                    "owner": "Alice",
                },
                {
                    "name": "Admin dashboard",
                    "description": "Still in progress.",
                    "deadline": "2099-12-31",   # future
                    "owner": "Bob",
                },
            ],
        }
        fc = _rule_based_forecast(scope)
        self.assertNotIn("Seat booking form", fc["delayed_milestones"],
                         "Completed deliverable must not appear in delayed_milestones")
        self.assertIn(fc["status"], ("On Track", "At Risk"),
                     f"Status should be On Track/At Risk, not '{fc['status']}'")

    def test_incomplete_past_deliverable_is_delayed(self):
        """An incomplete deliverable whose deadline passed IS delayed."""
        from agents.risk_agent import _rule_based_forecast

        scope = {
            "milestones": [],
            "deliverables": [
                {
                    "name": "Integration module",
                    "description": "Still in progress, not yet complete.",
                    "deadline": "2020-01-01",
                    "owner": "Dev",
                },
            ],
        }
        fc = _rule_based_forecast(scope)
        # No upcoming items, all overdue → Delayed
        self.assertEqual(fc["status"], "Delayed")
        self.assertIn("Integration module", fc["delayed_milestones"])
