from contextlib import redirect_stdout
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import evaluate_answers
from evaluate_answers import _normalize_for_match, _sources_match_case


class AnswerEvaluationTests(unittest.TestCase):
    def test_rate_limit_skips_case_and_continues_evaluation(self):
        class FakeRateLimitError(Exception):
            pass

        cases = [
            {"name": "Limited", "question": "first", "expected": "answer", "url": None},
            {"name": "Continued", "question": "second", "expected": "insufficient", "url": None},
        ]
        output = io.StringIO()
        with (
            patch.object(evaluate_answers, "CASES", cases),
            patch.object(evaluate_answers, "RateLimitError", FakeRateLimitError),
            patch.object(evaluate_answers, "load_resources", return_value=([], object(), object(), object(), object())),
            patch.object(evaluate_answers, "_retrieval_query", side_effect=lambda question, history: question),
            patch.object(evaluate_answers, "retrieve_chunks", return_value=[]),
            patch.object(
                evaluate_answers,
                "generate_answer",
                side_effect=[FakeRateLimitError(), "insufficient information"],
            ) as generate,
            redirect_stdout(output),
        ):
            evaluate_answers.main()

        self.assertEqual(generate.call_count, 2)
        self.assertIn("SKIP Limited: Groq rate limit reached", output.getvalue())
        self.assertIn("Answer cases: 1 passed, 0 failed, 1 skipped (2 total)", output.getvalue())

    def test_two_year_expectation_accepts_typographic_dash_variants(self):
        expected = _normalize_for_match("two-year")
        for answer_phrase in ("two-year", "two\N{NON-BREAKING HYPHEN}year", "two\N{EN DASH}year", "two year"):
            with self.subTest(answer_phrase=answer_phrase):
                self.assertIn(expected, _normalize_for_match(answer_phrase))

    def test_normalization_does_not_accept_a_different_duration(self):
        self.assertNotIn(
            _normalize_for_match("two-year"),
            _normalize_for_match("The programme lasts three years."),
        )

    def test_unsupported_case_requires_empty_sources(self):
        case = {"url": None}
        self.assertTrue(_sources_match_case(case, []))
        self.assertFalse(_sources_match_case(case, [SimpleNamespace(url="https://www.rnsit.ac.in/cse/")]))

    def test_supported_case_requires_expected_source_url(self):
        case = {"url": "/cse/"}
        self.assertTrue(_sources_match_case(case, [SimpleNamespace(url="https://www.rnsit.ac.in/cse/")]))
        self.assertFalse(_sources_match_case(case, []))

    def test_supported_case_requires_expected_fact_in_source_excerpt(self):
        case = {"url": "/cse/", "source_terms": ["cse", "720"]}
        correct = SimpleNamespace(url="https://www.rnsit.ac.in/cse/", excerpt="The CSE intake is 720 students.")
        mismatched = SimpleNamespace(url="https://www.rnsit.ac.in/cse/", excerpt="CSE is a department with faculty.")
        self.assertTrue(_sources_match_case(case, [correct]))
        self.assertFalse(_sources_match_case(case, [mismatched]))


if __name__ == "__main__":
    unittest.main()
