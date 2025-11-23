import unittest

from src.llm_runners import parse_arbiter_json
from src.prompts import (
    build_arbiter_prompt,
    build_followup_prompt,
    build_initial_review_prompt,
)
from src.auditors import Auditor


class TestPrompts(unittest.TestCase):
    def test_initial_review_prompt_includes_required_sections(self) -> None:
        prompt = build_initial_review_prompt(
            reviewer_name="ReviewerA",
            task_description="Do a test review.",
            context_text="CODE",
            memo_text="memo contents",
        )
        self.assertIn("ReviewerA", prompt)
        self.assertIn("Do a test review.", prompt)
        self.assertIn("```text", prompt)
        self.assertIn("MEMO_JSON", prompt)

    def test_followup_prompt_references_initial_review_and_question(self) -> None:
        prompt = build_followup_prompt(
            reviewer_name="ReviewerB",
            task_description="Check edge cases.",
            context_text="CTX",
            initial_review="Initial review text",
            question="Is this safe?",
            qa_snippet="Prev QA",
            memo_text="some memo",
        )
        self.assertIn("ReviewerB", prompt)
        self.assertIn("Initial review text", prompt)
        self.assertIn("Is this safe?", prompt)
        self.assertIn("MEMO_JSON", prompt)

    def test_arbiter_prompt_mentions_auditors_and_limits(self) -> None:
        auditors = [
            Auditor(
                name="Auditor1",
                kind="codex",
                model_name="dummy",
                workdir="/tmp",
                reasoning_effort="high",
            )
        ]
        prompt = build_arbiter_prompt(
            arbiter_name="ArbiterX",
            task_description="Review everything.",
            context_text="CTX",
            auditors=auditors,
            initial_reviews={"Auditor1": "Some review"},
            qa_history=[],
            max_queries=3,
            query_count=1,
        )
        self.assertIn("ArbiterX", prompt)
        self.assertIn("Auditor1", prompt)
        self.assertIn("hard limit of 3 clarification questions", prompt)
        self.assertIn("NEEDS HUMAN REVIEW", prompt)


class TestParseArbiterJson(unittest.TestCase):
    def test_parse_valid_json(self) -> None:
        raw = '{"state": "final", "final_markdown": "ok"}'
        parsed = parse_arbiter_json(raw)
        self.assertEqual(parsed["state"], "final")
        self.assertEqual(parsed["final_markdown"], "ok")

    def test_parse_json_embedded_in_text(self) -> None:
        raw = "noise {\"state\": \"query\", \"target_reviewer\": \"R\", \"question\": \"Q?\", \"reason\": \"because\"} trailing"
        parsed = parse_arbiter_json(raw)
        self.assertEqual(parsed["state"], "query")
        self.assertEqual(parsed["target_reviewer"], "R")

    def test_parse_fallback_to_final_markdown(self) -> None:
        raw = "not json at all"
        parsed = parse_arbiter_json(raw)
        self.assertEqual(parsed["state"], "final")
        self.assertEqual(parsed["final_markdown"], raw)


if __name__ == "__main__":
    unittest.main()
