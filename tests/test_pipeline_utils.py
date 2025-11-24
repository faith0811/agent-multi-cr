import unittest

from agent_multi_cr.pipeline import build_qa_snippet_for_reviewer


class TestPipelineUtils(unittest.TestCase):
    def test_build_qa_snippet_empty(self) -> None:
        history = []
        snippet = build_qa_snippet_for_reviewer(history, "A")
        self.assertIn("no previous Q&A with you", snippet)

    def test_build_qa_snippet_filters_by_reviewer(self) -> None:
        history = [
            {"reviewer": "A", "question": "Q1", "answer": "A1"},
            {"reviewer": "B", "question": "Q2", "answer": "A2"},
            {"reviewer": "A", "question": "Q3", "answer": "A3"},
        ]
        snippet = build_qa_snippet_for_reviewer(history, "A")
        self.assertIn("Q1", snippet)
        self.assertIn("A1", snippet)
        self.assertIn("Q3", snippet)
        self.assertIn("A3", snippet)
        self.assertNotIn("Q2", snippet)
        self.assertNotIn("A2", snippet)
        self.assertIn("---", snippet)
