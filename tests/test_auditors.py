import os
import tempfile
import unittest

from agent_multi_cr.auditors import (
    MEMO_JSON_PREFIX,
    Auditor,
    extract_and_update_memo,
    load_memo,
    slugify,
)


class TestSlugify(unittest.TestCase):
    def test_slugify_removes_unsafe_chars(self) -> None:
        name = "Codex[gpt-5.1|high]"
        slug = slugify(name)
        self.assertTrue(slug)
        self.assertNotIn("[", slug)
        self.assertNotIn("]", slug)
        self.assertNotIn("|", slug)
        self.assertNotIn(" ", slug)

    def test_slugify_empty_falls_back(self) -> None:
        self.assertEqual(slugify(""), "auditor")


class TestMemoExtraction(unittest.TestCase):
    def test_extract_and_update_memo_append_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = Auditor(
                name="TestAuditor",
                kind="codex",
                model_name="dummy-model",
                workdir=tmpdir,
                reasoning_effort="high",
            )

            # Start with empty memo; append text.
            current_memo = load_memo(auditor)
            raw_output = (
                f"Line 1\n{MEMO_JSON_PREFIX} "
                '{"append": "note1", "overwrite": false}\nLine 2'
            )
            cleaned, new_memo = extract_and_update_memo(auditor, raw_output, current_memo)

            self.assertEqual(cleaned, "Line 1\nLine 2")
            self.assertEqual(new_memo, "note1")

            memo_file_path = os.path.join(tmpdir, "memo.txt")
            with open(memo_file_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "note1")

            # Append again without overwrite should add a newline and then text.
            current_memo = new_memo
            raw_output2 = (
                "Something\n"
                f"{MEMO_JSON_PREFIX} "
                '{"append": "note2", "overwrite": false}'
            )
            cleaned2, new_memo2 = extract_and_update_memo(auditor, raw_output2, current_memo)

            self.assertEqual(cleaned2, "Something")
            self.assertEqual(new_memo2, "note1\nnote2")

            with open(memo_file_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "note1\nnote2")

            # Overwrite should replace the memo entirely.
            current_memo = new_memo2
            raw_output3 = (
                "Header\n"
                f"{MEMO_JSON_PREFIX} "
                '{"append": "fresh", "overwrite": true}'
            )
            cleaned3, new_memo3 = extract_and_update_memo(auditor, raw_output3, current_memo)

            self.assertEqual(cleaned3, "Header")
            self.assertEqual(new_memo3, "fresh")

            with open(memo_file_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "fresh")

    def test_extract_and_update_memo_ignores_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = Auditor(
                name="TestAuditor",
                kind="codex",
                model_name="dummy-model",
                workdir=tmpdir,
                reasoning_effort="high",
            )
            current_memo = ""
            raw_output = f"Line\n{MEMO_JSON_PREFIX} not-a-json\nMore"
            cleaned, new_memo = extract_and_update_memo(auditor, raw_output, current_memo)

            # Memo unchanged and MEMO_JSON line removed from cleaned output.
            self.assertEqual(new_memo, "")
            self.assertEqual(cleaned, "Line\nMore")
