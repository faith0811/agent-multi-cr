import io
import unittest
from unittest import mock

from src.context import get_stdin_context, resolve_context_text


class TestContextHelpers(unittest.TestCase):
    def test_get_stdin_context_reads_data(self) -> None:
        fake_stdin = io.StringIO("some input\n")
        with mock.patch("sys.stdin", fake_stdin):
            result = get_stdin_context()
        self.assertEqual(result, "some input\n")

    def test_get_stdin_context_raises_on_empty(self) -> None:
        fake_stdin = io.StringIO("   \n")
        with mock.patch("sys.stdin", fake_stdin):
            with self.assertRaises(SystemExit):
                get_stdin_context()

    def test_resolve_context_unknown_mode_raises(self) -> None:
        with self.assertRaises(SystemExit):
            resolve_context_text(
                context_mode="unknown",
                use_cached=False,
                repo_root=".",
                max_context_files=10,
                max_context_bytes_per_file=1000,
            )


if __name__ == "__main__":
    unittest.main()

