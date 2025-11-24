import io
import unittest
from unittest import mock

from agent_multi_cr.context import get_git_diff, get_stdin_context, resolve_context_text


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
            )

    @mock.patch("agent_multi_cr.context.run_shell")
    def test_get_git_diff_is_deprecated_and_returns_diff(self, mock_run_shell) -> None:
        # Simulate a non-empty diff so get_git_diff does not exit.
        mock_run_shell.return_value = "diff content"
        with self.assertWarns(DeprecationWarning):
            diff = get_git_diff(use_cached=False)
        self.assertEqual(diff, "diff content")
        mock_run_shell.assert_called_once()


if __name__ == "__main__":
    unittest.main()
