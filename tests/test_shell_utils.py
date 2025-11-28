import subprocess
import unittest
from unittest import mock

from agent_multi_cr.shell_utils import run_shell


class TestRunShellRetries(unittest.TestCase):
    def test_run_shell_success_no_retry(self) -> None:
        fake_result = subprocess.CompletedProcess(args=["echo"], returncode=0, stdout="ok", stderr="")

        with mock.patch("agent_multi_cr.shell_utils.subprocess.run", return_value=fake_result) as mock_run:
            output = run_shell(["echo"])

        self.assertEqual(output, "ok")
        mock_run.assert_called_once()

    def test_run_shell_retries_then_success(self) -> None:
        results = [
            subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout="fail1", stderr="err1"),
            subprocess.CompletedProcess(args=["cmd"], returncode=2, stdout="fail2", stderr="err2"),
            subprocess.CompletedProcess(args=["cmd"], returncode=0, stdout="ok", stderr=""),
        ]

        with mock.patch(
            "agent_multi_cr.shell_utils.subprocess.run",
            side_effect=results,
        ) as mock_run:
            output = run_shell(["cmd"])

        self.assertEqual(output, "ok")
        self.assertEqual(mock_run.call_count, 3)

    def test_run_shell_exhausts_retries_and_raises(self) -> None:
        results = [
            subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout="fail1", stderr="err1"),
            subprocess.CompletedProcess(args=["cmd"], returncode=2, stdout="fail2", stderr="err2"),
            subprocess.CompletedProcess(args=["cmd"], returncode=3, stdout="fail3", stderr="err3"),
            subprocess.CompletedProcess(args=["cmd"], returncode=4, stdout="fail4", stderr="err4"),
        ]

        with mock.patch(
            "agent_multi_cr.shell_utils.subprocess.run",
            side_effect=results,
        ) as mock_run:
            with self.assertRaises(RuntimeError) as ctx:
                run_shell(["cmd"])

        self.assertIn("after 4 attempts", str(ctx.exception))
        self.assertEqual(mock_run.call_count, 4)


