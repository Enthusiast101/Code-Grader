import json
import tempfile
import unittest
from pathlib import Path

from grading_agent.preflight import run_preflight


class FakeSandbox:
    def __init__(self, compile_result, run_result=None):
        self.compile_result = compile_result
        self.run_result = run_result or {
            "exit_code": 0,
            "output": "ok\\n",
            "timed_out": False,
            "output_may_be_truncated": False,
        }
        self.written = []
        self.commands = []

    def write_file(self, filename, content):
        self.written.append((filename, content))

    def run(self, command, timeout_s=None, stdin=None):
        self.commands.append((command, timeout_s, stdin))
        if command[0] == "gcc":
            return self.compile_result
        return self.run_result


class PreflightTests(unittest.TestCase):
    def test_compile_failure_is_reported(self):
        sandbox = FakeSandbox({
            "exit_code": 1, "output": "syntax error", "timed_out": False,
            "output_may_be_truncated": False,
        })
        result = run_preflight(
            sandbox, source_code="bad", source_filename="main.c",
            rules={"time_limit_s": 2, "metrics": []},
        )
        self.assertEqual(result["status"], "compile_failed")
        self.assertFalse(result["compiled"])

    def test_smoke_timeout_sets_suspected_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            testset = Path(tmp) / "tests.json"
            testset.write_text(json.dumps([{"input": "1\n", "expected": "1\n"}]), encoding="utf-8")
            sandbox = FakeSandbox(
                {"exit_code": 0, "output": "", "timed_out": False, "output_may_be_truncated": False},
                {"exit_code": 124, "output": "", "timed_out": True, "output_may_be_truncated": False},
            )
            result = run_preflight(
                sandbox, source_code="int main(void){return 0;}", source_filename="main.c",
                rules={
                    "time_limit_s": 2,
                    "metrics": [{
                        "enabled": True, "analyzer": "io_tests",
                        "params": {"resolved_testset_path": str(testset)},
                    }],
                },
            )
            self.assertEqual(result["status"], "timeout")
            self.assertTrue(result["suspected_infinite_loop"])


if __name__ == "__main__":
    unittest.main()
