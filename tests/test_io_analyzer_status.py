import unittest

from grading_agent.analyzers.io_tests import run_io_tests


class FakeSandbox:
    def __init__(self):
        self.calls = 0

    def write_file(self, filename, content):
        self.filename = filename
        self.content = content

    def run(self, command, timeout_s, stdin=None):
        self.calls += 1
        if command[0] == "gcc":
            return {"exit_code": 0, "timed_out": False, "output": ""}
        return {
            "exit_code": 0,
            "timed_out": False,
            "output": "42\n",
            "output_may_be_truncated": False,
        }


class IOAnalyzerStatusTests(unittest.TestCase):
    def test_successful_io_run_has_completed_status(self):
        result = run_io_tests(
            FakeSandbox(),
            "int main(void) { return 0; }",
            [{"input": "", "expected": "42"}],
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["tests_passed"], 1)


if __name__ == "__main__":
    unittest.main()
