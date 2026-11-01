import unittest
from unittest.mock import Mock

from grading_agent.analyzers.memory_safety import run_memory_safety, _parse_valgrind_log


CLEAN_VALGRIND_LOG = """==123== Memcheck, a memory error detector
==123== Command: ./student_program_memcheck
==123== HEAP SUMMARY:
==123==     in use at exit: 0 bytes in 0 blocks
==123==   total heap usage: 2 allocs, 2 frees, 8,192 bytes allocated
==123== All heap blocks were freed -- no leaks are possible
==123== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
"""


class ValgrindCaptureTests(unittest.TestCase):
    def test_parser_recognizes_clean_valgrind_log(self):
        parsed = _parse_valgrind_log(CLEAN_VALGRIND_LOG)
        self.assertEqual(parsed["error_count"], 0)
        self.assertTrue(parsed["all_heap_blocks_freed"])
        self.assertEqual(parsed["definitely_lost"]["bytes"], 0)

    def test_valgrind_log_is_captured_from_exec_output(self):
        sandbox = Mock()
        sandbox.write_file.return_value = None
        sandbox.run.side_effect = [
            {"exit_code": 0, "output": "", "timed_out": False, "output_may_be_truncated": False},
            {"exit_code": 0, "output": CLEAN_VALGRIND_LOG, "timed_out": False, "output_may_be_truncated": False},
        ]

        result = run_memory_safety(
            sandbox,
            "int main(void) { return 0; }",
            [{"input": "", "expected": ""}],
            run_sanitizers=False,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["runtime_tools"]["valgrind_score"], 100.0)
        self.assertEqual(result["results"][0]["parsed_summary"]["error_count"], 0)
        self.assertIn("ERROR SUMMARY", result["results"][0]["raw_log"])
        # The analyzer no longer tries Docker's archive API to obtain a log.
        self.assertEqual(sandbox.read_file.call_count, 0)


if __name__ == "__main__":
    unittest.main()
