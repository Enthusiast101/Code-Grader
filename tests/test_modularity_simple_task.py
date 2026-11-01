import unittest

from grading_agent.analyzers.modularity import run_modularity


class SimpleTaskModularityTests(unittest.TestCase):
    def test_single_main_is_not_penalized_when_helpers_are_not_required(self):
        source = """#include <stdio.h>\nint main(void) {\n    int a, b;\n    if (scanf(\"%d %d\", &a, &b) != 2) return 1;\n    printf(\"%d\\n\", a + b);\n    return 0;\n}\n"""
        result = run_modularity(source, min_functions=1, expect_helpers=False)
        self.assertEqual(result["function_count"], 1)
        self.assertEqual(result["helper_function_count"], 0)
        self.assertEqual(result["score"], 100.0)


if __name__ == "__main__":
    unittest.main()
