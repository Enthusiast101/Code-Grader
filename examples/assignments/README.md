# Code Checking Assistant — Example Grading Assignments

These five small assignments are deliberately supplied with **buggy starter C programs** so you can exercise the grading pipeline. They are regression fixtures, not secure or correct reference solutions.

| Folder | Main issue | What the grader should notice |
|---|---|---|
| `01-memory-errors` | Heap use-after-free and memory leak | Valgrind/ASan evidence, exact offending statements |
| `02-logical-leap-year` | Incorrect Gregorian leap-year condition | 1900 and 2100 fail; 2000 succeeds |
| `03-computation-average` | Integer division truncates average | Fractional averages are printed incorrectly |
| `04-bruteforce-triplets` | Cubic brute-force enumeration | Correct small answers but poor scaling; runtime sample included |
| `05-buffer-overflow` | Unbounded `%s` into `char username[8]` | Overlong input risks stack-buffer overflow |

## Files in each assignment folder

- `assignment.md`: student-facing task description.
- `submission.c`: intentionally flawed C submission.
- `rules.json`: project-compatible grading configuration.
- `testcases.json`: normal correctness tests.
- `edge_cases.json`: boundary and corner tests.
- `memory_cases.json`: inputs used by memory-safety checks.
- `runtime_cases.json`: a large input for the brute-force example; the complexity metric covers scaling now that runtime efficiency is no longer graded.
- `instructor_expected_findings.json`: hidden instructor oracle with expected defect category, line, and excerpt. Do not pass this file to the LLM.

## Use

Extract the `examples/` folder into the root of your Code Checking Assistant repository. In Gradio's **Grade a C submission** tab, upload the selected folder's `rules.json` and `submission.c`. The rules' test paths are relative to the repository root, so preserve the directory layout.

For direct grading, the application must be launched with the repository root as its working directory. Keep the `instructor_expected_findings.json` separate from model inputs; it is intended for evaluating whether the grader detects known defects.

## Safety and reproducibility

The buffer-overflow and use-after-free examples intentionally invoke undefined behaviour. Run them only inside the project's restricted Docker sandbox. Their visible output may vary; Valgrind/ASan results, not the printed output, are the primary oracle for these cases. The brute-force runtime threshold is illustrative and should be calibrated on the same machine/container used for grading.
