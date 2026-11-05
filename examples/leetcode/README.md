# LeetCode-style grading examples

Medium and hard LeetCode problems rewritten as stdin/stdout C assignments. Each folder has `assignment.md`, a student `submission.c` with a realistic planted defect, `rules.json` (all nine metrics), correctness, edge and memory test files, a hidden `instructor_expected_findings.json` oracle, a `reference_solution.c` and `validate_reference.py`.

| Folder | Level | Planted defect in `submission.c` | Similarity setup | Expected similarity points |
|---|---|---|---|---|
| `0042-trapping-rain-water` | Hard | 32-bit total overflows on large inputs | `submissions/` with a renamed copy of the submission | 0/10 (similarity 1.00) |
| `0056-merge-intervals` | Medium | Touching intervals not merged (`<` instead of `<=`) | `submissions/` with a partial copy | about 3.6/10 (similarity 0.67) |
| `0200-number-of-islands` | Medium | Grid never freed (memory leak) | `submissions/`; the submission is original, but two cohort files copy each other | 10/10 (cohort pair flagged) |
| `0239-sliding-window-maximum` | Hard | O(n*k) window rescans time out on the large case | No folder (`corpus_dir: null`) | 10/10 |
| `1192-critical-connections` | Hard | Bridge test uses `>=` instead of `>` | `originality_corpus/` (unrelated baseline) | 10/10 |

The `submissions/` folders simulate a class of submissions for the same assignment. The similarity report lists the graded submission's similarity to every file and the full pairwise matrix, so copied pairs between other students are flagged too.

To grade one in Gradio, upload the folder's `rules.json` and `submission.c`. Paths in the rules are relative to the repository root. Run `python examples/leetcode/<folder>/validate_reference.py` (with GCC installed) to re-check the stored expected outputs.
