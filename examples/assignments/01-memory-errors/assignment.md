# Heap use-after-free and memory leak

## Student task

Allocate an integer array, print a short status message, and release all allocated memory safely. The starter submission intentionally reads from a freed allocation and leaks a second allocation.

## Submission

`submission.c` is an intentionally flawed starter submission used to test the grading pipeline. Fix the defects while preserving the specified input/output contract.

## How to grade

Upload this assignment's `rules.json` and your `.c` submission through the Gradio **Grade a C submission** tab. The rules refer to test files using paths relative to the repository root.

## Scope

The intended issue category is listed in `instructor_expected_findings.json`. That file is an instructor-side test oracle; do not show it to the model or student during blind evaluation.
