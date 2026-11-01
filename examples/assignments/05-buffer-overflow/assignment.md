# Bounded username input (buffer overflow)

## Student task

Read a username of at most 7 characters into a fixed-size buffer and print Hello followed by the name. Reject longer input safely; never write beyond the buffer.

## Submission

`submission.c` is an intentionally flawed starter submission used to test the grading pipeline. Fix the defects while preserving the specified input/output contract.

## How to grade

Upload this assignment's `rules.json` and your `.c` submission through the Gradio **Grade a C submission** tab. The rules refer to test files using paths relative to the repository root.

## Scope

The intended issue category is listed in `instructor_expected_findings.json`. That file is an instructor-side test oracle; do not show it to the model or student during blind evaluation.
