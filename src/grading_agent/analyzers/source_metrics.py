"""C source metrics using Lizard where available, with a lexical fallback.

Lizard supplies cyclomatic complexity and per-function NLOC. If the package
is missing or cannot parse a file, lightweight lexical estimates are used
and the returned measurement note identifies that fallback. Compiler and
cppcheck diagnostics remain the authoritative syntax/static checks.
"""
from __future__ import annotations

import re
from typing import Any

from grading_agent.config import settings


_CONTROL_WORDS = {
    "if", "for", "while", "case", "catch", "switch", "sizeof",
    "return", "_Alignof",
}
_C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile",
    "while", "_Bool", "_Complex", "_Imaginary",
}
_FUNCTION_RE = re.compile(
    r"(?m)^[ \t]*(?:(?:static|inline|extern|const|volatile|unsigned|signed|long|short|struct\s+\w+|enum\s+\w+|union\s+\w+|[A-Za-z_]\w*)\s+|\*\s*)+"
    r"\**\s*([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*\{"
)
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|&&|\|\||[^\s]")


def strip_comments_and_literals(source: str) -> str:
    """Replace comments and literals with spaces while preserving newlines."""
    pattern = re.compile(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        re.MULTILINE,
    )

    def replace(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return pattern.sub(replace, source)


def _matching_brace_end(code: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(code) - 1



def _lizard_metrics(filename: str, source: str) -> dict[str, Any] | None:
    """Use Lizard when installed; return None for the local lexical fallback."""
    try:
        import lizard  # type: ignore
    except ImportError:
        return None
    try:
        info = lizard.analyze_file.analyze_source_code(filename, source)
        functions = []
        for function in info.function_list:
            nloc = int(getattr(function, "nloc", 0) or 0)
            functions.append({
                "name": str(getattr(function, "name", "<anonymous>")),
                "start_line": int(getattr(function, "start_line", 0) or 0),
                "end_line": None,
                "length": nloc,
                "nloc": nloc,
                "cyclomatic_complexity_estimate": int(getattr(function, "cyclomatic_complexity", 1) or 1),
                "parameter_count_estimate": int(getattr(function, "parameter_count", 0) or 0),
                "token_count": int(getattr(function, "token_count", 0) or 0),
            })
        return {
            "functions": functions,
            "function_count": len(functions),
            "max_cyclomatic_complexity_estimate": max((item["cyclomatic_complexity_estimate"] for item in functions), default=0),
            "max_function_length": max((item["length"] for item in functions), default=0),
            "measurement_note": "Lizard-based cyclomatic complexity and per-function NLOC.",
        }
    except Exception:
        # A parse error should not silently crash all grading. The explicit
        # note makes the fallback visible in the report.
        return None


def analyze_source(source: str, filename: str = settings.DEFAULT_SOURCE_FILENAME) -> dict[str, Any]:
    """Return heuristic function, complexity, line and identifier metrics."""
    clean = strip_comments_and_literals(source)
    lizard_result = _lizard_metrics(filename, source)
    functions: list[dict[str, Any]] = []

    for match in _FUNCTION_RE.finditer(clean):
        name = match.group(1)
        if name in _CONTROL_WORDS:
            continue
        open_brace = clean.find("{", match.start(), match.end())
        if open_brace < 0:
            continue
        close_brace = _matching_brace_end(clean, open_brace)
        body = clean[open_brace + 1:close_brace]
        start_line = clean.count("\n", 0, match.start()) + 1
        end_line = clean.count("\n", 0, close_brace) + 1
        nloc = sum(1 for line in clean[match.start():close_brace + 1].splitlines() if line.strip())
        decision_count = len(re.findall(r"\b(?:if|for|while|case)\b", body))
        decision_count += len(re.findall(r"&&|\|\||\?", body))
        functions.append({
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "length": max(1, end_line - start_line + 1),
            "nloc": nloc,
            "cyclomatic_complexity_estimate": 1 + decision_count,
            "parameter_count_estimate": 0 if not match.group(2).strip() or match.group(2).strip() == "void" else match.group(2).count(",") + 1,
        })

    if lizard_result is not None:
        functions = lizard_result["functions"]

    lines = source.splitlines()
    nonblank_lines = [line for line in lines if line.strip()]
    comment_lines = sum(1 for line in lines if line.strip().startswith(("//", "/*", "*", "*/")))
    code_lines = [line for line in strip_comments_and_literals(source).splitlines() if line.strip()]
    identifiers = [
        token for token in _TOKEN_RE.findall(clean)
        if re.fullmatch(r"[A-Za-z_]\w*", token) and token not in _C_KEYWORDS
    ]

    return {
        "line_count": len(lines),
        "nonblank_line_count": len(nonblank_lines),
        "comment_line_count": comment_lines,
        "function_count": len(functions),
        "functions": functions,
        "max_cyclomatic_complexity_estimate": max(
            (item["cyclomatic_complexity_estimate"] for item in functions), default=0
        ),
        "max_function_length": max((item["length"] for item in functions), default=0),
        "identifier_count": len(identifiers),
        "single_character_identifier_count": sum(1 for item in identifiers if len(item) == 1),
        "long_line_count": sum(1 for line in lines if len(line) > settings.ANALYZER_MAX_LINE_LENGTH),
        "average_line_length": round(
            sum(len(line) for line in nonblank_lines) / len(nonblank_lines), 2
        ) if nonblank_lines else 0.0,
        "comment_ratio": round(comment_lines / max(1, len(code_lines)), 4),
        "measurement_note": (
            lizard_result["measurement_note"] if lizard_result is not None else
            "Lizard unavailable or unable to parse this file; function and cyclomatic-complexity values are lexical estimates, not a full compiler AST analysis."
        ),
    }
