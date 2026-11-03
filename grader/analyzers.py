from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_CONTROL = {"if", "for", "while", "switch", "catch", "sizeof"}
_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
    "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "_Imaginary",
}


def strip_comments_literals(source: str) -> str:
    """Blank comments and literals without changing offsets or line numbers."""
    out = list(source)
    i = 0
    state = "code"
    while i < len(source):
        char = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if char == '"':
                out[i] = " "
                state = "string"
            elif char == "'":
                out[i] = " "
                state = "char"
            elif char == "/" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 1
                state = "line_comment"
            elif char == "/" and nxt == "*":
                out[i] = out[i + 1] = " "
                i += 1
                state = "block_comment"
        elif state in {"string", "char"}:
            if char != "\n":
                out[i] = " "
            if char == "\\" and i + 1 < len(source):
                i += 1
                if source[i] != "\n":
                    out[i] = " "
            elif (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                out[i] = " "
        elif state == "block_comment":
            if char == "*" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 1
                state = "code"
            elif char != "\n":
                out[i] = " "
        i += 1
    return "".join(out)


def _brace_end(code: str, start: int) -> int:
    depth = 0
    for index in range(start, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(code) - 1


def analyze_source(source: str) -> dict[str, Any]:
    clean = strip_comments_literals(source)
    functions: list[dict[str, Any]] = []
    signature_pattern = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*\{")
    for match in signature_pattern.finditer(clean):
        name = match.group(1)
        if name in _CONTROL:
            continue
        # A function definition must begin after a declaration-like prefix,
        # not after assignment/member/call syntax.
        prefix = clean[max(0, match.start() - 160):match.start()].split("\n")[-1]
        if "=" in prefix or "." in prefix or "->" in prefix:
            continue
        opening = clean.find("{", match.start(), match.end())
        closing = _brace_end(clean, opening)
        body = clean[opening + 1:closing]
        start_line = clean.count("\n", 0, match.start()) + 1
        end_line = clean.count("\n", 0, closing) + 1
        nloc = sum(bool(line.strip()) for line in clean[match.start():closing + 1].splitlines())
        decisions = len(re.findall(r"\b(?:if|for|while|case)\b", body))
        decisions += len(re.findall(r"&&|\|\||\?", body))
        functions.append({
            "name": name,
            "line_start": start_line,
            "line_end": end_line,
            "length": max(1, end_line - start_line + 1),
            "nloc": nloc,
            "cyclomatic": decisions + 1,
            "parameters": 0 if not match.group(2).strip() or match.group(2).strip() == "void" else match.group(2).count(",") + 1,
        })
    # Drop false positives that are nested within an already-detected function.
    unique: list[dict[str, Any]] = []
    for function in sorted(functions, key=lambda x: (x["line_start"], x["line_end"])):
        if any(parent["line_start"] <= function["line_start"] <= parent["line_end"] and parent["name"] != function["name"] for parent in unique):
            continue
        if not any(item["name"] == function["name"] and item["line_start"] == function["line_start"] for item in unique):
            unique.append(function)
    lines = source.splitlines()
    nonblank = [line for line in lines if line.strip()]
    long_lines = [index for index, line in enumerate(lines, 1) if len(line) > 100]
    tokens = re.findall(r"[A-Za-z_]\w*", clean)
    identifiers = sorted({token for token in tokens if token not in _KEYWORDS})
    comments = len(re.findall(r"(?m)^\s*(?://|/\*|\*|\*/)", source))
    return {
        "line_count": len(lines),
        "nonblank_lines": len(nonblank),
        "function_count": len(unique),
        "functions": unique,
        "has_main": any(item["name"] == "main" for item in unique),
        "max_cyclomatic": max((item["cyclomatic"] for item in unique), default=0),
        "max_function_length": max((item["length"] for item in unique), default=0),
        "long_lines": long_lines,
        "identifier_count": len(identifiers),
        "single_character_identifiers": [name for name in identifiers if len(name) == 1],
        "comment_lines": comments,
    }


def score_structure(metrics: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    max_cc = int(params.get("max_cyclomatic", 12))
    max_length = int(params.get("max_function_length", 80))
    min_functions = int(params.get("min_functions", 1))
    penalty = float(params.get("violation_penalty", 15))
    checks = [
        ("main function defined", bool(metrics["has_main"])),
        (f"at least {min_functions} function(s)", metrics["function_count"] >= min_functions),
        (f"maximum cyclomatic complexity <= {max_cc}", metrics["max_cyclomatic"] <= max_cc),
        (f"maximum function length <= {max_length} lines", metrics["max_function_length"] <= max_length),
    ]
    failed = [label for label, passed in checks if not passed]
    score = max(0.0, 100.0 - penalty * len(failed))
    return {
        "status": "completed", "score": round(score, 2), "checks": [{"criterion": k, "passed": v} for k, v in checks],
        "findings": failed, "source_metrics": metrics,
        "rubric": f"100 points minus {penalty:g} for each failed configured structural check; floor 0.",
        "rationale": "All configured structural checks passed." if not failed else "Failed checks: " + "; ".join(failed) + ".",
    }


def _similarity_tokens(source: str) -> list[str]:
    """Tokenize C while making identifier renaming and comment edits less effective."""
    clean = strip_comments_literals(source)
    tokens = re.findall(
        r"==|!=|<=|>=|\+\+|--|->|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|"
        r"[A-Za-z_]\w*|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?)|[^\s]",
        clean,
    )
    preprocessor = {"include", "define", "undef", "ifdef", "ifndef", "if", "elif", "else", "endif", "pragma"}
    renamed: dict[str, str] = {}
    normalized: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z_]\w*", token):
            if token in _KEYWORDS or token in preprocessor:
                normalized.append(token)
            else:
                normalized.append(renamed.setdefault(token, f"ID{len(renamed)}"))
        elif token[:1].isdigit():
            normalized.append("NUM")
        else:
            normalized.append(token)
    return normalized


def originality_similarity(
    source: str, source_path: str | Path, corpus_dir: str | Path, *, threshold: float = 0.85, min_tokens: int = 80, max_files: int = 500, max_file_bytes: int = 1_000_000
) -> dict[str, Any]:
    """Flag high code-token overlap against a local C corpus; this is triage, not proof."""
    current_path = Path(source_path).resolve()
    corpus = Path(corpus_dir).resolve()
    if not corpus.is_dir():
        return {"status": "unavailable", "flagged": False, "reason": f"Comparison corpus not found: {corpus}"}

    def shingles(code: str) -> set[tuple[str, ...]]:
        tokens = _similarity_tokens(code)
        if len(tokens) < min_tokens:
            return set()
        width = 6
        return {tuple(tokens[index:index + width]) for index in range(len(tokens) - width + 1)}

    current = shingles(source)
    if not current:
        return {"status": "unavailable", "flagged": False, "reason": f"Submission has fewer than {min_tokens} normalized C tokens; comparison skipped."}

    candidates = sorted(
        (path for path in corpus.rglob("*.c") if path.is_file() and not path.is_symlink() and path.resolve() != current_path),
        key=lambda path: str(path.relative_to(corpus)),
    )[:max_files]
    comparisons = []
    for path in candidates:
        try:
            if path.stat().st_size > max_file_bytes:
                continue
            peer_source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        peer = shingles(peer_source)
        if not peer:
            continue
        shared = len(current & peer)
        union = len(current | peer)
        containment = shared / min(len(current), len(peer))
        jaccard = shared / union if union else 0.0
        comparisons.append({
            "file": str(path.relative_to(corpus)),
            "containment_similarity": round(containment, 4),
            "jaccard_similarity": round(jaccard, 4),
            "shared_shingles": shared,
        })
    if not comparisons:
        return {"status": "unavailable", "flagged": False, "comparisons": 0,
                "reason": "No comparable C files were found in the configured corpus."}
    comparisons.sort(key=lambda item: (-item["containment_similarity"], -item["jaccard_similarity"], item["file"]))
    best = comparisons[0]
    flagged = best["containment_similarity"] >= threshold
    return {
        "status": "completed", "flagged": flagged, "threshold": threshold,
        "similarity_method": "identifier-normalized six-token shingles; containment ratio",
        "max_similarity": best["containment_similarity"], "max_jaccard_similarity": best["jaccard_similarity"],
        "matched_file": best["file"] if flagged else None, "comparisons": len(comparisons),
        "top_matches": comparisons[:5],
        "reason": (f"High lexical overlap with {best['file']} ({best['containment_similarity']:.1%}); instructor review recommended."
                   if flagged else f"No corpus match reached the {threshold:.0%} review threshold; this does not establish originality."),
        "disclaimer": "Similarity is a review signal, not proof of plagiarism or LLM use. Results depend on the completeness and relevance of the supplied corpus.",
    }
