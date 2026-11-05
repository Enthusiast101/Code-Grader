"""Deterministic C syntax-tree analysis with pycparser.

The submission is parsed into an abstract syntax tree and reduced to facts
that the model can rely on instead of re-reading source text:

* functions (signature, line range, calls, recursion);
* loops (kind, line, nesting depth, condition) and the deepest nesting;
* branch conditions, every comparison and the constants it tests against;
* input/output calls (format strings, whether scanf's result is checked);
* heap allocations and frees per variable, and reads after a free;
* integer divisions, narrow integer accumulators and fixed-size arrays.

From these, ``edge_case_targets`` lists concrete, line-anchored boundaries
worth testing (e.g. ``a < b`` at line 31 -> test a == b). The same facts are
added to the grading evidence ledger. They are structural observations, not
proven defects.

pycparser needs preprocessed C, so ``#include`` lines are blanked (keeping
line numbers), simple object-like ``#define`` constants are substituted, and
common library typedefs are declared before a ``#line 1`` marker.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

try:
    from pycparser import c_ast, c_generator, c_parser
except ImportError:  # pragma: no cover - reported as unavailable
    c_ast = c_generator = c_parser = None

_PRELUDE_TYPES = (
    "size_t ssize_t ptrdiff_t intptr_t uintptr_t intmax_t uintmax_t "
    "int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t "
    "int_least8_t int_least16_t int_least32_t int_least64_t "
    "uint_least8_t uint_least16_t uint_least32_t uint_least64_t "
    "int_fast8_t int_fast16_t int_fast32_t int_fast64_t "
    "uint_fast8_t uint_fast16_t uint_fast32_t uint_fast64_t "
    "time_t clock_t wchar_t wint_t va_list off_t"
).split()
_PRELUDE = "".join(f"typedef int {name};\n" for name in _PRELUDE_TYPES) + (
    "typedef _Bool bool;\n"
    "typedef struct __pycparser_FILE FILE;\n"
    "typedef struct __pycparser_div div_t;\n"
)
_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([^\n]+?)\s*$")
_DIRECTIVE = re.compile(r"^\s*#")
_ALLOCATORS = {"malloc", "calloc", "realloc", "strdup", "aligned_alloc"}
_INPUT_FUNCS = {"scanf", "fscanf", "sscanf", "getchar", "fgets", "getline", "fgetc", "getc", "gets"}
_OUTPUT_FUNCS = {"printf", "fprintf", "puts", "putchar", "fputs", "sprintf", "snprintf"}
_NARROW_INT = {"int", "short", "char", "unsigned int", "unsigned short", "signed int", "int32_t", "uint32_t", "int16_t"}
_COMPARISONS = {"<", "<=", ">", ">=", "==", "!="}
MAX_TARGETS = 30


_FORMAT_MACRO = re.compile(r"\b(?:PRI|SCN)[diouxX](?:8|16|32|64|MAX|PTR|LEAST(?:8|16|32|64)|FAST(?:8|16|32|64))\b")


def strip_comments(source: str) -> str:
    """Remove // and /* */ comments, keeping string/char literals and every newline."""
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch in "\"'":
            quote, j = ch, i + 1
            while j < n and source[j] != quote and source[j] != "\n":
                j += 2 if source[j] == "\\" else 1
            out.append(source[i:j + 1])
            i = j + 1
        elif ch == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(" " + "\n" * source.count("\n", i, end))
            i = end
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def preprocess(source: str) -> str:
    """Make student C parseable by pycparser while preserving line numbers."""
    defines: dict[str, str] = {}
    lines = []
    text = strip_comments(source.replace("\r\n", "\n").replace("\r", "\n"))
    # <inttypes.h> format macros expand to string literals; any literal keeps the
    # format string well formed for parsing (the exact conversion is irrelevant here).
    text = _FORMAT_MACRO.sub('"d"', text)
    for line in text.split("\n"):
        match = _DEFINE.match(line)
        if match and "(" not in match.group(1):
            value = re.sub(r"//.*$|/\*.*?\*/", "", match.group(2)).strip()
            if value:
                defines[match.group(1)] = value
            lines.append("")
        elif _DIRECTIVE.match(line):
            lines.append("")
        else:
            lines.append(line)
    text = "\n".join(lines)
    for name, value in defines.items():
        # A function replacement: macro values may contain backslashes ("\x1b").
        text = re.sub(rf"\b{re.escape(name)}\b", lambda _match, value=value: f"({value})", text)
    return _PRELUDE + '#line 1 "submission.c"\n' + text


def _line(node: Any) -> int | None:
    coord = getattr(node, "coord", None)
    return coord.line if coord is not None and coord.line else None


def _type_name(node: Any) -> str:
    """Readable C type of a declaration node (pointer/array aware)."""
    if isinstance(node, c_ast.TypeDecl):
        inner = node.type
        if isinstance(inner, c_ast.IdentifierType):
            return " ".join(inner.names)
        if isinstance(inner, (c_ast.Struct, c_ast.Union)):
            return f"struct {inner.name or ''}".strip()
        if isinstance(inner, c_ast.Enum):
            return f"enum {inner.name or ''}".strip()
    if isinstance(node, c_ast.PtrDecl):
        return _type_name(node.type) + " *"
    if isinstance(node, c_ast.ArrayDecl):
        return _type_name(node.type) + " []"
    if isinstance(node, c_ast.FuncDecl):
        return _type_name(node.type) + " ()"
    return "?"


class _FactCollector(c_ast.NodeVisitor if c_ast else object):
    def __init__(self) -> None:
        self.generator = c_generator.CGenerator()
        self.functions: list[dict[str, Any]] = []
        self.loops: list[dict[str, Any]] = []
        self.conditions: list[dict[str, Any]] = []
        self.comparisons: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.allocations: list[dict[str, Any]] = []
        self.frees: list[dict[str, Any]] = []
        self.divisions: list[dict[str, Any]] = []
        self.accumulators: list[dict[str, Any]] = []
        self.arrays: list[dict[str, Any]] = []
        self.globals: list[str] = []
        self.identifiers: set[str] = set()
        self.types: dict[str, str] = {}
        self._function: str | None = None
        self._loop_depth = 0
        self._in_condition = 0
        self._max_line = 0
        # Input data flow: which variables scanf-style calls write, and where
        # each identifier is read outside input calls and their validation.
        self.input_targets: list[dict[str, Any]] = []
        self.non_input_uses: dict[str, list[int]] = defaultdict(list)
        self.params: set[str] = set()
        self._induction: list[str] = []
        self._input_context = 0

    # -- helpers -------------------------------------------------------
    def text(self, node: Any) -> str:
        try:
            return self.generator.visit(node)[:160]
        except Exception:
            return "?"

    def _note_line(self, node: Any) -> None:
        line = _line(node)
        if line and line > self._max_line:
            self._max_line = line

    def generic_visit(self, node: Any) -> None:
        self._note_line(node)
        super().generic_visit(node)

    def _expr_type(self, node: Any) -> str:
        if isinstance(node, c_ast.ID):
            return self.types.get(node.name, "?")
        if isinstance(node, c_ast.Constant):
            return "double" if node.type in {"double", "float"} else ("int" if node.type == "int" else node.type)
        if isinstance(node, c_ast.Cast):
            return _type_name(node.to_type.type)
        if isinstance(node, c_ast.ArrayRef):
            base = self._expr_type(node.name)
            return base.replace(" []", "").replace(" *", "")
        if isinstance(node, c_ast.BinaryOp) and node.op in {"+", "-", "*", "/", "%"}:
            left, right = self._expr_type(node.left), self._expr_type(node.right)
            if "double" in (left, right) or "float" in (left, right):
                return "double"
            return left if left != "?" else right
        return "?"

    # -- visitors ------------------------------------------------------
    def visit_FuncDef(self, node: Any) -> None:
        decl = node.decl
        name = decl.name
        params = []
        args = getattr(decl.type, "args", None)
        for param in getattr(args, "params", []) or []:
            if isinstance(param, c_ast.Decl):
                params.append(f"{_type_name(param.type)} {param.name}")
                self.types[param.name] = _type_name(param.type)
                self.params.add(param.name)
        previous, self._function = self._function, name
        before_calls = len(self.calls)
        start = _line(node)
        self._max_line = start or 0
        self.visit(node.body)
        calls = sorted({call["name"] for call in self.calls[before_calls:]})
        self.functions.append({
            "name": name,
            "returns": _type_name(decl.type.type),
            "params": params,
            "line_start": start,
            "line_end": self._max_line or start,
            "calls": calls,
            "recursive": name in calls,
        })
        self._function = previous

    def visit_Decl(self, node: Any) -> None:
        if node.name:
            type_name = _type_name(node.type)
            self.types[node.name] = type_name
            self.identifiers.add(node.name)
            if self._function is None and not isinstance(node.type, c_ast.FuncDecl):
                self.globals.append(node.name)
            if isinstance(node.type, c_ast.ArrayDecl):
                self.arrays.append({
                    "name": node.name, "line": _line(node), "element_type": _type_name(node.type.type),
                    "size": self.text(node.type.dim) if node.type.dim is not None else "unsized",
                    "function": self._function,
                })
        if node.init is not None:
            self._check_allocation(node.name, node.init, node)
        self.generic_visit(node)

    def visit_Assignment(self, node: Any) -> None:
        target = self.text(node.lvalue)
        self._check_allocation(target, node.rvalue, node)
        if node.op in {"+=", "*=", "-="} and self._loop_depth > 0:
            declared = self._expr_type(node.lvalue)
            if declared in _NARROW_INT:
                self.accumulators.append({
                    "variable": target, "type": declared, "line": _line(node), "operator": node.op,
                    "function": self._function,
                })
        self.generic_visit(node)

    def _check_allocation(self, target: str | None, value: Any, node: Any) -> None:
        inner = value.expr if isinstance(value, c_ast.Cast) else value
        if isinstance(inner, c_ast.FuncCall) and isinstance(inner.name, c_ast.ID) and inner.name.name in _ALLOCATORS:
            self.allocations.append({
                "variable": target, "allocator": inner.name.name, "line": _line(node),
                "function": self._function, "loop_depth": self._loop_depth,
            })

    def _loop(self, node: Any, kind: str, condition: Any, induction: list[str] | None = None) -> None:
        self._loop_depth += 1
        induction = induction or []
        self._induction.extend(induction)
        self.loops.append({
            "kind": kind, "line": _line(node), "depth": self._loop_depth,
            "condition": self.text(condition) if condition is not None else "(none)",
            "function": self._function,
            **({"induction_variables": induction} if induction else {}),
        })
        if condition is not None:
            self._visit_condition(condition)
        for child_name, child in node.children():
            if child is not condition:
                self.visit(child)
        if induction:
            del self._induction[-len(induction):]
        self._loop_depth -= 1

    def _visit_condition(self, condition: Any) -> None:
        # A condition that performs input (``scanf(...) != 1 || n > 30``)
        # validates that input; its comparisons are not program logic.
        is_input = int(_calls_input(condition))
        self._in_condition += 1
        self._input_context += is_input
        self.visit(condition)
        self._input_context -= is_input
        self._in_condition -= 1

    def visit_For(self, node: Any) -> None:
        induction: list[str] = []
        init = node.init
        items = list(getattr(init, "decls", None) or getattr(init, "exprs", None) or ([init] if init is not None else []))
        for item in items:
            if isinstance(item, c_ast.Decl) and item.name:
                induction.append(item.name)
            elif isinstance(item, c_ast.Assignment) and isinstance(item.lvalue, c_ast.ID):
                induction.append(item.lvalue.name)
        self._loop(node, "for", node.cond, induction)

    def visit_While(self, node: Any) -> None:
        self._loop(node, "while", node.cond)

    def visit_DoWhile(self, node: Any) -> None:
        self._loop(node, "do-while", node.cond)

    def visit_If(self, node: Any) -> None:
        self.conditions.append({"line": _line(node), "condition": self.text(node.cond), "function": self._function})
        self._visit_condition(node.cond)
        if node.iftrue is not None:
            self.visit(node.iftrue)
        if node.iffalse is not None:
            self.visit(node.iffalse)

    def visit_TernaryOp(self, node: Any) -> None:
        self.conditions.append({"line": _line(node), "condition": self.text(node.cond), "function": self._function})
        self.generic_visit(node)

    def visit_BinaryOp(self, node: Any) -> None:
        if node.op in _COMPARISONS:
            constants = [self.text(side) for side in (node.left, node.right) if isinstance(side, c_ast.Constant)]
            sides = (node.left, node.right)
            self.comparisons.append({
                "line": _line(node), "operator": node.op, "left": self.text(node.left),
                "right": self.text(node.right), "constants": constants, "function": self._function,
                "loop_depth": self._loop_depth,
                # ``i < n`` on a loop counter: the counter is not an input value.
                "on_loop_counter": any(isinstance(side, c_ast.ID) and side.name in self._induction for side in sides),
                "input_validation": self._input_context > 0,
                "tests_input_call_result": any(_calls_input(side) for side in sides),
            })
        if node.op == "/":
            left, right = self._expr_type(node.left), self._expr_type(node.right)
            integral = lambda t: t not in {"?", "double", "float", "long double"} and "double" not in t and "float" not in t  # noqa: E731
            if integral(left) and integral(right):
                self.divisions.append({"line": _line(node), "expression": self.text(node), "function": self._function})
        self.generic_visit(node)

    def visit_FuncCall(self, node: Any) -> None:
        name = node.name.name if isinstance(node.name, c_ast.ID) else self.text(node.name)
        args = list(getattr(node.args, "exprs", []) or [])
        record: dict[str, Any] = {"name": name, "line": _line(node), "function": self._function, "loop_depth": self._loop_depth}
        if name in _INPUT_FUNCS or name in _OUTPUT_FUNCS:
            fmt = next((a.value for a in args if isinstance(a, c_ast.Constant) and a.type == "string"), None)
            if fmt:
                record["format"] = fmt[:80]
        if name in {"scanf", "fscanf", "sscanf"}:
            record["result_checked"] = self._in_condition > 0
            destinations = args[2:] if name in {"fscanf", "sscanf"} else args[1:]
            for arg in destinations:
                base = _input_destination(arg)
                if base:
                    self.input_targets.append({
                        "variable": base, "line": _line(node), "call": name, "function": self._function,
                    })
        if name == "free" and args:
            self.frees.append({"variable": self.text(args[0]), "line": _line(node), "function": self._function})
        self.calls.append(record)
        is_input = int(name in _INPUT_FUNCS)
        self._input_context += is_input
        self.generic_visit(node)
        self._input_context -= is_input

    def visit_ID(self, node: Any) -> None:
        self.identifiers.add(node.name)
        if not self._input_context:
            self.non_input_uses[node.name].append(_line(node) or 0)
        self._note_line(node)


def _calls_input(node: Any) -> bool:
    """Whether an expression contains a call to an input function."""
    if node is None:
        return False
    if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name in _INPUT_FUNCS:
        return True
    return any(_calls_input(child) for _, child in node.children())


def _input_destination(arg: Any) -> str | None:
    """Base variable a scanf argument writes: ``&x``, ``&a[i][j]`` or ``&s.f``.

    Bare pointers (``scanf("%s", buf)`` or a pointer parameter) are skipped:
    their reads may happen through another name, so "never used" is unprovable.
    """
    if not (isinstance(arg, c_ast.UnaryOp) and arg.op == "&"):
        return None
    node = arg.expr
    while isinstance(node, (c_ast.ArrayRef, c_ast.StructRef)):
        if isinstance(node, c_ast.StructRef) and node.type == "->":
            return None
        node = node.name
    return node.name if isinstance(node, c_ast.ID) else None


def _ignored_inputs(collector: "_FactCollector") -> list[dict[str, Any]]:
    """Variables filled from input that nothing outside input/validation reads.

    For example ``scanf("%d", &blocked[i][j])`` where ``blocked`` never appears
    in the computation: the program cannot depend on those values, so inputs
    where they should change the answer are the most direct tests.
    """
    ignored: dict[str, dict[str, Any]] = {}
    for target in collector.input_targets:
        name = target["variable"]
        if name in collector.params:
            continue
        if not collector.non_input_uses.get(name):
            ignored.setdefault(name, target)
    return list(ignored.values())


def _uses_after_free(source_lines: list[str], frees: list[dict[str, Any]], functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lines in the same function that read a pointer after it was freed (no reassignment in between)."""
    findings = []
    spans = {f["name"]: (f["line_start"], f["line_end"]) for f in functions}
    for item in frees:
        name = item["variable"]
        if not re.fullmatch(r"[A-Za-z_]\w*", name or ""):
            continue
        start, end = spans.get(item["function"], (item["line"], item["line"]))
        for number in range(item["line"] + 1, (end or item["line"]) + 1):
            text = source_lines[number - 1] if 0 < number <= len(source_lines) else ""
            if re.search(rf"\b{name}\s*=(?!=)", text):
                break  # reassigned
            if re.search(rf"\b{name}\s*(\[|->)|\*\s*{name}\b", text):
                findings.append({"variable": name, "freed_line": item["line"], "used_line": number})
                break
            if re.search(r"\breturn\b|\bexit\s*\(", text):
                break  # the free was on an exit path (e.g. an error branch)
    return findings


def _edge_targets(facts: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []

    def add(line: Any, kind: str, detail: str, suggestion: str) -> None:
        if len(targets) < MAX_TARGETS * 3:
            targets.append({"line": line, "kind": kind, "detail": detail, "suggestion": suggestion})

    for item in facts.get("ignored_inputs", []):
        add(item["line"], "ignored_input",
            f"{item['variable']} is read by {item['call']} at line {item['line']} but never used outside input and its validation",
            f"small inputs where the values read into {item['variable']} must change the expected output")
    for item in facts["comparisons"]:
        op, left, right = item["operator"], item["left"], item["right"]
        if item.get("tests_input_call_result"):
            continue  # ``scanf(...) != 1`` compares a return value, not an input value
        if item.get("on_loop_counter"):
            continue  # ``i < n``: loop_bounds covers the size; i itself is not an input
        if item.get("input_validation"):
            if op not in {"<", "<=", ">", ">="}:
                continue  # ``v != 0 && v != 1`` enumerates allowed values, not a limit
            for constant in item["constants"]:
                if re.fullmatch(r"-?\d+", constant):
                    add(item["line"], "validated_input_limit", f"{left} {op} {right}",
                        f"the valid value nearest {constant} (exactly at the limit); values beyond it are invalid input, "
                        "and a size limit counts only if the input stays within the generated-input length")
            continue
        has_constant = any(re.fullmatch(r"-?\d+", constant) for constant in item["constants"])
        if op in {"<", "<=", ">", ">="} and not has_constant:
            kind = "loop_comparison_boundary" if item.get("loop_depth", 0) > 0 else "comparison_boundary"
            add(item["line"], kind, f"{left} {op} {right}",
                f"inputs where {left} equals {right} and differs by one on each side")
        modulus = re.search(r"%\s*\(?\s*(\d+)", f"{left} {right}")
        if modulus:
            k = int(modulus.group(1))
            add(item["line"], "modulus_condition", f"{left} {op} {right}",
                f"values that are multiples of {k} and values that are not, including 0 and negatives if allowed")
            continue
        for constant in item["constants"]:
            if re.fullmatch(r"-?\d+", constant):
                value = int(constant)
                add(item["line"], "constant_boundary", f"{left} {op} {right}",
                    f"values {value - 1}, {value} and {value + 1} for the compared quantity, only those the assignment allows")
    for loop in facts["loops"]:
        add(loop["line"], "loop_bounds", f"{loop['kind']} ({loop['condition']})",
            "zero iterations (if valid), exactly one iteration, and a few iterations")
    for acc in facts["accumulators"]:
        add(acc["line"], "integer_overflow", f"{acc['variable']} ({acc['type']}) {acc['operator']} inside a loop",
            "the largest allowed values, with as many of them as fit the generated-input length, to push the total past the 32-bit range")
    for div in facts["divisions"]:
        add(div["line"], "integer_division", div["expression"],
            "operands that do not divide evenly, and negative operands")
    for array in facts["arrays"]:
        add(array["line"], "fixed_capacity", f"{array['name']}[{array['size']}]",
            "an input reaching the array capacity only if that input fits the generated-input length; "
            "otherwise skip it (instructor performance cases cover maximum sizes)")
    for call in facts["input_calls"]:
        if call["name"] in {"scanf", "fscanf"} and not call.get("result_checked"):
            add(call["line"], "unchecked_input", f"{call['name']}({call.get('format', '')}) result not checked",
                "only if the assignment defines behaviour for missing or malformed input")
    for fn in facts["functions"]:
        if fn["recursive"]:
            add(fn["line_start"], "recursion_depth", f"{fn['name']} calls itself",
                "the deepest recursion that fits the generated-input length")
    # Most informative first; input-validation comparisons (scanf(...) != 1) last.
    priority = {
        "ignored_input": -1,
        "validated_input_limit": 6,
        "integer_overflow": 0, "integer_division": 1, "modulus_condition": 1, "loop_comparison_boundary": 2, "fixed_capacity": 3,
        "comparison_boundary": 4, "recursion_depth": 4, "loop_bounds": 5, "constant_boundary": 6,
        "unchecked_input": 7,
    }
    def rank(item: dict[str, Any]) -> tuple[int, int]:
        about_input = item["kind"] != "ignored_input" and ("scanf" in item["detail"] or "getchar" in item["detail"])
        return (priority.get(item["kind"], 9) + (10 if about_input else 0), item["line"] or 0)
    unique: dict[tuple[Any, str, str], dict[str, Any]] = {}
    for item in targets:
        unique.setdefault((item["line"], item["kind"], item["detail"]), item)
    return sorted(unique.values(), key=rank)[:MAX_TARGETS]


_PARSE_ERROR_AT = re.compile(r"submission\.c:(\d+):(\d+): before: (\S+)")
_C_WORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
    "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Bool",
}
MAX_TYPEDEF_RETRIES = 12


def parse_c(source: str) -> tuple[Any | None, str | None]:
    """Parse with pycparser; unknown library type names are declared and retried.

    Library headers are not expanded, so a declaration such as ``pid_t pid;``
    fails at ``pid``. The identifier just before the failure point is declared
    as an opaque typedef and parsing is retried (bounded).
    """
    if c_parser is None:
        return None, "pycparser is not installed."
    text = preprocess(source)
    body_lines = text.split('#line 1 "submission.c"\n', 1)[-1].split("\n")
    declared: list[str] = []
    error = ""
    for _ in range(MAX_TYPEDEF_RETRIES + 1):
        prelude = "".join(f"typedef struct __opaque_{name} {name};\n" for name in declared)
        try:
            return c_parser.CParser().parse(prelude + text, filename="submission.c"), None
        except Exception as exc:  # pycparser raises ParseError (and occasionally others)
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            match = _PARSE_ERROR_AT.search(str(exc))
            if not match:
                break
            line_no, column = int(match.group(1)), int(match.group(2))
            line = body_lines[line_no - 1] if 0 < line_no <= len(body_lines) else ""
            before = re.findall(r"[A-Za-z_]\w*", line[:max(0, column - 1)])
            candidate = before[-1] if before else None
            if (not candidate or candidate in _C_WORDS or candidate in declared
                    or candidate in _PRELUDE_TYPES):
                break
            declared.append(candidate)
    return None, error


def analyze_c_source(source: str) -> dict[str, Any]:
    """Parse C source and return structural facts plus edge-case targets."""
    if c_parser is None:
        return {"status": "unavailable", "error": "pycparser is not installed."}
    tree, error = parse_c(source)
    if tree is None:
        return {"status": "parse_error", "error": error}
    collector = _FactCollector()
    collector.visit(tree)
    source_lines = source.replace("\r\n", "\n").split("\n")
    freed_vars = {item["variable"] for item in collector.frees}
    leaks = [
        alloc for alloc in collector.allocations
        if alloc["variable"] and alloc["variable"] not in freed_vars
        and not re.search(r"\breturn\b[^;]*\b" + re.escape(str(alloc["variable"]).split("[")[0]) + r"\b",
                          "\n".join(source_lines))
    ]
    facts: dict[str, Any] = {
        "status": "completed",
        "parser": "pycparser",
        "functions": collector.functions,
        "loops": collector.loops,
        "max_loop_depth": max((loop["depth"] for loop in collector.loops), default=0),
        "conditions": collector.conditions[:40],
        "comparisons": collector.comparisons[:60],
        "input_calls": [c for c in collector.calls if c["name"] in _INPUT_FUNCS],
        "output_calls": [c for c in collector.calls if c["name"] in _OUTPUT_FUNCS][:20],
        "allocations": collector.allocations,
        "frees": collector.frees,
        "allocations_never_freed": leaks,
        "uses_after_free": _uses_after_free(source_lines, collector.frees, collector.functions),
        "integer_divisions": collector.divisions,
        "narrow_accumulators": collector.accumulators,
        "fixed_arrays": collector.arrays,
        "globals": collector.globals,
        "called_functions": sorted({c["name"] for c in collector.calls}),
        "identifiers": sorted(collector.identifiers),
        "ignored_inputs": _ignored_inputs(collector),
    }
    facts["accumulators"] = facts["narrow_accumulators"]
    facts["divisions"] = facts["integer_divisions"]
    facts["arrays"] = facts["fixed_arrays"]
    facts["edge_case_targets"] = _edge_targets(facts)
    for key in ("accumulators", "divisions", "arrays"):
        facts.pop(key)
    # Loop nesting that follows calls: a loop that calls a function containing
    # a loop nests them (e.g. rescanning a window inside the main loop).
    own_depth = defaultdict(int)
    for loop in collector.loops:
        own_depth[loop["function"]] = max(own_depth[loop["function"]], loop["depth"])
    calls_by_function = defaultdict(list)
    for call in collector.calls:
        calls_by_function[call["function"]].append(call)
    defined = {fn["name"] for fn in collector.functions}
    memo: dict[str, int] = {}

    def effective(name: str, stack: frozenset = frozenset()) -> int:
        if name in memo:
            return memo[name]
        if name in stack:
            return own_depth[name]
        best = own_depth[name]
        for call in calls_by_function[name]:
            if call["name"] in defined:
                best = max(best, call["loop_depth"] + effective(call["name"], stack | {name}))
        memo[name] = best
        return best

    effective_depth = max([effective(fn["name"]) for fn in collector.functions] + [facts["max_loop_depth"]])
    facts["effective_loop_depth"] = effective_depth
    for fn in facts["functions"]:
        fn["effective_loop_depth"] = effective(fn["name"])
    facts["complexity_hint"] = (
        "recursive" if any(f["recursive"] for f in collector.functions)
        else f"O(n^{effective_depth})" if effective_depth > 1
        else "O(n)" if effective_depth == 1 else "O(1)"
    ) + (
        " from loop nesting including called functions. This is a structural upper bound, not a proof: "
        "amortized loops (monotonic stacks or deques, two pointers) nest syntactically but run in O(n) overall."
    )
    return facts


def compact_ast_summary(facts: dict[str, Any]) -> dict[str, Any]:
    """The subset of AST facts sent to the model (bounded size)."""
    if facts.get("status") != "completed":
        return {"status": facts.get("status"), "error": facts.get("error")}
    return {
        "status": "completed",
        "functions": [
            {k: f[k] for k in ("name", "returns", "params", "line_start", "line_end", "calls", "recursive")}
            for f in facts["functions"]
        ],
        "loops": [{k: loop[k] for k in ("kind", "line", "depth", "condition")} for loop in facts["loops"][:15]],
        "max_loop_depth": facts["max_loop_depth"],
        "effective_loop_depth": facts["effective_loop_depth"],
        "complexity_hint": facts["complexity_hint"],
        "branch_conditions": [{"line": c["line"], "condition": c["condition"]} for c in facts["conditions"][:20]],
        "input_calls": facts["input_calls"][:8],
        "ignored_inputs": facts["ignored_inputs"][:8],
        "allocations_never_freed": facts["allocations_never_freed"][:8],
        "uses_after_free": facts["uses_after_free"][:8],
        "integer_divisions": facts["integer_divisions"][:8],
        "narrow_accumulators": facts["narrow_accumulators"][:8],
        "fixed_arrays": facts["fixed_arrays"][:8],
        "called_functions": facts["called_functions"],
        "edge_case_targets": facts["edge_case_targets"][:20],
    }
