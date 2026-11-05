"""Syntax-tree checks for C safety properties (the classes ESBMC verifies).

Deterministic, line-anchored checks over the pycparser tree:

* ``unbounded_input``: ``scanf``/``fscanf``/``sscanf`` ``%s`` or ``%[`` without
  a field width, and ``gets`` -> buffer overflow from input.
* ``unchecked_allocation``: a ``malloc``/``calloc``/``realloc``/``strdup``
  result dereferenced before any condition tests it -> NULL dereference.
* ``loop_out_of_bounds``: a ``for`` loop whose bound lets the index reach the
  declared size of a fixed array it indexes (``i <= N`` over ``a[N]``).
* ``unchecked_input_index`` / ``unchecked_input_divisor`` /
  ``unchecked_input_arithmetic``: a value read from input used as an array
  index, a divisor, or a signed arithmetic operand before any condition tests it.
* ``memory_leak``: heap memory that is never freed, reported at the end of
  ``main`` (where it is lost) with the allocation line in the description.
* ``use_after_free``: a pointer read after ``free`` (from ``c_ast``).

Findings are flow-insensitive approximations within one function, ordered by
line. They are evidence for the benchmark and the reviewer, not proofs.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from grading_agent.analyzers.c_ast import _FactCollector, _uses_after_free, parse_c

try:
    from pycparser import c_ast, c_generator
except ImportError:  # pragma: no cover
    c_ast = c_generator = None

_ALLOCATORS = {"malloc", "calloc", "realloc", "strdup", "strndup"}
_INPUT_SCANF = {"scanf": 0, "fscanf": 1, "sscanf": 1}
# Library calls that read or write through every pointer argument.
_DEREF_CALLS = {
    "strcpy", "strncpy", "strcat", "strncat", "memcpy", "memmove", "memset", "memcmp", "strlen", "strcmp",
    "strncmp", "strchr", "strrchr", "strstr", "sprintf", "snprintf", "fgets", "fread", "fwrite", "sscanf",
    "atoi", "atol", "atof", "strtol", "strtoul", "strtod", "qsort", "puts", "fputs", "strcspn", "strspn",
}
_TAINT_CALLS = {"atoi", "atol", "atoll", "atof", "strtol", "strtoll", "strtoul", "strtod", "getchar", "fgetc", "getc"}
_SIGNED_INTS = {"int", "long", "short", "long long", "long int", "signed", "signed int", "short int", "long long int",
                "int32_t", "int64_t", "int16_t", "int8_t", "char", "signed char"}
_CONVERSION = re.compile(r"%(\*?)(\d*)(?:hh|h|ll|l|L|j|z|t)?(\[[^\]]*\]|[a-zA-Z%])")


def _text(node: Any) -> str:
    try:
        return c_generator.CGenerator().visit(node)
    except Exception:
        return ""


def _line(node: Any) -> int | None:
    coord = getattr(node, "coord", None)
    return coord.line if coord is not None and coord.line else None


def _identifiers(node: Any) -> set[str]:
    names: set[str] = set()

    class _Ids(c_ast.NodeVisitor):
        def visit_ID(self, item: Any) -> None:
            names.add(item.name)

        def visit_FuncCall(self, item: Any) -> None:
            if item.args is not None:
                self.visit(item.args)

        def visit_StructRef(self, item: Any) -> None:
            self.visit(item.name)  # the field name is not a variable

    if node is not None:
        _Ids().visit(node)
    return names


def _root(node: Any) -> str | None:
    """Base variable of an lvalue: ``a[i].x->y`` -> ``a``."""
    while node is not None:
        if isinstance(node, c_ast.ID):
            return node.name
        if isinstance(node, (c_ast.ArrayRef, c_ast.StructRef)):
            node = node.name
        elif isinstance(node, c_ast.UnaryOp):
            node = node.expr
        elif isinstance(node, c_ast.Cast):
            node = node.expr
        else:
            return None
    return None


def _call_name(node: Any) -> str | None:
    return node.name.name if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) else None


def _strip_cast(node: Any) -> Any:
    while isinstance(node, c_ast.Cast):
        node = node.expr
    return node


def _const_int(node: Any) -> int | None:
    """Evaluate a small constant integer expression (after #define substitution)."""
    if isinstance(node, c_ast.Constant) and node.type in {"int", "long int", "unsigned int"}:
        try:
            return int(node.value.rstrip("uUlL"), 0)
        except ValueError:
            return None
    if isinstance(node, c_ast.BinaryOp) and node.op in {"+", "-", "*"}:
        left, right = _const_int(node.left), _const_int(node.right)
        if left is None or right is None:
            return None
        return {"+": left + right, "-": left - right, "*": left * right}[node.op]
    if isinstance(node, c_ast.UnaryOp) and node.op == "-":
        value = _const_int(node.expr)
        return -value if value is not None else None
    return None


class _FunctionFacts(c_ast.NodeVisitor):
    """Ordered events inside one function body."""

    def __init__(self, name: str, global_arrays: dict[str, int]) -> None:
        self.name = name
        self.allocs: list[tuple[int, str]] = []           # (line, target text)
        self.checks: list[tuple[int, str]] = []           # (line, condition text)
        self.derefs: list[tuple[int, str, str]] = []      # (line, base text, how)
        self.assigns: list[tuple[int, str]] = []          # (line, target text)
        self.taints: list[tuple[int, str]] = []           # (line, variable)
        self.index_uses: list[tuple[int, str, set[str]]] = []   # (line, array text, subscript ids)
        self.divisions: list[tuple[int, set[str], str]] = []    # (line, divisor ids, text)
        self.arith: list[tuple[int, set[str], str]] = []        # (line, operand ids, text)
        self.unbounded_input: list[tuple[int, str]] = []
        self.loop_oob: list[tuple[int, str]] = []
        self.signed_vars: set[str] = set()
        self.arrays: dict[str, int] = dict(global_arrays)
        self.loop_bounds: list[tuple[str, int]] = []      # active (index var, inclusive max)

    # --- declarations and assignments -------------------------------------
    def visit_Decl(self, node: Any) -> None:
        if node.name:
            kind = node.type
            if isinstance(kind, c_ast.TypeDecl) and isinstance(kind.type, c_ast.IdentifierType):
                if " ".join(kind.type.names) in _SIGNED_INTS:
                    self.signed_vars.add(node.name)
            if isinstance(kind, c_ast.ArrayDecl):
                size = _const_int(kind.dim) if kind.dim is not None else None
                if size is not None:
                    self.arrays[node.name] = size
            if node.init is not None:
                self._assigned(_line(node), node.name, node.init)
        self.generic_visit(node)

    def visit_Assignment(self, node: Any) -> None:
        target = _text(node.lvalue)
        line = _line(node)
        if node.op == "=":
            self._assigned(line, target, node.rvalue)
        elif node.op in {"+=", "-=", "*="}:
            ids = _identifiers(node.rvalue) | {_root(node.lvalue) or ""}
            self.arith.append((line, ids, _text(node)))
        elif node.op in {"/=", "%="}:
            self.divisions.append((line, _identifiers(node.rvalue), _text(node)))
        self.generic_visit(node)

    def _assigned(self, line: int | None, target: str, value: Any) -> None:
        if line is None:
            return
        self.assigns.append((line, target))
        inner = _strip_cast(value)
        name = _call_name(inner)
        if name in _ALLOCATORS:
            self.allocs.append((line, target))
        if name in _TAINT_CALLS:
            self.taints.append((line, target))

    # --- conditions -------------------------------------------------------
    def _condition(self, node: Any) -> None:
        if node is not None and _line(node) is not None:
            self.checks.append((_line(node), _text(node)))

    def visit_If(self, node: Any) -> None:
        self._condition(node.cond)
        self.generic_visit(node)

    def visit_While(self, node: Any) -> None:
        self._condition(node.cond)
        self.generic_visit(node)

    def visit_DoWhile(self, node: Any) -> None:
        self._condition(node.cond)
        self.generic_visit(node)

    def visit_TernaryOp(self, node: Any) -> None:
        self._condition(node.cond)
        self.generic_visit(node)

    def visit_Switch(self, node: Any) -> None:
        self._condition(node.cond)
        self.generic_visit(node)

    def visit_For(self, node: Any) -> None:
        self._condition(node.cond)
        bound = self._loop_bound(node)
        if bound:
            self.loop_bounds.append(bound)
            self.generic_visit(node)
            self.loop_bounds.pop()
        else:
            self.generic_visit(node)

    @staticmethod
    def _loop_bound(node: Any) -> tuple[str, int] | None:
        cond = node.cond
        if not isinstance(cond, c_ast.BinaryOp) or not isinstance(cond.left, c_ast.ID):
            return None
        limit = _const_int(cond.right)
        if limit is None:
            return None
        if cond.op == "<":
            return cond.left.name, limit - 1
        if cond.op == "<=":
            return cond.left.name, limit
        return None

    # --- expressions --------------------------------------------------------
    def visit_StructRef(self, node: Any) -> None:
        if node.type == "->":
            self.derefs.append((_line(node) or 0, _text(node.name), "->"))
        self.generic_visit(node)

    def visit_UnaryOp(self, node: Any) -> None:
        if node.op == "*":
            self.derefs.append((_line(node) or 0, _text(node.expr), "*"))
        if node.op in {"p++", "p--", "++", "--"}:
            self.arith.append((_line(node) or 0, {_root(node.expr) or ""}, _text(node)))
        self.generic_visit(node)

    def visit_ArrayRef(self, node: Any) -> None:
        line = _line(node) or 0
        base = _text(node.name)
        self.derefs.append((line, base, "[]"))
        self.index_uses.append((line, base, _identifiers(node.subscript)))
        if isinstance(node.name, c_ast.ID) and isinstance(node.subscript, c_ast.ID):
            size = self.arrays.get(node.name.name)
            for var, inclusive_max in self.loop_bounds:
                if var == node.subscript.name and size is not None and inclusive_max >= size:
                    self.loop_oob.append((line, f"{base}[{var}] with {var} up to {inclusive_max} but {base} has {size} elements"))
        self.generic_visit(node)

    def visit_BinaryOp(self, node: Any) -> None:
        line = _line(node) or 0
        if node.op in {"/", "%"}:
            self.divisions.append((line, _identifiers(node.right), _text(node)))
        elif node.op in {"+", "-", "*"}:
            self.arith.append((line, _identifiers(node.left) | _identifiers(node.right), _text(node)))
        elif node.op in {"&&", "||"}:
            self._condition(node)
        self.generic_visit(node)

    def visit_FuncCall(self, node: Any) -> None:
        name = _call_name(node)
        line = _line(node) or 0
        args = list(node.args.exprs) if node.args is not None else []
        if name in {"assert"} and args:
            self._condition(args[0])
        if name in _DEREF_CALLS:
            for arg in args:
                self.derefs.append((line, _text(_strip_cast(arg)), f"{name}()"))
        if name == "gets":
            self.unbounded_input.append((line, "gets() reads an unbounded line"))
        if name in _INPUT_SCANF and len(args) > _INPUT_SCANF[name]:
            fmt = args[_INPUT_SCANF[name]]
            conversions = []
            if isinstance(fmt, c_ast.Constant) and fmt.type == "string":
                conversions = _CONVERSION.findall(fmt.value)
                for suppressed, width, conversion in conversions:
                    if not suppressed and not width and (conversion == "s" or conversion.startswith("[")):
                        self.unbounded_input.append((line, f"{name} %{conversion} without a field width"))
                        break
            # Every &target written by scanf now holds untrusted input.
            for arg in args[_INPUT_SCANF[name] + 1:]:
                if isinstance(arg, c_ast.UnaryOp) and arg.op == "&":
                    root = _root(arg.expr)
                    if root:
                        self.taints.append((line, root))
        self.generic_visit(node)


def _mentions(text: str, target: str) -> bool:
    return re.search(r"(?<![\w.>])" + re.escape(target) + r"(?![\w])", text) is not None


def _first_check(facts: _FunctionFacts, target: str, after: int) -> int | None:
    lines = [line for line, text in facts.checks if line >= after and _mentions(text, target)]
    return min(lines) if lines else None


def _unchecked_allocations(facts: _FunctionFacts) -> list[dict[str, Any]]:
    findings = []
    for alloc_line, target in facts.allocs:
        check = _first_check(facts, target, alloc_line)
        reassigned = [line for line, t in facts.assigns if t == target and line > alloc_line]
        horizon = min([value for value in (check, min(reassigned) if reassigned else None) if value is not None],
                      default=None)
        for line, base, how in sorted(facts.derefs):
            if base != target or line < alloc_line or (line == alloc_line and how != "->"):
                continue
            if horizon is not None and line >= horizon:
                break
            findings.append({
                "line": line, "family": "null_dereference", "check": "unchecked_allocation",
                "description": f"'{target}' from the allocation on line {alloc_line} is dereferenced ({how}) "
                               "without checking it for NULL.",
            })
            break
    return findings


def _tainted_uses(facts: _FunctionFacts) -> list[dict[str, Any]]:
    findings = []
    taint_lines: dict[str, int] = {}
    for line, var in sorted(facts.taints):
        taint_lines.setdefault(var, line)

    def open_taint(ids: set[str], line: int) -> str | None:
        for var in sorted(ids):
            start = taint_lines.get(var)
            if start is None or line <= start:
                continue
            check = _first_check(facts, var, start)
            if check is None or check > line:
                return var
        return None

    reported: set[tuple[str, int]] = set()
    for line, base, subscript_ids in facts.index_uses:
        var = open_taint(subscript_ids, line)
        if var and ("idx", line) not in reported:
            reported.add(("idx", line))
            findings.append({"line": line, "family": "out_of_bounds", "check": "unchecked_input_index",
                             "description": f"'{var}' comes from input and indexes '{base}' without a bounds check."})
    for line, ids, text in facts.divisions:
        var = open_taint(ids, line)
        if var and ("div", line) not in reported:
            reported.add(("div", line))
            findings.append({"line": line, "family": "division_by_zero", "check": "unchecked_input_divisor",
                             "description": f"'{var}' comes from input and is used as a divisor in '{text[:60]}' without a zero check."})
    for line, ids, text in facts.arith:
        var = open_taint({name for name in ids if name in facts.signed_vars}, line)
        if var and ("ari", line) not in reported:
            reported.add(("ari", line))
            findings.append({"line": line, "family": "integer_overflow", "check": "unchecked_input_arithmetic",
                             "description": f"Signed '{var}' comes from input and is used in '{text[:60]}' without a range check, "
                                            "so the arithmetic can overflow."})
    return findings


def check_c_properties(source: str) -> dict[str, Any]:
    """Run every check; returns ``status`` plus line-anchored ``findings``."""
    if c_ast is None:
        return {"status": "unavailable", "error": "pycparser is not installed.", "findings": []}
    tree, error = parse_c(source)
    if tree is None:
        return {"status": "parse_error", "error": error, "findings": []}

    global_arrays: dict[str, int] = {}
    for item in tree.ext:
        if isinstance(item, c_ast.Decl) and isinstance(item.type, c_ast.ArrayDecl) and item.type.dim is not None:
            size = _const_int(item.type.dim)
            if size is not None and item.name:
                global_arrays[item.name] = size

    findings: list[dict[str, Any]] = []
    all_facts: list[_FunctionFacts] = []
    main_end: int | None = None
    reachable = _reachable_functions(tree)
    for item in tree.ext:
        if not isinstance(item, c_ast.FuncDef) or item.decl.name not in reachable:
            # Code no execution can reach (never called, directly or through a
            # function pointer, from main) cannot fail at run time.
            continue
        facts = _FunctionFacts(item.decl.name, global_arrays)
        facts.visit(item.body)
        all_facts.append(facts)
        if item.decl.name == "main":
            ends = [_line(node) for node in _walk(item.body) if _line(node)]
            main_end = max(ends, default=_line(item)) + 1 if ends else _line(item)
        for line, detail in facts.unbounded_input:
            findings.append({"line": line, "family": "input_buffer_overflow", "check": "unbounded_input",
                             "description": f"{detail}: input longer than the buffer overflows it."})
        for line, detail in facts.loop_oob:
            findings.append({"line": line, "family": "out_of_bounds", "check": "loop_out_of_bounds",
                             "description": f"Loop index can reach the array size: {detail}."})
        findings.extend(_unchecked_allocations(facts))
        findings.extend(_tainted_uses(facts))

    # Leaks: memory allocated anywhere and never released by any free() call.
    collector = _FactCollector()
    collector.visit(tree)
    source_lines = source.replace("\r\n", "\n").split("\n")
    freed_roots = {str(item["variable"]).split("[")[0].split("->")[0].split(".")[0].lstrip("*&( ")
                   for item in collector.frees}
    any_free = bool(collector.frees)
    if main_end is not None:
        main_end = min(main_end, len(source_lines))
        # The closing brace of main is where ESBMC reports forgotten memory.
        for offset in range(main_end, min(len(source_lines), main_end + 3) + 1):
            if source_lines[offset - 1].strip().startswith("}"):
                main_end = offset
                break
    for alloc in collector.allocations:
        if alloc.get("function") not in reachable:
            continue
        variable = str(alloc.get("variable") or "")
        root = variable.split("[")[0].split("->")[0].split(".")[0].lstrip("*&( ")
        returned = re.search(r"\breturn\b[^;]*\b" + re.escape(root) + r"\b", "\n".join(source_lines)) if root else None
        if not root or root in freed_roots or returned or (any_free and alloc.get("function") != "main"):
            continue
        findings.append({
            "line": main_end or alloc.get("line"), "family": "memory_leak", "check": "memory_leak",
            "description": f"Memory allocated on line {alloc.get('line')} ('{variable}') is never freed.",
        })
        break  # one leak finding per program is enough evidence

    for item in _uses_after_free(source_lines, collector.frees, collector.functions):
        if not any(fn["name"] in reachable and fn["line_start"] <= item["used_line"] <= fn["line_end"]
                   for fn in collector.functions):
            continue
        findings.append({"line": item["used_line"], "family": "invalid_pointer", "check": "use_after_free",
                         "description": f"'{item['variable']}' is used after being freed on line {item['freed_line']}."})

    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for finding in findings:
        if isinstance(finding.get("line"), int):
            unique.setdefault((finding["line"], finding["family"]), finding)
    ordered = sorted(unique.values(), key=lambda f: (f["line"], f["family"]))
    return {"status": "completed", "findings": ordered, "functions": [f.name for f in all_facts]}


def _reachable_functions(tree: Any) -> set[str]:
    """Functions reachable from ``main`` by calls or function-pointer references."""
    bodies = {item.decl.name: item.body for item in tree.ext if isinstance(item, c_ast.FuncDef)}
    if "main" not in bodies:
        return set(bodies)
    edges = {name: _identifiers_all(body) & set(bodies) for name, body in bodies.items()}
    # Function pointers stored in global initialisers (dispatch tables) are reachable too.
    global_refs: set[str] = set()
    for item in tree.ext:
        if isinstance(item, c_ast.Decl) and item.init is not None:
            global_refs |= _identifiers_all(item.init) & set(bodies)
    seen, stack = set(), ["main", *global_refs]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(edges.get(name, ()))
    return seen


def _identifiers_all(node: Any) -> set[str]:
    """Every identifier in a subtree, including called function names."""
    return {child.name for child in _walk(node) if isinstance(child, c_ast.ID)}


def _walk(node: Any):
    yield node
    for _, child in node.children():
        yield from _walk(child)
