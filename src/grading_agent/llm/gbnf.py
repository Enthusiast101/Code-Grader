"""Legacy utility: compile the project's small JSON Schema subset to GBNF.

Production grading and compression calls use llama.cpp's documented
``response_format: {"type": "json_schema", "schema": ...}`` contract on
``/v1/chat/completions``. This converter remains available for local utilities
and focused tests; production requests do not send a raw ``grammar`` field.
"""
from __future__ import annotations

import json
from typing import Any

_JSON_PRIMITIVES = r'''ws ::= (" " | "\t" | "\n" | "\r")*
json-string ::= "\"" json-char* "\""
json-char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
json-integer ::= "-"? ("0" | [1-9] [0-9]*)
json-boolean ::= "true" | "false"
json-null ::= "null"'''


def _literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_schema_to_gbnf(schema: dict[str, Any] | None) -> str:
    """Compile object/array/primitive/enum/union JSON Schema nodes to GBNF.

    Every declared object property is emitted in schema order. This is stricter
    than JSON Schema's optional-property semantics but remains schema-valid and
    keeps local-model output shape stable. Numeric/string bounds are checked by
    the Python normalizers after generation.
    """
    if not isinstance(schema, dict):
        schema = {"type": "object"}

    rules = [_JSON_PRIMITIVES]
    counter = 0

    def add_rule(expression: str, prefix: str = "value") -> str:
        nonlocal counter
        counter += 1
        name = f"{prefix}-{counter}"
        rules.append(f"{name} ::= {expression}")
        return name

    def build(node: Any) -> str:
        if not isinstance(node, dict):
            return "json-string"

        if "const" in node:
            return add_rule(_literal(node["const"]), "const")

        enum_values = node.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return add_rule(" | ".join(_literal(item) for item in enum_values), "enum")

        node_type = node.get("type")
        if isinstance(node_type, list):
            variants = [build({**node, "type": variant, "enum": None}) for variant in node_type]
            return add_rule(" | ".join(variants), "union")

        if node_type == "object" or (node_type is None and isinstance(node.get("properties"), dict)):
            properties = node.get("properties", {})
            if not isinstance(properties, dict) or not properties:
                return "json-generic-object"
            fields = []
            for key, child in properties.items():
                fields.append(_literal(str(key)) + ' ws ":" ws ' + build(child))
            return add_rule('"{" ws ' + ' ws "," ws '.join(fields) + ' ws "}"', "object")

        if node_type == "array":
            item = build(node.get("items", {}))
            min_items = max(0, int(node.get("minItems", 0) or 0))
            max_value = node.get("maxItems")
            max_items = int(max_value) if isinstance(max_value, int) and max_value >= 0 else None
            if max_items is not None:
                max_items = max(max_items, min_items)
            if max_items == 0:
                expression = '"[" ws "]"'
            elif min_items == 0:
                if max_items is None:
                    contents = f'({item} (ws "," ws {item})*)?'
                else:
                    contents = f'({item} (ws "," ws {item}){{0,{max_items - 1}}})?'
                expression = f'"[" ws {contents} ws "]"'
            else:
                required = " ".join([item] + [f'ws "," ws {item}' for _ in range(min_items - 1)])
                if max_items is None:
                    tail = f' (ws "," ws {item})*'
                else:
                    tail_count = max(0, max_items - min_items)
                    tail = f' (ws "," ws {item}){{0,{tail_count}}}' if tail_count else ""
                expression = f'"[" ws {required}{tail} ws "]"'
            return add_rule(expression, "array")

        if node_type == "string":
            return "json-string"
        if node_type == "integer":
            return "json-integer"
        if node_type == "number":
            return "json-number"
        if node_type == "boolean":
            return "json-boolean"
        if node_type == "null":
            return "json-null"
        return "json-value"

    rules.extend([
        'json-value ::= json-string | json-number | json-boolean | json-null | json-generic-object | json-generic-array',
        'json-generic-object ::= "{" ws (json-string ws ":" ws json-value (ws "," ws json-string ws ":" ws json-value)*)? ws "}"',
        'json-generic-array ::= "[" ws (json-value (ws "," ws json-value)*)? ws "]"',
    ])
    root = build(schema)
    return "\n".join([f"root ::= {root} ws", *rules]) + "\n"
