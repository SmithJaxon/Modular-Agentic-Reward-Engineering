"""
Summary: Load and validate agentic run specs from JSON or constrained YAML files.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rewardlab.schemas.agentic_run import AgenticRunSpec


@dataclass(slots=True, frozen=True)
class _YamlToken:
    """
    Represent one normalized YAML line token with indentation metadata.
    """

    indent: int
    content: str


def load_run_spec(path: Path) -> AgenticRunSpec:
    """
    Load and validate an agentic run spec from JSON or constrained YAML.
    """
    raw_text = path.read_text(encoding="utf-8")
    data = _load_spec_payload(path, raw_text)
    if not isinstance(data, dict):
        raise ValueError("run spec must be a top-level object")
    return AgenticRunSpec.model_validate(data)


def _load_spec_payload(path: Path, text: str) -> dict[str, Any]:
    """
    Resolve file format and parse run spec payload data.
    """
    lowered_suffix = path.suffix.lower()
    if lowered_suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("JSON run spec must be a top-level object")
        return payload
    return _parse_yaml_mapping(text)


def _parse_yaml_mapping(text: str) -> dict[str, Any]:
    """
    Parse constrained YAML containing nested mappings and list scalars.
    """
    tokens = _tokenize_yaml(text)
    if not tokens:
        return {}
    value, next_index = _parse_block(tokens, 0, tokens[0].indent)
    if next_index != len(tokens):
        raise ValueError("unexpected trailing YAML content")
    if not isinstance(value, dict):
        raise ValueError("YAML run spec must start with a mapping")
    return value


def _tokenize_yaml(text: str) -> list[_YamlToken]:
    """
    Normalize YAML lines into indentation-aware tokens.
    """
    tokens: list[_YamlToken] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"unsupported YAML indentation: {line!r}")
        tokens.append(_YamlToken(indent=indent, content=line.strip()))
    return tokens


def _parse_block(
    tokens: list[_YamlToken],
    start_index: int,
    indent: int,
) -> tuple[object, int]:
    """
    Parse one YAML block at the requested indentation level.
    """
    if start_index >= len(tokens):
        return {}, start_index
    if tokens[start_index].content.startswith("- "):
        return _parse_list(tokens, start_index, indent)
    return _parse_mapping(tokens, start_index, indent)


def _parse_mapping(
    tokens: list[_YamlToken],
    start_index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    """
    Parse mapping entries at the requested indentation level.
    """
    payload: dict[str, Any] = {}
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if token.indent < indent:
            break
        if token.indent > indent:
            raise ValueError(f"unexpected nested indentation near: {token.content!r}")
        if token.content.startswith("- "):
            break
        key, value = _split_mapping(token.content)
        index += 1
        if value != "":
            payload[key] = _parse_scalar(value)
            continue
        if index >= len(tokens) or tokens[index].indent <= indent:
            payload[key] = {}
            continue
        child_indent = tokens[index].indent
        child, index = _parse_block(tokens, index, child_indent)
        payload[key] = child
    return payload, index


def _parse_list(
    tokens: list[_YamlToken],
    start_index: int,
    indent: int,
) -> tuple[list[Any], int]:
    """
    Parse list entries at the requested indentation level.
    """
    values: list[Any] = []
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if token.indent < indent:
            break
        if token.indent > indent:
            raise ValueError(f"unexpected nested indentation near: {token.content!r}")
        if not token.content.startswith("- "):
            break
        item_content = token.content[2:].strip()
        index += 1
        if item_content == "":
            if index >= len(tokens) or tokens[index].indent <= indent:
                values.append(None)
                continue
            nested_indent = tokens[index].indent
            nested_value, index = _parse_block(tokens, index, nested_indent)
            values.append(nested_value)
            continue
        if ":" in item_content and not item_content.startswith(("'", '"')):
            key, value = _split_mapping(item_content)
            entry: dict[str, Any] = {}
            if value != "":
                entry[key] = _parse_scalar(value)
            elif index < len(tokens) and tokens[index].indent > indent:
                nested_indent = tokens[index].indent
                nested_value, index = _parse_block(tokens, index, nested_indent)
                entry[key] = nested_value
            else:
                entry[key] = {}
            if index < len(tokens) and tokens[index].indent > indent:
                nested_indent = tokens[index].indent
                nested_value, index = _parse_block(tokens, index, nested_indent)
                if not isinstance(nested_value, dict):
                    raise ValueError("inline list mapping must expand to an object")
                entry.update(nested_value)
            values.append(entry)
            continue
        values.append(_parse_scalar(item_content))
        if index < len(tokens) and tokens[index].indent > indent:
            raise ValueError("scalar list entries cannot contain nested blocks")
    return values, index


def _split_mapping(content: str) -> tuple[str, str]:
    """
    Split one constrained YAML mapping line.
    """
    key, separator, value = content.partition(":")
    if separator == "":
        raise ValueError(f"invalid mapping entry: {content!r}")
    trimmed_key = key.strip()
    if not trimmed_key:
        raise ValueError(f"mapping key cannot be blank: {content!r}")
    return trimmed_key, value.strip()


def _strip_comment(line: str) -> str:
    """
    Strip YAML comments while preserving quoted scalar content.
    """
    in_single_quote = False
    in_double_quote = False
    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]
    return line


def _parse_scalar(value: str) -> object:
    """
    Parse one constrained YAML scalar value.
    """
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if value.startswith(("'", '"')):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
