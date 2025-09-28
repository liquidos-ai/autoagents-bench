from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Optional

EXPECTED_AVERAGE_TRIP_MINUTES = 25.70
_ROUND_DECIMALS = 2
_JSON_CHUNK_PATTERN = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_PREFERRED_KEYS = (
    "value",
    "average_trip_duration_minutes",
    "average_minutes",
    "answer",
    "result",
)

def evaluate_structured_result(payload: Any) -> tuple[bool, Optional[float]]:
    for text in _iter_text_candidates(payload):
        value = _parse_structured_value(text)
        if value is None:
            continue
        rounded = round(value, _ROUND_DECIMALS)
        return math.isclose(rounded, EXPECTED_AVERAGE_TRIP_MINUTES, abs_tol=1e-9), value
    return False, None


def _iter_text_candidates(payload: Any, depth: int = 0) -> Iterable[str]:
    if payload is None or depth > 5:
        return
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped:
            yield stripped
        return
    if isinstance(payload, dict):
        if "messages" in payload:
            messages = payload["messages"]
            if isinstance(messages, list):
                for message in reversed(messages):
                    yield from _iter_text_candidates(message, depth + 1)
        for key in ("content", "result", "return", "output", "response", "final_output"):
            if key in payload:
                yield from _iter_text_candidates(payload[key], depth + 1)
        for value in payload.values():
            yield from _iter_text_candidates(value, depth + 1)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _iter_text_candidates(item, depth + 1)
        return
    content = getattr(payload, "content", None)
    if content is not None:
        yield from _iter_text_candidates(content, depth + 1)
        return
    text = getattr(payload, "text", None)
    if text is not None:
        yield from _iter_text_candidates(text, depth + 1)
        return


def _parse_structured_value(text: str) -> Optional[float]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for match in _JSON_CHUNK_PATTERN.finditer(text):
            chunk = match.group(1)
            try:
                payload = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            value = _dig_numeric(payload)
            if value is not None:
                return value
        return None
    return _dig_numeric(payload)


def _dig_numeric(node: Any) -> Optional[float]:
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        for key in _PREFERRED_KEYS:
            if key in node:
                value = _dig_numeric(node[key])
                if value is not None:
                    return value
        for value in node.values():
            numeric = _dig_numeric(value)
            if numeric is not None:
                return numeric
        return None
    if isinstance(node, list):
        for item in node:
            value = _dig_numeric(item)
            if value is not None:
                return value
    return None
