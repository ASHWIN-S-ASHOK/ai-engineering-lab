"""Unit tests for SSE and NDJSON stream framing parsers."""

from collections.abc import AsyncIterator

import pytest

from structured_stream_lab.models import NonRetryableStreamError
from structured_stream_lab.streaming import iter_ndjson, iter_sse_json


async def _lines(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


async def test_iter_sse_json_skips_comments_and_done_sentinel() -> None:
    raw = [
        ": keep-alive",
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","delta":"Hello"}',
        "",
        "data: [DONE]",
        "",
    ]
    parsed = [item async for item in iter_sse_json(_lines(raw))]
    assert parsed == [{"type": "response.output_text.delta", "delta": "Hello"}]


async def test_iter_sse_json_rejects_malformed_json() -> None:
    raw = ["data: {not-valid-json", ""]
    with pytest.raises(NonRetryableStreamError, match="Malformed SSE JSON"):
        _ = [item async for item in iter_sse_json(_lines(raw))]


async def test_iter_ndjson_parses_multiple_objects() -> None:
    raw = [
        "",
        '{"message":{"content":"Hi "},"done":false}',
        '{"message":{"content":"there"},"done":true,"eval_count":4}',
        "",
    ]
    parsed = [item async for item in iter_ndjson(_lines(raw))]
    assert len(parsed) == 2
    assert parsed[1]["done"] is True
    assert parsed[1]["eval_count"] == 4
