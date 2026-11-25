"""Offline integration tests for StructuredStreamingClient."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import BaseModel

from structured_stream_lab.client import StructuredStreamingClient
from structured_stream_lab.config import AppSettings
from structured_stream_lab.models import GenerationRequest, SchemaValidationError
from structured_stream_lab.streaming import ResilientHttpTransport


class _Summary(BaseModel):
    topic: str
    score: int


class _ByteChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def _settings() -> AppSettings:
    return AppSettings(
        _env_file=None,
        llm_provider="gemini",
        gemini_api_key="gem-key",
        openai_api_key="oai-key",
        anthropic_api_key="ant-key",
    )


async def test_generate_structured_parses_and_validates_model() -> None:
    valid_json = json.dumps({"topic": "RAG", "score": 5})
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": valid_json}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 8,
                        "totalTokenCount": 20,
                    },
                },
            )
        ),
    )
    async with transport:
        client = StructuredStreamingClient(settings=_settings(), http_transport=transport)
        result = await client.generate_structured(
            prompt="Summarize topic",
            response_model=_Summary,
        )

    assert result.data == _Summary(topic="RAG", score=5)
    assert result.usage.total_tokens == 20
    assert result.finish_reason == "STOP"


async def test_generate_structured_raises_on_truncated_json() -> None:
    truncated = '{"topic": "RAG"'  # Missing closing brace due to token limit
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": truncated}]}, "finishReason": "MAX_TOKENS"}
                    ],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                },
            )
        ),
    )
    async with transport:
        client = StructuredStreamingClient(settings=_settings(), http_transport=transport)
        with pytest.raises(SchemaValidationError) as exc_info:
            await client.generate_structured(prompt="Summarize", response_model=_Summary)

    assert exc_info.value.finish_reason == "MAX_TOKENS"
    assert exc_info.value.raw_text == truncated


async def test_stream_generate_sse_accumulates_anthropic_lifecycle() -> None:
    sse_bytes = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":15}}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello "}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}\n\n',
        (
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":2}}\n\n'
        ),
        b'data: {"type":"message_stop"}\n\n',
    ]
    mock = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ByteChunks(sse_bytes),
        )
    )
    async with StructuredStreamingClient(settings=_settings(), mock_transport=mock) as client:
        events = [
            ev
            async for ev in client.stream_generate(
                GenerationRequest(prompt="Greet", stream=True),
                provider="anthropic",
            )
        ]

    deltas = [ev.delta_text for ev in events if ev.delta_text]
    assert "".join(deltas) == "Hello world"
    terminal = events[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "end_turn"
    assert terminal.ttft_ms is not None
    assert terminal.total_ms is not None
    assert terminal.usage is not None
    assert terminal.usage.input_tokens == 15
    assert terminal.usage.output_tokens == 2
    assert terminal.usage.total_tokens == 17


async def test_stream_generate_ndjson_for_ollama() -> None:
    ndjson_bytes = [
        b'{"message":{"content":"Local "},"done":false}\n',
        (
            b'{"message":{"content":"stream"},"done":true,'
            b'"done_reason":"stop","prompt_eval_count":9,"eval_count":2}\n'
        ),
    ]
    mock = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=_ByteChunks(ndjson_bytes),
        )
    )
    async with StructuredStreamingClient(settings=_settings(), mock_transport=mock) as client:
        events = [
            ev
            async for ev in client.stream_generate(
                GenerationRequest(prompt="Greet", stream=True),
                provider="ollama",
            )
        ]

    text = "".join(ev.delta_text for ev in events)
    assert text == "Local stream"
    assert events[-1].done is True
    assert events[-1].usage is not None
    assert events[-1].usage.total_tokens == 11
