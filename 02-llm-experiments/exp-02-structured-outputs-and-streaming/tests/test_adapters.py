"""Unit tests for structured schema requests and stream event parsing across all 4 adapters."""

from pydantic import BaseModel, SecretStr

from structured_stream_lab.adapters import ADAPTER_REGISTRY, StructuredStreamProvider
from structured_stream_lab.models import (
    GenerationRequest,
    ProviderRuntimeConfig,
    schema_from_model,
)


class _Contact(BaseModel):
    name: str
    email: str


def _runtime(provider: str, base_url: str, model: str) -> ProviderRuntimeConfig:
    return ProviderRuntimeConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        base_url=base_url,
        has_api_key=(provider != "ollama"),
    )


def test_schema_from_model_enforces_additional_properties_false() -> None:
    spec = schema_from_model(_Contact)
    assert spec.name == "_Contact"
    assert spec.json_schema["additionalProperties"] is False
    assert set(spec.json_schema["required"]) == {"name", "email"}


def test_all_adapters_conform_to_protocol() -> None:
    assert set(ADAPTER_REGISTRY.keys()) == {"gemini", "openai", "anthropic", "ollama"}
    for adapter in ADAPTER_REGISTRY.values():
        assert isinstance(adapter, StructuredStreamProvider)


def test_gemini_structured_and_stream_request_shapes() -> None:
    adapter = ADAPTER_REGISTRY["gemini"]
    rt = _runtime("gemini", "https://generativelanguage.googleapis.com", "gemini-2.5-flash")
    spec = schema_from_model(_Contact)

    req_struct = GenerationRequest(prompt="Extract", schema_spec=spec, stream=False)
    prep_struct = adapter.build_request(req_struct, rt, SecretStr("g-key"))
    assert prep_struct.url.endswith(":generateContent")
    gen_cfg = prep_struct.json_body["generationConfig"]
    assert gen_cfg["responseMimeType"] == "application/json"
    assert gen_cfg["responseJsonSchema"]["additionalProperties"] is False

    req_stream = GenerationRequest(prompt="Stream", stream=True)
    prep_stream = adapter.build_request(req_stream, rt, SecretStr("g-key"))
    assert prep_stream.url.endswith(":streamGenerateContent?alt=sse")

    chunk = adapter.parse_stream_event(
        {
            "candidates": [
                {"content": {"parts": [{"text": "chunk"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
        }
    )
    assert chunk.delta_text == "chunk"
    assert chunk.done is True
    assert chunk.total_tokens == 7


def test_openai_structured_and_stream_shapes() -> None:
    adapter = ADAPTER_REGISTRY["openai"]
    rt = _runtime("openai", "https://api.openai.com/v1", "gpt-4o-mini")
    req = GenerationRequest(prompt="Extract", schema_spec=schema_from_model(_Contact), stream=True)
    prep = adapter.build_request(req, rt, SecretStr("o-key"))

    assert prep.json_body["stream"] is True
    assert prep.json_body["text"]["format"]["type"] == "json_schema"
    assert prep.json_body["text"]["format"]["strict"] is True

    delta = adapter.parse_stream_event({"type": "response.output_text.delta", "delta": "Hi"})
    assert delta.delta_text == "Hi"
    assert delta.done is False

    done = adapter.parse_stream_event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
            },
        }
    )
    assert done.done is True
    assert done.finish_reason == "completed"
    assert done.total_tokens == 9


def test_anthropic_structured_and_stream_shapes() -> None:
    adapter = ADAPTER_REGISTRY["anthropic"]
    rt = _runtime("anthropic", "https://api.anthropic.com/v1", "claude-3-5-haiku-latest")
    req = GenerationRequest(prompt="Extract", schema_spec=schema_from_model(_Contact), stream=True)
    prep = adapter.build_request(req, rt, SecretStr("a-key"))

    assert prep.json_body["output_config"]["format"]["type"] == "json_schema"
    assert prep.json_body["stream"] is True

    start = adapter.parse_stream_event(
        {"type": "message_start", "message": {"usage": {"input_tokens": 11}}}
    )
    delta = adapter.parse_stream_event(
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "{"}}
    )
    msg_delta = adapter.parse_stream_event(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4},
        }
    )
    stop = adapter.parse_stream_event({"type": "message_stop"})

    assert start.input_tokens == 11
    assert delta.delta_text == "{"
    assert msg_delta.finish_reason == "end_turn"
    assert msg_delta.output_tokens == 4
    assert stop.done is True


def test_ollama_structured_and_stream_shapes() -> None:
    adapter = ADAPTER_REGISTRY["ollama"]
    rt = _runtime("ollama", "http://localhost:11434", "llama3.1")
    req = GenerationRequest(prompt="Extract", schema_spec=schema_from_model(_Contact), stream=True)
    prep = adapter.build_request(req, rt, None)

    assert prep.json_body["stream"] is True
    assert prep.json_body["format"]["type"] == "object"

    chunk = adapter.parse_stream_event(
        {
            "message": {"content": "}"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "eval_count": 5,
        }
    )
    assert chunk.delta_text == "}"
    assert chunk.done is True
    assert chunk.input_tokens == 8
    assert chunk.output_tokens == 5
