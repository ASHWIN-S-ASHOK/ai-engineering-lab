"""Unit tests for multi-turn tool-calling wire formats across all 4 providers."""

import json

from pydantic import SecretStr

from tool_router_lab.adapters import ADAPTER_REGISTRY, ToolCallingAdapter
from tool_router_lab.models import (
    AssistantToolTurn,
    ProviderRuntimeConfig,
    ToolCallRequest,
    ToolExecutionResult,
    ToolResultTurn,
    UserTurn,
)
from tool_router_lab.tools import build_logistics_tool_registry


def _runtime(provider: str, base_url: str, model: str) -> ProviderRuntimeConfig:
    return ProviderRuntimeConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        base_url=base_url,
        has_api_key=(provider != "ollama"),
    )


def _sample_turns() -> list[UserTurn | AssistantToolTurn | ToolResultTurn]:
    return [
        UserTurn(text="Check SHP-1042"),
        AssistantToolTurn(
            tool_calls=[
                ToolCallRequest(
                    call_id="call-1",
                    name="lookup_shipment_status",
                    arguments={"tracking_id": "SHP-1042"},
                )
            ]
        ),
        ToolResultTurn(
            results=[
                ToolExecutionResult(
                    call_id="call-1",
                    name="lookup_shipment_status",
                    output={"status": "delayed"},
                )
            ]
        ),
    ]


def test_all_adapters_conform_to_protocol() -> None:
    assert set(ADAPTER_REGISTRY.keys()) == {"gemini", "openai", "anthropic", "ollama"}
    for adapter in ADAPTER_REGISTRY.values():
        assert isinstance(adapter, ToolCallingAdapter)


def test_gemini_multi_turn_tool_serialization_and_parsing() -> None:
    adapter = ADAPTER_REGISTRY["gemini"]
    specs = build_logistics_tool_registry().specs()
    rt = _runtime("gemini", "https://generativelanguage.googleapis.com", "gemini-2.5-flash")
    prep = adapter.build_request(
        system_prompt="Sys",
        turns=_sample_turns(),
        tools=specs,
        runtime=rt,
        api_key=SecretStr("g-key"),
    )

    assert "functionDeclarations" in prep.json_body["tools"][0]
    fc_name = prep.json_body["contents"][1]["parts"][0]["functionCall"]["name"]
    fr_name = prep.json_body["contents"][2]["parts"][0]["functionResponse"]["name"]
    assert fc_name == "lookup_shipment_status"
    assert fr_name == "lookup_shipment_status"

    parsed = adapter.parse_response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "lookup_shipment_status",
                                    "args": {"tracking_id": "SHP-1042"},
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        },
        requested_model="gemini-2.5-flash",
    )
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {"tracking_id": "SHP-1042"}


def test_openai_responses_tool_serialization_and_parsing() -> None:
    adapter = ADAPTER_REGISTRY["openai"]
    specs = build_logistics_tool_registry().specs()
    rt = _runtime("openai", "https://api.openai.com/v1", "gpt-4o-mini")
    prep = adapter.build_request(
        system_prompt="Sys",
        turns=_sample_turns(),
        tools=specs,
        runtime=rt,
        api_key=SecretStr("o-key"),
    )

    assert prep.json_body["tools"][0]["strict"] is True
    assert prep.json_body["input"][1]["type"] == "function_call"
    assert prep.json_body["input"][2]["type"] == "function_call_output"

    parsed = adapter.parse_response(
        {
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_99",
                    "name": "lookup_shipment_status",
                    "arguments": json.dumps({"tracking_id": "SHP-1042"}),
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        },
        requested_model="gpt-4o-mini",
    )
    assert parsed.tool_calls[0].call_id == "call_99"
    assert parsed.tool_calls[0].arguments["tracking_id"] == "SHP-1042"


def test_anthropic_and_ollama_tool_serialization_and_parsing() -> None:
    specs = build_logistics_tool_registry().specs()

    ant = ADAPTER_REGISTRY["anthropic"]
    ant_rt = _runtime("anthropic", "https://api.anthropic.com/v1", "claude-3-5-haiku-latest")
    ant_prep = ant.build_request(
        system_prompt="Sys",
        turns=_sample_turns(),
        tools=specs,
        runtime=ant_rt,
        api_key=SecretStr("a-key"),
    )
    assert ant_prep.json_body["tools"][0]["input_schema"]["type"] == "object"
    assert ant_prep.json_body["messages"][1]["content"][0]["type"] == "tool_use"
    assert ant_prep.json_body["messages"][2]["content"][0]["type"] == "tool_result"

    ant_parsed = ant.parse_response(
        {
            "model": "claude-3-5-haiku-latest",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "check_warehouse_stock",
                    "input": {"sku": "SKU-4812-X", "hub": "BLR-01"},
                }
            ],
            "usage": {"input_tokens": 14, "output_tokens": 9},
        },
        requested_model="claude-3-5-haiku-latest",
    )
    assert ant_parsed.tool_calls[0].call_id == "toolu_1"

    oll = ADAPTER_REGISTRY["ollama"]
    oll_rt = _runtime("ollama", "http://localhost:11434", "llama3.1")
    oll_prep = oll.build_request(
        system_prompt="Sys",
        turns=_sample_turns(),
        tools=specs,
        runtime=oll_rt,
        api_key=None,
    )
    assert oll_prep.json_body["messages"][-1]["role"] == "tool"
    oll_parsed = oll.parse_response(
        {
            "model": "llama3.1",
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_shipment_status",
                            "arguments": {"tracking_id": "SHP-1042"},
                        }
                    }
                ],
            },
            "prompt_eval_count": 10,
            "eval_count": 4,
        },
        requested_model="llama3.1",
    )
    assert oll_parsed.tool_calls[0].name == "lookup_shipment_status"
