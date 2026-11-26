"""Offline integration tests for BoundedToolRouter execution, errors, and step limits."""

import json

import httpx
import pytest

from tool_router_lab.config import AppSettings
from tool_router_lab.main import _build_offline_gemini_transport
from tool_router_lab.models import MaxStepsExceededError
from tool_router_lab.router import BoundedToolRouter
from tool_router_lab.tools import build_logistics_tool_registry
from tool_router_lab.transport import ResilientHttpTransport


async def test_router_executes_multi_step_logistics_workflow() -> None:
    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    registry = build_logistics_tool_registry()
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=_build_offline_gemini_transport(),
    )
    async with transport:
        router = BoundedToolRouter(settings=settings, registry=registry, transport=transport)
        result = await router.run("Check SHP-1042 and SLA refund for $200.")

    assert result.steps_taken == 3
    assert result.trace[0].requested_tools == ["lookup_shipment_status"]
    assert result.trace[1].requested_tools == ["check_warehouse_stock", "calculate_sla_refund"]
    assert result.trace[2].requested_tools == []
    assert result.total_usage.total_tokens == 94 + 151 + 192
    assert "$30.00" in result.answer


async def test_router_feeds_validation_error_back_and_recovers() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content.decode("utf-8"))
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "lookup_shipment_status",
                                    "arguments": {"tracking_id": "INVALID"},
                                }
                            }
                        ],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 5,
                },
            )
        # Ensure the error payload was forwarded in the tool turn.
        tool_msg = body["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert "Invalid arguments" in tool_msg["content"]
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Please provide a valid tracking ID in SHP-1234 format.",
                },
                "prompt_eval_count": 30,
                "eval_count": 10,
            },
        )

    settings = AppSettings(_env_file=None, llm_provider="ollama")
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )
    async with transport:
        router = BoundedToolRouter(
            settings=settings,
            registry=build_logistics_tool_registry(),
            transport=transport,
        )
        result = await router.run("Check INVALID", provider="ollama")

    assert result.steps_taken == 2
    assert result.trace[0].tool_results[0].is_error is True
    assert "SHP-1234" in result.answer


async def test_router_enforces_max_steps_budget() -> None:
    def infinite_tool_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
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
        )

    settings = AppSettings(_env_file=None, llm_provider="ollama", router_max_steps=2)
    transport = ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(infinite_tool_handler),
    )
    async with transport:
        router = BoundedToolRouter(
            settings=settings,
            registry=build_logistics_tool_registry(),
            transport=transport,
        )
        with pytest.raises(MaxStepsExceededError) as exc_info:
            await router.run("Loop forever", provider="ollama")

    assert len(exc_info.value.trace) == 2
