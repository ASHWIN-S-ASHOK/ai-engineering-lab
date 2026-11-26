"""Offline integration tests for HITL pause/resume state machine and security gates."""

import json

import httpx
import pytest

from hitl_agent_poc.agent import BoundedHitlAgent
from hitl_agent_poc.config import AppSettings
from hitl_agent_poc.main import _build_offline_transport
from hitl_agent_poc.models import (
    AgentRunState,
    ApprovalDecision,
    ApprovalMismatchError,
    MaxStepsExceededError,
)
from hitl_agent_poc.tools import OperationsLedger, build_operations_registry
from hitl_agent_poc.transport import ResilientHttpTransport


async def test_hitl_approval_executes_mutating_tool_and_completes() -> None:
    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger)

    async with ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=_build_offline_transport(),
    ) as transport:
        agent = BoundedHitlAgent(settings=settings, registry=registry, transport=transport)
        paused = await agent.start("Check SHP-1042 and issue SLA refund.")

        assert paused.status == "PENDING_APPROVAL"
        assert paused.pending_approval is not None
        assert paused.pending_approval.tool_call.name == "issue_customer_refund"
        assert ledger.refunds_issued == []

        # Verify state survives full JSON round-trip serialization.
        rehydrated = AgentRunState.model_validate_json(paused.model_dump_json())
        completed = await agent.resume(
            rehydrated,
            ApprovalDecision(
                action_id=rehydrated.pending_approval.action_id,  # type: ignore[union-attr]
                approved=True,
                reviewer="ashwin.ops",
                reason="Confirmed 18h delay",
            ),
        )

    assert completed.status == "COMPLETED"
    assert len(ledger.refunds_issued) == 1
    assert ledger.refunds_issued[0]["amount_usd"] == 30.0
    assert any(e.event == "approval_granted" for e in completed.audit_log)


async def test_indirect_prompt_injection_is_blocked_without_side_effect() -> None:
    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger)

    async with ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=_build_offline_transport(),
    ) as transport:
        agent = BoundedHitlAgent(settings=settings, registry=registry, transport=transport)
        paused = await agent.start("Check status of shipment SHP-6660.")
        assert paused.status == "PENDING_APPROVAL"
        assert paused.pending_approval is not None
        assert ledger.refunds_issued == []

        rejected = await agent.resume(
            paused,
            ApprovalDecision(
                action_id=paused.pending_approval.action_id,
                approved=False,
                reviewer="ashwin.ops",
                reason="Injection attempt inside customer note",
            ),
        )

    assert rejected.status == "COMPLETED"
    assert ledger.refunds_issued == []
    assert "blocked" in (rejected.final_answer or "").lower()
    assert any(e.event == "approval_rejected" for e in rejected.audit_log)


async def test_mismatched_action_id_raises_approval_mismatch() -> None:
    settings = AppSettings(_env_file=None, gemini_api_key="gem-key")
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger)

    async with ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=_build_offline_transport(),
    ) as transport:
        agent = BoundedHitlAgent(settings=settings, registry=registry, transport=transport)
        paused = await agent.start("Check SHP-1042 and issue SLA refund.")

        with pytest.raises(ApprovalMismatchError):
            await agent.resume(
                paused,
                ApprovalDecision(
                    action_id="act-wrong-id",
                    approved=True,
                    reviewer="ashwin.ops",
                    reason="Stale action id",
                ),
            )


async def test_mixed_read_and_mutating_turn_preserves_all_call_ids() -> None:
    seen_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append(body)
        if len(seen_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-read",
                                "function": {
                                    "name": "lookup_shipment_status",
                                    "arguments": {"tracking_id": "SHP-1042"},
                                },
                            },
                            {
                                "id": "call-mut",
                                "function": {
                                    "name": "dispatch_replacement_shipment",
                                    "arguments": {
                                        "tracking_id": "SHP-1042",
                                        "sku": "SKU-4812-X",
                                        "hub": "BLR-01",
                                    },
                                },
                            },
                        ],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 10,
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "Dispatched replacement."},
                "prompt_eval_count": 30,
                "eval_count": 5,
            },
        )

    settings = AppSettings(_env_file=None, llm_provider="ollama")
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger)
    async with ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    ) as transport:
        agent = BoundedHitlAgent(settings=settings, registry=registry, transport=transport)
        paused = await agent.start("Lookup SHP-1042 and dispatch replacement.", provider="ollama")
        assert paused.status == "PENDING_APPROVAL"
        assert len(paused.completed_turn_results) == 1
        assert paused.completed_turn_results[0].name == "lookup_shipment_status"
        assert paused.pending_approval is not None

        done = await agent.resume(
            paused,
            ApprovalDecision(
                action_id=paused.pending_approval.action_id,
                approved=True,
                reviewer="ashwin.ops",
                reason="Approved replacement",
            ),
        )

    assert done.status == "COMPLETED"
    assert len(ledger.replacements_dispatched) == 1
    # Both tool results must appear in the resumed Ollama messages list.
    tool_msgs = [m for m in seen_requests[1]["messages"] if m["role"] == "tool"]  # type: ignore[index]
    assert [m["tool_name"] for m in tool_msgs] == [
        "lookup_shipment_status",
        "dispatch_replacement_shipment",
    ]


async def test_max_steps_enforced() -> None:
    def loop_handler(_: httpx.Request) -> httpx.Response:
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

    settings = AppSettings(_env_file=None, llm_provider="ollama", agent_max_steps=2)
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger)
    async with ResilientHttpTransport(
        base_backoff_seconds=0.0,
        transport=httpx.MockTransport(loop_handler),
    ) as transport:
        agent = BoundedHitlAgent(settings=settings, registry=registry, transport=transport)
        with pytest.raises(MaxStepsExceededError) as exc_info:
            await agent.start("Loop", provider="ollama")

    assert exc_info.value.state.status == "FAILED"
