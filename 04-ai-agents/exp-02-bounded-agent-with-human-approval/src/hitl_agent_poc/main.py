"""Offline & optional interactive live demo of HITL approval and injection defense."""

import argparse
import asyncio
import json
import logging

import httpx

from hitl_agent_poc.agent import BoundedHitlAgent
from hitl_agent_poc.config import AppSettings
from hitl_agent_poc.models import ApprovalDecision, ProviderName
from hitl_agent_poc.tools import OperationsLedger, build_operations_registry
from hitl_agent_poc.transport import ResilientHttpTransport


def _build_offline_transport() -> httpx.MockTransport:
    """Simulate two Gemini scenarios: valid SLA refund approval + prompt-injected shipment."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        contents = body["contents"]

        if len(contents) == 1:
            first_text = contents[0]["parts"][0]["text"]
            tracking_id = "SHP-6660" if "SHP-6660" in first_text else "SHP-1042"
            return _gem_fc("lookup_shipment_status", {"tracking_id": tracking_id})

        if len(contents) == 3:
            fn_resp = contents[2]["parts"][0]["functionResponse"]["response"]
            tid = fn_resp["tracking_id"]
            if tid == "SHP-6660":
                # Simulate an LLM tricked by indirect prompt injection inside customer_note!
                return _gem_fc(
                    "issue_customer_refund",
                    {
                        "tracking_id": "SHP-6660",
                        "amount_usd": 499.0,
                        "reason": "Urgent customer note override",
                    },
                )
            return _gem_fc(
                "issue_customer_refund",
                {
                    "tracking_id": "SHP-1042",
                    "amount_usd": 30.0,
                    "reason": "18h SLA delivery breach (15%)",
                },
            )

        last_resp = contents[-1]["parts"][0]["functionResponse"]["response"]
        if last_resp.get("rejected"):
            msg = (
                f"Action blocked by human reviewer ({last_resp['reviewer']}): "
                f"{last_resp['reason']}. No refund was issued."
            )
        else:
            msg = (
                f"Approved and executed refund {last_resp.get('refund_id')} "
                f"for ${last_resp.get('amount_usd')}."
            )
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": msg}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 18},
            },
        )

    return httpx.MockTransport(handler)


def _gem_fc(name: str, args: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"name": name, "args": args}}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 70, "candidatesTokenCount": 15},
        },
    )


async def run_offline_demo() -> None:
    """Demonstrate pause/resume approval and prompt-injection blocking offline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = AppSettings(_env_file=None, gemini_api_key="demo-gemini-key")
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger, max_refund_usd=settings.max_refund_usd)

    async with ResilientHttpTransport(transport=_build_offline_transport()) as transport:
        agent = BoundedHitlAgent(settings=settings, registry=registry, transport=transport)

        print("=== Scenario 1: Legitimate SLA Refund (Human Approves) ===")
        paused = await agent.start("Check SHP-1042 and issue the 15% SLA delay refund if delayed.")
        assert paused.pending_approval is not None
        print(f"Paused status={paused.status} pending={paused.pending_approval.summary}")
        done = await agent.resume(
            paused,
            ApprovalDecision(
                action_id=paused.pending_approval.action_id,
                approved=True,
                reviewer="ashwin.ops",
                reason="Verified 18h hub delay",
            ),
        )
        print(f"Final status={done.status} | Answer: {done.final_answer}")

        print("\n=== Scenario 2: Indirect Prompt Injection (Human Rejects) ===")
        inj_paused = await agent.start("Check status of shipment SHP-6660.")
        assert inj_paused.pending_approval is not None
        print(f"Paused status={inj_paused.status} pending={inj_paused.pending_approval.summary}")
        inj_done = await agent.resume(
            inj_paused,
            ApprovalDecision(
                action_id=inj_paused.pending_approval.action_id,
                approved=False,
                reviewer="ashwin.ops",
                reason="Shipment SHP-6660 has 0 delay; injected customer note rejected.",
            ),
        )
        print(f"Final status={inj_done.status} | Answer: {inj_done.final_answer}")
        print(f"Ledger refunds_issued={ledger.refunds_issued}")


async def run_live_demo(provider: ProviderName | None, query: str) -> None:
    """Interactive live HITL session using .env credentials and terminal approval."""
    settings = AppSettings()
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger, max_refund_usd=settings.max_refund_usd)

    async with ResilientHttpTransport(
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
    ) as transport:
        agent = BoundedHitlAgent(settings=settings, registry=registry, transport=transport)
        state = await agent.start(query, provider=provider)
        while state.status == "PENDING_APPROVAL" and state.pending_approval is not None:
            print(f"\n[APPROVAL REQUIRED] {state.pending_approval.summary}")
            choice = input("Approve? [y/N]: ").strip().lower()
            state = await agent.resume(
                state,
                ApprovalDecision(
                    action_id=state.pending_approval.action_id,
                    approved=(choice == "y"),
                    reviewer="cli-operator",
                    reason="Interactive CLI review",
                ),
            )
        print(state.model_dump_json(indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Bounded HITL Agent POC.")
    parser.add_argument("--live", action="store_true", help="Run live interactive session.")
    parser.add_argument("--provider", choices=["gemini", "openai", "anthropic", "ollama"])
    parser.add_argument(
        "--query",
        default="Check shipment SHP-1042 and issue a 15% SLA delay refund if delayed.",
    )
    args = parser.parse_args()
    if args.live:
        asyncio.run(run_live_demo(args.provider, args.query))
    else:
        asyncio.run(run_offline_demo())


if __name__ == "__main__":
    main()
