"""Offline & optional live CLI demo for bounded multi-step tool routing."""

import argparse
import asyncio
import json
import logging

import httpx

from tool_router_lab.config import AppSettings
from tool_router_lab.models import ProviderName
from tool_router_lab.router import BoundedToolRouter
from tool_router_lab.tools import build_logistics_tool_registry
from tool_router_lab.transport import ResilientHttpTransport


def _build_offline_gemini_transport() -> httpx.MockTransport:
    """Simulate a 3-step Gemini logistics workflow (shipment lookup -> stock & refund -> answer)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        contents = body["contents"]

        if len(contents) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "lookup_shipment_status",
                                            "args": {"tracking_id": "SHP-1042"},
                                        }
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 80,
                        "candidatesTokenCount": 14,
                        "totalTokenCount": 94,
                    },
                    "modelVersion": "gemini-2.5-flash",
                },
            )

        if len(contents) == 3:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "check_warehouse_stock",
                                            "args": {"sku": "SKU-4812-X", "hub": "BLR-01"},
                                        }
                                    },
                                    {
                                        "functionCall": {
                                            "name": "calculate_sla_refund",
                                            "args": {"order_total_usd": 200.0, "delay_hours": 18},
                                        }
                                    },
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 125,
                        "candidatesTokenCount": 26,
                        "totalTokenCount": 151,
                    },
                    "modelVersion": "gemini-2.5-flash",
                },
            )

        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": (
                                        "Shipment SHP-1042 is delayed by 18 hours at BLR-01. "
                                        "14 units of SKU-4812-X are available in stock, and the "
                                        "15% SLA delay credit is $30.00."
                                    )
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 160,
                    "candidatesTokenCount": 32,
                    "totalTokenCount": 192,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        )

    return httpx.MockTransport(handler)


async def run_demo(*, live: bool, provider: ProviderName | None, query: str) -> None:
    """Execute the bounded tool-calling router and print the audited trace."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = (
        AppSettings()
        if live
        else AppSettings(_env_file=None, gemini_api_key="demo-gemini-key")
    )
    mock = None if live else _build_offline_gemini_transport()
    registry = build_logistics_tool_registry()

    async with ResilientHttpTransport(
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        transport=mock,
    ) as transport:
        router = BoundedToolRouter(settings=settings, registry=registry, transport=transport)
        result = await router.run(query, provider=provider)

    print(result.model_dump_json(indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Bounded multi-provider tool-calling router.")
    parser.add_argument("--live", action="store_true", help="Call live provider from .env.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic", "ollama"],
        default=None,
    )
    parser.add_argument(
        "--query",
        default=(
            "Check shipment SHP-1042 (order total $200). Verify replacement stock at its hub "
            "and compute any SLA delay credit."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run_demo(live=args.live, provider=args.provider, query=args.query))


if __name__ == "__main__":
    main()
