"""Unit tests for ToolRegistry allow-listing and Pydantic argument validation."""

from tool_router_lab.models import ToolCallRequest
from tool_router_lab.tools import build_logistics_tool_registry


def test_registry_executes_valid_tool_call() -> None:
    registry = build_logistics_tool_registry()
    res = registry.execute(
        ToolCallRequest(
            call_id="c1",
            name="lookup_shipment_status",
            arguments={"tracking_id": "SHP-1042"},
        )
    )
    assert res.is_error is False
    assert res.output["status"] == "delayed"
    assert res.output["hub"] == "BLR-01"


def test_registry_blocks_unknown_tool_safely() -> None:
    registry = build_logistics_tool_registry()
    res = registry.execute(
        ToolCallRequest(call_id="c2", name="drop_database", arguments={})
    )
    assert res.is_error is True
    assert "not registered" in res.output["error"]


def test_registry_returns_structured_error_on_invalid_arguments() -> None:
    registry = build_logistics_tool_registry()
    res = registry.execute(
        ToolCallRequest(
            call_id="c3",
            name="lookup_shipment_status",
            arguments={"tracking_id": "BAD-FORMAT"},
        )
    )
    assert res.is_error is True
    assert "Invalid arguments" in res.output["error"]


def test_tool_specs_enforce_strict_additional_properties_false() -> None:
    registry = build_logistics_tool_registry()
    for spec in registry.specs():
        assert spec.json_schema["additionalProperties"] is False
