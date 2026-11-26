"""Unit tests for risk-tiered tool validation, policy caps, and untrusted note wrapping."""

from hitl_agent_poc.models import ToolCallRequest
from hitl_agent_poc.tools import OperationsLedger, build_operations_registry


def test_read_only_lookup_wraps_untrusted_customer_note() -> None:
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger, max_refund_usd=500.0)
    call = ToolCallRequest(
        call_id="c1",
        name="lookup_shipment_status",
        arguments={"tracking_id": "SHP-6660"},
    )
    risk, validated, err = registry.validate_call(call)
    assert risk == "read_only"
    assert err is None
    assert validated is not None

    outcome = registry.execute_validated(call, validated)
    assert outcome.is_error is False
    assert "<untrusted_tool_data>" in outcome.output["untrusted_customer_note"]


def test_refund_policy_cap_blocks_excessive_amount_before_approval() -> None:
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger, max_refund_usd=250.0)
    call = ToolCallRequest(
        call_id="c2",
        name="issue_customer_refund",
        arguments={"tracking_id": "SHP-1042", "amount_usd": 400.0, "reason": "Manual override"},
    )
    risk, validated, err = registry.validate_call(call)
    assert risk == "mutating"
    assert validated is None
    assert err is not None
    assert err.is_error is True
    assert "exceeds policy cap" in err.output["error"]


def test_unknown_tool_is_blocked() -> None:
    ledger = OperationsLedger()
    registry = build_operations_registry(ledger=ledger)
    risk, validated, err = registry.validate_call(
        ToolCallRequest(call_id="c3", name="delete_all_orders", arguments={})
    )
    assert risk is None
    assert validated is None
    assert err is not None
    assert err.is_error is True
