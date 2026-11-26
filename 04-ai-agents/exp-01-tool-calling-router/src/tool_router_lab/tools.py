"""Allow-listed tool registry with Pydantic argument validation and sample logistics tools."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tool_router_lab.models import (
    ToolCallRequest,
    ToolExecutionResult,
    ToolSpec,
    build_strict_schema,
)


@dataclass(frozen=True)
class RegisteredTool:
    """Internal binding of a ToolSpec, Pydantic arg validator, and pure Python handler."""

    spec: ToolSpec
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], dict[str, Any]]


class ToolRegistry:
    """Explicit allow-list dispatching validated tool calls safely."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[[Any], dict[str, Any]],
    ) -> None:
        """Register a typed tool under an explicit allow-listed name."""
        spec = ToolSpec(
            name=name,
            description=description,
            json_schema=build_strict_schema(args_model),
        )
        self._tools[name] = RegisteredTool(spec=spec, args_model=args_model, handler=handler)

    def specs(self) -> list[ToolSpec]:
        """Return all registered tool specifications."""
        return [entry.spec for entry in self._tools.values()]

    def execute(self, call: ToolCallRequest) -> ToolExecutionResult:
        """Validate ``call.arguments`` and invoke allow-listed handler safely."""
        registered = self._tools.get(call.name)
        if registered is None:
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                output={"error": f"Tool '{call.name}' is not registered or permitted."},
                is_error=True,
            )

        try:
            validated_args = registered.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                output={"error": f"Invalid arguments for '{call.name}': {exc.errors()}"},
                is_error=True,
            )

        try:
            result = registered.handler(validated_args)
        except Exception as exc:  # noqa: BLE001 — return tool error cleanly to model
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                output={"error": f"Tool execution failed: {exc}"},
                is_error=True,
            )

        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            output=result,
            is_error=False,
        )


class ShipmentLookupArgs(BaseModel):
    """Input schema for ``lookup_shipment_status``."""

    model_config = ConfigDict(extra="forbid")

    tracking_id: str = Field(pattern=r"^SHP-[0-9]{4}$", description="Format SHP-1234.")


class StockCheckArgs(BaseModel):
    """Input schema for ``check_warehouse_stock``."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(min_length=3, description="Warehouse SKU identifier, e.g. SKU-4812-X.")
    hub: str = Field(min_length=2, description="Warehouse hub code, e.g. BLR-01.")


class RefundEstimateArgs(BaseModel):
    """Input schema for ``calculate_sla_refund``."""

    model_config = ConfigDict(extra="forbid")

    order_total_usd: float = Field(gt=0.0, le=10000.0)
    delay_hours: int = Field(ge=0, le=720)


_SHIPMENTS: dict[str, dict[str, Any]] = {
    "SHP-1042": {"status": "delayed", "hub": "BLR-01", "delay_hours": 18, "sku": "SKU-4812-X"},
    "SHP-2088": {"status": "delivered", "hub": "DEL-02", "delay_hours": 0, "sku": "SKU-9001-A"},
}

_STOCK: dict[tuple[str, str], int] = {
    ("SKU-4812-X", "BLR-01"): 14,
    ("SKU-9001-A", "DEL-02"): 0,
}


def build_logistics_tool_registry() -> ToolRegistry:
    """Create a synthetic retail/logistics tool registry for offline & live experiments."""
    registry = ToolRegistry()

    def lookup_shipment(args: ShipmentLookupArgs) -> dict[str, Any]:
        record = _SHIPMENTS.get(args.tracking_id)
        if record is None:
            return {"found": False, "tracking_id": args.tracking_id}
        return {"found": True, "tracking_id": args.tracking_id, **record}

    def check_stock(args: StockCheckArgs) -> dict[str, Any]:
        qty = _STOCK.get((args.sku, args.hub), 0)
        return {"sku": args.sku, "hub": args.hub, "available_units": qty}

    def calculate_refund(args: RefundEstimateArgs) -> dict[str, Any]:
        rate = 0.15 if args.delay_hours >= 12 else 0.0
        return {
            "order_total_usd": args.order_total_usd,
            "delay_hours": args.delay_hours,
            "sla_breach": args.delay_hours >= 12,
            "credit_usd": round(args.order_total_usd * rate, 2),
        }

    registry.register(
        name="lookup_shipment_status",
        description="Look up shipment status, hub, delay hours, and SKU by tracking ID.",
        args_model=ShipmentLookupArgs,
        handler=lookup_shipment,
    )
    registry.register(
        name="check_warehouse_stock",
        description="Check available replacement inventory for a SKU at a warehouse hub.",
        args_model=StockCheckArgs,
        handler=check_stock,
    )
    registry.register(
        name="calculate_sla_refund",
        description="Compute SLA delay credit (15% when delay_hours >= 12).",
        args_model=RefundEstimateArgs,
        handler=calculate_refund,
    )
    return registry
