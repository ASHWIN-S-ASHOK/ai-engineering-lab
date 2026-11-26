"""Bounded multi-provider tool-calling router with Pydantic validation."""

from tool_router_lab.adapters import ADAPTER_REGISTRY, ToolCallingAdapter
from tool_router_lab.config import AppSettings
from tool_router_lab.models import (
    AssistantTextTurn,
    AssistantToolTurn,
    ConversationTurn,
    MaxStepsExceededError,
    ModelStepOutput,
    NonRetryableApiError,
    ProviderName,
    RetryableApiError,
    RouterResult,
    RouterTraceStep,
    TokenUsage,
    ToolCallRequest,
    ToolExecutionResult,
    ToolResultTurn,
    ToolRouterError,
    ToolSpec,
    UserTurn,
)
from tool_router_lab.router import BoundedToolRouter
from tool_router_lab.tools import ToolRegistry, build_logistics_tool_registry

__all__ = [
    "ADAPTER_REGISTRY",
    "AppSettings",
    "AssistantTextTurn",
    "AssistantToolTurn",
    "BoundedToolRouter",
    "ConversationTurn",
    "MaxStepsExceededError",
    "ModelStepOutput",
    "NonRetryableApiError",
    "ProviderName",
    "RetryableApiError",
    "RouterResult",
    "RouterTraceStep",
    "TokenUsage",
    "ToolCallRequest",
    "ToolCallingAdapter",
    "ToolExecutionResult",
    "ToolRegistry",
    "ToolResultTurn",
    "ToolRouterError",
    "ToolSpec",
    "UserTurn",
    "build_logistics_tool_registry",
]
