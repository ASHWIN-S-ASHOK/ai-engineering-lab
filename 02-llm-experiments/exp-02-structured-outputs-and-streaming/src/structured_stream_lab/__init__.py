"""Structured JSON-schema outputs and SSE/NDJSON streaming across switchable LLM providers."""

from structured_stream_lab.adapters import ADAPTER_REGISTRY, StructuredStreamProvider
from structured_stream_lab.client import StructuredStreamingClient, create_client
from structured_stream_lab.config import AppSettings
from structured_stream_lab.models import (
    GenerationRequest,
    NonRetryableStreamError,
    ProviderName,
    ProviderRuntimeConfig,
    RetryableStreamError,
    SchemaValidationError,
    StreamEvent,
    StreamFraming,
    StructuredResult,
    StructuredSchemaSpec,
    StructuredStreamError,
    TokenUsage,
    schema_from_model,
)

__all__ = [
    "ADAPTER_REGISTRY",
    "AppSettings",
    "GenerationRequest",
    "NonRetryableStreamError",
    "ProviderName",
    "ProviderRuntimeConfig",
    "RetryableStreamError",
    "SchemaValidationError",
    "StreamEvent",
    "StreamFraming",
    "StructuredResult",
    "StructuredSchemaSpec",
    "StructuredStreamError",
    "StructuredStreamProvider",
    "StructuredStreamingClient",
    "TokenUsage",
    "create_client",
    "schema_from_model",
]
