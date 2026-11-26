"""Multi-turn tool-calling REST adapters for Gemini, OpenAI, Anthropic, and Ollama."""

import json
from typing import Any, Protocol, runtime_checkable

from pydantic import SecretStr

from tool_router_lab.models import (
    AssistantTextTurn,
    AssistantToolTurn,
    ConversationTurn,
    ModelStepOutput,
    NonRetryableApiError,
    PreparedHttpRequest,
    ProviderName,
    ProviderRuntimeConfig,
    TokenUsage,
    ToolCallRequest,
    ToolResultTurn,
    ToolSpec,
    UserTurn,
)


@runtime_checkable
class ToolCallingAdapter(Protocol):
    """Protocol translating canonical multi-turn tool conversations per vendor wire spec."""

    @property
    def provider_name(self) -> ProviderName:
        """Identifier of the model provider."""
        ...

    def build_request(
        self,
        *,
        system_prompt: str,
        turns: list[ConversationTurn],
        tools: list[ToolSpec],
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        """Build a multi-turn HTTP request declaring ``tools``."""
        ...

    def parse_response(self, payload: dict[str, Any], *, requested_model: str) -> ModelStepOutput:
        """Extract text or tool calls from provider response."""
        ...


def _require_key(api_key: SecretStr | None, provider: ProviderName) -> str:
    if api_key is None or not api_key.get_secret_value().strip():
        raise NonRetryableApiError(f"Missing API key for provider '{provider}'.")
    return api_key.get_secret_value()


class GeminiToolAdapter:
    """Gemini REST tool-calling adapter (``functionDeclarations`` / ``functionCall``)."""

    provider_name: ProviderName = "gemini"

    def build_request(
        self,
        *,
        system_prompt: str,
        turns: list[ConversationTurn],
        tools: list[ToolSpec],
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        contents: list[dict[str, Any]] = []
        for turn in turns:
            if isinstance(turn, UserTurn):
                contents.append({"role": "user", "parts": [{"text": turn.text}]})
            elif isinstance(turn, AssistantToolTurn):
                parts: list[dict[str, Any]] = []
                if turn.text:
                    parts.append({"text": turn.text})
                parts.extend(
                    {"functionCall": {"name": tc.name, "args": tc.arguments}}
                    for tc in turn.tool_calls
                )
                contents.append({"role": "model", "parts": parts})
            elif isinstance(turn, ToolResultTurn):
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": res.name,
                                    "response": res.output,
                                }
                            }
                            for res in turn.results
                        ],
                    }
                )
            elif isinstance(turn, AssistantTextTurn):
                contents.append({"role": "model", "parts": [{"text": turn.text}]})

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
        }
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.json_schema,
                        }
                        for t in tools
                    ]
                }
            ]

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/v1beta/models/{runtime.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json_body=body,
        )

    def parse_response(self, payload: dict[str, Any], *, requested_model: str) -> ModelStepOutput:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise NonRetryableApiError("Gemini response missing 'candidates'.")
        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])

        text_chunks: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for idx, part in enumerate(parts, start=1):
            if "text" in part:
                text_chunks.append(str(part["text"]))
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCallRequest(
                        call_id=f"gem-call-{idx}",
                        name=str(fc["name"]),
                        arguments=dict(fc.get("args") or {}),
                    )
                )

        usage_raw = payload.get("usageMetadata", {})
        in_tok = int(usage_raw.get("promptTokenCount", 0))
        out_tok = int(usage_raw.get("candidatesTokenCount", 0))
        tot_tok = int(usage_raw.get("totalTokenCount", in_tok + out_tok))
        return ModelStepOutput(
            model=str(payload.get("modelVersion", requested_model)),
            text="".join(text_chunks),
            tool_calls=tool_calls,
            finish_reason=first.get("finishReason"),
            usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok),
        )


class OpenAIToolAdapter:
    """OpenAI Responses API tool adapter (``function_call`` / ``function_call_output``)."""

    provider_name: ProviderName = "openai"

    def build_request(
        self,
        *,
        system_prompt: str,
        turns: list[ConversationTurn],
        tools: list[ToolSpec],
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        items: list[dict[str, Any]] = []
        for turn in turns:
            if isinstance(turn, UserTurn):
                items.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantToolTurn):
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": tc.call_id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                    for tc in turn.tool_calls
                )
            elif isinstance(turn, ToolResultTurn):
                items.extend(
                    {
                        "type": "function_call_output",
                        "call_id": res.call_id,
                        "output": json.dumps(res.output),
                    }
                    for res in turn.results
                )
            elif isinstance(turn, AssistantTextTurn):
                items.append({"role": "assistant", "content": turn.text})

        body: dict[str, Any] = {
            "model": runtime.model,
            "instructions": system_prompt,
            "input": items,
            "temperature": 0.0,
            "max_output_tokens": 512,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema,
                    "strict": True,
                }
                for t in tools
            ]

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/responses",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json_body=body,
        )

    def parse_response(self, payload: dict[str, Any], *, requested_model: str) -> ModelStepOutput:
        output_items = payload.get("output")
        if not isinstance(output_items, list):
            raise NonRetryableApiError("OpenAI response missing 'output'.")

        text_chunks: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for item in output_items:
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and "text" in part:
                        text_chunks.append(str(part["text"]))
            elif item_type == "function_call":
                raw_args = item.get("arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except ValueError as exc:
                    raise NonRetryableApiError(
                        f"OpenAI returned invalid tool JSON arguments: {exc}"
                    ) from exc
                tool_calls.append(
                    ToolCallRequest(
                        call_id=str(item["call_id"]),
                        name=str(item["name"]),
                        arguments=dict(parsed_args or {}),
                    )
                )

        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))
        tot_tok = int(usage_raw.get("total_tokens", in_tok + out_tok))
        return ModelStepOutput(
            model=str(payload.get("model", requested_model)),
            text="".join(text_chunks),
            tool_calls=tool_calls,
            finish_reason=payload.get("status"),
            usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok),
        )


class AnthropicToolAdapter:
    """Anthropic Messages API tool adapter (``tool_use`` / ``tool_result``)."""

    provider_name: ProviderName = "anthropic"

    def build_request(
        self,
        *,
        system_prompt: str,
        turns: list[ConversationTurn],
        tools: list[ToolSpec],
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        key = _require_key(api_key, self.provider_name)
        messages: list[dict[str, Any]] = []
        for turn in turns:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantToolTurn):
                blocks: list[dict[str, Any]] = []
                if turn.text:
                    blocks.append({"type": "text", "text": turn.text})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tc.call_id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in turn.tool_calls
                )
                messages.append({"role": "assistant", "content": blocks})
            elif isinstance(turn, ToolResultTurn):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": res.call_id,
                                "content": json.dumps(res.output),
                                "is_error": res.is_error,
                            }
                            for res in turn.results
                        ],
                    }
                )
            elif isinstance(turn, AssistantTextTurn):
                messages.append({"role": "assistant", "content": turn.text})

        body: dict[str, Any] = {
            "model": runtime.model,
            "system": system_prompt,
            "max_tokens": 512,
            "temperature": 0.0,
            "messages": messages,
        }
        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.json_schema,
                }
                for t in tools
            ]

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": runtime.anthropic_version,
            },
            json_body=body,
        )

    def parse_response(self, payload: dict[str, Any], *, requested_model: str) -> ModelStepOutput:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise NonRetryableApiError("Anthropic response missing 'content'.")

        text_chunks: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for block in blocks:
            btype = block.get("type")
            if btype == "text" and "text" in block:
                text_chunks.append(str(block["text"]))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        call_id=str(block["id"]),
                        name=str(block["name"]),
                        arguments=dict(block.get("input") or {}),
                    )
                )

        usage_raw = payload.get("usage", {})
        in_tok = int(usage_raw.get("input_tokens", 0))
        out_tok = int(usage_raw.get("output_tokens", 0))
        return ModelStepOutput(
            model=str(payload.get("model", requested_model)),
            text="".join(text_chunks),
            tool_calls=tool_calls,
            finish_reason=payload.get("stop_reason"),
            usage=TokenUsage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            ),
        )


class OllamaToolAdapter:
    """Local Ollama ``/api/chat`` tool-calling adapter."""

    provider_name: ProviderName = "ollama"

    def build_request(
        self,
        *,
        system_prompt: str,
        turns: list[ConversationTurn],
        tools: list[ToolSpec],
        runtime: ProviderRuntimeConfig,
        api_key: SecretStr | None,
    ) -> PreparedHttpRequest:
        del api_key
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for turn in turns:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantToolTurn):
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.text,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments},
                            }
                            for tc in turn.tool_calls
                        ],
                    }
                )
            elif isinstance(turn, ToolResultTurn):
                messages.extend(
                    {
                        "role": "tool",
                        "tool_name": res.name,
                        "content": json.dumps(res.output),
                    }
                    for res in turn.results
                )
            elif isinstance(turn, AssistantTextTurn):
                messages.append({"role": "assistant", "content": turn.text})

        body: dict[str, Any] = {
            "model": runtime.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.json_schema,
                    },
                }
                for t in tools
            ]

        return PreparedHttpRequest(
            url=f"{runtime.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json_body=body,
        )

    def parse_response(self, payload: dict[str, Any], *, requested_model: str) -> ModelStepOutput:
        message = payload.get("message")
        if not isinstance(message, dict):
            raise NonRetryableApiError("Ollama response missing 'message'.")

        tool_calls: list[ToolCallRequest] = []
        for idx, raw_tc in enumerate(message.get("tool_calls") or [], start=1):
            fn = raw_tc.get("function") or {}
            tool_calls.append(
                ToolCallRequest(
                    call_id=str(raw_tc.get("id") or f"oll-call-{idx}"),
                    name=str(fn["name"]),
                    arguments=dict(fn.get("arguments") or {}),
                )
            )

        in_tok = int(payload.get("prompt_eval_count", 0))
        out_tok = int(payload.get("eval_count", 0))
        return ModelStepOutput(
            model=str(payload.get("model", requested_model)),
            text=str(message.get("content", "")),
            tool_calls=tool_calls,
            finish_reason=payload.get("done_reason"),
            usage=TokenUsage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            ),
        )


ADAPTER_REGISTRY: dict[ProviderName, ToolCallingAdapter] = {
    "gemini": GeminiToolAdapter(),
    "openai": OpenAIToolAdapter(),
    "anthropic": AnthropicToolAdapter(),
    "ollama": OllamaToolAdapter(),
}
