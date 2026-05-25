"""DSPy backend that uses the Claude Agent SDK (claude-agent-sdk).

This backend invokes Claude Code as a subprocess via the official Python
Agent SDK, giving DSPy access to Claude Code's agentic capabilities:
filesystem I/O, code execution, MCP tools, and structured output via
constrained decoding (``--json-schema``).

Usage::

    import dspy
    from dspy.clients._claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend(max_turns=3)
    lm = dspy.LM("claude-code/claude-sonnet-4-6", backend=backend)
    dspy.configure(lm=lm)

Works with both Anthropic API keys and Pro/Max subscription auth —
whichever the local ``claude`` CLI is configured with.
"""

from __future__ import annotations

import asyncio
import anyio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

from dspy.clients._request_utils import StreamChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context-window error
# ---------------------------------------------------------------------------

class ContextWindowError(Exception):
    """Raised when the prompt exceeds Claude's context window."""


# ---------------------------------------------------------------------------
# Capability queries (module-level, called by BaseLM via _backend_capability)
# ---------------------------------------------------------------------------

def supports_function_calling(model: str) -> bool:
    # Claude Code uses tools internally; it doesn't expose them as
    # OpenAI-style function_calling in the response.
    return False


def supports_reasoning(model: str) -> bool:
    return True


def supports_response_schema(model: str) -> bool:
    # Supported via --json-schema / output_format in the SDK.
    return True


def supported_params(model: str) -> set[str]:
    return {"temperature", "max_tokens", "stop", "response_schema"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_prefix(model: str) -> str:
    """Remove the ``claude-code/`` prefix and normalize the model name.

    Converts shorthand like ``sonnet-4.6`` to the full Claude API model ID
    ``claude-sonnet-4-6`` so callers can write ``claude-code/sonnet-4.6``
    instead of ``claude-code/claude-sonnet-4-6``.
    """
    if model.startswith("claude-code/"):
        model = model[len("claude-code/"):]
    if model and not model.startswith("claude-"):
        model = "claude-" + model.replace(".", "-")
    return model


def _extract_messages(request: dict) -> tuple[str | None, str]:
    """Split request messages into a system prompt and user prompt string."""
    messages = request.get("messages", [])
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            # Few-shot assistant turns — include as context in prompt.
            user_parts.append(f"[assistant]: {content}")
        else:
            user_parts.append(content)

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    prompt = "\n\n".join(user_parts)
    return system_prompt, prompt


def _build_completion(
    content: str | None,
    usage: dict[str, int],
    model: str,
    cost_usd: float | None = None,
) -> ChatCompletion:
    """Assemble an OpenAI-shaped ChatCompletion from Claude Code results."""
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(finish_reason="stop", index=0, message=message)
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    comp = ChatCompletion(
        id=f"cc-{uuid.uuid4().hex[:12]}",
        choices=[choice],
        created=int(time.time()),
        model=model,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    if cost_usd is not None:
        if not hasattr(comp, "_hidden_params"):
            comp._hidden_params = {}
        comp._hidden_params["response_cost"] = cost_usd
    return comp


# ---------------------------------------------------------------------------
# Stream wrapper
# ---------------------------------------------------------------------------

class _ClaudeCodeStreamWrapper:
    """Async iterator that yields ``StreamChunk``s from the Agent SDK.

    After exhaustion, ``.assembled`` holds the final ``ChatCompletion``.
    """

    def __init__(self, aiter, model: str):
        self._aiter = aiter
        self._model = model
        self._text_parts: list[str] = []
        self._usage: dict[str, int] = {}
        self._cost: float | None = None
        self._structured: Any = None
        self.assembled: ChatCompletion | None = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamChunk:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ThinkingBlock, StreamEvent

        while True:
            try:
                msg = await self._aiter.__anext__()
            except StopAsyncIteration:
                content = self._structured if self._structured else "\n".join(self._text_parts) or None
                self.assembled = _build_completion(
                    content=json.dumps(content) if isinstance(content, dict) else content,
                    usage=self._usage,
                    model=self._model,
                    cost_usd=self._cost,
                )
                raise

            if isinstance(msg, StreamEvent):
                # Partial text deltas from include_partial_messages.
                event = msg.event if hasattr(msg, "event") else msg.data if hasattr(msg, "data") else None
                if event and isinstance(event, dict):
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        return StreamChunk(content=delta.get("text", ""))
                    if delta.get("type") == "thinking_delta":
                        return StreamChunk(reasoning_content=delta.get("thinking", ""))

            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        self._text_parts.append(block.text)
                        return StreamChunk(content=block.text)
                    elif isinstance(block, ThinkingBlock):
                        return StreamChunk(reasoning_content=block.thinking)

            elif isinstance(msg, ResultMessage):
                self._usage = msg.usage or {}
                self._cost = msg.total_cost_usd
                self._structured = msg.structured_output
                return StreamChunk(finish_reason="stop")


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

@dataclass
class ClaudeCodeBackend:
    """DSPy backend powered by Claude Code via the Agent SDK.

    Parameters
    ----------
    model : str | None
        Override the model passed to Claude Code.  If ``None``, the model
        from the DSPy ``LM`` is used (with ``claude-code/`` prefix stripped).
    max_turns : int
        Maximum agentic turns per call.  ``1`` behaves like a single
        LLM completion; higher values let Claude Code use tools.
    allowed_tools : list[str] | None
        Restrict which tools Claude Code can use (e.g. ``["Read", "Bash"]``).
    disallowed_tools : list[str] | None
        Block specific tools.
    cwd : str | None
        Working directory for the Claude Code subprocess.
    permission_mode : str
        One of the Claude Code permission modes.
    system_prompt_mode : str
        How to handle DSPy's system prompt:
        ``"append"`` (default) adds it alongside Claude Code's built-in
        system prompt.  ``"replace"`` uses only DSPy's system prompt.
    persistent : bool
        If ``True``, use a ``ClaudeSDKClient`` that keeps a subprocess
        alive across calls (lower latency, conversation state preserved).
        If ``False`` (default), each call uses a fresh ``query()`` invocation.
    """

    model: str | None = None
    max_turns: int = 1
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    cwd: str | None = None
    permission_mode: str = "auto"
    system_prompt_mode: str = "append"
    persistent: bool = False

    # Internal state for persistent mode.
    _client: Any = field(default=None, init=False, repr=False)
    _client_system: str | None = field(default=None, init=False, repr=False)

    # Expose ContextWindowError at the instance/class level as required by the
    # DSPy backend protocol.
    ContextWindowError = ContextWindowError

    def _make_options(self, request: dict):
        from claude_agent_sdk import ClaudeAgentOptions

        system_prompt, _ = _extract_messages(request)
        model = self.model or _strip_prefix(request.get("model", ""))

        opts_kwargs: dict[str, Any] = {}
        if model:
            opts_kwargs["model"] = model
        if system_prompt:
            opts_kwargs["system_prompt"] = system_prompt
        opts_kwargs["max_turns"] = self.max_turns
        if self.allowed_tools is not None:
            opts_kwargs["allowed_tools"] = self.allowed_tools
        if self.disallowed_tools is not None:
            opts_kwargs["disallowed_tools"] = self.disallowed_tools
        if self.cwd:
            opts_kwargs["cwd"] = self.cwd
        opts_kwargs["permission_mode"] = self.permission_mode

        # Structured output via json_schema.
        schema = request.get("response_schema")
        if schema is not None:
            opts_kwargs["output_format"] = {"type": "json_schema", "schema": schema}

        return ClaudeAgentOptions(**opts_kwargs)

    # ------------------------------------------------------------------
    # Stateless path
    # ------------------------------------------------------------------

    async def _query_once(self, request: dict) -> ChatCompletion:
        from claude_agent_sdk import query, AssistantMessage, ResultMessage, TextBlock, ThinkingBlock

        opts = self._make_options(request)
        _, prompt = _extract_messages(request)
        model = opts.model if hasattr(opts, "model") and opts.model else _strip_prefix(request.get("model", ""))

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, int] = {}
        cost: float | None = None
        structured: Any = None

        try:
            async for msg in query(prompt=prompt, options=opts):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ThinkingBlock):
                            reasoning_parts.append(block.thinking)
                elif isinstance(msg, ResultMessage):
                    usage = msg.usage or {}
                    cost = msg.total_cost_usd
                    structured = msg.structured_output
                    if msg.is_error:
                        err_text = msg.result or "\n".join(text_parts) or "unknown error"
                        if "context" in err_text.lower() and "window" in err_text.lower():
                            raise ContextWindowError(err_text)
                        raise Exception(f"Claude Code error: {err_text}")
        except Exception as e:
            if "context" in str(e).lower() and "window" in str(e).lower():
                raise ContextWindowError(str(e)) from e
            raise

        content = structured if structured else "\n".join(text_parts) or None
        if isinstance(content, dict):
            content = json.dumps(content)

        comp = _build_completion(content=content, usage=usage, model=model, cost_usd=cost)

        # Attach reasoning_content if present.
        if reasoning_parts and comp.choices:
            comp.choices[0].message.reasoning_content = "\n".join(reasoning_parts)

        return comp

    # ------------------------------------------------------------------
    # Persistent path
    # ------------------------------------------------------------------

    async def _ensure_client(self, request: dict):
        from claude_agent_sdk import ClaudeSDKClient

        opts = self._make_options(request)
        system_prompt, _ = _extract_messages(request)

        if self._client is None or self._client_system != system_prompt:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
            self._client = ClaudeSDKClient(opts)
            await self._client.connect()
            self._client_system = system_prompt

    async def _query_persistent(self, request: dict) -> ChatCompletion:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ThinkingBlock

        await self._ensure_client(request)
        _, prompt = _extract_messages(request)
        model = self.model or _strip_prefix(request.get("model", ""))

        await self._client.query(prompt)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, int] = {}
        cost: float | None = None

        try:
            async for msg in self._client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ThinkingBlock):
                            reasoning_parts.append(block.thinking)
                elif isinstance(msg, ResultMessage):
                    usage = msg.usage or {}
                    cost = msg.total_cost_usd
        except Exception as e:
            if "context" in str(e).lower() and "window" in str(e).lower():
                raise ContextWindowError(str(e)) from e
            raise

        content = "\n".join(text_parts) or None
        comp = _build_completion(content=content, usage=usage, model=model, cost_usd=cost)
        if reasoning_parts and comp.choices:
            comp.choices[0].message.reasoning_content = "\n".join(reasoning_parts)
        return comp

    # ------------------------------------------------------------------
    # Backend protocol
    # ------------------------------------------------------------------

    def complete_request(
        self, request: dict[str, Any], model_type: str, num_retries: int
    ) -> ChatCompletion:
        if model_type != "chat":
            raise ValueError(
                f"ClaudeCodeBackend only supports model_type='chat', got {model_type!r}."
            )
        last_err = None
        for attempt in range(num_retries + 1):
            try:
                return asyncio.run(self._dispatch(request))
            except ContextWindowError:
                raise
            except Exception as e:
                last_err = e
                if attempt < num_retries:
                    time.sleep(2 ** attempt)
                    logger.warning("ClaudeCodeBackend retry %d/%d: %s", attempt + 1, num_retries, e)
        raise last_err

    async def acomplete_request(
        self, request: dict[str, Any], model_type: str, num_retries: int
    ) -> ChatCompletion:
        if model_type != "chat":
            raise ValueError(
                f"ClaudeCodeBackend only supports model_type='chat', got {model_type!r}."
            )
        last_err = None
        for attempt in range(num_retries + 1):
            try:
                return await self._dispatch(request)
            except ContextWindowError:
                raise
            except Exception as e:
                last_err = e
                if attempt < num_retries:
                    await anyio.sleep(2 ** attempt)
                    logger.warning("ClaudeCodeBackend retry %d/%d: %s", attempt + 1, num_retries, e)
        raise last_err

    async def astream_complete(
        self, request: dict[str, Any], num_retries: int
    ) -> _ClaudeCodeStreamWrapper:
        from claude_agent_sdk import query

        opts = self._make_options(request)
        _, prompt = _extract_messages(request)
        model = self.model or _strip_prefix(request.get("model", ""))

        aiter = query(prompt=prompt, options=opts)
        return _ClaudeCodeStreamWrapper(aiter, model)

    async def _dispatch(self, request: dict) -> ChatCompletion:
        if self.persistent:
            return await self._query_persistent(request)
        return await self._query_once(request)


# ---------------------------------------------------------------------------
# Module-level backend protocol (used when _resolve_backend returns this module)
# ---------------------------------------------------------------------------

_default_backend = ClaudeCodeBackend()


def complete_request(request: dict[str, Any], model_type: str, num_retries: int) -> ChatCompletion:
    return _default_backend.complete_request(request, model_type, num_retries)


async def acomplete_request(request: dict[str, Any], model_type: str, num_retries: int) -> ChatCompletion:
    return await _default_backend.acomplete_request(request, model_type, num_retries)


async def astream_complete(request: dict[str, Any], num_retries: int) -> _ClaudeCodeStreamWrapper:
    return await _default_backend.astream_complete(request, num_retries)
