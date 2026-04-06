"""Tests for the ClaudeCodeBackend DSPy client.

These tests verify the behavior of the Claude Code backend, which uses the
Claude Agent SDK to interact with Claude Code's agentic capabilities.
"""

import json
import pytest
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.chat import ChatCompletion


# ---------------------------------------------------------------------------
# Test capability functions
# ---------------------------------------------------------------------------


def test_supports_function_calling_returns_false():
    """Claude Code uses tools internally but doesn't expose OpenAI-style function_calling."""
    from dspy.clients._claude_code import supports_function_calling

    assert supports_function_calling("claude-sonnet-4-6") is False
    assert supports_function_calling("claude-opus-4") is False


def test_supports_reasoning_returns_true():
    """Claude Code supports reasoning capabilities."""
    from dspy.clients._claude_code import supports_reasoning

    assert supports_reasoning("claude-sonnet-4-6") is True
    assert supports_reasoning("claude-opus-4") is True


def test_supports_response_schema_returns_true():
    """Claude Code supports structured output via --json-schema."""
    from dspy.clients._claude_code import supports_response_schema

    assert supports_response_schema("claude-sonnet-4-6") is True
    assert supports_response_schema("any-model") is True


def test_supported_params_returns_expected_set():
    """Verify the set of supported parameters."""
    from dspy.clients._claude_code import supported_params

    params = supported_params("claude-sonnet-4-6")
    assert params == {"temperature", "max_tokens", "stop", "response_schema"}


# ---------------------------------------------------------------------------
# Test helper functions
# ---------------------------------------------------------------------------


def test_strip_prefix_removes_claude_code_prefix():
    """_strip_prefix should remove 'claude-code/' prefix from model names."""
    from dspy.clients._claude_code import _strip_prefix

    assert _strip_prefix("claude-code/claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert _strip_prefix("claude-code/claude-opus-4") == "claude-opus-4"


def test_strip_prefix_preserves_model_without_prefix():
    """_strip_prefix should return model unchanged if no prefix present."""
    from dspy.clients._claude_code import _strip_prefix

    assert _strip_prefix("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert _strip_prefix("gpt-4") == "gpt-4"


def test_extract_messages_separates_system_and_user():
    """_extract_messages should split messages into system prompt and user prompt."""
    from dspy.clients._claude_code import _extract_messages

    request = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
    }

    system_prompt, prompt = _extract_messages(request)

    assert system_prompt == "You are a helpful assistant."
    assert prompt == "Hello!"


def test_extract_messages_handles_multiple_system_messages():
    """_extract_messages should join multiple system messages."""
    from dspy.clients._claude_code import _extract_messages

    request = {
        "messages": [
            {"role": "system", "content": "First system instruction."},
            {"role": "system", "content": "Second system instruction."},
            {"role": "user", "content": "Query"},
        ]
    }

    system_prompt, prompt = _extract_messages(request)

    assert system_prompt == "First system instruction.\n\nSecond system instruction."
    assert prompt == "Query"


def test_extract_messages_handles_assistant_turns():
    """_extract_messages should include assistant turns as context."""
    from dspy.clients._claude_code import _extract_messages

    request = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
    }

    system_prompt, prompt = _extract_messages(request)

    assert system_prompt is None
    assert "What is 2+2?" in prompt
    assert "[assistant]: 4" in prompt
    assert "And 3+3?" in prompt


def test_extract_messages_returns_none_for_no_system():
    """_extract_messages should return None for system prompt if none present."""
    from dspy.clients._claude_code import _extract_messages

    request = {
        "messages": [
            {"role": "user", "content": "Hello!"},
        ]
    }

    system_prompt, prompt = _extract_messages(request)

    assert system_prompt is None
    assert prompt == "Hello!"


def test_build_completion_returns_chat_completion():
    """_build_completion should return a properly structured ChatCompletion."""
    from dspy.clients._claude_code import _build_completion

    completion = _build_completion(
        content="Hello, world!",
        usage={"input_tokens": 10, "output_tokens": 5},
        model="claude-sonnet-4-6",
    )

    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Hello, world!"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.model == "claude-sonnet-4-6"
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 5
    assert completion.usage.total_tokens == 15


def test_build_completion_handles_none_content():
    """_build_completion should handle None content."""
    from dspy.clients._claude_code import _build_completion

    completion = _build_completion(
        content=None,
        usage={},
        model="claude-sonnet-4-6",
    )

    assert completion.choices[0].message.content is None


def test_build_completion_includes_cost():
    """_build_completion should include cost in _hidden_params if provided."""
    from dspy.clients._claude_code import _build_completion

    completion = _build_completion(
        content="Response",
        usage={"input_tokens": 100, "output_tokens": 50},
        model="claude-sonnet-4-6",
        cost_usd=0.05,
    )

    assert hasattr(completion, "_hidden_params")
    assert completion._hidden_params["response_cost"] == 0.05


# ---------------------------------------------------------------------------
# Test ContextWindowError
# ---------------------------------------------------------------------------


def test_context_window_error_is_exception():
    """ContextWindowError should be a proper exception class."""
    from dspy.clients._claude_code import ContextWindowError

    error = ContextWindowError("Prompt exceeds context window")
    assert isinstance(error, Exception)
    assert str(error) == "Prompt exceeds context window"


def test_backend_exposes_context_window_error():
    """ClaudeCodeBackend should expose ContextWindowError as class attribute."""
    from dspy.clients._claude_code import ClaudeCodeBackend, ContextWindowError

    backend = ClaudeCodeBackend()
    assert backend.ContextWindowError is ContextWindowError


# ---------------------------------------------------------------------------
# Test ClaudeCodeBackend initialization
# ---------------------------------------------------------------------------


def test_backend_default_initialization():
    """ClaudeCodeBackend should initialize with sensible defaults."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()

    assert backend.model is None
    assert backend.max_turns == 1
    assert backend.allowed_tools is None
    assert backend.disallowed_tools is None
    assert backend.cwd is None
    assert backend.permission_mode == "auto"
    assert backend.system_prompt_mode == "append"
    assert backend.persistent is False


def test_backend_custom_initialization():
    """ClaudeCodeBackend should accept custom configuration."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend(
        model="claude-opus-4",
        max_turns=5,
        allowed_tools=["Read", "Write"],
        cwd="/tmp",
        permission_mode="bypassPermissions",
        persistent=True,
    )

    assert backend.model == "claude-opus-4"
    assert backend.max_turns == 5
    assert backend.allowed_tools == ["Read", "Write"]
    assert backend.cwd == "/tmp"
    assert backend.permission_mode == "bypassPermissions"
    assert backend.persistent is True


# ---------------------------------------------------------------------------
# Test complete_request
# ---------------------------------------------------------------------------


def test_complete_request_rejects_non_chat_model_type():
    """complete_request should raise ValueError for non-chat model types."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()

    with pytest.raises(ValueError, match="only supports model_type='chat'"):
        backend.complete_request(
            request={"messages": [{"role": "user", "content": "Hi"}]},
            model_type="text",
            num_retries=0,
        )


@patch("dspy.clients._claude_code.asyncio.run")
def test_complete_request_calls_dispatch(mock_asyncio_run):
    """complete_request should call _dispatch via asyncio.run."""
    from dspy.clients._claude_code import ClaudeCodeBackend, _build_completion

    mock_completion = _build_completion(
        content="Hello",
        usage={"input_tokens": 5, "output_tokens": 3},
        model="claude-sonnet-4-6",
    )
    mock_asyncio_run.return_value = mock_completion

    backend = ClaudeCodeBackend()
    result = backend.complete_request(
        request={"model": "claude-code/claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
        model_type="chat",
        num_retries=0,
    )

    assert result == mock_completion
    mock_asyncio_run.assert_called_once()


@patch("dspy.clients._claude_code.asyncio.run")
def test_complete_request_retries_on_error(mock_asyncio_run):
    """complete_request should retry on transient errors."""
    from dspy.clients._claude_code import ClaudeCodeBackend, _build_completion

    mock_completion = _build_completion(
        content="Success",
        usage={},
        model="claude-sonnet-4-6",
    )
    # Fail twice, then succeed
    mock_asyncio_run.side_effect = [
        RuntimeError("Transient error 1"),
        RuntimeError("Transient error 2"),
        mock_completion,
    ]

    backend = ClaudeCodeBackend()

    with patch("dspy.clients._claude_code.time.sleep"):
        result = backend.complete_request(
            request={"model": "claude-code/claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
            model_type="chat",
            num_retries=2,
        )

    assert result == mock_completion
    assert mock_asyncio_run.call_count == 3


@patch("dspy.clients._claude_code.asyncio.run")
def test_complete_request_raises_context_window_error_immediately(mock_asyncio_run):
    """complete_request should not retry ContextWindowError."""
    from dspy.clients._claude_code import ClaudeCodeBackend, ContextWindowError

    mock_asyncio_run.side_effect = ContextWindowError("Context exceeded")

    backend = ClaudeCodeBackend()

    with pytest.raises(ContextWindowError):
        backend.complete_request(
            request={"messages": [{"role": "user", "content": "Hi"}]},
            model_type="chat",
            num_retries=3,
        )

    # Should only be called once - no retries
    assert mock_asyncio_run.call_count == 1


# ---------------------------------------------------------------------------
# Test acomplete_request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acomplete_request_rejects_non_chat_model_type():
    """acomplete_request should raise ValueError for non-chat model types."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()

    with pytest.raises(ValueError, match="only supports model_type='chat'"):
        await backend.acomplete_request(
            request={"messages": [{"role": "user", "content": "Hi"}]},
            model_type="responses",
            num_retries=0,
        )


@pytest.mark.asyncio
async def test_acomplete_request_calls_dispatch():
    """acomplete_request should call _dispatch and return completion."""
    from dspy.clients._claude_code import ClaudeCodeBackend, _build_completion

    mock_completion = _build_completion(
        content="Async response",
        usage={"input_tokens": 10, "output_tokens": 8},
        model="claude-sonnet-4-6",
    )

    backend = ClaudeCodeBackend()
    backend._dispatch = AsyncMock(return_value=mock_completion)

    result = await backend.acomplete_request(
        request={"model": "claude-code/claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
        model_type="chat",
        num_retries=0,
    )

    assert result == mock_completion
    backend._dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_acomplete_request_retries_on_error():
    """acomplete_request should retry on transient errors."""
    from dspy.clients._claude_code import ClaudeCodeBackend, _build_completion

    mock_completion = _build_completion(
        content="Success after retry",
        usage={},
        model="claude-sonnet-4-6",
    )

    backend = ClaudeCodeBackend()
    backend._dispatch = AsyncMock(
        side_effect=[
            RuntimeError("Error 1"),
            mock_completion,
        ]
    )

    with patch("dspy.clients._claude_code.asyncio.sleep", new_callable=AsyncMock):
        result = await backend.acomplete_request(
            request={"messages": [{"role": "user", "content": "Hi"}]},
            model_type="chat",
            num_retries=1,
        )

    assert result == mock_completion
    assert backend._dispatch.await_count == 2


@pytest.mark.asyncio
async def test_acomplete_request_raises_context_window_error_immediately():
    """acomplete_request should not retry ContextWindowError."""
    from dspy.clients._claude_code import ClaudeCodeBackend, ContextWindowError

    backend = ClaudeCodeBackend()
    backend._dispatch = AsyncMock(side_effect=ContextWindowError("Context exceeded"))

    with pytest.raises(ContextWindowError):
        await backend.acomplete_request(
            request={"messages": [{"role": "user", "content": "Hi"}]},
            model_type="chat",
            num_retries=5,
        )

    # Should only be called once - no retries
    assert backend._dispatch.await_count == 1


# ---------------------------------------------------------------------------
# Test _make_options
# ---------------------------------------------------------------------------


def test_make_options_creates_agent_options():
    """_make_options should create ClaudeAgentOptions with correct parameters."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    # Mock the claude_agent_sdk module
    mock_options_class = MagicMock()
    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.ClaudeAgentOptions = mock_options_class

    backend = ClaudeCodeBackend(
        model="claude-opus-4",
        max_turns=3,
        allowed_tools=["Read"],
        cwd="/workspace",
    )

    request = {
        "model": "claude-code/claude-sonnet-4-6",
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ],
    }

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        backend._make_options(request)

    mock_options_class.assert_called_once()
    call_kwargs = mock_options_class.call_args.kwargs

    assert call_kwargs["model"] == "claude-opus-4"
    assert call_kwargs["max_turns"] == 3
    assert call_kwargs["allowed_tools"] == ["Read"]
    assert call_kwargs["cwd"] == "/workspace"
    assert call_kwargs["system_prompt"] == "Be helpful"


def test_make_options_includes_response_schema():
    """_make_options should include output_format when response_schema is provided."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    mock_options_class = MagicMock()
    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.ClaudeAgentOptions = mock_options_class

    backend = ClaudeCodeBackend()

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    request = {
        "model": "claude-code/claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Hello"}],
        "response_schema": schema,
    }

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        backend._make_options(request)

    call_kwargs = mock_options_class.call_args.kwargs
    assert call_kwargs["output_format"] == {"type": "json_schema", "schema": schema}


def test_make_options_uses_request_model_when_backend_model_is_none():
    """_make_options should use model from request if backend.model is None."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    mock_options_class = MagicMock()
    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.ClaudeAgentOptions = mock_options_class

    backend = ClaudeCodeBackend(model=None)

    request = {
        "model": "claude-code/claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        backend._make_options(request)

    call_kwargs = mock_options_class.call_args.kwargs
    # Should strip the prefix
    assert call_kwargs["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Test _query_once (stateless path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_once_processes_assistant_messages():
    """_query_once should extract text from AssistantMessage blocks."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    # Create mock SDK types
    mock_text_block = MagicMock()
    mock_text_block.text = "Hello from Claude!"

    mock_assistant_msg = MagicMock()
    mock_assistant_msg.content = [mock_text_block]

    mock_result_msg = MagicMock()
    mock_result_msg.usage = {"input_tokens": 10, "output_tokens": 5}
    mock_result_msg.total_cost_usd = 0.001
    mock_result_msg.structured_output = None

    async def mock_query(prompt, options):
        yield mock_assistant_msg
        yield mock_result_msg

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.query = mock_query
    mock_claude_agent_sdk.AssistantMessage = type(mock_assistant_msg)
    mock_claude_agent_sdk.ResultMessage = type(mock_result_msg)
    mock_claude_agent_sdk.TextBlock = type(mock_text_block)
    mock_claude_agent_sdk.ThinkingBlock = MagicMock
    mock_claude_agent_sdk.ClaudeAgentOptions = MagicMock

    backend = ClaudeCodeBackend()

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        result = await backend._query_once({
            "model": "claude-code/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
        })

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello from Claude!"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_query_once_handles_structured_output():
    """_query_once should return structured output as JSON when present."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    # Create a unique class for ResultMessage to avoid isinstance issues
    class MockResultMessage:
        usage = {"input_tokens": 5, "output_tokens": 10}
        total_cost_usd = None
        structured_output = {"answer": "42", "confidence": 0.95}

    class MockAssistantMessage:
        pass

    class MockTextBlock:
        pass

    class MockThinkingBlock:
        pass

    async def mock_query(prompt, options):
        yield MockResultMessage()

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.query = mock_query
    mock_claude_agent_sdk.AssistantMessage = MockAssistantMessage
    mock_claude_agent_sdk.ResultMessage = MockResultMessage
    mock_claude_agent_sdk.TextBlock = MockTextBlock
    mock_claude_agent_sdk.ThinkingBlock = MockThinkingBlock
    mock_claude_agent_sdk.ClaudeAgentOptions = MagicMock

    backend = ClaudeCodeBackend()

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        result = await backend._query_once({
            "model": "claude-code/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "What is the answer?"}],
            "response_schema": {"type": "object"},
        })

    # Structured output should be JSON serialized
    content = result.choices[0].message.content
    parsed = json.loads(content)
    assert parsed == {"answer": "42", "confidence": 0.95}


@pytest.mark.asyncio
async def test_query_once_captures_reasoning():
    """_query_once should capture reasoning content from ThinkingBlock."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    mock_thinking_block = MagicMock()
    mock_thinking_block.thinking = "Let me think about this..."

    mock_text_block = MagicMock()
    mock_text_block.text = "The answer is 42."

    mock_assistant_msg = MagicMock()
    mock_assistant_msg.content = [mock_thinking_block, mock_text_block]

    mock_result_msg = MagicMock()
    mock_result_msg.usage = {}
    mock_result_msg.total_cost_usd = None
    mock_result_msg.structured_output = None

    async def mock_query(prompt, options):
        yield mock_assistant_msg
        yield mock_result_msg

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.query = mock_query
    mock_claude_agent_sdk.AssistantMessage = type(mock_assistant_msg)
    mock_claude_agent_sdk.ResultMessage = type(mock_result_msg)
    mock_claude_agent_sdk.TextBlock = type(mock_text_block)
    mock_claude_agent_sdk.ThinkingBlock = type(mock_thinking_block)
    mock_claude_agent_sdk.ClaudeAgentOptions = MagicMock

    backend = ClaudeCodeBackend()

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        result = await backend._query_once({
            "model": "claude-code/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Think about this"}],
        })

    assert result.choices[0].message.content == "The answer is 42."
    assert result.choices[0].message.reasoning_content == "Let me think about this..."


@pytest.mark.asyncio
async def test_query_once_raises_context_window_error():
    """_query_once should raise ContextWindowError for context window errors."""
    from dspy.clients._claude_code import ClaudeCodeBackend, ContextWindowError

    async def mock_query(prompt, options):
        raise RuntimeError("Input exceeds context window limit")
        yield  # Make it an async generator

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.query = mock_query
    mock_claude_agent_sdk.AssistantMessage = MagicMock
    mock_claude_agent_sdk.ResultMessage = MagicMock
    mock_claude_agent_sdk.TextBlock = MagicMock
    mock_claude_agent_sdk.ThinkingBlock = MagicMock
    mock_claude_agent_sdk.ClaudeAgentOptions = MagicMock

    backend = ClaudeCodeBackend()

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        with pytest.raises(ContextWindowError):
            await backend._query_once({
                "messages": [{"role": "user", "content": "Very long prompt..."}],
            })


# ---------------------------------------------------------------------------
# Test _dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_uses_query_once_by_default():
    """_dispatch should use _query_once when persistent=False."""
    from dspy.clients._claude_code import ClaudeCodeBackend, _build_completion

    mock_completion = _build_completion(content="Response", usage={}, model="test")

    backend = ClaudeCodeBackend(persistent=False)
    backend._query_once = AsyncMock(return_value=mock_completion)
    backend._query_persistent = AsyncMock()

    result = await backend._dispatch({"messages": []})

    assert result == mock_completion
    backend._query_once.assert_awaited_once()
    backend._query_persistent.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_uses_query_persistent_when_enabled():
    """_dispatch should use _query_persistent when persistent=True."""
    from dspy.clients._claude_code import ClaudeCodeBackend, _build_completion

    mock_completion = _build_completion(content="Persistent response", usage={}, model="test")

    backend = ClaudeCodeBackend(persistent=True)
    backend._query_once = AsyncMock()
    backend._query_persistent = AsyncMock(return_value=mock_completion)

    result = await backend._dispatch({"messages": []})

    assert result == mock_completion
    backend._query_persistent.assert_awaited_once()
    backend._query_once.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test streaming (_ClaudeCodeStreamWrapper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_wrapper_yields_text_chunks():
    """_ClaudeCodeStreamWrapper should yield StreamChunks with text content."""
    from dspy.clients._claude_code import _ClaudeCodeStreamWrapper

    # Create unique classes for proper isinstance checks
    class MockTextBlock:
        def __init__(self, text):
            self.text = text

    class MockAssistantMessage:
        def __init__(self, content):
            self.content = content

    class MockResultMessage:
        def __init__(self):
            self.usage = {"input_tokens": 5, "output_tokens": 2}
            self.total_cost_usd = 0.001
            self.structured_output = None

    class MockThinkingBlock:
        pass

    class MockStreamEvent:
        pass

    async def mock_aiter():
        yield MockAssistantMessage([MockTextBlock("Hello ")])
        yield MockAssistantMessage([MockTextBlock("World!")])
        yield MockResultMessage()

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.AssistantMessage = MockAssistantMessage
    mock_claude_agent_sdk.ResultMessage = MockResultMessage
    mock_claude_agent_sdk.TextBlock = MockTextBlock
    mock_claude_agent_sdk.ThinkingBlock = MockThinkingBlock
    mock_claude_agent_sdk.StreamEvent = MockStreamEvent

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        wrapper = _ClaudeCodeStreamWrapper(mock_aiter(), "claude-sonnet-4-6")

        chunks = []
        async for chunk in wrapper:
            chunks.append(chunk)

    # Should have text chunks plus finish
    assert len(chunks) == 3
    assert chunks[0].content == "Hello "
    assert chunks[1].content == "World!"
    assert chunks[2].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_wrapper_assembles_final_completion():
    """_ClaudeCodeStreamWrapper should assemble final ChatCompletion after exhaustion."""
    from dspy.clients._claude_code import _ClaudeCodeStreamWrapper

    # Create unique classes for proper isinstance checks
    class MockTextBlock:
        def __init__(self, text):
            self.text = text

    class MockAssistantMessage:
        def __init__(self, content):
            self.content = content

    class MockResultMessage:
        def __init__(self):
            self.usage = {"input_tokens": 10, "output_tokens": 5}
            self.total_cost_usd = 0.002
            self.structured_output = None

    class MockThinkingBlock:
        pass

    class MockStreamEvent:
        pass

    async def mock_aiter():
        yield MockAssistantMessage([MockTextBlock("Complete response")])
        yield MockResultMessage()

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.AssistantMessage = MockAssistantMessage
    mock_claude_agent_sdk.ResultMessage = MockResultMessage
    mock_claude_agent_sdk.TextBlock = MockTextBlock
    mock_claude_agent_sdk.ThinkingBlock = MockThinkingBlock
    mock_claude_agent_sdk.StreamEvent = MockStreamEvent

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        wrapper = _ClaudeCodeStreamWrapper(mock_aiter(), "claude-sonnet-4-6")

        # Exhaust the iterator
        async for _ in wrapper:
            pass

    # Check assembled completion
    assert wrapper.assembled is not None
    assert isinstance(wrapper.assembled, ChatCompletion)
    assert wrapper.assembled.choices[0].message.content == "Complete response"
    assert wrapper.assembled.usage.prompt_tokens == 10
    assert wrapper.assembled.usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_stream_wrapper_handles_thinking_blocks():
    """_ClaudeCodeStreamWrapper should yield reasoning_content for ThinkingBlocks."""
    from dspy.clients._claude_code import _ClaudeCodeStreamWrapper

    # Create unique classes for proper isinstance checks
    class MockThinkingBlock:
        def __init__(self, thinking):
            self.thinking = thinking

    class MockTextBlock:
        pass

    class MockAssistantMessage:
        def __init__(self, content):
            self.content = content

    class MockResultMessage:
        def __init__(self):
            self.usage = {}
            self.total_cost_usd = None
            self.structured_output = None

    class MockStreamEvent:
        pass

    async def mock_aiter():
        yield MockAssistantMessage([MockThinkingBlock("Reasoning...")])
        yield MockResultMessage()

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.AssistantMessage = MockAssistantMessage
    mock_claude_agent_sdk.ResultMessage = MockResultMessage
    mock_claude_agent_sdk.TextBlock = MockTextBlock
    mock_claude_agent_sdk.ThinkingBlock = MockThinkingBlock
    mock_claude_agent_sdk.StreamEvent = MockStreamEvent

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        wrapper = _ClaudeCodeStreamWrapper(mock_aiter(), "claude-sonnet-4-6")

        chunks = []
        async for chunk in wrapper:
            chunks.append(chunk)

    assert chunks[0].reasoning_content == "Reasoning..."


# ---------------------------------------------------------------------------
# Test astream_complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_complete_returns_stream_wrapper():
    """astream_complete should return a _ClaudeCodeStreamWrapper."""
    from dspy.clients._claude_code import ClaudeCodeBackend, _ClaudeCodeStreamWrapper

    async def mock_query(prompt, options):
        yield MagicMock()

    mock_claude_agent_sdk = MagicMock()
    mock_claude_agent_sdk.query = mock_query
    mock_claude_agent_sdk.ClaudeAgentOptions = MagicMock

    backend = ClaudeCodeBackend()

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_claude_agent_sdk}):
        result = await backend.astream_complete(
            request={
                "model": "claude-code/claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Stream this"}],
            },
            num_retries=0,
        )

    assert isinstance(result, _ClaudeCodeStreamWrapper)


# ---------------------------------------------------------------------------
# Test integration with DSPy LM
# ---------------------------------------------------------------------------


def test_backend_can_be_passed_to_dspy_lm():
    """ClaudeCodeBackend should be usable as a DSPy LM backend."""
    from dspy.clients._claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend(max_turns=3)

    # This tests that the backend has the expected interface
    assert hasattr(backend, "complete_request")
    assert hasattr(backend, "acomplete_request")
    assert hasattr(backend, "astream_complete")
    assert hasattr(backend, "ContextWindowError")
