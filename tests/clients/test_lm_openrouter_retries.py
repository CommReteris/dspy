import time

import httpx
import pytest
from litellm import LlmProviders
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

import dspy


@pytest.mark.asyncio
async def test_openrouter_gateway_errors_use_exponential_backoff() -> None:
    attempt_times: list[float] = []

    def openrouter_gateway(request: httpx.Request) -> httpx.Response:
        attempt_times.append(time.monotonic())
        return httpx.Response(
            status_code=502,
            headers={"content-type": "text/html"},
            text="<title>openrouter.ai | 502: Bad gateway</title>",
            request=request,
        )

    openrouter_client = get_async_httpx_client(
        llm_provider=LlmProviders.OPENROUTER,
        params={"ssl_verify": None},
    )
    provider_http_client = openrouter_client.client
    openrouter_client.client = httpx.AsyncClient(transport=httpx.MockTransport(openrouter_gateway))

    try:
        lm = dspy.LM(
            model="openrouter/qwen/qwen3.5-9b",
            api_key="test-openrouter-key",
            cache=False,
            num_retries=3,
        )

        with pytest.raises(dspy.LMServerError):
            await lm.acall(prompt="Resolve the annotation.")
    finally:
        await openrouter_client.close()
        openrouter_client.client = provider_http_client

    assert len(attempt_times) == 4
    assert attempt_times[1] - attempt_times[0] >= 0.9
    assert attempt_times[2] - attempt_times[1] >= 1.9
    assert attempt_times[3] - attempt_times[2] >= 3.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_attempts", "error_type"),
    [
        (429, 2, dspy.LMRateLimitError),
        (408, 2, dspy.LMTimeoutError),
        (400, 1, dspy.LMInvalidRequestError),
    ],
)
async def test_openrouter_only_retries_transient_statuses(
    status_code: int,
    expected_attempts: int,
    error_type: type[dspy.LMProviderError],
) -> None:
    attempts = 0

    def openrouter(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            status_code=status_code,
            headers={"content-type": "application/json"},
            json={"error": {"message": "provider error", "code": status_code}},
            request=request,
        )

    openrouter_client = get_async_httpx_client(
        llm_provider=LlmProviders.OPENROUTER,
        params={"ssl_verify": None},
    )
    provider_http_client = openrouter_client.client
    openrouter_client.client = httpx.AsyncClient(transport=httpx.MockTransport(openrouter))

    try:
        lm = dspy.LM(
            model="openrouter/qwen/qwen3.5-9b",
            api_key="test-openrouter-key",
            cache=False,
            num_retries=1,
        )

        with pytest.raises(error_type):
            await lm.acall(prompt="Resolve the annotation.")
    finally:
        await openrouter_client.close()
        openrouter_client.client = provider_http_client

    assert attempts == expected_attempts
