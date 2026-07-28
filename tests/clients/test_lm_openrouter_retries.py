import time

import httpx
import pytest
from litellm import APIError, LlmProviders
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

        with pytest.raises(APIError):
            await lm.acall(prompt="Resolve the annotation.")
    finally:
        await openrouter_client.close()
        openrouter_client.client = provider_http_client

    assert len(attempt_times) == 4
    assert attempt_times[2] - attempt_times[1] >= 1
    assert attempt_times[3] - attempt_times[2] >= 2
