"""Verda backend with transparent async endpoint lifecycle.

Lifts the anyio concurrency pattern from outernet's DedicatedEndpointLM,
providing idempotent, leader/follower resilient lifecycle management
that is completely transparent to callers.

Usage:
    backend = VerdaBackend(deployment_name="my-deploy", inference_key="...")
    lm = dspy.LM("verda/my-model", backend=backend)
    result = await lm.aforward(...)  # Endpoint started automatically!

    # Or with explicit lifecycle control:
    async with VerdaBackend(...) as backend:
        lm = dspy.LM("verda/my-model", backend=backend)
        await lm.aforward(...)
    # Cleanup after context
"""

from __future__ import annotations

import functools
import os
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any

import anyio
import openai
from tenacity import retry, retry_if_exception_type, stop, wait

from dspy.clients._openai import (
    _TRANSIENT,
    _assemble_chat_chunks,
    _make_client,
    _prepare,
    normalize_chunk,
)
from dspy.clients._request_utils import acall_with_retries, call_with_retries
from dspy.clients.openai import OpenAIProvider

log = getLogger(__name__)


class TheEndpointIsSlowError(TimeoutError):
    """Raised when a follower times out waiting for the leader."""


class VerdaBackend:
    """Stateful backend with transparent lifecycle management.

    Implements the DSPy backend protocol while providing automatic endpoint
    lifecycle management. The lifecycle is completely transparent - users
    don't need to call launch() or kill().

    Owns the anyio-based concurrency pattern:
    - anyio.Event signals endpoint readiness
    - anyio.Lock ensures exactly-once bootstrap under concurrent access
    - _ensure_endpoint() is the idempotent lazy entry point
    - eat_popcorn() implements distributed responsibility (leader/follower)
    - up(tg) provides nursery-based warm-start for task groups
    """

    # Backend protocol: error type
    ContextWindowError = openai.BadRequestError

    def __init__(
        self,
        deployment_name: str | None = None,
        deployment: Any | None = None,
        deployment_template: Any | None = None,
        verda_client: Any | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        inference_key: str | None = None,
        verda_base_url: str = "https://api.verda.com/v1",
        readiness_timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 15.0,
    ):
        # Lifecycle state
        self._endpoint_online: anyio.Event = anyio.Event()
        self._bootstrap_lock: anyio.Lock = anyio.Lock()

        # Deployment state
        self._deployment = deployment
        self._deployment_template = deployment_template
        self._deployment_name = (
            deployment_name or getattr(deployment, "name", None) or getattr(deployment_template, "name", None)
        )
        self._deployment_status: str | None = None
        self._endpoint_url: str | None = None
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

        # Auth
        self._inference_key = inference_key or os.getenv("VERDA_INFERENCE_KEY") or os.getenv("VERDA_API_KEY")
        if not self._inference_key:
            raise ValueError("inference_key is required (or set VERDA_INFERENCE_KEY)")

        # Client
        self._verda = verda_client
        if self._verda is None:
            self._verda = self._build_client(
                client_id or os.getenv("VERDA_CLIENT_ID"),
                client_secret or os.getenv("VERDA_CLIENT_SECRET"),
                verda_base_url,
            )

        # OpenAI client for inference (created when endpoint is ready)
        self._openai_client: openai.AsyncOpenAI | None = None

    def _build_client(self, client_id: str | None, client_secret: str | None, base_url: str) -> Any:
        if not client_id or not client_secret:
            return None  # Can't create client without credentials

        try:
            from verda import VerdaClient
        except ImportError as e:
            raise ImportError("Install `verda` package: pip install verda") from e

        return VerdaClient(
            client_id=client_id,
            client_secret=client_secret,
            base_url=base_url,
            inference_key=self._inference_key,
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """True when the endpoint is ready for inference."""
        return self._endpoint_online.is_set()

    @ready.setter
    def ready(self, value: bool) -> None:
        if value:
            self._endpoint_online.set()
        else:
            self._endpoint_online = anyio.Event()

    # ── Backend protocol: capability queries ─────────────────────────────
    # Note: this needs to be implemented and depends on the Deployment utilized
    def supports_function_calling(self, model: str) -> bool:
        return True

    def supports_reasoning(self, model: str) -> bool:
        return True

    def supports_response_schema(self, model: str) -> bool:
        return True

    def supported_params(self, model: str) -> set[str]:
        return {
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "logprobs",
            "top_logprobs",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream",
            "stream_options",
        }

    # ── Backend protocol: completion ─────────────────────────────────────

    def complete_request(self, request: dict[str, Any], model_type: str, num_retries: int):
        """Sync completion - runs async version via syncify."""
        import asyncer

        return asyncer.syncify(self.acomplete_request)(request, model_type, num_retries)

    async def acomplete_request(self, request: dict[str, Any], model_type: str, num_retries: int):
        """Async completion with transparent lifecycle management."""
        await self._ensure_endpoint()  # TRANSPARENT!
        return await self._openai_acomplete(request, model_type, num_retries)

    async def astream_complete(self, request: dict[str, Any], num_retries: int):
        """Streaming completion with transparent lifecycle."""
        await self._ensure_endpoint()
        request = _prepare(request)
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
        request["api_base"] = self._endpoint_url
        request["api_key"] = self._inference_key
        client = _make_client(request, async_=True)
        stream = await acall_with_retries(client.chat.completions.create, num_retries, _TRANSIENT, **request)
        return _VerdaStreamWrapper(stream)

    async def _openai_acomplete(self, request: dict[str, Any], model_type: str, num_retries: int):
        """Delegate to OpenAI-compatible API."""
        request = _prepare(request)
        request["api_base"] = self._endpoint_url
        request["api_key"] = self._inference_key
        client = _make_client(request, async_=True)

        if model_type == "chat":
            return await acall_with_retries(client.chat.completions.create, num_retries, _TRANSIENT, **request)
        elif model_type == "text":
            prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])
            request.pop("model", None)
            return await acall_with_retries(
                client.completions.create, num_retries, _TRANSIENT, prompt=prompt, **request
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    # ── Lifecycle: the anyio pattern from DedicatedEndpointLM ────────────

    def up(self, tg: anyio.abc.TaskGroup) -> None:
        """Start background bootstrap via task group. Returns immediately.

        Usage:
            async with anyio.create_task_group() as tg:
                backend.up(tg)
                # ... other parallel work ...
            # Endpoint is ready here
        """
        if self.ready:
            return
        tg.start_soon(self._ensure_endpoint)

    async def _ensure_endpoint(self) -> None:
        """Ensure endpoint is initialized and ready. Blocks until ready.

        This method is idempotent and safe to call concurrently. Uses a
        leader/follower pattern: the first caller acquires the lock and
        does the bootstrap; others wait (eat_popcorn) with resilience.
        """

        # NOTE: anyio.Event is duck-type compatible with threading.Event for is_set()
        @retry(
            retry=retry_if_exception_type(TheEndpointIsSlowError),
            stop=(stop.stop_after_attempt(10) | stop.stop_when_event_set(self._endpoint_online)),  # type: ignore[arg-type]
            wait=wait.wait_random(min=1, max=5),
        )
        async def eat_popcorn():
            """Wait for someone else to bring the endpoint online.

            Occasionally checks if we need to take over (leader failed/slow).
            Implements distributed responsibility in leader/follower pattern.
            """
            with anyio.move_on_after(120):
                await self._endpoint_online.wait()
            if not self.ready:
                await self._resume_endpoint()
                raise TheEndpointIsSlowError("Leader is slow, trying to help")

        if not self.ready:
            acquired = False
            try:
                self._bootstrap_lock.acquire_nowait()
                acquired = True
                if self.ready:  # re-check after acquiring lock
                    return

                if self._deployment is None:
                    endpoint = await self._find_endpoint()

                    if endpoint is None:
                        endpoint = await self._create_endpoint()

                    if endpoint is None:
                        raise RuntimeError("Failed to find or create a Verda deployment")
                    self._deployment = endpoint

                await self._resume_endpoint()
                await self._wait()

                self.ready = True
                log.info("Verda endpoint ready: %s", self._endpoint_url)

            except anyio.WouldBlock:
                await eat_popcorn()

            finally:
                if acquired:
                    self._bootstrap_lock.release()

    # ── Verda operations ─────────────────────────────────────────────────

    async def _call_verda(self, method: Any, *args: Any) -> Any:
        """Run a blocking Verda SDK call in the thread pool."""
        return await anyio.to_thread.run_sync(functools.partial(method, *args))

    @staticmethod
    def _normalize_status(status: Any) -> str:
        return getattr(status, "value", str(status)).strip().lower()

    async def _refresh_status(self) -> str:
        if self._deployment_name is None:
            raise RuntimeError("Deployment name not set")
        if self._verda is None:
            raise RuntimeError("Verda client not configured")
        status = await self._call_verda(self._verda.containers.get_deployment_status, self._deployment_name)
        self._deployment_status = self._normalize_status(status)
        return self._deployment_status

    async def _find_endpoint(self) -> Any | None:
        """Find an existing Verda deployment."""
        if self._verda is None:
            return self._deployment

        if self._deployment is not None:
            self._deployment_name = getattr(self._deployment, "name", self._deployment_name)
            await self._refresh_status()
            return self._deployment

        if self._deployment_name is not None:
            try:
                self._deployment = await self._call_verda(
                    self._verda.containers.get_deployment_by_name, self._deployment_name
                )
                await self._refresh_status()
                return self._deployment
            except Exception:
                log.info("Deployment not found by name: %s", self._deployment_name)

        return None

    async def _create_endpoint(self) -> Any:
        """Create a new Verda deployment from template."""
        if self._deployment_template is None:
            raise RuntimeError(f"No deployment {self._deployment_name!r} found and no deployment_template provided")
        if self._verda is None:
            raise RuntimeError("Verda client required to create deployment")
        self._deployment = await self._call_verda(self._verda.containers.create_deployment, self._deployment_template)
        self._deployment_name = self._deployment.name
        await self._refresh_status()
        log.info("Created Verda deployment: %s", self._deployment_name)
        return self._deployment

    async def _resume_endpoint(self) -> None:
        """Issue resume if deployment is paused. Returns immediately."""
        if self._deployment_name is None or self._verda is None:
            return
        status = self._deployment_status or await self._refresh_status()
        if status == "paused":
            await self._call_verda(self._verda.containers.resume_deployment, self._deployment_name)

    async def _wait(self) -> None:
        """Poll until deployment is healthy, then capture URL."""
        if self._verda is None:
            raise RuntimeError("Verda client required to wait for deployment")
        deadline = time.monotonic() + self._readiness_timeout_seconds

        while True:
            status = await self._refresh_status()

            if status == "healthy":
                self._deployment = await self._call_verda(
                    self._verda.containers.get_deployment_by_name, self._deployment_name
                )
                base = self._deployment.endpoint_base_url.rstrip("/")
                self._endpoint_url = base if base.endswith("/v1") else f"{base}/v1"
                return

            if status in {"unhealthy", "quota_reached"}:
                raise RuntimeError(f"Deployment {self._deployment_name} not usable: {status}")

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timeout waiting for {self._deployment_name} (status: {status})")

            log.info("Waiting for deployment %s (status: %s)", self._deployment_name, status)
            await anyio.sleep(self._poll_interval_seconds)

    # ── Context manager ──────────────────────────────────────────────────

    async def __aenter__(self) -> VerdaBackend:
        await self._ensure_endpoint()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.ready:
            log.warning("Endpoint still online - consider pausing/scaling down")
        await self.aclose()

    async def aclose(self) -> None:
        """Clean up resources. Does NOT pause the deployment."""
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None


class _VerdaStreamWrapper:
    """Wraps an OpenAI stream, normalizing chunks and collecting them."""

    def __init__(self, stream):
        self._stream = stream
        self._raw_chunks: list = []
        self.assembled = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            raw = await self._stream.__anext__()
        except StopAsyncIteration:
            self.assembled = _assemble_chat_chunks(self._raw_chunks)
            raise
        self._raw_chunks.append(raw)
        return normalize_chunk(raw)


class VerdaProvider(OpenAIProvider):
    """Provider interface for Verda models.

    Inherits finetuning from OpenAIProvider. For inference, use VerdaBackend
    which provides transparent lifecycle management.
    """

    @staticmethod
    def is_provider_model(model: str) -> bool:
        return model.startswith("verda/") or model.startswith("verda:")
