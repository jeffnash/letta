import httpx
import openai
import pytest

from letta.adapters.letta_llm_stream_adapter import LettaLLMStreamAdapter
from letta.errors import LLMConnectionError, LLMRateLimitError, LLMServerError
from letta.llm_api.openai_client import OpenAIClient
from letta.schemas.llm_config import LLMConfig


@pytest.mark.asyncio
async def test_letta_llm_stream_adapter_converts_openai_api_error_during_streaming(monkeypatch):
    """Regression: provider APIError raised *during* streaming iteration should be converted via handle_llm_error."""

    # Create a generic APIError that might occur during streaming
    # (not a specific subtype like APIStatusError)
    # Use a message that doesn't trigger transport-like detection (avoid: connection, stream, network, timeout, closed, reset, peer, broken)
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "Backend processing failed", "type": "server_error"}}
    error = openai.APIError("Backend processing failed", request=request, body=body)

    class FakeAsyncStream:
        """Mimics openai.AsyncStream enough for SimpleOpenAIStreamingInterface (async cm + async iterator)."""

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

    async def fake_stream_async(self, request_data: dict, llm_config: LLMConfig):
        return FakeAsyncStream()

    monkeypatch.setattr(OpenAIClient, "stream_async", fake_stream_async, raising=True)

    llm_client = OpenAIClient()
    llm_config = LLMConfig(model="gpt-4", model_endpoint_type="openai", context_window=128000)
    adapter = LettaLLMStreamAdapter(llm_client=llm_client, llm_config=llm_config)

    gen = adapter.invoke_llm(request_data={}, messages=[], tools=[], use_assistant_message=True)

    # Should raise LLMServerError (not generic LLMError)
    with pytest.raises(LLMServerError) as exc_info:
        async for _ in gen:
            pass

    # Verify error has proper structure
    assert "Backend processing failed" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("provider_exception_type") == "APIError"


@pytest.mark.asyncio
async def test_letta_llm_stream_adapter_converts_openai_api_error_with_status_code(monkeypatch):
    """Regression: APIError with status code metadata should be mapped to appropriate error type."""

    # Create an APIError with a 500 status code in the body
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "The server had an error while processing your request", "type": "server_error", "param": None, "code": None}}
    error = openai.APIError("Internal server error", request=request, body=body)
    # Manually set status_code-like attribute if present
    error.body = body

    class FakeAsyncStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

    async def fake_stream_async(self, request_data: dict, llm_config: LLMConfig):
        return FakeAsyncStream()

    monkeypatch.setattr(OpenAIClient, "stream_async", fake_stream_async, raising=True)

    llm_client = OpenAIClient()
    llm_config = LLMConfig(model="gpt-4", model_endpoint_type="openai", context_window=128000)
    adapter = LettaLLMStreamAdapter(llm_client=llm_client, llm_config=llm_config)

    gen = adapter.invoke_llm(request_data={}, messages=[], tools=[], use_assistant_message=True)

    # Should raise LLMServerError for 500 errors
    with pytest.raises(LLMServerError):
        async for _ in gen:
            pass


@pytest.mark.asyncio
async def test_letta_llm_stream_adapter_converts_openai_api_error_rate_limit(monkeypatch):
    """Regression: APIError with rate limit indication should be mapped to LLMRateLimitError."""

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}
    error = openai.APIError("Rate limit exceeded", request=request, body=body)
    error.body = body

    class FakeAsyncStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

    async def fake_stream_async(self, request_data: dict, llm_config: LLMConfig):
        return FakeAsyncStream()

    monkeypatch.setattr(OpenAIClient, "stream_async", fake_stream_async, raising=True)

    llm_client = OpenAIClient()
    llm_config = LLMConfig(model="gpt-4", model_endpoint_type="openai", context_window=128000)
    adapter = LettaLLMStreamAdapter(llm_client=llm_client, llm_config=llm_config)

    gen = adapter.invoke_llm(request_data={}, messages=[], tools=[], use_assistant_message=True)

    # Should raise LLMRateLimitError for rate limit errors
    with pytest.raises(LLMRateLimitError):
        async for _ in gen:
            pass


@pytest.mark.asyncio
async def test_letta_llm_stream_adapter_converts_openai_api_error_transport_like(monkeypatch):
    """Regression: APIError with transport-like message should be mapped to LLMConnectionError."""

    # Simulate a stream connection error
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "Connection closed unexpectedly while reading stream"}}
    error = openai.APIError(
        "Connection closed unexpectedly while reading stream",
        request=request,
        body=body,
    )

    class FakeAsyncStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

    async def fake_stream_async(self, request_data: dict, llm_config: LLMConfig):
        return FakeAsyncStream()

    monkeypatch.setattr(OpenAIClient, "stream_async", fake_stream_async, raising=True)

    llm_client = OpenAIClient()
    llm_config = LLMConfig(model="gpt-4", model_endpoint_type="openai", context_window=128000)
    adapter = LettaLLMStreamAdapter(llm_client=llm_client, llm_config=llm_config)

    gen = adapter.invoke_llm(request_data={}, messages=[], tools=[], use_assistant_message=True)

    # Should raise LLMConnectionError for transport-like errors
    with pytest.raises(LLMConnectionError) as exc_info:
        async for _ in gen:
            pass

    assert "Connection error during OpenAI streaming" in str(exc_info.value)


def test_openai_client_handle_llm_error_generic_api_error():
    """Test that handle_llm_error correctly converts generic APIError to LLMServerError."""
    client = OpenAIClient()

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    # Use a message that doesn't trigger transport-like detection (avoid: connection, stream, network, timeout, closed, reset, peer, broken)
    body = {"error": {"message": "Backend processing failed", "type": "server_error"}}
    error = openai.APIError("Backend processing failed", request=request, body=body)

    result = client.handle_llm_error(error)

    assert isinstance(result, LLMServerError)
    assert "OpenAI API error" in result.message
    assert result.details is not None
    assert result.details.get("provider_exception_type") == "APIError"


def test_openai_client_handle_llm_error_api_error_with_request_id():
    """Test that handle_llm_error extracts request_id from APIError."""
    client = OpenAIClient()

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "Internal server error", "type": "server_error"}}
    error = openai.APIError("Internal server error", request=request, body=body)
    # Simulate request_id attribute that may be present
    error.request_id = "req_1234567890"

    result = client.handle_llm_error(error)

    assert isinstance(result, LLMServerError)
    assert result.details.get("request_id") == "req_1234567890"
