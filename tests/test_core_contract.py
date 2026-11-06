"""Exercise typed Core requests and the hooks-only coordinator."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_perplexity import PerplexityProvider


@pytest.mark.asyncio
async def test_typed_request_output_bound_and_hooks():
    hooks = SimpleNamespace(emit=AsyncMock())
    p = PerplexityProvider(
        api_key="test-key",
        config={"raw": True},
        coordinator=SimpleNamespace(hooks=hooks),
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            id="test-response",
            model="sonar-pro",
            citations=[],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="partial"), finish_reason="length"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10),
        )
    )
    p.client.chat.completions.create = create
    result = await p.complete(
        ChatRequest(
            model="sonar-pro",
            max_output_tokens=100,
            temperature=0,
            messages=[Message(role="user", content="Question")],
        )
    )
    assert create.call_args.kwargs["max_tokens"] == 100
    assert create.call_args.kwargs["temperature"] == 0
    assert result.finish_reason == "length" and result.usage.total_tokens == 40
    assert result.metadata["perplexity:response_id"] == "test-response"
    assert [c.args[0] for c in hooks.emit.call_args_list] == [
        "llm:request",
        "llm:response",
    ]
    event = hooks.emit.call_args_list[0].args[1]
    assert event["raw"]["messages"] == create.call_args.kwargs["messages"]
    assert "api_key" not in event["raw"]
    assert p.client.timeout.read is None and p.client.timeout.connect == 5
    await p.close()
