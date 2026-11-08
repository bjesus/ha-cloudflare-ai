"""Model names must reach the Workers AI URL with their slashes intact.

cloudflare SDK >= 5.x percent-encodes path params, so `ai.run("@cf/a/b")`
becomes `/ai/run/%40cf%2Fa%2Fb` and the API answers 400 "No route for that URI".
"""

from __future__ import annotations

import json

import httpx
from cloudflare import AsyncCloudflare

from custom_components.cloudflare_ai.client import CloudflareAIClient

MODEL = "@cf/openai/gpt-oss-20b"
EXPECTED_PATH = f"/client/v4/accounts/acct/ai/run/{MODEL}"


def _client(seen: list[httpx.Request], body: bytes, ctype: str) -> CloudflareAIClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers={"content-type": ctype})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cf = AsyncCloudflare(api_token="tok", http_client=http)
    return CloudflareAIClient(cf=cf, account_id="acct", api_token="tok")


def _raw_path(request: httpx.Request) -> str:
    return request.url.raw_path.decode()


async def test_run_model_keeps_slashes_in_model_path() -> None:
    seen: list[httpx.Request] = []
    body = json.dumps({"success": True, "result": {"response": "hi"}}).encode()
    client = _client(seen, body, "application/json")

    result = await client.run_model(
        MODEL, {"messages": [], "chat_template_kwargs": {"x": 1}}
    )

    assert _raw_path(seen[0]) == EXPECTED_PATH
    assert json.loads(seen[0].content)["chat_template_kwargs"] == {"x": 1}
    assert seen[0].headers["authorization"] == "Bearer tok"
    assert result == {"response": "hi"}


async def test_run_model_binary_keeps_slashes_in_model_path() -> None:
    seen: list[httpx.Request] = []
    client = _client(seen, b"\xff\xfbaudio", "audio/mpeg")

    result = await client.run_model_binary(MODEL, {"text": "hi"})

    assert _raw_path(seen[0]) == EXPECTED_PATH
    assert result == b"\xff\xfbaudio"


async def test_stream_model_keeps_slashes_in_model_path() -> None:
    seen: list[httpx.Request] = []
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    client = _client(seen, sse, "text/event-stream")

    events = [e async for e in client.stream_model(MODEL, {"messages": []})]

    assert _raw_path(seen[0]) == EXPECTED_PATH
    assert json.loads(seen[0].content)["stream"] is True
    assert events == [{"choices": [{"delta": {"content": "hi"}}]}]
