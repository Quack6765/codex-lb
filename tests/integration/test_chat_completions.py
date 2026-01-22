from __future__ import annotations

import base64
import json

import pytest

import app.modules.proxy.chat_completions_service as chat_service_module
from app.core.auth import generate_unique_account_id

pytestmark = pytest.mark.integration


def _encode_jwt(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    return f"header.{body}.sig"


def _make_auth_json(account_id: str, email: str) -> dict:
    payload = {
        "email": email,
        "chatgpt_account_id": account_id,
        "https://api.openai.com/auth": {"chatgpt_plan_type": "plus"},
    }
    return {
        "tokens": {
            "idToken": _encode_jwt(payload),
            "accessToken": "access-token",
            "refreshToken": "refresh-token",
            "accountId": account_id,
        },
    }


def _extract_error_from_sse(lines: list[str]) -> dict | None:
    for line in lines:
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if "error" in data:
                    return data
            except json.JSONDecodeError:
                continue
    return None


def _extract_chunks_from_sse(lines: list[str]) -> list[dict]:
    chunks = []
    for line in lines:
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
                if "choices" in data:
                    chunks.append(data)
            except json.JSONDecodeError:
                continue
    return chunks


@pytest.mark.asyncio
async def test_chat_completions_no_accounts(async_client):
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    async with async_client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
    ) as resp:
        assert resp.status_code == 200
        lines = [line async for line in resp.aiter_lines() if line]

    error_event = _extract_error_from_sse(lines)
    assert error_event is not None
    assert error_event["error"]["code"] == "no_accounts"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming_no_accounts(async_client):
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 503
    data = response.json()
    assert data["error"]["code"] == "no_accounts"


@pytest.mark.asyncio
async def test_chat_completions_streams_via_responses_api(async_client, monkeypatch):
    email = "chat-streamer@example.com"
    raw_account_id = "acc_chat_live"
    auth_json = _make_auth_json(raw_account_id, email)
    files = {"auth_json": ("auth.json", json.dumps(auth_json), "application/json")}
    response = await async_client.post("/api/accounts/import", files=files)
    assert response.status_code == 200

    seen = {}

    async def fake_responses_stream(payload, headers, access_token, account_id, base_url=None, raise_for_status=False):
        seen["access_token"] = access_token
        seen["account_id"] = account_id
        seen["payload_model"] = payload.model
        seen["payload_instructions"] = payload.instructions
        yield 'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n'
        yield 'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        yield 'data: {"type":"response.output_text.delta","delta":" there!"}\n\n'
        yield (
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed",'
            '"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}\n\n'
        )

    monkeypatch.setattr(chat_service_module, "core_stream_responses", fake_responses_stream)

    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": True,
    }
    async with async_client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
    ) as resp:
        assert resp.status_code == 200
        lines = [line async for line in resp.aiter_lines() if line]

    chunks = _extract_chunks_from_sse(lines)
    assert len(chunks) >= 1
    assert seen["access_token"] == "access-token"
    assert seen["account_id"] == raw_account_id
    assert "You are helpful" in seen["payload_instructions"]

    has_done = any("data: [DONE]" in line for line in lines)
    assert has_done


@pytest.mark.asyncio
async def test_chat_to_responses_conversion(async_client):
    from app.core.openai.chat_completions import ChatCompletionRequest, ChatCompletionMessage
    from app.core.openai.chat_to_responses import chat_to_responses_request

    chat_req = ChatCompletionRequest(
        model="gpt-4",
        messages=[
            ChatCompletionMessage(role="system", content="Be concise."),
            ChatCompletionMessage(role="user", content="What is 2+2?"),
            ChatCompletionMessage(role="assistant", content="4"),
            ChatCompletionMessage(role="user", content="And 3+3?"),
        ],
    )

    responses_req = chat_to_responses_request(chat_req)

    assert responses_req.model == "gpt-4"
    assert "Be concise" in responses_req.instructions
    assert len(responses_req.input) == 3
    assert responses_req.input[0]["role"] == "user"
    assert responses_req.input[1]["role"] == "assistant"
    assert responses_req.input[2]["role"] == "user"
