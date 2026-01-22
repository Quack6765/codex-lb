from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

from app.core.openai.chat_completions import (
    ChatCompletionChunk,
    ChatCompletionError,
    ChatCompletionErrorEnvelope,
    ChatCompletionResponse,
)
from app.core.openai.models import OpenAIError, OpenAIErrorEnvelope, OpenAIEvent, OpenAIResponsePayload

_EVENT_ADAPTER = TypeAdapter(OpenAIEvent)
_ERROR_ADAPTER = TypeAdapter(OpenAIErrorEnvelope)
_RESPONSE_ADAPTER = TypeAdapter(OpenAIResponsePayload)
_CHAT_CHUNK_ADAPTER = TypeAdapter(ChatCompletionChunk)
_CHAT_RESPONSE_ADAPTER = TypeAdapter(ChatCompletionResponse)
_CHAT_ERROR_ADAPTER = TypeAdapter(ChatCompletionErrorEnvelope)


def parse_sse_event(line: str) -> OpenAIEvent | None:
    data = None
    if line.startswith("data:"):
        data = line[5:].strip()
    elif "\n" in line:
        for part in line.splitlines():
            if part.startswith("data:"):
                data = part[5:].strip()
                break
    if data is None:
        return None
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _EVENT_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


def parse_error_payload(payload: object) -> OpenAIError | None:
    if not isinstance(payload, dict):
        return None
    try:
        envelope = _ERROR_ADAPTER.validate_python(payload)
    except ValidationError:
        return None
    return envelope.error


def parse_response_payload(payload: object) -> OpenAIResponsePayload | None:
    if not isinstance(payload, dict):
        return None
    try:
        return _RESPONSE_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


def parse_chat_chunk(line: str) -> ChatCompletionChunk | None:
    data = None
    if line.startswith("data:"):
        data = line[5:].strip()
    elif "\n" in line:
        for part in line.splitlines():
            if part.startswith("data:"):
                data = part[5:].strip()
                break
    if data is None:
        return None
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _CHAT_CHUNK_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


def is_chat_stream_done(line: str) -> bool:
    if line.startswith("data:"):
        data = line[5:].strip()
        return data == "[DONE]"
    return False


def parse_chat_response(payload: object) -> ChatCompletionResponse | None:
    if not isinstance(payload, dict):
        return None
    try:
        return _CHAT_RESPONSE_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


def parse_chat_error_payload(payload: object) -> ChatCompletionError | None:
    if not isinstance(payload, dict):
        return None
    try:
        envelope = _CHAT_ERROR_ADAPTER.validate_python(payload)
    except ValidationError:
        return None
    return envelope.error
