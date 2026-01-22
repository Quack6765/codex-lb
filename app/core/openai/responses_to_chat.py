from __future__ import annotations

import json
import time
from typing import Iterator

from app.core.openai.chat_completions import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from app.core.types import JsonValue


class ResponsesStreamConverter:
    def __init__(self, request_id: str, model: str) -> None:
        self._request_id = request_id
        self._model = model
        self._created = int(time.time())
        self._accumulated_content = ""
        self._accumulated_tool_calls: list[JsonValue] = []
        self._first_chunk_sent = False
        self._usage: ChatCompletionUsage | None = None

    def convert_sse_line(self, line: str) -> Iterator[str]:
        if not line.startswith("data:"):
            return

        data = line[5:].strip()
        if not data or data == "[DONE]":
            return

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return

        if not isinstance(payload, dict):
            return

        event_type = payload.get("type", "")

        if event_type == "response.created":
            if not self._first_chunk_sent:
                self._first_chunk_sent = True
                yield self._format_chunk(ChatCompletionChunkDelta(role="assistant", content=""))

        elif event_type == "response.output_text.delta":
            delta_text = payload.get("delta", "")
            if delta_text:
                self._accumulated_content += delta_text
                yield self._format_chunk(ChatCompletionChunkDelta(content=delta_text))

        elif event_type == "response.content_part.delta":
            delta = payload.get("delta", {})
            if isinstance(delta, dict) and delta.get("type") == "output_text":
                text = delta.get("text", "")
                if text:
                    self._accumulated_content += text
                    yield self._format_chunk(ChatCompletionChunkDelta(content=text))

        elif event_type == "response.function_call_arguments.delta":
            pass

        elif event_type == "response.output_item.done":
            item = payload.get("item", {})
            if item.get("type") == "function_call":
                tool_call = {
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                }
                self._accumulated_tool_calls.append(tool_call)
                yield self._format_chunk(ChatCompletionChunkDelta(
                    tool_calls=[{
                        "index": len(self._accumulated_tool_calls) - 1,
                        **tool_call,
                    }]
                ))

        elif event_type == "response.completed":
            response = payload.get("response", {})
            usage = response.get("usage", {})
            if usage:
                self._usage = ChatCompletionUsage(
                    prompt_tokens=usage.get("input_tokens"),
                    completion_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("total_tokens"),
                )
            yield self._format_final_chunk("stop")

        elif event_type == "response.failed":
            error = payload.get("response", {}).get("error", {})
            error_message = error.get("message", "Request failed")
            yield self._format_error(error.get("code", "error"), error_message)

        elif event_type == "error":
            error = payload.get("error", {})
            error_message = error.get("message", "Request failed")
            yield self._format_error(error.get("code", "error"), error_message)

    def _format_chunk(self, delta: ChatCompletionChunkDelta, finish_reason: str | None = None) -> str:
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{self._request_id}",
            object="chat.completion.chunk",
            created=self._created,
            model=self._model,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )],
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def _format_final_chunk(self, finish_reason: str) -> str:
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{self._request_id}",
            object="chat.completion.chunk",
            created=self._created,
            model=self._model,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason=finish_reason,
            )],
            usage=self._usage,
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def _format_error(self, code: str, message: str) -> str:
        error_data = {
            "error": {
                "message": message,
                "type": "server_error",
                "code": code,
            }
        }
        return f"data: {json.dumps(error_data)}\n\n"

    def format_done(self) -> str:
        return "data: [DONE]\n\n"


def responses_to_chat_response(
    response_data: dict,
    request_id: str,
    model: str,
) -> ChatCompletionResponse:
    created = int(time.time())
    content = ""
    tool_calls: list[JsonValue] = []

    output = response_data.get("output", [])
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content += part.get("text", "")
                elif part.get("type") == "text":
                    content += part.get("text", "")
        elif item.get("type") == "function_call":
            tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                },
            })

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"

    message = ChatCompletionChoiceMessage(
        role="assistant",
        content=content if content else None,
        tool_calls=tool_calls if tool_calls else None,
    )

    usage_data = response_data.get("usage", {})
    usage = None
    if usage_data:
        usage = ChatCompletionUsage(
            prompt_tokens=usage_data.get("input_tokens"),
            completion_tokens=usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        object="chat.completion",
        created=created,
        model=model,
        choices=[ChatCompletionChoice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )],
        usage=usage,
    )
