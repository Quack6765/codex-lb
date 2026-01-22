from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from app.core.types import JsonObject, JsonValue


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str | list[JsonValue] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[JsonValue] | None = None


class ChatCompletionTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: JsonObject


class ChatCompletionResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonObject | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatCompletionMessage]

    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = Field(default=None, ge=1, le=128)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    response_format: ChatCompletionResponseFormat | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: JsonObject | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    tools: list[ChatCompletionTool] | None = None
    tool_choice: str | JsonObject | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    reasoning_effort: str | None = None

    def to_payload(self) -> JsonObject:
        return self.model_dump(mode="json", exclude_none=True)


class ChatCompletionUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: StrictInt | None = None
    completion_tokens: StrictInt | None = None
    total_tokens: StrictInt | None = None
    prompt_tokens_details: JsonObject | None = None
    completion_tokens_details: JsonObject | None = None


class ChatCompletionChoiceMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: StrictStr = "assistant"
    content: StrictStr | None = None
    tool_calls: list[JsonValue] | None = None
    refusal: StrictStr | None = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: StrictInt
    message: ChatCompletionChoiceMessage
    finish_reason: StrictStr | None = None
    logprobs: JsonValue | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: StrictStr
    object: Literal["chat.completion"] = "chat.completion"
    created: StrictInt
    model: StrictStr
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None
    system_fingerprint: StrictStr | None = None
    service_tier: StrictStr | None = None


class ChatCompletionChunkDelta(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: StrictStr | None = None
    content: StrictStr | None = None
    tool_calls: list[JsonValue] | None = None
    refusal: StrictStr | None = None


class ChatCompletionChunkChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: StrictInt
    delta: ChatCompletionChunkDelta
    finish_reason: StrictStr | None = None
    logprobs: JsonValue | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: StrictStr
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: StrictInt
    model: StrictStr
    choices: list[ChatCompletionChunkChoice]
    usage: ChatCompletionUsage | None = None
    system_fingerprint: StrictStr | None = None
    service_tier: StrictStr | None = None


class ChatCompletionError(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: StrictStr | None = None
    type: StrictStr | None = None
    code: StrictStr | None = None
    param: StrictStr | None = None


class ChatCompletionErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: ChatCompletionError | None = None
