from __future__ import annotations

from app.core.openai.chat_completions import ChatCompletionRequest
from app.core.openai.requests import ResponsesReasoning, ResponsesRequest
from app.core.types import JsonValue


def chat_to_responses_request(chat_req: ChatCompletionRequest) -> ResponsesRequest:
    instructions = ""
    input_items: list[JsonValue] = []

    for msg in chat_req.messages:
        if msg.role == "system" or msg.role == "developer":
            if instructions:
                instructions += "\n\n"
            instructions += _extract_content_text(msg.content)
        elif msg.role == "user":
            content_parts = _normalize_content_for_input(msg.content)
            input_items.append({
                "type": "message",
                "role": "user",
                "content": content_parts,
            })
        elif msg.role == "assistant":
            content_parts = _normalize_content_for_output(msg.content)
            if content_parts:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": content_parts,
                })

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict) and tool_call.get("type") == "function":
                        func = tool_call.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "call_id": tool_call.get("id"),
                            "name": func.get("name"),
                            "arguments": func.get("arguments"),
                        })
        elif msg.role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id or "",
                "output": _extract_content_text(msg.content),
            })

    reasoning = None
    if chat_req.reasoning_effort:
        reasoning = ResponsesReasoning(effort=chat_req.reasoning_effort)

    tools: list[JsonValue] = []
    if chat_req.tools:
        for tool in chat_req.tools:
            if tool.type == "function":
                tools.append({
                    "type": "function",
                    **tool.function,
                })

    responses_req = ResponsesRequest(
        model=chat_req.model,
        instructions=instructions or "You are a helpful assistant.",
        input=input_items,
        tools=tools if tools else [],
        tool_choice=_convert_tool_choice(chat_req.tool_choice),
        parallel_tool_calls=chat_req.parallel_tool_calls,
        reasoning=reasoning,
        store=False,
        stream=True,
    )

    return responses_req


def _extract_content_text(content: str | list[JsonValue] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text_parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(part.get("text", ""))
        elif isinstance(part, str):
            text_parts.append(part)
    return "".join(text_parts)


def _normalize_content_for_input(content: str | list[JsonValue] | None) -> str | list[JsonValue]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if all(isinstance(x, str) or (isinstance(x, dict) and x.get("type") == "text") for x in content):
        return _extract_content_text(content)
        
    result = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                result.append({"type": "text", "text": part.get("text", "")})
            else:
                result.append(part)
        elif isinstance(part, str):
            result.append({"type": "text", "text": part})
    return result if result else ""


def _normalize_content_for_output(content: str | list[JsonValue] | None) -> str | list[JsonValue]:
    return _normalize_content_for_input(content)


def _convert_tool_choice(choice: str | dict | None) -> str | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            return choice.get("function", {}).get("name")
    return None
