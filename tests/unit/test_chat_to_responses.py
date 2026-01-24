
from app.core.openai.chat_completions import ChatCompletionRequest, ChatCompletionMessage
from app.core.openai.chat_to_responses import chat_to_responses_request

def test_chat_to_responses_assistant_tool_calls():
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[
            ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }
                ]
            )
        ]
    )
    
    response_req = chat_to_responses_request(request)
    
    assert len(response_req.input) == 1
    item = response_req.input[0]
    assert item["type"] == "function_call"
    assert item["call_id"] == "call_123"
    assert item["name"] == "get_weather"

def test_chat_to_responses_assistant_content_and_tool_calls():
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[
            ChatCompletionMessage(
                role="assistant",
                content="Checking weather...",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }
                ]
            )
        ]
    )
    
    response_req = chat_to_responses_request(request)
    
    assert len(response_req.input) == 2
    msg_item = response_req.input[0]
    assert msg_item["type"] == "message"
    assert msg_item["role"] == "assistant"
    assert msg_item["content"] == "Checking weather..."
    
    tool_item = response_req.input[1]
    assert tool_item["type"] == "function_call"
    assert tool_item["call_id"] == "call_123"
