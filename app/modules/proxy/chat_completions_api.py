from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Body, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.clients.proxy import ProxyResponseError
from app.core.openai.chat_completions import ChatCompletionRequest
from app.dependencies import ChatCompletionsContext, get_chat_completions_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat-completions"])


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    payload: ChatCompletionRequest = Body(...),
    context: ChatCompletionsContext = Depends(get_chat_completions_context),
) -> Response:
    rate_limit_headers = await context.rate_limit_headers()

    if payload.stream:
        stream = context.service.stream_chat_completions(
            payload,
            request.headers,
            propagate_http_errors=True,
        )
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            return StreamingResponse(
                _prepend_first(None, stream),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", **rate_limit_headers},
            )
        except ProxyResponseError as exc:
            logger.warning("Chat completions proxy error status=%d payload=%s", exc.status_code, exc.payload)
            return JSONResponse(status_code=exc.status_code, content=exc.payload, headers=rate_limit_headers)
        return StreamingResponse(
            _prepend_first(first, stream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", **rate_limit_headers},
        )

    try:
        result = await context.service.chat_completions(payload, request.headers)
    except ProxyResponseError as exc:
        logger.warning("Chat completions proxy error status=%d payload=%s", exc.status_code, exc.payload)
        return JSONResponse(status_code=exc.status_code, content=exc.payload, headers=rate_limit_headers)
    return JSONResponse(content=result.model_dump(exclude_none=True), headers=rate_limit_headers)


async def _prepend_first(first: str | None, stream: AsyncIterator[str]) -> AsyncIterator[str]:
    if first is not None:
        yield first
    async for line in stream:
        yield line
