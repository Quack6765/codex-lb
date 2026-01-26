from __future__ import annotations

import json
import logging
import time
from contextlib import aclosing
from typing import AsyncIterator, Mapping

import anyio

from app.core.auth.refresh import RefreshError
from app.core.balancer import PERMANENT_FAILURE_CODES
from app.core.balancer.types import UpstreamError
from app.core.clients.proxy import (
    ProxyResponseError,
    filter_inbound_headers,
)
from app.core.clients.proxy import stream_responses as core_stream_responses
from app.core.crypto import TokenEncryptor
from app.core.errors import openai_error
from app.core.openai.chat_completions import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from app.core.openai.chat_to_responses import chat_to_responses_request
from app.core.openai.parsing import parse_sse_event
from app.core.openai.responses_to_chat import (
    ResponsesStreamConverter,
    responses_to_chat_response,
)
from app.core.utils.request_id import ensure_request_id
from app.db.models import Account
from app.modules.accounts.auth_manager import AuthManager
from app.modules.accounts.repository import AccountsRepository
from app.modules.proxy.helpers import (
    _header_account_id,
    _normalize_error_code,
)
from app.modules.proxy.load_balancer import LoadBalancer
from app.modules.proxy.sticky_repository import StickySessionsRepository
from app.modules.request_logs.repository import RequestLogsRepository
from app.modules.settings.repository import SettingsRepository
from app.modules.usage.repository import UsageRepository

logger = logging.getLogger(__name__)


class _RetryableChatError(Exception):
    def __init__(self, code: str, error: UpstreamError) -> None:
        super().__init__(code)
        self.code = code
        self.error = error


class ChatCompletionsService:
    def __init__(
        self,
        accounts_repo: AccountsRepository,
        usage_repo: UsageRepository,
        logs_repo: RequestLogsRepository,
        sticky_repo: StickySessionsRepository,
        settings_repo: SettingsRepository,
    ) -> None:
        self._accounts_repo = accounts_repo
        self._usage_repo = usage_repo
        self._logs_repo = logs_repo
        self._settings_repo = settings_repo
        self._encryptor = TokenEncryptor()
        self._auth_manager = AuthManager(accounts_repo)
        self._load_balancer = LoadBalancer(accounts_repo, usage_repo, sticky_repo)

    def stream_chat_completions(
        self,
        payload: ChatCompletionRequest,
        headers: Mapping[str, str],
        *,
        propagate_http_errors: bool = False,
    ) -> AsyncIterator[str]:
        filtered = filter_inbound_headers(headers)
        return self._stream_with_retry(
            payload,
            filtered,
            propagate_http_errors=propagate_http_errors,
        )

    async def chat_completions(
        self,
        payload: ChatCompletionRequest,
        headers: Mapping[str, str],
    ) -> ChatCompletionResponse:
        request_id = ensure_request_id()
        filtered = filter_inbound_headers(headers)
        settings = await self._settings_repo.get_or_create()
        prefer_earlier_reset = settings.prefer_earlier_reset_accounts
        selection = await self._load_balancer.select_account(
            sticky_key=None,
            reallocate_sticky=False,
            prefer_earlier_reset_accounts=prefer_earlier_reset,
        )
        account = selection.account
        if not account:
            raise ProxyResponseError(
                503,
                openai_error("no_accounts", selection.error_message or "No active accounts available"),
            )
        account = await self._ensure_fresh(account)

        responses_request = chat_to_responses_request(payload)
        responses_request.stream = True

        full_response_data: dict = {}

        async def _call(target: Account) -> ChatCompletionResponse:
            nonlocal full_response_data
            access_token = self._encryptor.decrypt(target.access_token_encrypted)
            account_id_header = _header_account_id(target.chatgpt_account_id)

            stream = core_stream_responses(
                responses_request,
                filtered,
                access_token,
                account_id_header,
                raise_for_status=True,
            )

            async with aclosing(stream):
                async for line in stream:
                    event = parse_sse_event(line)
                    if event and event.type == "response.completed":
                        if event.response:
                            full_response_data = event.response.model_dump(exclude_none=True)
                    elif event and event.type in ("response.failed", "error"):
                        error = event.response.error if event.response else event.error
                        code = _normalize_error_code(
                            error.code if error else None,
                            error.type if error else None,
                        )
                        message = error.message if error else "Request failed"
                        raise ProxyResponseError(500, openai_error(code, message))

            return responses_to_chat_response(full_response_data, request_id, payload.model)

        try:
            return await _call(account)
        except ProxyResponseError as exc:
            if exc.status_code != 401:
                await self._handle_proxy_error(account, exc)
                raise
            try:
                account = await self._ensure_fresh(account, force=True)
            except RefreshError as refresh_exc:
                if refresh_exc.is_permanent:
                    await self._load_balancer.mark_permanent_failure(account, refresh_exc.code)
                raise exc
            try:
                return await _call(account)
            except ProxyResponseError as exc:
                await self._handle_proxy_error(account, exc)
                raise

    async def _stream_with_retry(
        self,
        payload: ChatCompletionRequest,
        headers: Mapping[str, str],
        *,
        propagate_http_errors: bool,
    ) -> AsyncIterator[str]:
        request_id = ensure_request_id()
        settings = await self._settings_repo.get_or_create()
        prefer_earlier_reset = settings.prefer_earlier_reset_accounts
        max_attempts = 3

        for attempt in range(max_attempts):
            selection = await self._load_balancer.select_account(
                sticky_key=None,
                prefer_earlier_reset_accounts=prefer_earlier_reset,
            )
            account = selection.account
            if not account:
                yield _format_chat_error("no_accounts", selection.error_message or "No active accounts available")
                return

            account_id_value = account.id
            try:
                account = await self._ensure_fresh(account)
                async for line in self._stream_once(
                    account,
                    payload,
                    headers,
                    request_id,
                    attempt < max_attempts - 1,
                ):
                    yield line
                return
            except _RetryableChatError as exc:
                await self._handle_stream_error(account, exc.error, exc.code)
                continue
            except ProxyResponseError as exc:
                if exc.status_code == 401:
                    try:
                        account = await self._ensure_fresh(account, force=True)
                    except RefreshError as refresh_exc:
                        if refresh_exc.is_permanent:
                            await self._load_balancer.mark_permanent_failure(account, refresh_exc.code)
                        continue
                    async for line in self._stream_once(account, payload, headers, request_id, False):
                        yield line
                    return
                error = _parse_openai_error(exc.payload)
                err_code = error.get("code") if error else None
                err_type = error.get("type") if error else None
                error_code = _normalize_error_code(err_code, err_type)
                error_message = error.get("message") if error else None
                await self._handle_stream_error(
                    account,
                    _upstream_error_from_dict(error),
                    error_code,
                )
                if propagate_http_errors:
                    raise
                yield _format_chat_error(error_code, error_message or "Upstream error")
                return
            except RefreshError as exc:
                if exc.is_permanent:
                    await self._load_balancer.mark_permanent_failure(account, exc.code)
                continue
            except Exception:
                try:
                    await self._load_balancer.record_error(account)
                except Exception:
                    logger.warning(
                        "Failed to record proxy error account_id=%s request_id=%s",
                        account_id_value,
                        request_id,
                        exc_info=True,
                    )
                if attempt == max_attempts - 1:
                    yield _format_chat_error("upstream_error", "Proxy streaming failed")
                    return

        yield _format_chat_error("no_accounts", "No available accounts after retries")

    async def _stream_once(
        self,
        account: Account,
        payload: ChatCompletionRequest,
        headers: Mapping[str, str],
        request_id: str,
        allow_retry: bool,
    ) -> AsyncIterator[str]:
        account_id_value = account.id
        access_token = self._encryptor.decrypt(account.access_token_encrypted)
        account_id_header = _header_account_id(account.chatgpt_account_id)
        model = payload.model
        reasoning_effort = payload.reasoning_effort
        start = time.monotonic()
        status = "success"
        error_code = None
        error_message = None
        input_tokens = None
        output_tokens = None

        responses_request = chat_to_responses_request(payload)
        converter = ResponsesStreamConverter(request_id, model)

        try:
            stream = core_stream_responses(
                responses_request,
                headers,
                access_token,
                account_id_header,
                raise_for_status=True,
            )
            async with aclosing(stream):
                iterator = stream.__aiter__()
                try:
                    first = await iterator.__anext__()
                except StopAsyncIteration:
                    return

                event = parse_sse_event(first)
                if event and event.type in ("response.failed", "error"):
                    if event.type == "response.failed":
                        response = event.response
                        error = response.error if response else None
                    else:
                        error = event.error
                    code = _normalize_error_code(
                        error.code if error else None,
                        error.type if error else None,
                    )
                    status = "error"
                    error_code = code
                    error_message = error.message if error else None
                    if allow_retry:
                        raise _RetryableChatError(code, UpstreamError())

                for chunk in converter.convert_sse_line(first):
                    yield chunk

                async for line in iterator:
                    event = parse_sse_event(line)
                    if event:
                        if event.type in ("response.failed", "error"):
                            status = "error"
                            if event.type == "response.failed":
                                response = event.response
                                error = response.error if response else None
                            else:
                                error = event.error
                            error_code = _normalize_error_code(
                                error.code if error else None,
                                error.type if error else None,
                            )
                            error_message = error.message if error else None
                        elif event.type == "response.completed":
                            if event.response and event.response.usage:
                                input_tokens = event.response.usage.input_tokens
                                output_tokens = event.response.usage.output_tokens

                    for chunk in converter.convert_sse_line(line):
                        yield chunk

                yield converter.format_done()

        except ProxyResponseError as exc:
            error = _parse_openai_error(exc.payload)
            status = "error"
            error_code = _normalize_error_code(
                error.get("code") if error else None,
                error.get("type") if error else None,
            )
            error_message = error.get("message") if error else None
            raise
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            with anyio.CancelScope(shield=True):
                try:
                    await self._logs_repo.add_log(
                        account_id=account_id_value,
                        request_id=request_id,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=None,
                        reasoning_tokens=None,
                        reasoning_effort=reasoning_effort,
                        latency_ms=latency_ms,
                        status=status,
                        error_code=error_code,
                        error_message=error_message,
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist request log account_id=%s request_id=%s",
                        account_id_value,
                        request_id,
                        exc_info=True,
                    )

    async def _ensure_fresh(self, account: Account, *, force: bool = False) -> Account:
        return await self._auth_manager.ensure_fresh(account, force=force)

    async def _handle_proxy_error(self, account: Account, exc: ProxyResponseError) -> None:
        error = _parse_openai_error(exc.payload)
        code = _normalize_error_code(
            error.get("code") if error else None,
            error.get("type") if error else None,
        )
        await self._handle_stream_error(account, _upstream_error_from_dict(error), code)

    async def _handle_stream_error(self, account: Account, error: UpstreamError, code: str) -> None:
        if code in {"rate_limit_exceeded", "usage_limit_reached"}:
            await self._load_balancer.mark_rate_limit(account, error)
            return
        if code in {"insufficient_quota", "usage_not_included", "quota_exceeded"}:
            await self._load_balancer.mark_quota_exceeded(account, error)
            return
        if code in PERMANENT_FAILURE_CODES:
            await self._load_balancer.mark_permanent_failure(account, code)
            return
        await self._load_balancer.record_error(account)


def _format_chat_error(code: str, message: str) -> str:
    error_data = {
        "error": {
            "message": message,
            "type": "server_error",
            "code": code,
        }
    }
    return f"data: {json.dumps(error_data)}\n\n"


def _parse_openai_error(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        return error
    return None


def _upstream_error_from_dict(error: dict | None) -> UpstreamError:
    if error is None:
        return UpstreamError()
    resets_at = error.get("resets_at")
    resets_in = error.get("resets_in_seconds")
    if isinstance(resets_at, (int, float)):
        return UpstreamError(reset_at=resets_at)
    if isinstance(resets_in, (int, float)):
        return UpstreamError(reset_in_seconds=resets_in)
    return UpstreamError()
