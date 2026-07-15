"""Generic OpenAI-compatible JSON transport."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from http.client import IncompleteRead
import json
import math
import os
from pathlib import Path
import time
from urllib import error, request
from urllib.parse import urlsplit

from robometanorm.models import Issue


_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_ISSUE_MESSAGES = {
    "VLM_CONFIG_MISSING": "VLM authentication is not configured.",
    "VLM_IMAGE_READ_FAILED": "A VLM input image could not be read.",
    "VLM_HTTP_ERROR": "The VLM service rejected the request.",
    "VLM_NETWORK_ERROR": "The VLM service could not be reached.",
    "VLM_RESPONSE_INVALID": "The VLM service returned an invalid response.",
}

_BASE_URL_ERROR = "base_url must be a valid HTTP(S) base URL"


class _InvalidResponse(ValueError):
    """Internal marker for a response that violates the transport contract."""


def _constant_credential(value: str) -> Callable[[], str]:
    """Keep an explicit credential out of inspectable transport properties."""

    def resolve() -> str:
        return value

    return resolve


def _issue(code: str, evidence: Mapping[str, object] | None = None) -> Issue:
    return Issue(
        code=code,
        message=_ISSUE_MESSAGES[code],
        scope="vlm",
        evidence=dict(evidence or {}),
    )


def _credential_is_usable(value: str) -> bool:
    return bool(value.strip()) and "\r" not in value and "\n" not in value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without outer whitespace")
    return value


def _normalize_base_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(_BASE_URL_ERROR)
    if any(
        character.isspace()
        or ord(character) < 32
        or 127 <= ord(character) <= 159
        for character in value
    ):
        raise ValueError(_BASE_URL_ERROR)

    normalized = value.rstrip("/")
    if not normalized or "?" in normalized or "#" in normalized:
        raise ValueError(_BASE_URL_ERROR)

    split_url = None
    hostname = None
    username = None
    password = None
    port = None
    parse_failed = False
    try:
        split_url = urlsplit(normalized)
        hostname = split_url.hostname
        username = split_url.username
        password = split_url.password
        port = split_url.port
    except ValueError:
        parse_failed = True

    if parse_failed or split_url is None:
        raise ValueError(_BASE_URL_ERROR)
    if (
        split_url.scheme not in {"http", "https"}
        or not hostname
        or username is not None
        or password is not None
        or split_url.netloc.endswith(":")
        or (port is not None and (type(port) is not int or not 1 <= port <= 65535))
    ):
        raise ValueError(_BASE_URL_ERROR)
    return normalized


def _require_finite_number(
    name: str, value: object, *, allow_zero: bool
) -> int | float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _require_exact_int(name: str, value: object, *, allow_zero: bool) -> int:
    if type(value) is not int or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _InvalidResponse("non-finite JSON number")
    return parsed


def _reject_json_constant(value: str) -> object:
    raise _InvalidResponse("non-standard JSON constant")


def _require_finite_json(value: object) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidResponse("non-finite JSON number")
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item)


def _strict_json_object(text: str) -> dict[str, object]:
    parsed = json.loads(
        text,
        parse_constant=_reject_json_constant,
        parse_float=_strict_float,
    )
    _require_finite_json(parsed)
    if not isinstance(parsed, dict):
        raise _InvalidResponse("JSON top level is not an object")
    return parsed


def _outer_response(raw: object) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise _InvalidResponse("response body is not bytes")
    return _strict_json_object(raw.decode("utf-8"))


def _chat_content(payload: Mapping[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _InvalidResponse("missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise _InvalidResponse("invalid choice")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise _InvalidResponse("invalid message")
    content = message.get("content")
    if not isinstance(content, str):
        raise _InvalidResponse("invalid message content")
    return content


def _responses_content(payload: Mapping[str, object]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        raise _InvalidResponse("missing output")

    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping):
            raise _InvalidResponse("invalid output item")
        item_type = item.get("type")
        if item_type is not None and item_type != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            raise _InvalidResponse("invalid output content")
        for part in content:
            if not isinstance(part, Mapping):
                raise _InvalidResponse("invalid output content part")
            if part.get("type") != "output_text":
                continue
            text = part.get("text")
            if not isinstance(text, str):
                raise _InvalidResponse("invalid output text")
            texts.append(text)

    if not texts:
        raise _InvalidResponse("missing output text")
    return "".join(texts)


class OpenAICompatibleTransport:
    """Send strict JSON requests through Chat Completions and Responses APIs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = _require_text("model", model)
        self.api_key_env = _require_text("api_key_env", api_key_env)
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("api_key must be a string or None")
        self.timeout_seconds = _require_finite_number(
            "timeout_seconds", timeout_seconds, allow_zero=False
        )
        self.max_retries = _require_exact_int(
            "max_retries", max_retries, allow_zero=True
        )
        self.retry_backoff_seconds = _require_finite_number(
            "retry_backoff_seconds", retry_backoff_seconds, allow_zero=True
        )
        self.max_tokens = _require_exact_int(
            "max_tokens", max_tokens, allow_zero=False
        )
        resolved_credential = (
            os.getenv(self.api_key_env, "") if api_key is None else api_key
        )
        self._credential_provider = _constant_credential(resolved_credential)

    def request_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[Path],
    ) -> tuple[Mapping[str, object] | None, Issue | None]:
        """Send a Chat Completions request and return its strict JSON object."""
        credential = self._credential_provider()
        if not _credential_is_usable(credential):
            return None, _issue(
                "VLM_CONFIG_MISSING", {"api_key_env": self.api_key_env}
            )

        user_content: list[dict[str, object]] = [
            {"type": "text", "text": user_prompt}
        ]
        for image_path in image_paths:
            mime_type = _IMAGE_MIME_TYPES.get(image_path.suffix.lower())
            if mime_type is None:
                return None, self._image_issue(image_path, "UnsupportedImageType")
            try:
                image_bytes = image_path.read_bytes()
            except OSError as image_error:
                return None, self._image_issue(
                    image_path, type(image_error).__name__
                )
            if not image_bytes:
                return None, self._image_issue(image_path, "EmptyImage")
            encoded = base64.b64encode(image_bytes).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                    },
                }
            )

        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        return self._post_json(
            self.base_url + "/chat/completions",
            payload,
            credential,
            _chat_content,
        )

    def request_web_json(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[Mapping[str, object] | None, Issue | None]:
        """Send a Responses web-search request and return its strict JSON object."""
        credential = self._credential_provider()
        if not _credential_is_usable(credential):
            return None, _issue(
                "VLM_CONFIG_MISSING", {"api_key_env": self.api_key_env}
            )

        payload: dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": [{"type": "web_search"}],
            "max_output_tokens": self.max_tokens,
        }
        return self._post_json(
            self.base_url + "/responses",
            payload,
            credential,
            _responses_content,
        )

    @staticmethod
    def _image_issue(image_path: Path, error_type: str) -> Issue:
        return _issue(
            "VLM_IMAGE_READ_FAILED",
            {"file_name": image_path.name, "error_type": error_type},
        )

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        credential: str,
        content_parser: Callable[[Mapping[str, object]], str],
    ) -> tuple[Mapping[str, object] | None, Issue | None]:
        try:
            encoded_payload = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError, RecursionError) as payload_error:
            return None, self._response_issue(payload_error, stage="request")

        for attempt in range(self.max_retries + 1):
            http_request = request.Request(
                url,
                data=encoded_payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {credential}",
                },
                method="POST",
            )
            try:
                with request.urlopen(
                    http_request, timeout=self.timeout_seconds
                ) as response:
                    response_payload = _outer_response(response.read())
                content = content_parser(response_payload)
                return _strict_json_object(content), None
            except error.HTTPError as http_error:
                self._close_http_error(http_error)
                if self._should_retry_http(http_error.code, attempt):
                    self._backoff(attempt)
                    continue
                return None, _issue("VLM_HTTP_ERROR", {"status": http_error.code})
            except (error.URLError, TimeoutError, OSError, IncompleteRead) as network_error:
                if attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                return None, _issue(
                    "VLM_NETWORK_ERROR",
                    {"error_type": type(network_error).__name__},
                )
            except (
                ValueError,
                UnicodeError,
                OverflowError,
                RecursionError,
            ) as response_error:
                return None, self._response_issue(response_error, stage="response")

        raise AssertionError("retry loop exhausted without returning")

    def _should_retry_http(self, status: int, attempt: int) -> bool:
        return attempt < self.max_retries and (
            status == 429 or 500 <= status < 600
        )

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay:
            time.sleep(delay)

    @staticmethod
    def _close_http_error(http_error: error.HTTPError) -> None:
        try:
            http_error.close()
        except OSError:
            pass

    @staticmethod
    def _response_issue(response_error: BaseException, *, stage: str) -> Issue:
        return _issue(
            "VLM_RESPONSE_INVALID",
            {"stage": stage, "error_type": type(response_error).__name__},
        )
