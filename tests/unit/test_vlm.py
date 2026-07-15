"""Generic OpenAI-compatible VLM transport contract tests."""

from __future__ import annotations

import base64
import hashlib
from http.client import IncompleteRead
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.models import Issue
from robometanorm.vlm import OpenAICompatibleTransport


class _HttpResponse:
    """Small context-managed HTTP response at the urllib boundary."""

    def __init__(
        self,
        payload: object | None = None,
        *,
        raw: bytes | object | None = None,
        read_error: BaseException | None = None,
        enter_error: BaseException | None = None,
        exit_error: BaseException | None = None,
    ) -> None:
        self.raw = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if raw is None
            else raw
        )
        self.read_error = read_error
        self.enter_error = enter_error
        self.exit_error = exit_error

    def __enter__(self) -> "_HttpResponse":
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> bool:
        if self.exit_error is not None:
            raise self.exit_error
        return False

    def read(self) -> bytes | object:
        if self.read_error is not None:
            raise self.read_error
        return self.raw


class _UnreadableHttpBody(BytesIO):
    """HTTPError body that makes accidental reads fail the test."""

    def read(self, *args: object, **kwargs: object) -> bytes:
        raise AssertionError("HTTPError body must not be read")


class _TrackedHttpBody(_UnreadableHttpBody):
    """Unreadable body that records resource cleanup."""

    def __init__(self, close_error: BaseException | None = None) -> None:
        super().__init__(b"private response body")
        self.close_calls = 0
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            close_error, self.close_error = self.close_error, None
            raise close_error
        super().close()


def _chat_response(content: object) -> _HttpResponse:
    return _HttpResponse({"choices": [{"message": {"content": content}}]})


def _responses_response(content: str) -> _HttpResponse:
    return _HttpResponse(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": content}],
                }
            ]
        }
    )


def _http_error(status_code: int, reason: str = "fixture failure") -> HTTPError:
    return HTTPError(
        "https://service.invalid/private",
        status_code,
        reason,
        {},
        _UnreadableHttpBody(b"private response body"),
    )


class VlmTransportTest(unittest.TestCase):
    """Verify transport behavior without making real network requests."""

    def make_transport(self, **overrides: object) -> OpenAICompatibleTransport:
        arguments: dict[str, object] = {
            "base_url": "https://example.test/v1",
            "model": "fixture-model",
            "api_key": "fixture-key",
            "retry_backoff_seconds": 0.0,
        }
        arguments.update(overrides)
        return OpenAICompatibleTransport(**arguments)

    def request_chat_content(
        self, content: object, *, max_retries: int = 2
    ) -> tuple[object | None, Issue | None, int]:
        transport = self.make_transport(max_retries=max_retries)
        with patch(
            "robometanorm.vlm.request.urlopen",
            return_value=_chat_response(content),
        ) as urlopen:
            payload, issue = transport.request_json("system", "user", ())
        return payload, issue, urlopen.call_count

    def test_constructor_rejects_invalid_text_configuration(self) -> None:
        for field, value in (
            ("base_url", ""),
            ("base_url", "   "),
            ("model", ""),
            ("model", "\t"),
            ("api_key_env", ""),
            ("api_key_env", "  "),
            ("api_key_env", None),
        ):
            with self.subTest(field=field, value=value):
                arguments: dict[str, object] = {
                    "base_url": "https://example.test/v1",
                    "model": "model",
                    "api_key": "key",
                }
                arguments[field] = value
                with self.assertRaises(ValueError):
                    OpenAICompatibleTransport(**arguments)

    def test_constructor_rejects_non_http_base_urls_or_url_extras(self) -> None:
        for base_url in (
            "example.test/v1",
            "ftp://example.test/v1",
            "https:///v1",
            "https://example.test/v1?private=query",
            "https://example.test/v1#private-fragment",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaises(ValueError):
                    self.make_transport(base_url=base_url)

    def test_constructor_rejects_invalid_numeric_configuration(self) -> None:
        invalid_values = (
            ("timeout_seconds", 0),
            ("timeout_seconds", -1),
            ("timeout_seconds", True),
            ("timeout_seconds", float("nan")),
            ("timeout_seconds", float("inf")),
            ("max_retries", -1),
            ("max_retries", True),
            ("max_retries", 1.0),
            ("retry_backoff_seconds", -0.1),
            ("retry_backoff_seconds", True),
            ("retry_backoff_seconds", float("nan")),
            ("retry_backoff_seconds", float("inf")),
            ("max_tokens", 0),
            ("max_tokens", -1),
            ("max_tokens", True),
            ("max_tokens", 1.0),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    self.make_transport(**{field: value})

    def test_missing_key_returns_safe_issue_before_file_or_http_access(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            transport = self.make_transport(
                api_key=None, api_key_env="ROBONORM_FIXTURE_KEY"
            )
            with (
                patch("pathlib.Path.read_bytes") as read_bytes,
                patch("robometanorm.vlm.request.urlopen") as urlopen,
            ):
                payload, issue = transport.request_json(
                    "private-system", "private-user", (Path("/private/image.png"),)
                )

        self.assertIsNone(payload)
        self.assertIsNotNone(issue)
        self.assertEqual(issue.code, "VLM_CONFIG_MISSING")
        self.assertEqual(issue.scope, "vlm")
        self.assertEqual(issue.severity, "review")
        self.assertEqual(issue.evidence, {"api_key_env": "ROBONORM_FIXTURE_KEY"})
        json.dumps(issue.evidence, allow_nan=False)
        read_bytes.assert_not_called()
        urlopen.assert_not_called()

    def test_explicit_key_even_empty_takes_priority_over_environment(self) -> None:
        with patch.dict(
            os.environ, {"ROBONORM_FIXTURE_KEY": "environment-secret"}, clear=True
        ):
            explicit = self.make_transport(
                api_key="explicit-secret", api_key_env="ROBONORM_FIXTURE_KEY"
            )
            empty = self.make_transport(
                api_key="", api_key_env="ROBONORM_FIXTURE_KEY"
            )
            with patch(
                "robometanorm.vlm.request.urlopen",
                return_value=_chat_response('{"ok": true}'),
            ) as urlopen:
                payload, issue = explicit.request_json("system", "user", ())
            empty_payload, empty_issue = empty.request_json("system", "user", ())

        self.assertEqual(payload, {"ok": True})
        self.assertIsNone(issue)
        self.assertEqual(
            urlopen.call_args.args[0].get_header("Authorization"),
            "Bearer explicit-secret",
        )
        self.assertIsNone(empty_payload)
        self.assertEqual(empty_issue.code, "VLM_CONFIG_MISSING")

    def test_environment_key_is_used_when_explicit_key_is_none(self) -> None:
        with patch.dict(
            os.environ, {"ROBONORM_FIXTURE_KEY": "environment-secret"}, clear=True
        ):
            transport = self.make_transport(
                api_key=None, api_key_env="ROBONORM_FIXTURE_KEY"
            )
            with patch(
                "robometanorm.vlm.request.urlopen",
                return_value=_chat_response('{"ok": true}'),
            ) as urlopen:
                payload, issue = transport.request_json("system", "user", ())

        self.assertEqual(payload, {"ok": True})
        self.assertIsNone(issue)
        self.assertEqual(
            urlopen.call_args.args[0].get_header("Authorization"),
            "Bearer environment-secret",
        )

    def test_environment_key_is_resolved_when_transport_is_constructed(self) -> None:
        with patch.dict(
            os.environ, {"ROBONORM_FIXTURE_KEY": "captured-secret"}, clear=True
        ):
            transport = self.make_transport(
                api_key=None, api_key_env="ROBONORM_FIXTURE_KEY"
            )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "robometanorm.vlm.request.urlopen",
                return_value=_chat_response('{"ok": true}'),
            ) as urlopen,
        ):
            payload, issue = transport.request_json("system", "user", ())

        self.assertEqual(payload, {"ok": True})
        self.assertIsNone(issue)
        self.assertEqual(
            urlopen.call_args.args[0].get_header("Authorization"),
            "Bearer captured-secret",
        )

    def test_blank_or_newline_credential_returns_config_issue_without_http(self) -> None:
        for api_key in ("   ", "prefix\rsecret", "prefix\nsecret"):
            with self.subTest(api_key=repr(api_key)):
                transport = self.make_transport(api_key=api_key)
                with patch("robometanorm.vlm.request.urlopen") as urlopen:
                    payload, issue = transport.request_json("system", "user", ())
                self.assertIsNone(payload)
                self.assertEqual(issue.code, "VLM_CONFIG_MISSING")
                urlopen.assert_not_called()

    def test_chat_request_has_standard_body_headers_and_ordered_mime_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suffixes = (".JPG", ".jpeg", ".PNG", ".WeBp", ".BMP")
            paths = tuple(root / f"frame-{index}{suffix}" for index, suffix in enumerate(suffixes))
            contents = tuple(f"image-{index}".encode("ascii") for index in range(len(paths)))
            for path, content in zip(paths, contents):
                path.write_bytes(content)
            before_hashes = tuple(
                hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
            )

            transport = self.make_transport(max_tokens=321, timeout_seconds=7.5)
            with patch(
                "robometanorm.vlm.request.urlopen",
                return_value=_chat_response('{"answer": "ok"}'),
            ) as urlopen:
                payload, issue = transport.request_json(
                    "system prompt", "user prompt", paths
                )

            after_hashes = tuple(
                hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
            )

        self.assertEqual(payload, {"answer": "ok"})
        self.assertIsNone(issue)
        self.assertEqual(before_hashes, after_hashes)
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 7.5})
        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(http_request.get_method(), "POST")
        self.assertEqual(http_request.get_header("Content-type"), "application/json")
        self.assertEqual(http_request.get_header("Authorization"), "Bearer fixture-key")
        body = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(
            set(body),
            {"model", "temperature", "max_tokens", "response_format", "messages"},
        )
        self.assertEqual(body["model"], "fixture-model")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 321)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["messages"][0], {"role": "system", "content": "system prompt"})
        user_content = body["messages"][1]["content"]
        self.assertEqual(user_content[0], {"type": "text", "text": "user prompt"})
        expected_mimes = (
            "image/jpeg",
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
        )
        self.assertEqual(
            user_content[1:],
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:"
                        + mime
                        + ";base64,"
                        + base64.b64encode(content).decode("ascii")
                    },
                }
                for mime, content in zip(expected_mimes, contents)
            ],
        )
        serialized_body = json.dumps(body)
        for path in paths:
            self.assertNotIn(str(path), serialized_body)

    def test_image_read_failure_is_local_safe_and_does_not_send_http(self) -> None:
        transport = self.make_transport()
        image_path = Path("/private/dataset/frame.png")
        with (
            patch(
                "pathlib.Path.read_bytes",
                side_effect=OSError("private filesystem detail"),
            ),
            patch("robometanorm.vlm.request.urlopen") as urlopen,
        ):
            payload, issue = transport.request_json(
                "private-system", "private-user", (image_path,)
            )

        self.assertIsNone(payload)
        self.assertEqual(issue.code, "VLM_IMAGE_READ_FAILED")
        self.assertEqual(
            issue.evidence, {"file_name": "frame.png", "error_type": "OSError"}
        )
        self.assertNotIn("private", issue.message.lower())
        self.assertNotIn(str(image_path), repr(issue))
        urlopen.assert_not_called()

    def test_unknown_image_extension_and_empty_image_are_not_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "frame.gif"
            empty = Path(directory) / "frame.png"
            unknown.write_bytes(b"GIF fixture")
            empty.write_bytes(b"")
            for path, expected_type in (
                (unknown, "UnsupportedImageType"),
                (empty, "EmptyImage"),
            ):
                with self.subTest(path=path.name):
                    transport = self.make_transport()
                    with patch("robometanorm.vlm.request.urlopen") as urlopen:
                        payload, issue = transport.request_json(
                            "system", "user", (path,)
                        )
                    self.assertIsNone(payload)
                    self.assertEqual(issue.code, "VLM_IMAGE_READ_FAILED")
                    self.assertEqual(
                        issue.evidence,
                        {"file_name": path.name, "error_type": expected_type},
                    )
                    urlopen.assert_not_called()

    def test_image_read_memory_error_propagates_without_http(self) -> None:
        transport = self.make_transport()
        with (
            patch("pathlib.Path.read_bytes", side_effect=MemoryError("memory")),
            patch("robometanorm.vlm.request.urlopen") as urlopen,
        ):
            with self.assertRaises(MemoryError):
                transport.request_json("system", "user", (Path("frame.png"),))
        urlopen.assert_not_called()

    def test_web_request_uses_responses_web_search_and_nested_output_text(self) -> None:
        transport = self.make_transport(max_tokens=654, timeout_seconds=9)
        with patch(
            "robometanorm.vlm.request.urlopen",
            return_value=_responses_response('{"found": true}'),
        ) as urlopen:
            payload, issue = transport.request_web_json("web system", "web user")

        self.assertEqual(payload, {"found": True})
        self.assertIsNone(issue)
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 9})
        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.full_url, "https://example.test/v1/responses")
        body = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(
            body,
            {
                "model": "fixture-model",
                "input": [
                    {"role": "system", "content": "web system"},
                    {"role": "user", "content": "web user"},
                ],
                "tools": [{"type": "web_search"}],
                "max_output_tokens": 654,
            },
        )

    def test_only_trailing_slashes_are_removed_from_base_url(self) -> None:
        for base_url, expected_base in (
            ("https://example.test/v1", "https://example.test/v1"),
            ("https://example.test/v1///", "https://example.test/v1"),
            (
                "https://example.test/v1/responses/",
                "https://example.test/v1/responses",
            ),
        ):
            with self.subTest(base_url=base_url):
                transport = self.make_transport(base_url=base_url)
                with patch(
                    "robometanorm.vlm.request.urlopen",
                    side_effect=[
                        _chat_response('{"chat": true}'),
                        _responses_response('{"web": true}'),
                    ],
                ) as urlopen:
                    transport.request_json("system", "user", ())
                    transport.request_web_json("system", "user")
                self.assertEqual(
                    urlopen.call_args_list[0].args[0].full_url,
                    expected_base + "/chat/completions",
                )
                self.assertEqual(
                    urlopen.call_args_list[1].args[0].full_url,
                    expected_base + "/responses",
                )

    def test_chat_parser_requires_one_whole_json_object(self) -> None:
        invalid_contents: tuple[object, ...] = (
            "not-json",
            '```json\n{"ok": true}\n```',
            'prefix {"ok": true}',
            '{"ok": true} suffix',
            '[{"ok": true}]',
            "null",
            '{"value": NaN}',
            '{"value": Infinity}',
            '{"value": 1e9999}',
            '{"one": 1}{"two": 2}',
            {"already": "decoded"},
        )
        for content in invalid_contents:
            with self.subTest(content=content):
                payload, issue, attempts = self.request_chat_content(content)
                self.assertIsNone(payload)
                self.assertEqual(issue.code, "VLM_RESPONSE_INVALID")
                self.assertEqual(issue.scope, "vlm")
                self.assertEqual(attempts, 1)
                json.dumps(issue.evidence, allow_nan=False)

    def test_outer_response_requires_utf8_strict_json_object(self) -> None:
        invalid_raw_values: tuple[bytes | object, ...] = (
            b"\xff",
            b"not-json",
            b"[]",
            b'{"choices": NaN}',
            "not-bytes",
        )
        for raw in invalid_raw_values:
            with self.subTest(raw=raw):
                transport = self.make_transport(max_retries=3)
                with patch(
                    "robometanorm.vlm.request.urlopen",
                    return_value=_HttpResponse(raw=raw),
                ) as urlopen:
                    payload, issue = transport.request_json("system", "user", ())
                self.assertIsNone(payload)
                self.assertEqual(issue.code, "VLM_RESPONSE_INVALID")
                self.assertEqual(urlopen.call_count, 1)

    def test_chat_response_structure_is_strict_and_not_retried(self) -> None:
        malformed_payloads = (
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": 3}}]},
        )
        for outer_payload in malformed_payloads:
            with self.subTest(outer_payload=outer_payload):
                transport = self.make_transport(max_retries=4)
                with patch(
                    "robometanorm.vlm.request.urlopen",
                    return_value=_HttpResponse(outer_payload),
                ) as urlopen:
                    payload, issue = transport.request_json("system", "user", ())
                self.assertIsNone(payload)
                self.assertEqual(issue.code, "VLM_RESPONSE_INVALID")
                self.assertEqual(urlopen.call_count, 1)

    def test_responses_parser_concatenates_ordered_output_text_parts_once(self) -> None:
        transport = self.make_transport()
        response = {
            "output": [
                {"type": "web_search_call", "status": "completed"},
                {
                    "content": [
                        {"type": "output_text", "text": '{"ordered":'},
                        {"type": "reasoning", "text": "ignored"},
                        {"type": "output_text", "text": " true"},
                    ]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "}"}],
                },
            ]
        }
        with patch(
            "robometanorm.vlm.request.urlopen",
            return_value=_HttpResponse(response),
        ):
            payload, issue = transport.request_web_json("system", "user")
        self.assertEqual(payload, {"ordered": True})
        self.assertIsNone(issue)

    def test_responses_parser_rejects_missing_or_invalid_text_structure(self) -> None:
        malformed_payloads = (
            {"output": []},
            {"output": [{"content": [{"type": "reasoning", "text": "ignored"}]}]},
            {
                "output": [
                    {
                        "type": "reasoning",
                        "content": [{"type": "output_text", "text": '{"bad": true}'}],
                    }
                ]
            },
            {
                "output_text": '{"top": true}',
                "output": [],
            },
            {"output": [{"content": [{"type": "output_text", "text": 7}]}]},
        )
        for outer_payload in malformed_payloads:
            with self.subTest(outer_payload=outer_payload):
                transport = self.make_transport(max_retries=3)
                with patch(
                    "robometanorm.vlm.request.urlopen",
                    return_value=_HttpResponse(outer_payload),
                ) as urlopen:
                    payload, issue = transport.request_web_json("system", "user")
                self.assertIsNone(payload)
                self.assertEqual(issue.code, "VLM_RESPONSE_INVALID")
                self.assertEqual(urlopen.call_count, 1)

    def test_http_retries_only_429_and_5xx(self) -> None:
        for status_code, expected_attempts in (
            (429, 3),
            (500, 3),
            (503, 3),
            (599, 3),
            (400, 1),
            (404, 1),
            (408, 1),
            (499, 1),
            (600, 1),
        ):
            with self.subTest(status_code=status_code):
                transport = self.make_transport(max_retries=2)
                with (
                    patch(
                        "robometanorm.vlm.request.urlopen",
                        side_effect=[_http_error(status_code) for _ in range(3)],
                    ) as urlopen,
                    patch("robometanorm.vlm.time.sleep") as sleep,
                ):
                    payload, issue = transport.request_json("system", "user", ())
                self.assertIsNone(payload)
                self.assertEqual(issue.code, "VLM_HTTP_ERROR")
                self.assertEqual(issue.evidence, {"status": status_code})
                self.assertEqual(urlopen.call_count, expected_attempts)
                sleep.assert_not_called()

    def test_network_errors_retry_to_max_retries_plus_one(self) -> None:
        network_errors = (
            URLError("private host detail"),
            TimeoutError("private timeout detail"),
            OSError("private socket detail"),
        )
        for network_error in network_errors:
            with self.subTest(error_type=type(network_error).__name__):
                transport = self.make_transport(max_retries=2)
                with patch(
                    "robometanorm.vlm.request.urlopen",
                    side_effect=[type(network_error)(str(network_error)) for _ in range(3)],
                ) as urlopen:
                    payload, issue = transport.request_json("system", "user", ())
                self.assertIsNone(payload)
                self.assertEqual(issue.code, "VLM_NETWORK_ERROR")
                self.assertEqual(
                    issue.evidence, {"error_type": type(network_error).__name__}
                )
                self.assertEqual(urlopen.call_count, 3)
                self.assertNotIn("private", repr(issue).lower())

    def test_incomplete_response_read_retries_as_network_failure(self) -> None:
        transport = self.make_transport(max_retries=2)
        responses = [
            _HttpResponse(read_error=IncompleteRead(b"partial", 10))
            for _ in range(3)
        ]
        with patch(
            "robometanorm.vlm.request.urlopen", side_effect=responses
        ) as urlopen:
            payload, issue = transport.request_json("system", "user", ())
        self.assertIsNone(payload)
        self.assertEqual(issue.code, "VLM_NETWORK_ERROR")
        self.assertEqual(issue.evidence, {"error_type": "IncompleteRead"})
        self.assertEqual(urlopen.call_count, 3)

    def test_exponential_backoff_occurs_only_before_another_attempt(self) -> None:
        transport = self.make_transport(
            max_retries=3, retry_backoff_seconds=0.25
        )
        with (
            patch(
                "robometanorm.vlm.request.urlopen",
                side_effect=[
                    URLError("first"),
                    TimeoutError("second"),
                    _chat_response('{"ok": true}'),
                ],
            ),
            patch("robometanorm.vlm.time.sleep") as sleep,
        ):
            payload, issue = transport.request_json("system", "user", ())
        self.assertEqual(payload, {"ok": True})
        self.assertIsNone(issue)
        self.assertEqual(sleep.call_args_list, [call(0.25), call(0.5)])

        terminal = self.make_transport(max_retries=0, retry_backoff_seconds=0.25)
        with (
            patch(
                "robometanorm.vlm.request.urlopen",
                side_effect=URLError("terminal"),
            ),
            patch("robometanorm.vlm.time.sleep") as terminal_sleep,
        ):
            terminal.request_json("system", "user", ())
        terminal_sleep.assert_not_called()

    def test_http_error_is_classified_before_url_error_without_reading_body(self) -> None:
        transport = self.make_transport(max_retries=0)
        error = _http_error(401, "authorization secret reason")
        with patch("robometanorm.vlm.request.urlopen", side_effect=error):
            payload, issue = transport.request_json("system", "user", ())
        self.assertIsNone(payload)
        self.assertEqual(issue.code, "VLM_HTTP_ERROR")
        self.assertEqual(issue.evidence, {"status": 401})
        self.assertNotIn("authorization", repr(issue).lower())
        self.assertNotIn("secret", repr(issue).lower())

    def test_http_error_body_is_closed_on_retry_and_terminal_failure(self) -> None:
        retry_body = _TrackedHttpBody()
        retry_error = HTTPError(
            "https://service.invalid/private",
            503,
            "private reason",
            {},
            retry_body,
        )
        transport = self.make_transport(max_retries=1)
        with patch(
            "robometanorm.vlm.request.urlopen",
            side_effect=[retry_error, _chat_response('{"ok": true}')],
        ):
            payload, issue = transport.request_json("system", "user", ())
        self.assertEqual(payload, {"ok": True})
        self.assertIsNone(issue)
        self.assertEqual(retry_body.close_calls, 1)

        terminal_body = _TrackedHttpBody()
        terminal_error = HTTPError(
            "https://service.invalid/private",
            400,
            "private reason",
            {},
            terminal_body,
        )
        with patch(
            "robometanorm.vlm.request.urlopen", side_effect=terminal_error
        ):
            payload, issue = transport.request_json("system", "user", ())
        self.assertIsNone(payload)
        self.assertEqual(issue.code, "VLM_HTTP_ERROR")
        self.assertEqual(terminal_body.close_calls, 1)

    def test_http_error_close_memory_error_propagates(self) -> None:
        body = _TrackedHttpBody(MemoryError("close"))
        http_error = HTTPError(
            "https://service.invalid/private", 400, "reason", {}, body
        )
        transport = self.make_transport(max_retries=0)
        with patch("robometanorm.vlm.request.urlopen", side_effect=http_error):
            with self.assertRaises(MemoryError):
                transport.request_json("system", "user", ())
        self.assertEqual(body.close_calls, 1)

    def test_http_error_close_oserror_is_redacted_and_keeps_http_result(self) -> None:
        body = _TrackedHttpBody(OSError("private close detail"))
        http_error = HTTPError(
            "https://service.invalid/private", 400, "reason", {}, body
        )
        transport = self.make_transport(max_retries=0)
        with patch("robometanorm.vlm.request.urlopen", side_effect=http_error):
            payload, issue = transport.request_json("system", "user", ())
        self.assertIsNone(payload)
        self.assertEqual(issue.code, "VLM_HTTP_ERROR")
        self.assertEqual(body.close_calls, 1)
        self.assertNotIn("private", repr(issue).lower())

    def test_response_and_context_failures_are_safely_classified(self) -> None:
        cases = (
            (_HttpResponse(read_error=UnicodeError("private decode")), "VLM_RESPONSE_INVALID", 1),
            (_HttpResponse(exit_error=ValueError("private context")), "VLM_RESPONSE_INVALID", 1),
            (_HttpResponse(enter_error=OSError("private connection")), "VLM_NETWORK_ERROR", 2),
        )
        for response, expected_code, expected_attempts in cases:
            with self.subTest(expected_code=expected_code):
                transport = self.make_transport(max_retries=1)
                with patch(
                    "robometanorm.vlm.request.urlopen", return_value=response
                ) as urlopen:
                    payload, issue = transport.request_json("system", "user", ())
                self.assertIsNone(payload)
                self.assertEqual(issue.code, expected_code)
                self.assertEqual(urlopen.call_count, expected_attempts)
                self.assertNotIn("private", repr(issue).lower())

    def test_unexpected_internal_attribute_error_propagates(self) -> None:
        transport = self.make_transport(max_retries=0)
        with (
            patch(
                "robometanorm.vlm.request.urlopen",
                return_value=_chat_response('{"ok": true}'),
            ),
            patch(
                "robometanorm.vlm._chat_content",
                side_effect=AttributeError("internal bug"),
            ),
        ):
            with self.assertRaises(AttributeError):
                transport.request_json("system", "user", ())

    def test_urlopen_and_response_read_memory_errors_propagate(self) -> None:
        transport = self.make_transport()
        with patch(
            "robometanorm.vlm.request.urlopen", side_effect=MemoryError("request")
        ):
            with self.assertRaises(MemoryError):
                transport.request_json("system", "user", ())

        with patch(
            "robometanorm.vlm.request.urlopen",
            return_value=_HttpResponse(read_error=MemoryError("read")),
        ):
            with self.assertRaises(MemoryError):
                transport.request_json("system", "user", ())

    def test_issues_and_transport_state_do_not_retain_sensitive_material(self) -> None:
        transport = self.make_transport(api_key="super-secret-key", max_retries=0)
        with patch(
            "robometanorm.vlm.request.urlopen",
            return_value=_HttpResponse(raw=b"private-server-body"),
        ):
            payload, issue = transport.request_json(
                "private-system-prompt", "private-user-prompt", ()
            )

        self.assertIsNone(payload)
        self.assertEqual(issue.code, "VLM_RESPONSE_INVALID")
        serialized_issue = repr(issue)
        state = getattr(transport, "__dict__", {})
        serialized_state = repr(state)
        for secret in (
            "super-secret-key",
            "private-system-prompt",
            "private-user-prompt",
            "private-server-body",
            "Authorization",
        ):
            self.assertNotIn(secret, serialized_issue)
            self.assertNotIn(secret, serialized_state)
        for forbidden_name in ("last_error", "api_key", "_api_key", "authorization", "_authorization"):
            self.assertNotIn(forbidden_name, state)
        self.assertFalse(any("cache" in name.lower() for name in state))


if __name__ == "__main__":
    unittest.main()
