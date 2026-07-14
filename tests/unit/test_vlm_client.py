"""OpenAI-compatible VLM 客户端韧性测试。"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.vlm import OpenAICompatibleVlmClassifier


class _Response:
    """最小 HTTP 响应替身。"""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class OpenAICompatibleVlmClientTest(unittest.TestCase):
    """验证参考客户端迁移后的连接韧性。"""

    def test_uses_explicit_key_then_configured_and_standard_environment_keys(self) -> None:
        with patch.dict(
            os.environ,
            {"P1_VLM_KEY": "configured", "DASHSCOPE_API_KEY": "dashscope", "OPENAI_API_KEY": "openai"},
            clear=True,
        ):
            explicit = OpenAICompatibleVlmClassifier(
                "http://localhost/v1", "test", "explicit", api_key_env="P1_VLM_KEY"
            )
            configured = OpenAICompatibleVlmClassifier(
                "http://localhost/v1", "test", None, api_key_env="P1_VLM_KEY"
            )
            fallback = OpenAICompatibleVlmClassifier(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "test",
                None,
            )

        self.assertEqual(explicit.api_key, "explicit")
        self.assertEqual(configured.api_key, "configured")
        self.assertEqual(fallback.api_key, "dashscope")

    def test_dashscope_endpoint_never_falls_back_to_openai_api_key(self) -> None:
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "openai-secret"}, clear=True
        ):
            dashscope = OpenAICompatibleVlmClassifier(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen3.7-plus",
                None,
            )
            openai = OpenAICompatibleVlmClassifier(
                "https://api.openai.com/v1", "test-model", None
            )

        self.assertEqual(dashscope.api_key, "")
        self.assertEqual(openai.api_key, "openai-secret")

    def test_retries_transient_http_error_and_extracts_embedded_json(self) -> None:
        response = _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "识别结果：\n```json\n"
                            '{"modality":"rgb","mount_type":"on_robot",'
                            '"direction_tokens":["left"],"body_part":"wrist",'
                            '"is_primary":false,"confidence":0.94,"ambiguous":false,'
                            '"alternatives":[],"need_human_review":false}\n```'
                        }
                    }
                ]
            }
        )
        transient_error = HTTPError("https://example.test", 503, "unavailable", {}, None)
        client = OpenAICompatibleVlmClassifier(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "test-vlm",
            "test-key",
            max_retries=2,
            retry_backoff_seconds=0,
            max_tokens=1024,
        )

        with patch(
            "robometanorm.camera.vlm.request.urlopen",
            side_effect=[transient_error, response],
        ) as urlopen:
            semantics = client.classify("system", "user", ())

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(semantics.body_part, "wrist")
        request_body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertFalse(request_body["enable_thinking"])
        self.assertEqual(
            request_body.get("response_format"), {"type": "json_object"}
        )
        self.assertNotIn("max_tokens", request_body)

    def test_keeps_standard_token_limit_for_non_dashscope_endpoint(self) -> None:
        response = _Response(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        )
        client = OpenAICompatibleVlmClassifier(
            "https://api.openai.com/v1",
            "test-vlm",
            "test-key",
            max_tokens=2048,
        )

        with patch(
            "robometanorm.camera.vlm.request.urlopen",
            return_value=response,
        ) as urlopen:
            payload = client.request_json("return JSON", "user", ())

        self.assertEqual(payload, {"ok": True})
        request_body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(request_body["max_tokens"], 2048)
        self.assertNotIn("enable_thinking", request_body)
        self.assertNotIn("response_format", request_body)

    def test_exposes_validated_json_transport_for_non_camera_semantics(self) -> None:
        response = _Response(
            {"choices": [{"message": {"content": "前缀 {\"semantic_type\": \"unknown\"} 后缀"}}]}
        )
        client = OpenAICompatibleVlmClassifier("http://localhost/v1", "test-vlm", "test-key")

        with patch("robometanorm.camera.vlm.request.urlopen", return_value=response):
            payload = client.request_json("system", "user", ())

        self.assertEqual(payload, {"semantic_type": "unknown"})

    def test_exposes_camera_schema_error_code_and_invalid_field(self) -> None:
        client = OpenAICompatibleVlmClassifier(
            "http://localhost/v1", "test-vlm", "test-key"
        )
        response_payload = {
            "modality": "rgb",
            "mount_type": "external",
            "direction_tokens": ["high"],
            "body_part": None,
            "is_primary": False,
            "confidence": 0.9,
            "ambiguous": False,
            "alternatives": [],
            "need_human_review": False,
        }

        with (
            patch.object(client, "request_json", return_value=response_payload),
            patch("robometanorm.camera.vlm.logger.warning"),
        ):
            semantics = client.classify("system", "user", ())

        self.assertIsNone(semantics)
        self.assertEqual(client.last_error_code, "VLM_SEMANTICS_INVALID")
        self.assertEqual(
            client.last_error_evidence,
            {"field": "direction_tokens", "value": ["high"]},
        )

    def test_missing_api_key_fails_without_sending_a_request(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAICompatibleVlmClassifier(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen3.7-plus",
                None,
                api_key_env="DASHSCOPE_API_KEY",
            )

        with (
            patch("robometanorm.camera.vlm.request.urlopen") as urlopen,
            patch("robometanorm.camera.vlm.logger.warning"),
        ):
            payload = client.request_json("system", "user", ())

        self.assertIsNone(payload)
        self.assertEqual(client.last_error_code, "VLM_CONFIG_MISSING")
        self.assertEqual(
            client.last_error_evidence,
            {"api_key_env": "DASHSCOPE_API_KEY"},
        )
        urlopen.assert_not_called()

    def test_requests_web_search_through_the_responses_api(self) -> None:
        response = _Response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '结果：{"robot_id":"airbot_mmk2","camera_mounts":[],"confidence":0.4,"ambiguous":true}',
                            }
                        ],
                    }
                ]
            }
        )
        client = OpenAICompatibleVlmClassifier(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen3.7-plus",
            "test-key",
        )

        with patch(
            "robometanorm.camera.vlm.request.urlopen", return_value=response
        ) as urlopen:
            payload = client.request_web_json("system", "user")

        self.assertEqual(payload["robot_id"], "airbot_mmk2")
        http_request = urlopen.call_args.args[0]
        self.assertEqual(
            http_request.full_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/responses",
        )
        request_body = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(request_body["tools"], [{"type": "web_search"}])
        self.assertEqual(request_body["model"], "qwen3.7-plus")


if __name__ == "__main__":
    unittest.main()
