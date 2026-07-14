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
                "http://localhost/v1", "test", None, api_key_env="MISSING_KEY"
            )

        self.assertEqual(explicit.api_key, "explicit")
        self.assertEqual(configured.api_key, "configured")
        self.assertEqual(fallback.api_key, "dashscope")

    def test_retries_transient_http_error_and_extracts_embedded_json(self) -> None:
        response = _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "识别结果：\n```json\n"
                            '{"modality":"rgb","mount_type":"body",'
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


if __name__ == "__main__":
    unittest.main()
