"""OpenAI-compatible VLM 语义分类接口。"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Protocol
from urllib import error, request

from robometanorm.camera.name_builder import BODY_PART_TOKENS, EXTERNAL_DIRECTION_TOKENS


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraSemantics:
    """VLM 返回的语义属性，不包含最终字段名。"""

    modality: str
    mount_type: str
    direction_tokens: tuple[str, ...]
    body_part: str | None
    is_primary: bool
    confidence: float
    ambiguous: bool
    alternatives: tuple[str, ...]
    need_human_review: bool


class VlmClassifier(Protocol):
    """可替换的相机语义分类器协议。"""

    def classify(
        self, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
    ) -> CameraSemantics | None:
        """返回语义，无法分类时返回空。"""


class DisabledVlmClassifier:
    """默认禁用 VLM，未知字段仅进入人工复核。"""

    def classify(
        self, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
    ) -> CameraSemantics | None:
        return None


class OpenAICompatibleVlmClassifier:
    """使用兼容 Chat Completions 的 HTTP 接口请求相机语义。"""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None,
        *,
        api_key_env: str | None = None,
        timeout_seconds: int = 120,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        max_tokens: int = 1024,
    ):
        """保存连接参数，并按显式值、指定环境变量、通用环境变量取密钥。"""
        if timeout_seconds <= 0 or max_retries < 0 or retry_backoff_seconds < 0 or max_tokens <= 0:
            raise ValueError("VLM 连接参数必须为正数，且重试次数不能为负数")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = (
            api_key
            or (os.getenv(api_key_env) if api_key_env else None)
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_tokens = max_tokens
        self.last_error: str | None = None

    def classify(
        self, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
    ) -> CameraSemantics | None:
        """发送文本和抽帧图；仅对临时网络或服务端错误重试。"""
        response_payload = self.request_json(system_prompt, user_prompt, image_paths)
        if response_payload is None:
            return None
        try:
            return parse_vlm_semantics(response_payload)
        except ValueError as response_error:
            return self._fail(f"相机 VLM 语义不合法: {response_error}")

    def request_json(
        self, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
    ) -> Mapping[str, object] | None:
        """发送通用多模态请求并返回未绑定业务 schema 的 JSON 对象。"""
        content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        # DashScope/DeepSeek 兼容端点沿用参考客户端的关闭思考配置。
        if any(provider in self.endpoint.lower() for provider in ("dashscope", "deepseek")):
            payload["enable_thinking"] = False
        return self._request_with_retry(payload)

    def _request_with_retry(self, payload: Mapping[str, object]) -> Mapping[str, object] | None:
        """执行请求并对 429、5xx 与网络异常进行指数退避。"""
        url = self.endpoint if self.endpoint.endswith("/chat/completions") else f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.max_retries + 1):
            http_request = request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                content_value = response_payload["choices"][0]["message"]["content"]
                self.last_error = None
                return _load_json_content(content_value)
            except error.HTTPError as request_error:
                if not self._should_retry(request_error.code, attempt):
                    return self._fail(f"HTTP {request_error.code}: {request_error.reason}")
            except (error.URLError, TimeoutError, OSError) as request_error:
                if attempt >= self.max_retries:
                    return self._fail(f"网络请求失败: {request_error}")
            except (KeyError, ValueError, json.JSONDecodeError) as response_error:
                return self._fail(f"VLM 响应格式无效: {response_error}")
            self._backoff(attempt)
        return self._fail("VLM 请求重试耗尽")

    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """仅重试限流和服务端异常，避免重复发送无效请求。"""
        return attempt < self.max_retries and (status_code == 429 or 500 <= status_code < 600)

    def _backoff(self, attempt: int) -> None:
        """按 1、2、4… 倍数等待；零等待适合测试和本地调试。"""
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay:
            time.sleep(delay)

    def _fail(self, message: str) -> None:
        """保留失败原因，供调用方记录为人工复核证据。"""
        self.last_error = message
        logger.warning("VLM 分类失败（模型 %s）: %s", self.model, message)
        return None


def parse_vlm_semantics(payload: Mapping[str, object]) -> CameraSemantics:
    """校验模型输出仅包含 P1 允许的语义属性。"""
    if "target_key" in payload:
        raise ValueError("VLM 不得输出最终字段名")
    modality = payload.get("modality")
    mount_type = payload.get("mount_type")
    direction_tokens = payload.get("direction_tokens")
    body_part = payload.get("body_part")
    confidence = payload.get("confidence")
    if modality not in {"rgb", "depth", "unknown"}:
        raise ValueError("VLM 模态不合法")
    if mount_type not in {"body", "external", "unknown"}:
        raise ValueError("VLM 安装类型不合法")
    if not isinstance(direction_tokens, list) or not all(
        isinstance(token, str) and token in EXTERNAL_DIRECTION_TOKENS
        for token in direction_tokens
    ):
        raise ValueError("VLM 方位词不合法")
    if body_part not in BODY_PART_TOKENS and body_part is not None:
        raise ValueError("VLM 本体部位不合法")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("VLM 置信度不合法")
    boolean_fields = ("is_primary", "ambiguous", "need_human_review")
    if not all(isinstance(payload.get(field), bool) for field in boolean_fields):
        raise ValueError("VLM 布尔字段不合法")
    alternatives = payload.get("alternatives")
    if not isinstance(alternatives, list) or not all(isinstance(item, str) for item in alternatives):
        raise ValueError("VLM 备选项不合法")
    return CameraSemantics(
        modality=modality,
        mount_type=mount_type,
        direction_tokens=tuple(direction_tokens),
        body_part=body_part,
        is_primary=payload["is_primary"],
        confidence=float(confidence),
        ambiguous=payload["ambiguous"],
        alternatives=tuple(alternatives),
        need_human_review=payload["need_human_review"],
    )


def _load_json_content(content: object) -> Mapping[str, object]:
    """解析纯文本、Markdown 围栏或嵌入文本中的 JSON 对象。"""
    if not isinstance(content, str):
        raise ValueError("VLM 返回内容不是文本")
    cleaned = content.strip()
    fenced_json = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fenced_json:
        cleaned = fenced_json.group(1)
    elif not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("VLM 返回顶层必须是对象")
    return parsed
