"""相机 VLM 提示词、兼容客户端与语义解析。"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Protocol
from urllib import error, request

from robometanorm.camera.media import MediaInfo
from robometanorm.camera.naming import (
    BODY_PART_TOKENS,
    DIRECTION_TOKENS,
    EXTERNAL_DIRECTION_TOKENS,
    MOUNT_TYPES,
    ON_ROBOT_DIRECTION_TOKENS,
)


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""你是机器人数据集相机语义分类器。
根据字段元数据和视频采样图判断模态、安装类型、方位、本体部位、主摄像头与歧义。
left/right 表示安装位置而非画面物体位置；wrist 表示末端执行器附近且随机械臂运动。
严格返回一个 JSON 对象，不得输出最终标准字段名：
- modality 仅允许 rgb、depth、unknown。
- mount_type 仅允许 on_robot、external。
- on_robot 的 direction_tokens 仅允许 {", ".join(sorted(ON_ROBOT_DIRECTION_TOKENS | {"ego"}))}。
- external 的 direction_tokens 仅允许 {", ".join(sorted(EXTERNAL_DIRECTION_TOKENS))}；未知时返回空数组 []。
- body_part 仅允许 {", ".join(sorted(BODY_PART_TOKENS))}；未知时返回 null。
- external 的 body_part 必须为 null；ego 不得与其他方位或部位组合。
- is_primary、ambiguous、need_human_review 必须是布尔值。
- confidence 必须是 0 到 1 的数字，alternatives 必须是字符串数组。
证据不足时设置 ambiguous=true 或 need_human_review=true。"""


def build_vlm_prompt(
    *,
    dataset_name: str,
    robot_type: str | None,
    source_key: str,
    feature: Mapping[str, object],
    declared_fps: object,
    media: MediaInfo | None,
    other_camera_keys: Sequence[str],
) -> tuple[str, str]:
    """返回系统提示词和带媒体证据的用户提示词。"""
    codec = media.codec if media else "unknown"
    resolution = f"{media.width}x{media.height}" if media else "unknown"
    actual_fps = media.fps if media else "unknown"
    user_prompt = "\n".join(
        [
            "请识别以下相机的语义：",
            f"dataset_name: {dataset_name}",
            f"robot_type: {robot_type or 'unknown'}",
            f"source_key: {source_key}",
            f"dtype: {feature.get('dtype')}",
            f"shape: {feature.get('shape')}",
            f"declared_fps: {declared_fps}",
            f"actual_codec: {codec}",
            f"actual_resolution: {resolution}",
            f"actual_fps: {actual_fps}",
            f"other_camera_keys: {list(other_camera_keys)}",
            "采样图按时间顺序排列。",
            "请返回 modality、mount_type、direction_tokens、body_part、is_primary、confidence、ambiguous、alternatives、need_human_review。",
        ]
    )
    return SYSTEM_PROMPT, user_prompt


@dataclass(frozen=True)
class CameraSemantics:
    """VLM 返回的语义属性，不包含最终字段名。"""

    modality: str
    mount_type: str | None
    direction_tokens: tuple[str, ...]
    body_part: str | None
    is_primary: bool
    confidence: float
    ambiguous: bool
    alternatives: tuple[str, ...]
    need_human_review: bool


class CameraSemanticsValidationError(ValueError):
    """携带非法字段和值的相机 VLM schema 错误。"""

    def __init__(self, message: str, *, field: str, value: object) -> None:
        super().__init__(message)
        self.field = field
        self.value = value


class VlmClassifier(Protocol):
    """可替换的相机语义分类器协议。"""

    def classify(
        self, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
    ) -> CameraSemantics | None:
        """返回语义，无法分类时返回空。"""


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
        self.api_key_env = api_key_env or (
            "DASHSCOPE_API_KEY"
            if "dashscope" in self.endpoint.lower()
            else "OPENAI_API_KEY"
        )
        self.api_key = api_key or os.getenv(self.api_key_env, "")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_tokens = max_tokens
        self.last_error: str | None = None
        self.last_error_code: str | None = None
        self.last_error_evidence: dict[str, object] = {}

    def classify(
        self, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
    ) -> CameraSemantics | None:
        """发送文本和抽帧图；仅对临时网络或服务端错误重试。"""
        response_payload = self.request_json(system_prompt, user_prompt, image_paths)
        if response_payload is None:
            return None
        try:
            return parse_vlm_semantics(response_payload)
        except CameraSemanticsValidationError as response_error:
            return self._fail(
                f"相机 VLM 语义不合法: {response_error}",
                code="VLM_SEMANTICS_INVALID",
                evidence={
                    "field": response_error.field,
                    "value": response_error.value,
                },
            )

    def request_json(
        self, system_prompt: str, user_prompt: str, image_paths: Sequence[Path]
    ) -> Mapping[str, object] | None:
        """发送通用多模态请求并返回未绑定业务 schema 的 JSON 对象。"""
        if not self._require_api_key():
            return None
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
        endpoint = self.endpoint.lower()
        if "dashscope" in endpoint:
            payload["enable_thinking"] = False
            payload["response_format"] = {"type": "json_object"}
            # DashScope JSON Mode 不限制输出 token，避免截断未闭合的 JSON。
            payload.pop("max_tokens", None)
        elif "deepseek" in endpoint:
            payload["enable_thinking"] = False
        return self._request_with_retry(payload)

    def request_web_json(
        self, system_prompt: str, user_prompt: str
    ) -> Mapping[str, object] | None:
        """通过 Responses API 的内置 web_search 返回结构化文本。"""
        if not self._require_api_key():
            return None
        payload: dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": [{"type": "web_search"}],
        }
        return self._post_json_with_retry(
            self._endpoint_url("responses"), payload, _responses_json_content
        )

    def _require_api_key(self) -> bool:
        """缺少认证时本地失败，避免发送无意义的外部请求。"""
        if self.api_key:
            return True
        api_key_env = self.api_key_env or "DASHSCOPE_API_KEY"
        self._fail(
            f"未配置 VLM API Key 环境变量 {api_key_env}",
            code="VLM_CONFIG_MISSING",
            evidence={"api_key_env": api_key_env},
        )
        return False

    def _request_with_retry(self, payload: Mapping[str, object]) -> Mapping[str, object] | None:
        """请求 Chat Completions 并解析消息内的 JSON。"""
        return self._post_json_with_retry(
            self._endpoint_url("chat/completions"), payload, _chat_json_content
        )

    def _post_json_with_retry(
        self,
        url: str,
        payload: Mapping[str, object],
        content_parser: Callable[[Mapping[str, object]], Mapping[str, object]],
    ) -> Mapping[str, object] | None:
        """复用认证、重试与错误脱敏发送 JSON 请求。"""
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
                if not isinstance(response_payload, Mapping):
                    raise ValueError("VLM 响应顶层必须是对象")
                parsed_content = content_parser(response_payload)
                self.last_error = None
                self.last_error_code = None
                self.last_error_evidence = {}
                return parsed_content
            except error.HTTPError as request_error:
                if not self._should_retry(request_error.code, attempt):
                    return self._fail(f"HTTP {request_error.code}: {request_error.reason}")
            except (error.URLError, TimeoutError, OSError) as request_error:
                if attempt >= self.max_retries:
                    return self._fail(f"网络请求失败: {request_error}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as response_error:
                return self._fail(f"VLM 响应格式无效: {response_error}")
            self._backoff(attempt)
        return self._fail("VLM 请求重试耗尽")

    def _endpoint_url(self, path: str) -> str:
        """兼容传入 base URL 或完整 API 路径。"""
        base_url = self.endpoint
        for suffix in ("/chat/completions", "/responses"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        return f"{base_url}/{path}"

    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """仅重试限流和服务端异常，避免重复发送无效请求。"""
        return attempt < self.max_retries and (status_code == 429 or 500 <= status_code < 600)

    def _backoff(self, attempt: int) -> None:
        """按 1、2、4… 倍数等待；零等待适合测试和本地调试。"""
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay:
            time.sleep(delay)

    def _fail(
        self,
        message: str,
        *,
        code: str = "VLM_UNAVAILABLE",
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        """保留失败原因，供调用方记录为人工复核证据。"""
        self.last_error = message
        self.last_error_code = code
        self.last_error_evidence = dict(evidence or {})
        logger.warning("VLM 分类失败（模型 %s）: %s", self.model, message)
        return None


def parse_vlm_semantics(payload: Mapping[str, object]) -> CameraSemantics:
    """校验模型输出仅包含 P1 允许的语义属性。"""
    if "target_key" in payload:
        raise CameraSemanticsValidationError(
            "VLM 不得输出最终字段名",
            field="target_key",
            value=payload.get("target_key"),
        )
    modality = payload.get("modality")
    mount_type = payload.get("mount_type")
    direction_tokens = payload.get("direction_tokens")
    body_part = payload.get("body_part")
    confidence = payload.get("confidence")
    if not isinstance(modality, str) or modality not in {"rgb", "depth", "unknown"}:
        raise CameraSemanticsValidationError(
            "VLM 模态不合法", field="modality", value=modality
        )
    if mount_type not in MOUNT_TYPES:
        raise CameraSemanticsValidationError(
            "VLM 安装类型不合法", field="mount_type", value=mount_type
        )
    if direction_tokens == ["unknown"]:
        direction_tokens = []
    elif not isinstance(direction_tokens, list) or not all(
        isinstance(token, str) and token in DIRECTION_TOKENS
        for token in direction_tokens
    ):
        raise CameraSemanticsValidationError(
            "VLM 方位词不合法",
            field="direction_tokens",
            value=direction_tokens,
        )
    if body_part == "unknown":
        body_part = None
    if body_part is not None and (
        not isinstance(body_part, str) or body_part not in BODY_PART_TOKENS
    ):
        raise CameraSemanticsValidationError(
            "VLM 本体部位不合法", field="body_part", value=body_part
        )
    allowed_directions = (
        ON_ROBOT_DIRECTION_TOKENS | {"ego"}
        if mount_type == "on_robot"
        else EXTERNAL_DIRECTION_TOKENS
    )
    invalid_direction = next(
        (token for token in direction_tokens if token not in allowed_directions),
        None,
    )
    if invalid_direction is not None or (
        "ego" in direction_tokens
        and (len(direction_tokens) != 1 or body_part is not None)
    ):
        raise CameraSemanticsValidationError(
            "VLM 方位词与安装类型不匹配",
            field="direction_tokens",
            value=direction_tokens,
        )
    if mount_type == "external" and body_part is not None:
        raise CameraSemanticsValidationError(
            "外部相机不得包含本体部位", field="body_part", value=body_part
        )
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise CameraSemanticsValidationError(
            "VLM 置信度不合法", field="confidence", value=confidence
        )
    boolean_fields = ("is_primary", "ambiguous", "need_human_review")
    invalid_boolean = next(
        (field for field in boolean_fields if not isinstance(payload.get(field), bool)),
        None,
    )
    if invalid_boolean is not None:
        raise CameraSemanticsValidationError(
            "VLM 布尔字段不合法",
            field=invalid_boolean,
            value=payload.get(invalid_boolean),
        )
    alternatives = payload.get("alternatives")
    if not isinstance(alternatives, list) or not all(
        isinstance(item, str) for item in alternatives
    ):
        raise CameraSemanticsValidationError(
            "VLM 备选项不合法", field="alternatives", value=alternatives
        )
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


def _chat_json_content(payload: Mapping[str, object]) -> Mapping[str, object]:
    """提取 Chat Completions 消息文本中的 JSON。"""
    choices = payload["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("VLM 响应缺少 choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("VLM choice 不是对象")
    message = first_choice["message"]
    if not isinstance(message, Mapping):
        raise ValueError("VLM message 不是对象")
    return _load_json_content(message["content"])


def _responses_json_content(payload: Mapping[str, object]) -> Mapping[str, object]:
    """提取 Responses API 最终 output_text 中的 JSON。"""
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses API 响应缺少 output")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, Mapping)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    if not texts:
        raise ValueError("Responses API 响应缺少 output_text")
    return _load_json_content("\n".join(texts))
