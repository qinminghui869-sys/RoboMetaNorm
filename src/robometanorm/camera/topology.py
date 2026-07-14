"""按机器人型号联网查询并校验本体相机拓扑。"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Protocol

from robometanorm.camera.models import CameraMount, RobotCameraTopology
from robometanorm.camera.naming import build_camera_key


TOPOLOGY_SYSTEM_PROMPT = """你是机器人硬件相机拓扑研究助手。
请联网检索机器人厂商、产品手册和可靠技术资料，判断指定型号出厂或本体集成的相机安装位。
只报告机器人本体相机，不报告数据采集现场额外架设的外部相机。
严格返回 JSON，不得返回数据集最终字段名或 target_key。
camera_mounts 每项只含 mount_type、direction_tokens、body_part；mount_type 必须为 on_robot。
证据冲突或型号不明确时设置 ambiguous=true 并降低 confidence。"""


class RobotCameraTopologyValidationError(ValueError):
    """携带非法字段和值的拓扑 schema 错误。"""

    def __init__(self, message: str, *, field: str, value: object) -> None:
        super().__init__(message)
        self.field = field
        self.value = value


class _WebJsonClient(Protocol):
    endpoint: str
    model: str
    last_error: str | None
    last_error_code: str | None
    last_error_evidence: dict[str, object]

    def request_web_json(
        self, system_prompt: str, user_prompt: str
    ) -> Mapping[str, object] | None:
        """联网搜索并返回 JSON 对象。"""


class RobotCameraTopologyResolver(Protocol):
    """可替换的机器人相机拓扑解析器。"""

    last_error: str | None
    last_error_code: str | None
    last_error_evidence: dict[str, object]

    def resolve(self, robot_id: str) -> RobotCameraTopology | None:
        """返回已校验拓扑，无法判断时返回空。"""


class OpenAICompatibleRobotCameraTopologyResolver:
    """使用共享 VLM 客户端的 Responses API 联网查询拓扑。"""

    def __init__(self, client: _WebJsonClient) -> None:
        self.client = client
        self._cache: dict[tuple[str, str, str], RobotCameraTopology | None] = {}
        self._cached_errors: dict[
            tuple[str, str, str], tuple[str | None, str | None, dict[str, object]]
        ] = {}
        self.last_error: str | None = None
        self.last_error_code: str | None = None
        self.last_error_evidence: dict[str, object] = {}

    def resolve(self, robot_id: str) -> RobotCameraTopology | None:
        """同一连接与型号在单次运行中只联网一次。"""
        cache_key = (self.client.endpoint, self.client.model, robot_id)
        if cache_key in self._cache:
            self._restore_cached_error(cache_key)
            return self._cache[cache_key]

        user_prompt = "\n".join(
            [
                "请查询以下机器人型号的本体相机拓扑：",
                f"robot_id: {json.dumps(robot_id, ensure_ascii=False)}",
                "返回 robot_id、camera_mounts、confidence、ambiguous。",
            ]
        )
        payload = self.client.request_web_json(TOPOLOGY_SYSTEM_PROMPT, user_prompt)
        if payload is None:
            self._set_error(
                self.client.last_error,
                self.client.last_error_code or "ROBOT_TOPOLOGY_UNAVAILABLE",
                self.client.last_error_evidence,
            )
            return self._cache_result(cache_key, None)
        try:
            topology = parse_robot_camera_topology(payload)
            if topology.robot_id != robot_id:
                raise RobotCameraTopologyValidationError(
                    "联网拓扑返回了不同机器人型号",
                    field="robot_id",
                    value=topology.robot_id,
                )
        except RobotCameraTopologyValidationError as topology_error:
            self._set_error(
                f"机器人相机拓扑不合法: {topology_error}",
                "ROBOT_TOPOLOGY_INVALID",
                {"field": topology_error.field, "value": topology_error.value},
            )
            return self._cache_result(cache_key, None)

        self._set_error(None, None, {})
        return self._cache_result(cache_key, topology)

    def _cache_result(
        self,
        cache_key: tuple[str, str, str],
        result: RobotCameraTopology | None,
    ) -> RobotCameraTopology | None:
        self._cache[cache_key] = result
        self._cached_errors[cache_key] = (
            self.last_error,
            self.last_error_code,
            dict(self.last_error_evidence),
        )
        return result

    def _restore_cached_error(self, cache_key: tuple[str, str, str]) -> None:
        message, code, evidence = self._cached_errors[cache_key]
        self._set_error(message, code, evidence)

    def _set_error(
        self,
        message: str | None,
        code: str | None,
        evidence: Mapping[str, object],
    ) -> None:
        self.last_error = message
        self.last_error_code = code
        self.last_error_evidence = dict(evidence)


def parse_robot_camera_topology(
    payload: Mapping[str, object],
) -> RobotCameraTopology:
    """把联网结果限制为内置规范允许的机器人本体相机槽位。"""
    if "target_key" in payload:
        raise RobotCameraTopologyValidationError(
            "联网结果不得包含最终字段名",
            field="target_key",
            value=payload.get("target_key"),
        )
    robot_id = payload.get("robot_id")
    mounts_payload = payload.get("camera_mounts")
    confidence = payload.get("confidence")
    ambiguous = payload.get("ambiguous")
    if not isinstance(robot_id, str) or not robot_id.strip():
        raise RobotCameraTopologyValidationError(
            "robot_id 不合法", field="robot_id", value=robot_id
        )
    if not isinstance(mounts_payload, list):
        raise RobotCameraTopologyValidationError(
            "camera_mounts 必须是数组",
            field="camera_mounts",
            value=mounts_payload,
        )
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise RobotCameraTopologyValidationError(
            "confidence 不合法", field="confidence", value=confidence
        )
    if not isinstance(ambiguous, bool):
        raise RobotCameraTopologyValidationError(
            "ambiguous 不合法", field="ambiguous", value=ambiguous
        )

    mounts: list[CameraMount] = []
    for index, value in enumerate(mounts_payload):
        field_prefix = f"camera_mounts[{index}]"
        if not isinstance(value, Mapping):
            raise RobotCameraTopologyValidationError(
                "相机槽位必须是对象", field=field_prefix, value=value
            )
        if "target_key" in value:
            raise RobotCameraTopologyValidationError(
                "相机槽位不得包含最终字段名",
                field=f"{field_prefix}.target_key",
                value=value.get("target_key"),
            )
        if value.get("mount_type") != "on_robot":
            raise RobotCameraTopologyValidationError(
                "机器人拓扑只能包含本体相机",
                field=f"{field_prefix}.mount_type",
                value=value.get("mount_type"),
            )
        directions = value.get("direction_tokens")
        body_part = value.get("body_part")
        if not isinstance(directions, list) or not all(
            isinstance(token, str) for token in directions
        ):
            raise RobotCameraTopologyValidationError(
                "direction_tokens 必须是字符串数组",
                field=f"{field_prefix}.direction_tokens",
                value=directions,
            )
        if body_part is not None and not isinstance(body_part, str):
            raise RobotCameraTopologyValidationError(
                "body_part 必须是字符串或 null",
                field=f"{field_prefix}.body_part",
                value=body_part,
            )
        mount = CameraMount("on_robot", tuple(directions), body_part)
        if build_camera_key(
            mount.mount_type, mount.direction_tokens, mount.body_part, "rgb"
        ) is None:
            raise RobotCameraTopologyValidationError(
                "相机槽位不符合内置命名规范", field=field_prefix, value=dict(value)
            )
        if mount not in mounts:
            mounts.append(mount)

    return RobotCameraTopology(
        robot_id=robot_id,
        camera_mounts=tuple(mounts),
        confidence=float(confidence),
        ambiguous=ambiguous,
    )
