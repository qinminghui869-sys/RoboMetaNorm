"""按机器人型号联网查询并校验标准相机配置。"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Protocol

from robometanorm.camera.models import (
    CameraMount,
    RobotCameraTopology,
    TopologyRejection,
)
from robometanorm.camera.naming import MOUNT_TYPES, build_camera_key


TOPOLOGY_SYSTEM_PROMPT = """你是机器人硬件相机拓扑研究助手。
请联网检索机器人厂商、产品手册和可靠技术资料，判断指定型号的标准相机配置。
报告机器人本体集成相机，以及标准平台配置中的固定外部相机；不要报告数据采集现场临时架设的任意相机。
严格返回 JSON，不得返回数据集最终字段名或 target_key。
camera_mounts 每项只含 mount_type、direction_tokens、body_part；mount_type 只能为 on_robot 或 external。
external 必须且只能给出一个主方位词，body_part 必须为 null。
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

        if topology.partial:
            rejected_mounts = [
                {
                    "field": rejection.field,
                    "value": rejection.value,
                    "reason": rejection.reason,
                }
                for rejection in topology.rejected_mounts
            ]
            self._set_error(
                f"机器人相机拓扑仅部分有效，拒绝 {len(rejected_mounts)} 个槽位",
                "ROBOT_TOPOLOGY_PARTIAL",
                {"rejected_mounts": rejected_mounts},
            )
        else:
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
    """校验标准相机配置；保留合法槽位并记录局部非法证据。"""
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
    rejected_mounts: list[TopologyRejection] = []
    for index, value in enumerate(mounts_payload):
        field_prefix = f"camera_mounts[{index}]"
        try:
            mount = _parse_camera_mount(value, field_prefix)
        except RobotCameraTopologyValidationError as topology_error:
            rejected_mounts.append(
                TopologyRejection(
                    field=topology_error.field,
                    value=topology_error.value,
                    reason=str(topology_error),
                )
            )
            continue
        if mount not in mounts:
            mounts.append(mount)

    return RobotCameraTopology(
        robot_id=robot_id,
        camera_mounts=tuple(mounts),
        confidence=float(confidence),
        ambiguous=ambiguous,
        rejected_mounts=tuple(rejected_mounts),
    )


def _parse_camera_mount(value: object, field_prefix: str) -> CameraMount:
    """解析单个槽位，使局部失败不会丢弃其余有效拓扑。"""
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
    mount_type = value.get("mount_type")
    if mount_type not in MOUNT_TYPES:
        raise RobotCameraTopologyValidationError(
            "mount_type 只能为 on_robot 或 external",
            field=f"{field_prefix}.mount_type",
            value=mount_type,
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
    mount = CameraMount(str(mount_type), tuple(directions), body_part)
    if build_camera_key(
        mount.mount_type, mount.direction_tokens, mount.body_part, "rgb"
    ) is None:
        raise RobotCameraTopologyValidationError(
            "相机槽位不符合内置命名规范", field=field_prefix, value=dict(value)
        )
    return mount
