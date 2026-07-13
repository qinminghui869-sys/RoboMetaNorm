"""P2 机器字段 VLM 语义协议与规则裁决。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from robometanorm.machine.prompt_builder import build_machine_prompt


SEMANTIC_TYPES = {
    "arm_joint", "hand_joint", "gripper_open", "eef_position", "eef_rotation_euler",
    "orientation_quaternion", "head_joint", "head_position", "head_rotation_euler",
    "head_orientation_quaternion", "torso_joint", "neck_joint", "base_position",
    "base_rotation_euler", "skeleton", "hand_keypoints", "unknown",
}
REPRESENTATIONS = {
    "scalar", "joint_vector", "position_xyz", "euler_xyz", "quaternion_xyzw", "keypoint_xyz", "unknown",
}


@dataclass(frozen=True)
class MachineSemantics:
    """机器 VLM 的受限语义输出，不包含最终字段名。"""

    semantic_type: str
    side: str
    body_part: str
    representation: str
    unit: str
    declared_name_status: str
    standardizable: str
    required_transform: str
    confidence: float
    alternatives: tuple[dict[str, object], ...]
    need_human_review: bool
    reason: str


class MachineVlmResolver(Protocol):
    """可替换的机器字段语义解析器。"""

    def resolve(self, evidence: Mapping[str, object]) -> MachineSemantics | None:
        """返回机器语义；失败或禁用时返回空。"""


class DisabledMachineVlmResolver:
    """默认不调用外部 VLM。"""

    def resolve(self, evidence: Mapping[str, object]) -> MachineSemantics | None:
        return None


class OpenAICompatibleMachineVlmResolver:
    """复用通用 OpenAI-compatible 客户端请求机器字段语义。"""

    def __init__(self, client: object):
        self.client = client

    def resolve(self, evidence: Mapping[str, object]) -> MachineSemantics | None:
        system_prompt, user_prompt = build_machine_prompt(evidence)
        request_json = getattr(self.client, "request_json", None)
        if not callable(request_json):
            return None
        payload = request_json(system_prompt, user_prompt, ())
        return parse_machine_semantics(payload) if isinstance(payload, Mapping) else None


def parse_machine_semantics(payload: Mapping[str, object]) -> MachineSemantics:
    """校验 VLM 只返回 P2 允许的枚举语义。"""
    if "target_key" in payload or "target_name" in payload:
        raise ValueError("机器 VLM 不得输出最终字段名")
    semantic_type = payload.get("semantic_type")
    representation = payload.get("representation")
    side = payload.get("side")
    body_part = payload.get("body_part")
    unit = payload.get("unit")
    declared_name_status = payload.get("declared_name_status")
    standardizable = payload.get("standardizable")
    required_transform = payload.get("required_transform")
    confidence = payload.get("confidence")
    alternatives = payload.get("alternatives")
    if semantic_type not in SEMANTIC_TYPES or representation not in REPRESENTATIONS:
        raise ValueError("机器 VLM 语义或表示形式不合法")
    if side not in {"left", "right", "both", "none", "unknown"}:
        raise ValueError("机器 VLM 左右侧不合法")
    if not isinstance(body_part, str) or not isinstance(unit, str):
        raise ValueError("机器 VLM 部位或单位不合法")
    if declared_name_status not in {"correct", "partially_correct", "misleading", "unknown"}:
        raise ValueError("机器 VLM 名称状态不合法")
    if standardizable not in {"direct", "needs_transform", "not_covered", "review"}:
        raise ValueError("机器 VLM 可规范化状态不合法")
    if not isinstance(required_transform, str):
        raise ValueError("机器 VLM 变换要求不合法")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("机器 VLM 置信度不合法")
    if not isinstance(alternatives, list) or not all(isinstance(item, dict) for item in alternatives):
        raise ValueError("机器 VLM 备选项不合法")
    if not isinstance(payload.get("need_human_review"), bool) or not isinstance(payload.get("reason"), str):
        raise ValueError("机器 VLM 复核字段不合法")
    return MachineSemantics(
        semantic_type=semantic_type,
        side=side,
        body_part=body_part,
        representation=representation,
        unit=unit,
        declared_name_status=declared_name_status,
        standardizable=standardizable,
        required_transform=required_transform,
        confidence=float(confidence),
        alternatives=tuple(alternatives),
        need_human_review=payload["need_human_review"],
        reason=payload["reason"],
    )


def can_apply_semantics(semantics: MachineSemantics, vector_length: int) -> bool:
    """执行 P2 的高置信度、无变换、无歧义自动采纳门槛。"""
    if (
        semantics.confidence < 0.92
        or semantics.need_human_review
        or semantics.standardizable != "direct"
        or semantics.required_transform != "none"
        or semantics.alternatives
    ):
        return False
    expected_representations = {
        "head_orientation_quaternion": "quaternion_xyzw",
        "head_position": "position_xyz",
        "arm_joint": "joint_vector",
        "hand_joint": "joint_vector",
        "eef_position": "position_xyz",
    }
    if semantics.representation != expected_representations.get(semantics.semantic_type):
        return False
    from robometanorm.machine.name_builder import build_names_from_semantics

    return build_names_from_semantics(semantics, vector_length) is not None
