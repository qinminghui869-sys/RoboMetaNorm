"""机器字段 VLM 分段语义协议、提示词与规则裁决。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Protocol

from robometanorm.machine.rules import build_names_from_semantics


SEMANTIC_TYPES = frozenset(
    {
        "arm_joint",
        "gripper_open",
        "eef_position",
        "eef_rotation_euler",
        "orientation_quaternion",
        "head_joint",
        "head_position",
        "head_rotation_euler",
        "head_orientation_quaternion",
        "torso_joint",
        "neck_joint",
        "base_position",
        "base_rotation_euler",
        "skeleton",
        "unknown",
    }
)

REPRESENTATIONS = frozenset(
    {
        "scalar",
        "joint_vector",
        "position_xyz",
        "euler_xyz",
        "quaternion_xyzw",
        "keypoint_xyz",
        "unknown",
    }
)

SIDES = frozenset({"left", "right", "both", "none", "unknown"})
DECLARED_NAME_STATUSES = frozenset(
    {"correct", "partially_correct", "misleading", "unknown"}
)
STANDARDIZABLE_STATUSES = frozenset(
    {"direct", "needs_transform", "not_covered", "review"}
)
REQUIRED_TRANSFORMS = frozenset({"none", "quaternion_to_euler", "unknown"})
UNITS = frozenset({"m", "rad", "none", "unknown"})


SYSTEM_PROMPT = """你是机器人数据集机器字段语义分类器。
本规范只覆盖机械臂末端为夹爪的机器字段。
灵巧手、手指关节和手部关键点不在覆盖范围；相关区段必须使用 semantic_type=unknown、standardizable=not_covered 和 need_human_review=true。
结构和 Parquet 数值事实优先于原始名称；名称仅为弱提示。
不得猜测单位，不得把四元数写成欧拉角，不得默认 wrist 是 EEF，也不得把三维关键点写成关节角。
证据不足的区段必须使用 unknown 或 need_human_review=true，不能省略任何维度。
只输出符合约定的合法 JSON，不得输出最终标准字段名。"""


def build_machine_prompt(evidence: Mapping[str, object]) -> tuple[str, str]:
    """把结构化字段证据转换为带完整分段协议的请求。"""
    vector_length = _vector_length(evidence.get("shape"))
    contract = {
        "segments": [
            {
                "local_slice": [0, vector_length],
                "semantic_type": "unknown",
                "side": "unknown",
                "body_part": "unknown",
                "representation": "unknown",
                "unit": "unknown",
                "declared_name_status": "unknown",
                "standardizable": "review",
                "required_transform": "none",
                "confidence": 0.0,
                "alternatives": [],
                "need_human_review": True,
                "reason": "证据不足",
            }
        ],
        "need_human_review": True,
        "reason": "整体判断理由",
    }
    composite_example = {
        "segments": [
            {
                "local_slice": [0, 3],
                "semantic_type": "eef_position",
                "side": "left",
                "body_part": "wrist",
                "representation": "position_xyz",
                "unit": "unknown",
                "declared_name_status": "partially_correct",
                "standardizable": "review",
                "required_transform": "none",
                "confidence": 0.91,
                "alternatives": [],
                "need_human_review": True,
                "reason": "缺少 wrist 等同 EEF 的外部证据",
            },
            {
                "local_slice": [3, 7],
                "semantic_type": "orientation_quaternion",
                "side": "left",
                "body_part": "wrist",
                "representation": "quaternion_xyzw",
                "unit": "none",
                "declared_name_status": "partially_correct",
                "standardizable": "review",
                "required_transform": "none",
                "confidence": 0.97,
                "alternatives": [],
                "need_human_review": True,
                "reason": "只能确认旋转表示，不能确认最终标准字段",
            },
        ],
        "need_human_review": True,
        "reason": "该字段包含多个语义区段",
    }
    user_prompt = "\n".join(
        [
            "请判断以下机器字段的真实语义，并只返回 JSON。",
            f"dataset_name: {evidence.get('dataset_name')}",
            f"robot_type: {evidence.get('robot_type')}",
            f"parent_feature: {evidence.get('parent_feature')}",
            f"source_feature: {evidence.get('source_feature')}",
            f"source_slice: {evidence.get('source_slice')}",
            f"shape: {evidence.get('shape')}",
            f"declared_names: {evidence.get('declared_names')}",
            f"numeric_profile: {json.dumps(evidence.get('numeric_profile', {}), ensure_ascii=False)}",
            f"relations: {json.dumps(evidence.get('relations', {}), ensure_ascii=False)}",
            f"rule_candidates: {evidence.get('rule_candidates')}",
            "JSON 协议约束：",
            f"semantic_type 只能取: {sorted(SEMANTIC_TYPES)}",
            f"representation 只能取: {sorted(REPRESENTATIONS)}",
            f"side 只能取: {sorted(SIDES)}",
            f"unit 只能取: {sorted(UNITS)}",
            f"declared_name_status 只能取: {sorted(DECLARED_NAME_STATUSES)}",
            f"standardizable 只能取: {sorted(STANDARDIZABLE_STATUSES)}",
            f"required_transform 只能取: {sorted(REQUIRED_TRANSFORMS)}",
            "字段类型：segments 和 alternatives 是数组；每个 segment 是 JSON 对象；local_slice 是两个整数的数组。",
            "semantic_type、side、body_part、representation、unit、declared_name_status、standardizable、required_transform 和 reason 是字符串。",
            "confidence 必须是 0 到 1 的数字；need_human_review 必须是布尔值；alternatives 必须是 JSON 对象数组。",
            f"segments 必须按 local_slice 连续覆盖 [0, {vector_length}]，不得空洞、重叠、逆序或越界。",
            "每个 segment 都必须包含示例中的全部字段；不确定区段也必须显式返回 unknown。",
            "单段或无法确认时可使用下列完整 JSON 结构：",
            json.dumps(contract, ensure_ascii=False, indent=2),
            "复合 JSON 示例（仅展示 7 维协议结构，不得复制其 slice 代替本次实际长度）：",
            json.dumps(composite_example, ensure_ascii=False, indent=2),
            "上述示例不代表 wrist 一定是 EEF。",
            "不得输出最终标准字段名。",
        ]
    )
    return SYSTEM_PROMPT, user_prompt


def build_machine_repair_prompt(
    original_user_prompt: str, validation_error: str, vector_length: int
) -> str:
    """在首次业务 schema 失败后构造一次受限纠错请求。"""
    concise_error = " ".join(validation_error.split())[:500]
    return "\n".join(
        [
            original_user_prompt,
            f"上次 JSON 不符合协议：{concise_error}",
            "请仅返回修正后的 JSON，不要解释。",
            f"不得省略区段；segments 必须连续覆盖 [0, {vector_length}]。",
        ]
    )


def _vector_length(shape: object) -> int:
    if (
        isinstance(shape, Sequence)
        and not isinstance(shape, (str, bytes))
        and shape
        and isinstance(shape[0], int)
        and not isinstance(shape[0], bool)
        and shape[0] > 0
    ):
        return shape[0]
    raise ValueError("机器字段 shape 缺少正向量长度")


@dataclass(frozen=True)
class MachineSemanticSegment:
    """复合机器字段中的一个连续局部语义区段。"""

    local_slice: tuple[int, int]
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


@dataclass(frozen=True)
class MachineSemantics:
    """完整覆盖一个源字段的机器语义分段结果。"""

    segments: tuple[MachineSemanticSegment, ...]
    need_human_review: bool
    reason: str


class MachineVlmResolutionError(RuntimeError):
    """机器 VLM 传输或语义协议无法恢复时的脱敏错误。"""


class MachineVlmResolver(Protocol):
    """可替换的机器字段语义解析器。"""

    def resolve(self, evidence: Mapping[str, object]) -> MachineSemantics | None:
        """返回机器语义；禁用时返回空，启用但失败时抛出脱敏异常。"""


class DisabledMachineVlmResolver:
    """默认不调用外部 VLM。"""

    def resolve(self, evidence: Mapping[str, object]) -> MachineSemantics | None:
        return None


class OpenAICompatibleMachineVlmResolver:
    """复用通用 OpenAI-compatible 客户端请求机器字段语义。"""

    def __init__(self, client: object):
        self.client = client
        self.last_error: str | None = None

    def resolve(self, evidence: Mapping[str, object]) -> MachineSemantics:
        system_prompt, user_prompt = build_machine_prompt(evidence)
        vector_length = _evidence_vector_length(evidence)
        request_json = getattr(self.client, "request_json", None)
        if not callable(request_json):
            raise MachineVlmResolutionError("VLM 客户端缺少 request_json")

        payload = request_json(system_prompt, user_prompt, ())
        if not isinstance(payload, Mapping):
            self._raise_transport_error()
        try:
            result = parse_machine_semantics(payload, vector_length=vector_length)
        except ValueError as first_error:
            repair_prompt = build_machine_repair_prompt(
                user_prompt, str(first_error), vector_length
            )
            repaired_payload = request_json(system_prompt, repair_prompt, ())
            if not isinstance(repaired_payload, Mapping):
                self._raise_transport_error(prefix="VLM schema 纠错请求失败")
            try:
                result = parse_machine_semantics(
                    repaired_payload, vector_length=vector_length
                )
            except ValueError as second_error:
                message = f"机器 VLM JSON 协议校验失败: {second_error}"
                self.last_error = message
                raise MachineVlmResolutionError(message) from None
        self.last_error = None
        return result

    def _raise_transport_error(self, prefix: str = "机器 VLM 请求失败") -> None:
        client_error = getattr(self.client, "last_error", None)
        detail = (
            _sanitize_error_detail(client_error)
            if client_error
            else "未返回 JSON 对象"
        )
        message = f"{prefix}: {detail}"
        self.last_error = message
        raise MachineVlmResolutionError(message)


def parse_machine_semantics(
    payload: Mapping[str, object], vector_length: int
) -> MachineSemantics:
    """校验新分段协议，并兼容旧的单语义对象。"""
    if not isinstance(vector_length, int) or isinstance(vector_length, bool) or vector_length <= 0:
        raise ValueError("vector_length 必须为正整数")
    _reject_forbidden_target_fields(payload)

    segments_payload = payload.get("segments")
    if segments_payload is None:
        segment = _parse_segment(payload, (0, vector_length), "legacy")
        result = MachineSemantics(
            segments=(segment,),
            need_human_review=_boolean(payload, "need_human_review", "root"),
            reason=_string(payload, "reason", "root"),
        )
    else:
        if not isinstance(segments_payload, list) or not segments_payload:
            raise ValueError("segments 必须是非空 JSON 对象数组")
        segments: list[MachineSemanticSegment] = []
        for index, item in enumerate(segments_payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"segments[{index}] 必须是 JSON 对象")
            local_slice = _parse_slice(item.get("local_slice"), index)
            segments.append(_parse_segment(item, local_slice, f"segments[{index}]"))
        result = MachineSemantics(
            segments=tuple(segments),
            need_human_review=_boolean(payload, "need_human_review", "root"),
            reason=_string(payload, "reason", "root"),
        )

    _validate_coverage(result.segments, vector_length)
    return result


def can_apply_semantics(semantics: MachineSemantics, vector_length: int) -> bool:
    """所有区段均跨过旧有安全门槛时才允许自动采纳。"""
    if semantics.need_human_review or not semantics.segments:
        return False
    expected_start = 0
    generated_names: list[str] = []
    expected_representations = {
        "head_orientation_quaternion": "quaternion_xyzw",
        "head_position": "position_xyz",
        "arm_joint": "joint_vector",
        "eef_position": "position_xyz",
    }

    for segment in semantics.segments:
        start, end = segment.local_slice
        if start != expected_start or end <= start or end > vector_length:
            return False
        expected_start = end
        if (
            segment.confidence < 0.92
            or segment.need_human_review
            or segment.standardizable != "direct"
            or segment.required_transform != "none"
            or segment.alternatives
            or segment.representation
            != expected_representations.get(segment.semantic_type)
        ):
            return False
        names = build_names_from_semantics(segment, end - start)
        if names is None:
            return False
        generated_names.extend(names)
    return (
        expected_start == vector_length
        and len(generated_names) == vector_length
        and len(set(generated_names)) == len(generated_names)
    )


def _parse_segment(
    payload: Mapping[str, object], local_slice: tuple[int, int], label: str
) -> MachineSemanticSegment:
    semantic_type = payload.get("semantic_type")
    representation = payload.get("representation")
    side = payload.get("side")
    declared_name_status = payload.get("declared_name_status")
    standardizable = payload.get("standardizable")
    required_transform = payload.get("required_transform")
    unit = payload.get("unit")
    confidence = payload.get("confidence")
    alternatives = payload.get("alternatives")
    if semantic_type not in SEMANTIC_TYPES:
        raise ValueError(f"{label}.semantic_type 不合法: {semantic_type}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"{label}.representation 不合法: {representation}")
    if side not in SIDES:
        raise ValueError(f"{label}.side 不合法: {side}")
    if declared_name_status not in DECLARED_NAME_STATUSES:
        raise ValueError(f"{label}.declared_name_status 不合法: {declared_name_status}")
    if standardizable not in STANDARDIZABLE_STATUSES:
        raise ValueError(f"{label}.standardizable 不合法: {standardizable}")
    if required_transform not in REQUIRED_TRANSFORMS:
        raise ValueError(f"{label}.required_transform 不合法: {required_transform}")
    if unit not in UNITS:
        raise ValueError(f"{label}.unit 不合法: {unit}")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError(f"{label}.confidence 必须在 0 到 1 之间")
    if not isinstance(alternatives, list) or not all(
        isinstance(item, dict) for item in alternatives
    ):
        raise ValueError(f"{label}.alternatives 必须是 JSON 对象数组")
    return MachineSemanticSegment(
        local_slice=local_slice,
        semantic_type=str(semantic_type),
        side=str(side),
        body_part=_string(payload, "body_part", label),
        representation=str(representation),
        unit=str(unit),
        declared_name_status=str(declared_name_status),
        standardizable=str(standardizable),
        required_transform=str(required_transform),
        confidence=float(confidence),
        alternatives=tuple(alternatives),
        need_human_review=_boolean(payload, "need_human_review", label),
        reason=_string(payload, "reason", label),
    )


def _parse_slice(value: object, index: int) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"segments[{index}].local_slice 必须是两个整数")
    return value[0], value[1]


def _validate_coverage(
    segments: Sequence[MachineSemanticSegment], vector_length: int
) -> None:
    expected_start = 0
    for index, segment in enumerate(segments):
        start, end = segment.local_slice
        if start != expected_start:
            relation = "重叠或逆序" if start < expected_start else "存在空洞"
            raise ValueError(
                f"segments[{index}].local_slice {relation}: 期望从 {expected_start} 开始"
            )
        if end <= start:
            raise ValueError(f"segments[{index}].local_slice 终点必须大于起点")
        if end > vector_length:
            raise ValueError(f"segments[{index}].local_slice 越界: {end}>{vector_length}")
        expected_start = end
    if expected_start != vector_length:
        raise ValueError(f"segments 未完整覆盖向量: {expected_start}!={vector_length}")


def _reject_forbidden_target_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"target_key", "target_name"}:
                raise ValueError("机器 VLM 不得输出最终字段名")
            _reject_forbidden_target_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_target_fields(item)


def _boolean(payload: Mapping[str, object], key: str, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} 必须是布尔值")
    return value


def _string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} 必须是非空字符串")
    return value


def _evidence_vector_length(evidence: Mapping[str, object]) -> int:
    shape = evidence.get("shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or not shape
        or not isinstance(shape[0], int)
        or isinstance(shape[0], bool)
        or shape[0] <= 0
    ):
        raise MachineVlmResolutionError("机器字段 shape 缺少正向量长度")
    return shape[0]


def _sanitize_error_detail(error: object) -> str:
    """限制传输错误并移除可能混入的认证信息。"""
    message = " ".join(str(error).split())
    message = re.sub(
        r"(?i)authorization\s*:\s*bearer\s+[^;\s]+;?\s*",
        "",
        message,
    )
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "sk-[REDACTED]", message)
    return (message or type(error).__name__)[:500]
