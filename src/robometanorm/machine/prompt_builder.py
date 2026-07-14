"""构建只要求机器字段分段语义的 P2 VLM 提示词。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from robometanorm.machine.semantic_schema import (
    DECLARED_NAME_STATUSES,
    REPRESENTATIONS,
    REQUIRED_TRANSFORMS,
    SEMANTIC_TYPES,
    SIDES,
    STANDARDIZABLE_STATUSES,
    UNITS,
)


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
