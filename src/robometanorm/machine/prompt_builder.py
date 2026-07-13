"""构建只要求机器字段语义的 P2 VLM 提示词。"""

from __future__ import annotations

import json
from collections.abc import Mapping


SYSTEM_PROMPT = """你是机器人数据集机器字段语义分类器。
结构和 Parquet 数值事实优先于原始名称；名称仅为弱提示。
不得猜测单位，不得把四元数写成欧拉角，不得默认 wrist 是 EEF，也不得把三维关键点写成关节角。
证据不足时使用 unknown 或 need_human_review=true。只输出合法 JSON，不得输出最终标准字段名。"""


def build_machine_prompt(evidence: Mapping[str, object]) -> tuple[str, str]:
    """把结构化字段证据转换为机器语义分类请求。"""
    user_prompt = "\n".join(
        [
            "请判断以下机器字段的真实语义。",
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
            "请返回 semantic_type、side、body_part、representation、unit、declared_name_status、standardizable、required_transform、confidence、alternatives、need_human_review、reason。",
        ]
    )
    return SYSTEM_PROMPT, user_prompt
