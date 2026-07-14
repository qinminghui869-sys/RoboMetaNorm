"""机器字段维度、单位和语义风险的保守校验。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re


_DEXTEROUS_HAND_TOKEN = re.compile(
    r"(?<![a-z0-9])(?:dexterous|hand|finger)(?![a-z0-9])",
    re.IGNORECASE,
)


def is_dexterous_hand_field(
    source_feature: str, names: Sequence[str]
) -> bool:
    """判断机器字段是否明确属于夹爪末端规范之外的灵巧手字段。"""
    return any(
        _DEXTEROUS_HAND_TOKEN.search(value) is not None
        for value in (source_feature, *names)
    )


def declared_vector_length(feature: Mapping[str, object]) -> int | None:
    """读取 feature.shape 的一维机器向量长度。"""
    shape = feature.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape:
        return None
    return shape[0] if isinstance(shape[0], int) else None


def declared_names(feature: Mapping[str, object]) -> list[str] | None:
    """读取逐维 names，分组名称不视为可安全重命名输入。"""
    names = feature.get("names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        return None
    if not all(isinstance(name, str) for name in names):
        return None
    return list(names)


def risk_categories(names: Sequence[str]) -> set[str]:
    """从原名称提取风险信号，但不把它们作为最终命名依据。"""
    categories: set[str] = set()
    lowered = [name.lower() for name in names]
    if len(set(lowered)) != len(lowered):
        categories.add("DECLARED_NAME_CONFLICT")
    if any("wrist" in name for name in lowered):
        categories.add("WRIST_EEF_RELATION_UNKNOWN")
    if any("skeleton" in name or "keypoint" in name for name in lowered):
        categories.add("SKELETON_STANDARD_UNDEFINED")
    if any("gripper" in name for name in lowered):
        categories.update({"GRIPPER_RANGE_UNKNOWN", "GRIPPER_DIRECTION_UNKNOWN"})
    if any("pose" in name or "position" in name for name in lowered):
        categories.add("UNKNOWN_UNIT")
    if any("joint" in name and not name.endswith("_rad") for name in lowered):
        categories.add("UNKNOWN_UNIT")
    return categories
