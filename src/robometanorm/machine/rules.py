"""机器字段的确定性发现、布局、校验与命名规则。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np

from robometanorm.machine.models import ParquetProfile


PARENT_MACHINE_FEATURES = ("action", "observation.state")

_OUT_OF_SCOPE_TOKEN = re.compile(
    r"(?<![a-z0-9])(?:dexterous|hand|finger|skeleton|keypoint)(?![a-z0-9])",
    re.IGNORECASE,
)
_HEAD_QUATERNION_PATTERN = re.compile(r"^head_(?:rotation_)?quat_([xyzw])$")
_STANDARD_NAME_PATTERN = re.compile(
    r"^(?:"
    r"(?:left|right)_arm_joint_\d+_rad|"
    r"(?:left|right)_gripper_open|"
    r"(?:left|right)_eef_pos_[xyz]_m|"
    r"(?:left|right)_eef_rot_euler_[xyz]_rad|"
    r"head_pos_[xyz]_m|head_orient_quat_[xyzw]|"
    r"torso_joint_\d+_rad|neck_joint_\d+_rad|"
    r"base_pos_[xyz]_m|base_rot_euler_[xyz]_rad"
    r")$"
)


def discover_machine_features(info: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """返回机器父字段和 observation.state 子字段定义。"""
    features = info.get("features")
    if not isinstance(features, Mapping):
        return {}
    return {
        str(key): value
        for key, value in features.items()
        if isinstance(value, Mapping)
        and (str(key) in PARENT_MACHINE_FEATURES or str(key).startswith("observation.state."))
    }


def resolve_child_slices(
    parent_samples: np.ndarray, child_samples: Mapping[str, np.ndarray]
) -> dict[str, tuple[int, int]]:
    """查找每个子字段在父向量中的连续切片。"""
    if parent_samples.ndim != 2:
        raise ValueError("父字段样本必须是二维向量")
    resolved: dict[str, tuple[int, int]] = {}
    for child_name, samples in child_samples.items():
        if samples.ndim != 2 or samples.shape[0] != parent_samples.shape[0]:
            continue
        width = samples.shape[1]
        for start in range(parent_samples.shape[1] - width + 1):
            end = start + width
            if np.allclose(parent_samples[:, start:end], samples, equal_nan=True):
                resolved[child_name] = (start, end)
                break
    return resolved


def action_equals_state(profile: ParquetProfile | None) -> bool:
    """仅在样本形状和值均一致时认定 action 与 state 可复用。"""
    if profile is None:
        return False
    action = profile.samples.get("action")
    state = profile.samples.get("observation.state")
    return bool(
        action is not None
        and state is not None
        and action.shape == state.shape
        and np.array_equal(action, state, equal_nan=True)
    )


def is_out_of_scope_machine_field(
    source_feature: str, names: Sequence[str]
) -> bool:
    """判断字段是否属于当前夹爪末端规范未覆盖的灵巧手或骨架数据。"""
    return any(
        _OUT_OF_SCOPE_TOKEN.search(value) is not None
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
    if any("gripper" in name for name in lowered):
        categories.update({"GRIPPER_RANGE_UNKNOWN", "GRIPPER_DIRECTION_UNKNOWN"})
    if unknown_unit_indices(lowered):
        categories.add("UNKNOWN_UNIT")
    return categories


def unknown_unit_indices(names: Sequence[str]) -> tuple[int, ...]:
    """返回缺少所需显式单位 token 的字段维度。"""
    indices: list[int] = []
    for index, name in enumerate(names):
        lowered = name.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        if "joint" in lowered and "rad" not in tokens:
            indices.append(index)
        elif "position" in lowered and "m" not in tokens:
            indices.append(index)
        elif "pose" in lowered and not tokens.intersection({"m", "rad"}):
            indices.append(index)
    return tuple(indices)


def build_confirmed_machine_name(source_name: str) -> str | None:
    """仅返回不依赖未知单位或物理关系的确定性目标名。"""
    match = _HEAD_QUATERNION_PATTERN.fullmatch(source_name)
    if match:
        return f"head_orient_quat_{match.group(1)}"
    if _STANDARD_NAME_PATTERN.fullmatch(source_name):
        return source_name
    return None


def build_names_from_semantics(semantics: object, vector_length: int) -> list[str] | None:
    """仅为规则已覆盖且维度匹配的 VLM 语义生成名称。"""
    semantic_type = getattr(semantics, "semantic_type", None)
    side = getattr(semantics, "side", None)
    unit = getattr(semantics, "unit", None)
    if semantic_type == "head_orientation_quaternion" and vector_length == 4:
        return [f"head_orient_quat_{axis}" for axis in "xyzw"]
    if semantic_type == "head_position" and vector_length == 3 and unit == "m":
        return [f"head_pos_{axis}_m" for axis in "xyz"]
    if semantic_type == "arm_joint" and side in {"left", "right"} and unit == "rad":
        return [f"{side}_arm_joint_{index}_rad" for index in range(vector_length)]
    if semantic_type == "eef_position" and side in {"left", "right"} and unit == "m" and vector_length == 3:
        return [f"{side}_eef_pos_{axis}_m" for axis in "xyz"]
    return None
