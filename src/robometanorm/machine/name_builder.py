"""P2 可直接确认的机器字段名称构造规则。"""

from __future__ import annotations

import re


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
