"""机器字段 VLM 协议共享的受限枚举。"""

from __future__ import annotations


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
