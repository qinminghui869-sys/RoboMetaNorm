"""相机字段的确定性命名与冲突检测。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import re

from robometanorm.camera.models import CameraNameProposal


CAMERA_NAME_MAP = {
    "observation.images.image_left": "observation.images.cam_left_rgb",
    "observation.images.image_right": "observation.images.cam_right_rgb",
    "observation.images.image_top": "observation.images.cam_top_rgb",
    "observation.images.image_wrist": "observation.images.cam_wrist_rgb",
    "observation.images.image_wrist_left": "observation.images.cam_left_wrist_rgb",
    "observation.images.image_wrist_right": "observation.images.cam_right_wrist_rgb",
    "observation.images.image_top_depth": "observation.images.cam_top_depth",
    "observation.images.image_left_depth": "observation.images.cam_left_depth",
    "observation.images.image_right_depth": "observation.images.cam_right_depth",
    "observation.images.image_wrist_depth": "observation.images.cam_wrist_depth",
}

ROBOT_CAMERA_NAME_MAPS = {
    "airbot_mmk2": {
        "observation.images.cam_left_wrist_rgb": "observation.images.cam_left_wrist_rgb",
        "observation.images.cam_right_wrist_rgb": "observation.images.cam_right_wrist_rgb",
    },
    "agilex_cobot_magic": {
        "observation.images.cam_left_wrist_rgb": "observation.images.cam_left_wrist_rgb",
        "observation.images.cam_right_wrist_rgb": "observation.images.cam_right_wrist_rgb",
    },
    "galaxea": {
        "observation.images.image_top_left": "observation.images.cam_top_left_rgb",
        "observation.images.image_top_right": "observation.images.cam_top_right_rgb",
        "observation.images.image_wrist_left": "observation.images.cam_left_wrist_rgb",
        "observation.images.image_wrist_right": "observation.images.cam_right_wrist_rgb",
    },
    "galaxea_r1_lite": {
        "observation.images.image_top_left": "observation.images.cam_top_left_rgb",
        "observation.images.image_top_right": "observation.images.cam_top_right_rgb",
        "observation.images.image_wrist_left": "observation.images.cam_left_wrist_rgb",
        "observation.images.image_wrist_right": "observation.images.cam_right_wrist_rgb",
    },
    "galbot": {
        "observation.images.image_head_right": "observation.images.cam_right_head_rgb",
        "observation.images.image_head_left": "observation.images.cam_left_head_rgb",
        "observation.images.image_arm_right": "observation.images.cam_right_arm_rgb",
        "observation.images.image_arm_left": "observation.images.cam_left_arm_rgb",
    },
    "galbot_g1": {
        "observation.images.image_head_right": "observation.images.cam_right_head_rgb",
        "observation.images.image_head_left": "observation.images.cam_left_head_rgb",
        "observation.images.image_arm_right": "observation.images.cam_right_arm_rgb",
        "observation.images.image_arm_left": "observation.images.cam_left_arm_rgb",
    },
    "aloha": {
        "observation.images.image_top": "observation.images.cam_top_rgb",
        "observation.images.image_left": "observation.images.cam_left_rgb",
        "observation.images.image_right": "observation.images.cam_right_rgb",
    },
    "dexterous_hand_v1": {
        "observation.images.image_top": "observation.images.cam_top_rgb",
        "observation.images.image_left": "observation.images.cam_left_rgb",
        "observation.images.image_right": "observation.images.cam_right_rgb",
        "observation.images.left_image": "observation.images.cam_left_rgb",
        "observation.images.right_image": "observation.images.cam_right_rgb",
    },
}

EXTERNAL_DIRECTION_TOKENS = frozenset(
    {"front", "rear", "left", "right", "upper", "lower", "middle", "top", "side", "global", "env"}
)
BODY_PART_TOKENS = frozenset(
    {"wrist", "head", "chest", "arm", "leg", "torso", "fisheye", "ego"}
)
_TOKEN_GROUPS = (
    ("front", "rear"),
    ("upper", "lower", "middle", "top"),
    ("left", "right", "side"),
)


def build_camera_key(
    direction_tokens: Iterable[str], body_part: str | None, modality: str
) -> str | None:
    """用前后、垂直、左右、本体部位、模态的顺序生成字段名。"""
    tokens = set(direction_tokens)
    if modality not in {"rgb", "depth"}:
        return None
    if not tokens.issubset(EXTERNAL_DIRECTION_TOKENS):
        return None
    if body_part is not None and body_part not in BODY_PART_TOKENS:
        return None

    ordered_tokens: list[str] = []
    for group in _TOKEN_GROUPS:
        selected = [token for token in group if token in tokens]
        if len(selected) > 1:
            return None
        ordered_tokens.extend(selected)
    if body_part:
        ordered_tokens.append(body_part)
    if not ordered_tokens:
        return None
    return "observation.images.cam_" + "_".join([*ordered_tokens, modality])


def propose_camera_name(source_key: str) -> CameraNameProposal | None:
    """为相机源字段生成确定性目标名，无法判断时返回空。"""
    target_key = CAMERA_NAME_MAP.get(source_key)
    if target_key:
        return CameraNameProposal(
            source_key=source_key,
            target_key=target_key,
            modality=_modality_from_target(target_key),
            method="exact",
        )

    field_name = source_key.rsplit(".", maxsplit=1)[-1].lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", field_name) if token]
    direction_tokens = [token for token in tokens if token in EXTERNAL_DIRECTION_TOKENS]
    body_parts = [token for token in tokens if token in BODY_PART_TOKENS]
    if len(set(body_parts)) > 1:
        return None
    modality = "depth" if "depth" in tokens else "rgb"
    target_key = build_camera_key(direction_tokens, body_parts[0] if body_parts else None, modality)
    if target_key is None:
        return None
    return CameraNameProposal(
        source_key=source_key,
        target_key=target_key,
        modality=modality,
        method="regex",
    )


def propose_robot_camera_name(
    robot_id: str | None, source_key: str
) -> CameraNameProposal | None:
    """从已验证的机器人级别名生成相机名称。"""
    if robot_id is None:
        return None
    target_key = ROBOT_CAMERA_NAME_MAPS.get(robot_id, {}).get(source_key)
    if target_key is None:
        return None
    return CameraNameProposal(
        source_key=source_key,
        target_key=target_key,
        modality=_modality_from_target(target_key),
        method="robot",
    )


def _modality_from_target(target_key: str) -> str:
    """从标准目标名恢复已验证的模态后缀。"""
    return "depth" if target_key.endswith("_depth") else "rgb"


def find_colliding_sources(proposals: Iterable[CameraNameProposal]) -> set[str]:
    """返回解析到相同目标字段的全部源字段。"""
    by_target: dict[str, list[str]] = defaultdict(list)
    for proposal in proposals:
        by_target[proposal.target_key].append(proposal.source_key)
    return {
        source_key
        for source_keys in by_target.values()
        if len(source_keys) > 1
        for source_key in source_keys
    }
