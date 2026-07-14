"""相机字段的确定性命名与冲突检测。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from robometanorm.camera.models import CameraMount, CameraNameProposal


_DUAL_WRIST_CAMERA_ALIASES = {
    "observation.images.cam_left_wrist_rgb": "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb": "observation.images.cam_right_wrist_rgb",
}

ROBOT_CAMERA_NAME_MAPS = {
    "airbot_mmk2": _DUAL_WRIST_CAMERA_ALIASES,
    "agilex_cobot_magic": _DUAL_WRIST_CAMERA_ALIASES,
    "galaxea": {
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

_ROBOT_CAMERA_MAP_ALIASES = {
    "galaxea_r1_lite": "galaxea",
    "galbot_g1": "galbot",
}

MOUNT_TYPES = frozenset({"on_robot", "external"})
ON_ROBOT_DIRECTION_TOKENS = frozenset(
    {"front", "rear", "left", "right", "upper", "lower", "middle"}
)
EXTERNAL_DIRECTION_TOKENS = frozenset(
    {"front", "rear", "left", "right", "upper", "lower", "top", "side", "global", "env"}
)
DIRECTION_TOKENS = ON_ROBOT_DIRECTION_TOKENS | EXTERNAL_DIRECTION_TOKENS | {"ego"}
BODY_PART_TOKENS = frozenset(
    {"wrist", "head", "chest", "arm", "leg", "torso", "fisheye"}
)
_TOKEN_GROUPS = (
    ("front", "rear"),
    ("upper", "lower", "middle", "top"),
    ("left", "right", "side"),
    ("global", "env"),
)


def build_camera_key(
    mount_type: str,
    direction_tokens: Iterable[str],
    body_part: str | None,
    modality: str,
) -> str | None:
    """按内置的本体/外部相机规范生成字段名。"""
    tokens = set(direction_tokens)
    if mount_type not in MOUNT_TYPES or modality not in {"rgb", "depth"}:
        return None
    if mount_type == "on_robot":
        if tokens == {"ego"} and body_part is None:
            return f"observation.images.cam_ego_{modality}"
        if (
            body_part not in BODY_PART_TOKENS
            or not tokens.issubset(ON_ROBOT_DIRECTION_TOKENS)
        ):
            return None
    elif (
        body_part is not None
        or len(tokens) != 1
        or not tokens.issubset(EXTERNAL_DIRECTION_TOKENS)
    ):
        return None

    ordered_tokens = _ordered_direction_tokens(tokens)
    if ordered_tokens is None:
        return None
    if body_part:
        ordered_tokens.append(body_part)
    return "observation.images.cam_" + "_".join([*ordered_tokens, modality])


def propose_camera_name(source_key: str) -> CameraNameProposal | None:
    """仅确认已经严格符合内置规范的相机字段。"""
    parsed = parse_standard_camera_key(source_key)
    if parsed is None:
        return None
    _, modality = parsed
    return CameraNameProposal(
        source_key=source_key,
        target_key=source_key,
        modality=modality,
        method="standard",
    )


def parse_standard_camera_key(source_key: str) -> tuple[CameraMount, str] | None:
    """解析严格符合规范的字段，含糊或非法名称返回空。"""
    prefix = "observation.images.cam_"
    if not source_key.startswith(prefix):
        return None
    tokens = source_key[len(prefix) :].split("_")
    if len(tokens) < 2 or tokens[-1] not in {"rgb", "depth"}:
        return None
    modality = tokens[-1]
    position_tokens = tokens[:-1]
    if position_tokens == ["ego"]:
        mount = CameraMount("on_robot", ("ego",), None)
    elif position_tokens[-1] in BODY_PART_TOKENS:
        mount = CameraMount(
            "on_robot", tuple(position_tokens[:-1]), position_tokens[-1]
        )
    else:
        mount = CameraMount("external", tuple(position_tokens), None)
    if (
        build_camera_key(
            mount.mount_type,
            mount.direction_tokens,
            mount.body_part,
            modality,
        )
        != source_key
    ):
        return None
    return mount, modality


def _ordered_direction_tokens(tokens: set[str]) -> list[str] | None:
    """按前后、垂直、左右、全局语义稳定排序。"""
    ordered_tokens: list[str] = []
    for group in _TOKEN_GROUPS:
        selected = [token for token in group if token in tokens]
        if len(selected) > 1:
            return None
        ordered_tokens.extend(selected)
    return ordered_tokens if len(ordered_tokens) == len(tokens) else None


def propose_robot_camera_name(
    robot_id: str | None, source_key: str
) -> CameraNameProposal | None:
    """从已验证的机器人级别名生成相机名称。"""
    if robot_id is None:
        return None
    mapping_id = _ROBOT_CAMERA_MAP_ALIASES.get(robot_id, robot_id)
    target_key = ROBOT_CAMERA_NAME_MAPS.get(mapping_id, {}).get(source_key)
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
