"""按 P1 固定顺序构造标准相机字段。"""

from __future__ import annotations

from collections.abc import Iterable


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
