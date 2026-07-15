"""Strict rendering and parsing for canonical camera feature keys."""

from __future__ import annotations

from robometanorm.models import CameraSlot


CAMERA_PREFIX = "observation.images.cam_"

BODY_PARTS = frozenset(
    {"wrist", "head", "chest", "arm", "leg", "torso", "fisheye"}
)
ON_ROBOT_DIRECTIONS = frozenset(
    {"front", "rear", "left", "right", "upper", "lower", "middle"}
)
EXTERNAL_DIRECTIONS = frozenset(
    {
        "front",
        "rear",
        "left",
        "right",
        "upper",
        "lower",
        "middle",
        "top",
        "side",
        "global",
        "env",
    }
)
DIRECTION_ORDER = (
    "front",
    "rear",
    "upper",
    "lower",
    "middle",
    "top",
    "left",
    "right",
    "side",
    "global",
    "env",
)
CONFLICT_GROUPS = (
    frozenset({"front", "rear"}),
    frozenset({"upper", "lower", "middle", "top"}),
    frozenset({"left", "right", "side"}),
    frozenset({"global", "env"}),
)

_MODALITIES = frozenset({"rgb", "depth"})
_STANDALONE_EXTERNAL_DIRECTIONS = frozenset({"global", "env"})


def _has_conflict(direction_set: frozenset[str]) -> bool:
    return any(len(direction_set & group) > 1 for group in CONFLICT_GROUPS)


def _ordered_directions(direction_set: frozenset[str]) -> tuple[str, ...]:
    return tuple(token for token in DIRECTION_ORDER if token in direction_set)


def render_camera_key(slot: CameraSlot) -> str | None:
    """Render a camera slot when it conforms to the canonical camera grammar."""

    if slot.modality not in _MODALITIES:
        return None

    direction_tokens = slot.direction_tokens
    if len(direction_tokens) != len(set(direction_tokens)):
        return None

    direction_set = frozenset(direction_tokens)
    if _has_conflict(direction_set):
        return None

    if slot.mount_type == "on_robot":
        if direction_tokens == ("ego",):
            if slot.body_part is not None:
                return None
            key_tokens = ("ego", slot.modality)
            return CAMERA_PREFIX + "_".join(key_tokens)

        if "ego" in direction_set:
            return None
        if slot.body_part not in BODY_PARTS:
            return None
        if not direction_set <= ON_ROBOT_DIRECTIONS:
            return None

        key_tokens = (
            *_ordered_directions(direction_set),
            slot.body_part,
            slot.modality,
        )
        return CAMERA_PREFIX + "_".join(key_tokens)

    if slot.mount_type == "external":
        if slot.body_part is not None or not direction_tokens:
            return None
        if not direction_set <= EXTERNAL_DIRECTIONS:
            return None
        if (
            direction_set & _STANDALONE_EXTERNAL_DIRECTIONS
            and len(direction_tokens) != 1
        ):
            return None

        key_tokens = (*_ordered_directions(direction_set), slot.modality)
        return CAMERA_PREFIX + "_".join(key_tokens)

    return None


def parse_standard_camera_key(key: str) -> str | None:
    """Return a canonical camera key's modality, or ``None`` when invalid."""

    if not key.startswith(CAMERA_PREFIX):
        return None

    key_tokens = tuple(key[len(CAMERA_PREFIX) :].split("_"))
    if len(key_tokens) < 2:
        return None

    modality = key_tokens[-1]
    camera_tokens = key_tokens[:-1]
    if modality not in _MODALITIES:
        return None

    if camera_tokens == ("ego",):
        mount_type = "on_robot"
        direction_tokens = camera_tokens
        body_part = None
    elif camera_tokens[-1] in BODY_PARTS:
        mount_type = "on_robot"
        direction_tokens = camera_tokens[:-1]
        body_part = camera_tokens[-1]
    else:
        mount_type = "external"
        direction_tokens = camera_tokens
        body_part = None

    parsed_slot = CameraSlot(
        camera_id=key,
        interface_name=None,
        mount_type=mount_type,
        direction_tokens=direction_tokens,
        body_part=body_part,
        modality=modality,
        confidence=1.0,
        ambiguous=False,
        reason="parsed canonical camera key",
        source_ids=(),
    )
    if render_camera_key(parsed_slot) != key:
        return None
    return modality
