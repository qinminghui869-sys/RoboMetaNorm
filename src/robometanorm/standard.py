"""Strict rendering and parsing for canonical feature names."""

from __future__ import annotations

import re

from robometanorm.models import CameraSlot, MachineComponent


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

FIXED_COMPONENTS = {
    "eef_position": ("position_xyz", "m", 3),
    "eef_rotation": ("euler_xyz", "rad", 3),
    "head_rotation": ("euler_xyz", "rad", 3),
    "head_orientation": ("quaternion_xyzw", "unitless", 4),
    "base_position": ("position_xyz", "m", 3),
    "base_rotation": ("euler_xyz", "rad", 3),
}
SIDED_COMPONENTS = frozenset(
    {
        "arm_joint",
        "hand_joint",
        "gripper_open",
        "gripper_open_scale",
        "eef_position",
        "eef_rotation",
    }
)
JOINT_COMPONENTS = frozenset(
    {"arm_joint", "hand_joint", "head_joint", "torso_joint", "neck_joint"}
)
INDEXED_COMPONENTS = frozenset({"head_position"})

_GRIPPER_COMPONENTS = frozenset({"gripper_open", "gripper_open_scale"})
_MACHINE_COMPONENTS = (
    frozenset(FIXED_COMPONENTS)
    | SIDED_COMPONENTS
    | JOINT_COMPONENTS
    | INDEXED_COMPONENTS
)
_FIXED_NAME_FORMATS = {
    "eef_position": "eef_pos_{axis}_m",
    "eef_rotation": "eef_rot_euler_{axis}_rad",
    "head_rotation": "head_rot_euler_{axis}_rad",
    "head_orientation": "head_orient_quat_{axis}",
    "base_position": "base_pos_{axis}_m",
    "base_rotation": "base_rot_euler_{axis}_rad",
}
_INDEX_PATTERN = r"(?:0|[1-9][0-9]*)"
_MACHINE_NAME_PATTERN = re.compile(
    rf"(?:"
    rf"(?:left|right)_(?:arm|hand)_joint_{_INDEX_PATTERN}_rad"
    rf"|(?:left|right)_gripper_open(?:_scale)?"
    rf"|(?:left|right)_eef_pos_[xyz]_m"
    rf"|(?:left|right)_eef_rot_euler_[xyz]_rad"
    rf"|(?:head|torso|neck)_joint_{_INDEX_PATTERN}_rad"
    rf"|head_pos_{_INDEX_PATTERN}_m"
    rf"|head_rot_euler_[xyz]_rad"
    rf"|head_orient_quat_[xyzw]"
    rf"|base_pos_[xyz]_m"
    rf"|base_rot_euler_[xyz]_rad"
    rf")"
)
_NUMBERED_MACHINE_NAME_PATTERN = re.compile(
    rf"(?P<family>"
    rf"(?:left|right)_(?:arm|hand)_joint"
    rf"|(?:head|torso|neck)_joint"
    rf"|head_pos"
    rf")_(?P<index>{_INDEX_PATTERN})_(?:rad|m)"
)
_FIXED_MACHINE_NAME_PATTERN = re.compile(
    r"(?P<family>"
    r"(?:left|right)_eef_pos"
    r"|(?:left|right)_eef_rot_euler"
    r"|head_rot_euler"
    r"|head_orient_quat"
    r"|base_pos"
    r"|base_rot_euler"
    r")_(?P<axis>[xyzw])(?:_(?:m|rad))?"
)


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


def render_component_names(component: MachineComponent) -> tuple[str, ...] | None:
    """Render names for a machine component that exactly matches the standard."""

    kind = component.kind
    if not isinstance(kind, str) or kind not in _MACHINE_COMPONENTS:
        return None

    if kind in SIDED_COMPONENTS:
        if component.side not in ("left", "right"):
            return None
        side_prefix = f"{component.side}_"
    else:
        if component.side is not None:
            return None
        side_prefix = ""

    count = component.count
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        return None

    element_order = component.element_order
    if not isinstance(element_order, tuple) or len(element_order) != count:
        return None
    if any(
        not isinstance(element, str) or not element.strip()
        for element in element_order
    ):
        return None
    if len(set(element_order)) != count:
        return None

    if kind in FIXED_COMPONENTS:
        representation, unit, fixed_count = FIXED_COMPONENTS[kind]
        expected_order = (
            ("x", "y", "z", "w")
            if kind == "head_orientation"
            else ("x", "y", "z")
        )
        if (
            component.representation != representation
            or component.unit != unit
            or count != fixed_count
            or element_order != expected_order
        ):
            return None
        name_format = _FIXED_NAME_FORMATS[kind]
        return tuple(
            side_prefix + name_format.format(axis=axis) for axis in expected_order
        )

    if kind in JOINT_COMPONENTS:
        if component.representation != "joint_vector" or component.unit != "rad":
            return None
        return tuple(
            f"{side_prefix}{kind}_{index}_rad" for index in range(count)
        )

    if kind in INDEXED_COMPONENTS:
        if component.representation != "position_vector" or component.unit != "m":
            return None
        return tuple(f"head_pos_{index}_m" for index in range(count))

    if (
        kind not in _GRIPPER_COMPONENTS
        or component.representation != "scalar"
        or component.unit != "unitless"
        or count != 1
    ):
        return None
    return (side_prefix + kind,)


def is_standard_machine_name(name: str) -> bool:
    """Return whether one name exactly matches the canonical machine grammar."""

    return isinstance(name, str) and _MACHINE_NAME_PATTERN.fullmatch(name) is not None


def are_standard_machine_names(names: tuple[str, ...]) -> bool:
    """Validate one or more complete canonical machine-name families."""

    if isinstance(names, (str, bytes)):
        return False
    try:
        name_tuple = tuple(names)
    except TypeError:
        return False

    if not name_tuple or not all(
        is_standard_machine_name(name) for name in name_tuple
    ):
        return False
    if len(name_tuple) != len(set(name_tuple)):
        return False

    numbered_families: dict[str, list[str]] = {}
    fixed_families: dict[str, list[str]] = {}
    for name in name_tuple:
        numbered_match = _NUMBERED_MACHINE_NAME_PATTERN.fullmatch(name)
        if numbered_match is not None:
            family = numbered_match.group("family")
            numbered_families.setdefault(family, []).append(
                numbered_match.group("index")
            )
            continue

        fixed_match = _FIXED_MACHINE_NAME_PATTERN.fullmatch(name)
        if fixed_match is not None:
            family = fixed_match.group("family")
            fixed_families.setdefault(family, []).append(fixed_match.group("axis"))

    if any(
        indices != [str(index) for index in range(len(indices))]
        for indices in numbered_families.values()
    ):
        return False

    return all(
        axes
        == (
            ["x", "y", "z", "w"]
            if family == "head_orient_quat"
            else ["x", "y", "z"]
        )
        for family, axes in fixed_families.items()
    )
