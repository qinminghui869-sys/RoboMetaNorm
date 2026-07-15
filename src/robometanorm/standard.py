"""Strict rendering and parsing for canonical feature names."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Sequence
from urllib.parse import urlsplit

from robometanorm.models import (
    CameraAssignment,
    CameraEvidence,
    CameraSlot,
    DatasetEvidence,
    DatasetMapping,
    HardwareProfile,
    IdentityAssessment,
    Issue,
    MachineComponent,
    MappingRecord,
    MediaSample,
    NormalizationResult,
    RobotIdentityFact,
    SourceReference,
)


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
_OFFICIAL_SOURCE_KINDS = frozenset(
    {"manufacturer_site", "official_product", "official_manual"}
)
_LOCAL_IDENTITY_SOURCES = (
    "info_robot_type",
    "common_record",
    "tasks",
)

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


def _is_builtin_finite(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_finite_number(value: object) -> bool:
    return _is_builtin_finite(value) and 0 <= value <= 1


def _safe_text(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and not any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    )


def _safe_url(value: object) -> bool:
    if (
        not _safe_text(value)
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(hostname)
        and username is None
        and password is None
        and not parsed.netloc.endswith(":")
        and (port is None or 1 <= port <= 65535)
    )


def _slugify_robot_type(manufacturer: object, model: object) -> str | None:
    if not _safe_text(manufacturer) or not _safe_text(model):
        return None
    manufacturer_words = re.findall(r"[a-z0-9]+", manufacturer.casefold())
    model_words = re.findall(r"[a-z0-9]+", model.casefold())
    if not manufacturer_words or not model_words:
        return None
    return "_".join((*manufacturer_words, *model_words))


def _source_index(
    profile: HardwareProfile,
) -> dict[str, SourceReference] | None:
    if type(profile.sources) is not tuple:
        return None
    sources: dict[str, SourceReference] = {}
    for source in profile.sources:
        if (
            not isinstance(source, SourceReference)
            or not _safe_text(source.source_id)
            or not _safe_text(source.title)
            or not _safe_url(source.url)
            or not _safe_text(source.kind)
            or source.source_id in sources
        ):
            return None
        sources[source.source_id] = source
    return sources


def _referenced_sources(
    source_ids: object,
    sources: dict[str, SourceReference] | None,
) -> tuple[SourceReference, ...] | None:
    if sources is None or type(source_ids) is not tuple:
        return None
    references: list[SourceReference] = []
    seen: set[str] = set()
    for source_id in source_ids:
        if (
            not _safe_text(source_id)
            or source_id in seen
            or source_id not in sources
        ):
            return None
        seen.add(source_id)
        references.append(sources[source_id])
    return tuple(references)


def _citation_payloads(
    references: tuple[SourceReference, ...] | None,
) -> tuple[dict[str, object], ...]:
    if references is None:
        return ()
    return tuple(
        {
            "source_id": source.source_id,
            "title": source.title,
            "url": source.url,
            "kind": source.kind,
        }
        for source in references
    )


def _has_official_reference(
    references: tuple[SourceReference, ...] | None,
) -> bool:
    return references is not None and any(
        source.kind in _OFFICIAL_SOURCE_KINDS for source in references
    )


def _expected_assessment_relation(
    local_source: str,
    evidence: DatasetEvidence,
) -> frozenset[str] | None:
    identity = evidence.identity
    if local_source == "info_robot_type":
        state = identity.info_robot_type_state
        if any(issue.code == "INFO_ROBOT_TYPE_INVALID" for issue in identity.issues):
            return frozenset({"invalid"})
    elif local_source == "common_record":
        state = identity.common_record_state
    elif local_source == "tasks":
        state = identity.tasks_state
    else:
        return None

    if state == "missing":
        return frozenset({"missing"})
    if state in {"invalid", "unreadable"}:
        return frozenset({"invalid"})
    if state == "present":
        return frozenset({"supports", "conflicts", "unknown"})
    return None


def _identity_evidence_states_match_source(evidence: DatasetEvidence) -> bool:
    source_has_robot_type = "robot_type" in evidence.source_info
    identity = evidence.identity
    if (
        type(identity.info_robot_type_state) is not str
        or type(identity.common_record_state) is not str
        or type(identity.tasks_state) is not str
        or type(identity.tasks) is not tuple
    ):
        return False
    if source_has_robot_type:
        if identity.info_robot_type_state != "present":
            return False
        if identity.info_robot_type != evidence.source_info["robot_type"]:
            return False
    elif identity.info_robot_type_state != "missing" or identity.info_robot_type is not None:
        return False
    if identity.common_record_state in {"missing", "invalid", "unreadable"}:
        if identity.common_record is not None:
            return False
    elif identity.common_record_state != "present":
        return False
    if identity.tasks_state in {"missing", "unreadable"}:
        if identity.tasks:
            return False
    elif identity.tasks_state not in {"present", "invalid"}:
        return False
    return True


def _assessments_match(
    fact: RobotIdentityFact,
    evidence: DatasetEvidence,
) -> bool:
    if type(fact.assessments) is not tuple or len(fact.assessments) != 3:
        return False
    assessments: dict[str, IdentityAssessment] = {}
    for assessment in fact.assessments:
        if (
            not isinstance(assessment, IdentityAssessment)
            or not _safe_text(assessment.local_source)
            or assessment.local_source not in _LOCAL_IDENTITY_SOURCES
            or assessment.local_source in assessments
            or not _safe_text(assessment.relation)
            or not _safe_text(assessment.explanation)
        ):
            return False
        allowed_relations = _expected_assessment_relation(
            assessment.local_source, evidence
        )
        if allowed_relations is None or assessment.relation not in allowed_relations:
            return False
        assessments[assessment.local_source] = assessment
    if tuple(sorted(assessments)) != tuple(sorted(_LOCAL_IDENTITY_SOURCES)):
        return False

    relations = {assessment.relation for assessment in fact.assessments}
    if not _safe_text(fact.local_evidence_status):
        return False
    if fact.local_evidence_status == "consistent":
        return "supports" in relations and "conflicts" not in relations
    if fact.local_evidence_status == "conflicts_explained":
        return {"supports", "conflicts"} <= relations
    return False


def _identity_reliability(
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    confidence_threshold: float,
) -> tuple[
    bool,
    str,
    str | None,
    tuple[dict[str, object], ...],
]:
    if not isinstance(profile, HardwareProfile) or not isinstance(
        profile.identity, RobotIdentityFact
    ):
        return False, "硬件画像中的机器人身份结构无效", None, ()
    fact = profile.identity
    slug = _slugify_robot_type(fact.manufacturer, fact.model)
    sources = _source_index(profile)
    references = _referenced_sources(fact.source_ids, sources)
    citations = _citation_payloads(references)
    if slug is None:
        return False, "联网研究未提供安全且唯一的厂商与型号", None, citations
    if type(fact.ambiguous) is not bool or fact.ambiguous:
        return False, "联网研究的机器人身份仍有歧义", slug, citations
    if not _is_finite_number(fact.confidence) or fact.confidence < confidence_threshold:
        return False, "机器人身份置信度无效或低于门槛", slug, citations
    if not _safe_text(fact.reason):
        return False, "机器人身份缺少有效判断理由", slug, citations
    if not _identity_evidence_states_match_source(evidence):
        return False, "本地机器人身份状态与源信息不一致", slug, citations
    if not _assessments_match(fact, evidence):
        return False, "联网身份评估未逐项匹配本地证据状态", slug, citations
    if not _has_official_reference(references):
        return False, "机器人身份未引用厂商官网、官方产品页或官方手册", slug, citations
    return True, "机器人身份由一致的本地证据和官方来源确认", slug, citations


def _safe_fact_semantics(fact: RobotIdentityFact | None) -> dict[str, object]:
    if fact is None:
        return {}
    confidence = fact.confidence if _is_finite_number(fact.confidence) else None
    return {
        "manufacturer": fact.manufacturer if _safe_text(fact.manufacturer) else None,
        "model": fact.model if _safe_text(fact.model) else None,
        "confidence": confidence,
        "ambiguous": fact.ambiguous if type(fact.ambiguous) is bool else None,
        "local_evidence_status": (
            fact.local_evidence_status
            if _safe_text(fact.local_evidence_status)
            else None
        ),
        "reason": fact.reason if _safe_text(fact.reason) else None,
    }


def _identity_record_and_issue(
    normalized_info: dict[str, object],
    evidence: DatasetEvidence,
    profile: HardwareProfile | None,
    confidence_threshold: float,
    *,
    allow_change: bool,
) -> tuple[MappingRecord, Issue | None, bool]:
    source_exists = "robot_type" in evidence.source_info
    source_value = deepcopy(evidence.source_info.get("robot_type"))
    fact = profile.identity if isinstance(profile, HardwareProfile) else None
    if isinstance(profile, HardwareProfile):
        reliable, reason, candidate, citations = _identity_reliability(
            evidence, profile, confidence_threshold
        )
    else:
        reliable = False
        reason = "缺少联网硬件画像，机器人身份保持源值"
        candidate = None
        citations = ()
    profile_reliable = reliable

    if not allow_change:
        reliable = False
        reason = "缺少完整硬件画像或整体映射，机器人身份保持源值"
    elif not source_exists:
        reliable = False
        reason = "源 info.json 缺少 robot_type，未创建候选字段"
    elif not _safe_text(source_value):
        reliable = False
        reason = "源 robot_type 不是可安全替换的字符串"

    if reliable and candidate is not None:
        normalized_info["robot_type"] = candidate
        output = deepcopy(normalized_info["robot_type"])
        changed = source_value != output
        decision = "apply" if changed else "keep"
        issue = None
    else:
        output = deepcopy(source_value)
        changed = False
        decision = "review"
        issue = Issue(
            "ROBOT_IDENTITY_UNRESOLVED",
            reason,
            "robot_type",
            {"candidate": candidate} if candidate is not None else {},
        )

    return (
        MappingRecord(
            source_address="robot_type",
            source=source_value,
            output=output,
            candidate=candidate,
            changed=changed,
            vlm_semantics=_safe_fact_semantics(fact),
            citations=citations,
            decision=decision,
            reason=reason,
        ),
        issue,
        profile_reliable,
    )


def _has_usable_rgb_sample(camera: CameraEvidence) -> bool:
    if not isinstance(camera, CameraEvidence):
        return False
    schema = camera.schema
    shape = schema.shape
    return (
        type(schema.dtype) is str
        and schema.dtype in {"video", "image"}
        and type(shape) is tuple
        and len(shape) >= 3
        and type(shape[-1]) is int
        and shape[-1] in {3, 4}
        and type(camera.samples) is tuple
        and bool(camera.samples)
        and all(
            isinstance(sample, MediaSample) and sample.frame_path is not None
            for sample in camera.samples
        )
    )


def check_preconditions(evidence: DatasetEvidence) -> tuple[Issue, ...]:
    """Return ordered blocking issues for missing core dataset inputs."""

    machine_keys = {
        machine.schema.source_key
        for machine in evidence.machines
        if _safe_text(machine.schema.source_key)
    }
    has_action = "action" in machine_keys
    has_observation = any(
        key == "observation.state" or key.startswith("observation.state.")
        for key in machine_keys
    )
    has_primary_rgb = any(
        _has_usable_rgb_sample(camera) for camera in evidence.cameras
    )
    requirements = (
        (
            "MISSING_PRIMARY_CAMERA",
            has_primary_rgb,
            "缺少有媒体证据的 RGB 主摄像头",
        ),
        ("MISSING_ACTION", has_action, "缺少 action 机器字段"),
        (
            "MISSING_OBSERVATION",
            has_observation,
            "缺少 observation.state 机器字段",
        ),
    )
    return tuple(
        Issue(code, message, "preconditions", {}, "block")
        for code, present, message in requirements
        if not present
    )


def _feature_mapping(source_info: dict[str, object]) -> dict[str, object] | None:
    features = source_info.get("features")
    return features if type(features) is dict else None


def _schema_matches_source_feature(
    camera: CameraEvidence,
    source_feature: dict[str, object],
) -> bool:
    schema = camera.schema
    raw_dtype = source_feature.get("dtype")
    if type(raw_dtype) is not type(schema.dtype) or raw_dtype != schema.dtype:
        return False
    raw_shape = source_feature.get("shape")
    if (
        type(raw_shape) is not list
        or len(raw_shape) != len(schema.shape)
        or any(
            type(raw_dimension) is not type(schema_dimension)
            or raw_dimension != schema_dimension
            for raw_dimension, schema_dimension in zip(raw_shape, schema.shape)
        )
    ):
        return False
    raw_fps = source_feature.get("fps")
    if type(raw_fps) is not type(schema.fps) or raw_fps != schema.fps:
        return False
    if "codec" in source_feature:
        raw_codec = source_feature["codec"]
        if type(raw_codec) is not type(schema.codec) or raw_codec != schema.codec:
            return False
    elif schema.codec is not None:
        return False
    if "names" in source_feature:
        raw_names = source_feature["names"]
        if type(raw_names) is not list or tuple(raw_names) != schema.names:
            return False
    elif schema.names:
        return False
    return True


def _camera_media_check(
    camera: CameraEvidence,
    source_feature: dict[str, object],
    slot: CameraSlot,
) -> tuple[bool, bool, str]:
    schema = camera.schema
    shape = schema.shape
    if (
        type(schema.dtype) is not str
        or schema.dtype not in {"video", "image"}
        or type(shape) is not tuple
        or len(shape) < 3
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        or not _is_positive_media_number(schema.fps)
        or not _schema_matches_source_feature(camera, source_feature)
        or (schema.codec is not None and not _safe_text(schema.codec))
    ):
        return False, False, "相机 feature schema 无效或与源信息不一致"

    expected_height = shape[-3]
    expected_width = shape[-2]
    channels = shape[-1]
    if slot.modality == "rgb":
        target_codec = "av1"
        if channels not in {3, 4}:
            return False, False, "RGB 相机通道数必须为 3 或 4"
    elif slot.modality == "depth":
        target_codec = "ffv1"
        if channels != 1:
            return False, False, "Depth 相机通道数必须为 1"
    else:
        return False, False, "相机模态不属于 RGB 或 Depth"

    if type(camera.samples) is not tuple or not camera.samples:
        return False, False, "相机缺少可复核的本地媒体样本"
    needs_transcode = False
    has_frame = False
    for sample in camera.samples:
        if not isinstance(sample, MediaSample):
            return False, False, "本地媒体样本结构无效"
        if sample.frame_path is not None:
            if not isinstance(sample.frame_path, Path):
                return False, False, "本地媒体代表帧路径无效"
            has_frame = True
        if (
            sample.media_type != schema.dtype
            or not _is_positive_media_number(sample.fps)
            or sample.fps != schema.fps
            or type(sample.width) is not int
            or sample.width <= 0
            or sample.width != expected_width
            or type(sample.height) is not int
            or sample.height <= 0
            or sample.height != expected_height
            or (sample.codec is not None and not _safe_text(sample.codec))
        ):
            return False, False, "本地媒体样本与 feature 的类型、帧率或尺寸不一致"
        if sample.codec != target_codec:
            needs_transcode = True
    if not has_frame:
        return False, False, "相机缺少可复核的本地代表帧"
    return True, needs_transcode, "本地媒体与相机语义一致"


def _is_positive_media_number(value: object) -> bool:
    return _is_builtin_finite(value) and value > 0


def _safe_assignment_semantics(
    assignment: CameraAssignment | None,
    slot: CameraSlot | None,
    target_key: str | None,
) -> dict[str, object]:
    if assignment is None:
        return {}
    confidence = (
        assignment.confidence
        if _is_finite_number(assignment.confidence)
        else None
    )
    return {
        "source_key": assignment.source_key if _safe_text(assignment.source_key) else None,
        "camera_id": assignment.camera_id if _safe_text(assignment.camera_id) else None,
        "target_key": target_key,
        "confidence": confidence,
        "ambiguous": (
            assignment.ambiguous if type(assignment.ambiguous) is bool else None
        ),
        "reason": assignment.reason if _safe_text(assignment.reason) else None,
        "modality": slot.modality if slot is not None and _safe_text(slot.modality) else None,
    }


@dataclass
class _CameraPlan:
    camera: CameraEvidence
    source_key: str
    source_feature: dict[str, object] | None
    assignment: CameraAssignment | None
    slot: CameraSlot | None
    target_key: str | None
    target_codec: str | None
    citations: tuple[dict[str, object], ...]
    ready: bool
    needs_transcode: bool
    issue_code: str
    reason: str
    collision: bool = False


def _slot_is_safe_to_render(slot: CameraSlot) -> bool:
    return (
        _safe_text(slot.camera_id)
        and _safe_text(slot.mount_type)
        and type(slot.direction_tokens) is tuple
        and all(_safe_text(token) for token in slot.direction_tokens)
        and (slot.body_part is None or _safe_text(slot.body_part))
        and _safe_text(slot.modality)
        and _safe_text(slot.reason)
    )


def _mapping_inputs_are_unique(
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    mapping: DatasetMapping,
) -> bool:
    if (
        type(evidence.cameras) is not tuple
        or type(profile.cameras) is not tuple
        or type(mapping.cameras) is not tuple
    ):
        return False
    evidence_keys = [
        camera.schema.source_key
        for camera in evidence.cameras
        if isinstance(camera, CameraEvidence) and _safe_text(camera.schema.source_key)
    ]
    if len(evidence_keys) != len(evidence.cameras) or len(set(evidence_keys)) != len(evidence_keys):
        return False
    camera_ids = [
        slot.camera_id
        for slot in profile.cameras
        if isinstance(slot, CameraSlot) and _safe_text(slot.camera_id)
    ]
    if len(camera_ids) != len(profile.cameras) or len(set(camera_ids)) != len(camera_ids):
        return False
    assignment_keys = [
        assignment.source_key
        for assignment in mapping.cameras
        if isinstance(assignment, CameraAssignment) and _safe_text(assignment.source_key)
    ]
    return (
        len(assignment_keys) == len(mapping.cameras)
        and len(set(assignment_keys)) == len(assignment_keys)
        and set(assignment_keys) == set(evidence_keys)
    )


def _build_camera_plans(
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    mapping: DatasetMapping,
    confidence_threshold: float,
    *,
    identity_reliable: bool,
) -> list[_CameraPlan]:
    sources = _source_index(profile)
    features = _feature_mapping(evidence.source_info)
    structurally_unique = _mapping_inputs_are_unique(evidence, profile, mapping)
    assignments = (
        {assignment.source_key: assignment for assignment in mapping.cameras}
        if structurally_unique
        else {}
    )
    slots = (
        {slot.camera_id: slot for slot in profile.cameras}
        if structurally_unique
        else {}
    )
    plans: list[_CameraPlan] = []
    for camera in evidence.cameras:
        source_key = camera.schema.source_key
        source_value = features.get(source_key) if features is not None else None
        source_feature = source_value if type(source_value) is dict else None
        assignment = assignments.get(source_key)
        slot = (
            slots.get(assignment.camera_id)
            if assignment is not None and _safe_text(assignment.camera_id)
            else None
        )
        target_key = (
            render_camera_key(slot)
            if slot is not None and _slot_is_safe_to_render(slot)
            else None
        )
        references = (
            _referenced_sources(slot.source_ids, sources) if slot is not None else None
        )
        citations = _citation_payloads(references)
        target_codec = (
            "av1"
            if slot is not None and slot.modality == "rgb"
            else "ffv1"
            if slot is not None and slot.modality == "depth"
            else None
        )
        ready = False
        needs_transcode = False
        issue_code = "CAMERA_MAPPING_UNRESOLVED"
        reason = "整体映射未提供该源相机的唯一有效槽位"
        if not structurally_unique:
            reason = "相机证据、硬件槽位或整体映射包含缺失或重复标识"
        elif not identity_reliable:
            reason = "机器人身份未达到自动应用相机映射的条件"
        elif assignment is None:
            reason = "整体映射缺少该源相机"
        elif type(assignment.ambiguous) is not bool or assignment.ambiguous:
            reason = "源相机到硬件槽位的映射仍有歧义"
        elif not _is_finite_number(assignment.confidence) or assignment.confidence < confidence_threshold:
            reason = "相机映射置信度无效或低于门槛"
        elif not _safe_text(assignment.reason):
            reason = "相机映射缺少有效判断理由"
        elif slot is None:
            reason = "相机映射引用了不存在或不唯一的硬件槽位"
        elif type(slot.ambiguous) is not bool or slot.ambiguous:
            reason = "硬件画像中的相机槽位仍有歧义"
        elif not _is_finite_number(slot.confidence) or slot.confidence < confidence_threshold:
            reason = "相机槽位置信度无效或低于门槛"
        elif not _safe_text(slot.reason):
            reason = "相机槽位缺少有效判断理由"
        elif target_key is None or target_codec is None:
            reason = "相机槽位不能按标准渲染目标名称"
        elif not _has_official_reference(references):
            reason = "相机槽位自身未引用官方来源"
        elif source_feature is None:
            reason = "源 info.json 中缺少对应相机 feature"
        else:
            media_ok, needs_transcode, media_reason = _camera_media_check(
                camera, source_feature, slot
            )
            if not media_ok:
                issue_code = "CAMERA_MEDIA_MISMATCH"
                reason = media_reason
            else:
                ready = True
                reason = media_reason
        plans.append(
            _CameraPlan(
                camera=camera,
                source_key=source_key,
                source_feature=source_feature,
                assignment=assignment,
                slot=slot,
                target_key=target_key,
                target_codec=target_codec,
                citations=citations,
                ready=ready,
                needs_transcode=needs_transcode,
                issue_code=issue_code,
                reason=reason,
            )
        )
    return plans


def _mark_camera_collisions(
    plans: list[_CameraPlan],
    features: dict[str, object] | None,
) -> None:
    target_counts: dict[str, int] = {}
    for plan in plans:
        if plan.target_key is not None:
            target_counts[plan.target_key] = target_counts.get(plan.target_key, 0) + 1
    for plan in plans:
        if plan.target_key is not None and target_counts[plan.target_key] > 1:
            plan.collision = True

    if features is None:
        return
    plans_by_source = {plan.source_key: plan for plan in plans}
    changed = True
    while changed:
        changed = False
        for plan in plans:
            target_key = plan.target_key
            if (
                not plan.ready
                or plan.collision
                or target_key is None
                or target_key == plan.source_key
                or target_key not in features
            ):
                continue
            occupant = plans_by_source.get(target_key)
            occupant_vacates = (
                occupant is not None
                and occupant.ready
                and not occupant.collision
                and occupant.target_key is not None
                and occupant.target_key != target_key
            )
            if not occupant_vacates:
                plan.collision = True
                changed = True


def _fallback_camera_records(
    evidence: DatasetEvidence,
) -> tuple[tuple[MappingRecord, ...], tuple[Issue, ...]]:
    features = _feature_mapping(evidence.source_info)
    records: list[MappingRecord] = []
    issues: list[Issue] = []
    for camera in evidence.cameras:
        source_key = camera.schema.source_key
        source_feature = (
            deepcopy(features.get(source_key)) if features is not None else None
        )
        is_standard = parse_standard_camera_key(source_key) is not None
        if is_standard:
            decision = "keep"
            reason = "源相机名称已符合标准；缺少完整研究映射时不修改元数据"
        else:
            decision = "review"
            reason = "缺少完整硬件画像或整体映射，源相机名称保持不变"
            issues.append(
                Issue(
                    "CAMERA_MAPPING_UNRESOLVED",
                    reason,
                    f"features.{source_key}",
                )
            )
        records.append(
            MappingRecord(
                source_address=f"features.{source_key}",
                source=source_feature,
                output=deepcopy(source_feature),
                candidate=None,
                changed=False,
                vlm_semantics={},
                citations=(),
                decision=decision,
                reason=reason,
            )
        )
    return tuple(records), tuple(issues)


def _apply_camera_plans(
    normalized_info: dict[str, object],
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    mapping: DatasetMapping,
    confidence_threshold: float,
    *,
    identity_reliable: bool,
) -> tuple[tuple[MappingRecord, ...], tuple[Issue, ...]]:
    source_features = _feature_mapping(evidence.source_info)
    output_features = _feature_mapping(normalized_info)
    plans = _build_camera_plans(
        evidence,
        profile,
        mapping,
        confidence_threshold,
        identity_reliable=identity_reliable,
    )
    _mark_camera_collisions(plans, source_features)

    if output_features is not None:
        for plan in plans:
            if (
                plan.ready
                and not plan.collision
                and plan.target_key is not None
                and plan.target_key != plan.source_key
            ):
                output_features.pop(plan.source_key, None)
        for plan in plans:
            if (
                plan.ready
                and not plan.collision
                and plan.target_key is not None
                and plan.target_codec is not None
                and plan.source_feature is not None
            ):
                output_feature = deepcopy(plan.source_feature)
                output_feature["codec"] = plan.target_codec
                output_features[plan.target_key] = output_feature

    records: list[MappingRecord] = []
    issues: list[Issue] = []
    for plan in plans:
        scope = f"features.{plan.source_key}"
        if plan.ready and not plan.collision and plan.target_key is not None:
            output_feature = (
                deepcopy(output_features.get(plan.target_key))
                if output_features is not None
                else deepcopy(plan.source_feature)
            )
            changed = (
                plan.source_key != plan.target_key
                or plan.source_feature != output_feature
            )
            if plan.needs_transcode:
                decision = "review"
                reason = "相机名称与目标 metadata 已应用，但实际媒体仍需转码"
                issues.append(
                    Issue(
                        "MEDIA_TRANSCODE_REQUIRED",
                        reason,
                        scope,
                        {
                            "target_key": plan.target_key,
                            "target_codec": plan.target_codec,
                        },
                    )
                )
            else:
                decision = "apply" if changed else "keep"
                reason = "相机语义、媒体证据和官方来源均满足自动应用条件"
        else:
            output_feature = deepcopy(plan.source_feature)
            changed = False
            decision = "review"
            if plan.collision:
                issue_code = "CAMERA_NAME_COLLISION"
                reason = "目标相机名称重复或被未迁移的源 feature 占用"
            else:
                issue_code = plan.issue_code
                reason = plan.reason
            evidence_payload = (
                {"candidate": plan.target_key}
                if plan.target_key is not None
                else {}
            )
            issues.append(Issue(issue_code, reason, scope, evidence_payload))
        records.append(
            MappingRecord(
                source_address=scope,
                source=deepcopy(plan.source_feature),
                output=output_feature,
                candidate=plan.target_key,
                changed=changed,
                vlm_semantics=_safe_assignment_semantics(
                    plan.assignment, plan.slot, plan.target_key
                ),
                citations=plan.citations,
                decision=decision,
                reason=reason,
            )
        )
    return tuple(records), tuple(issues)


def apply_standard(
    evidence: DatasetEvidence,
    profile: HardwareProfile | None,
    mapping: DatasetMapping | None,
    *,
    confidence_threshold: float,
    extra_issues: Sequence[Issue] = (),
) -> NormalizationResult:
    """Apply only fully sourced identity and camera changes to a source copy."""

    if not _is_finite_number(confidence_threshold):
        raise ValueError("confidence_threshold 必须是 0 到 1 的有限内置数字")
    normalized_info = deepcopy(evidence.source_info)
    issues = [*evidence.issues, *extra_issues]
    allow_change = profile is not None and mapping is not None
    identity_record, identity_issue, identity_reliable = _identity_record_and_issue(
        normalized_info,
        evidence,
        profile,
        confidence_threshold,
        allow_change=allow_change,
    )
    if identity_issue is not None:
        issues.append(identity_issue)

    if not allow_change:
        camera_records, camera_issues = _fallback_camera_records(evidence)
    else:
        camera_records, camera_issues = _apply_camera_plans(
            normalized_info,
            evidence,
            profile,
            mapping,
            confidence_threshold,
            identity_reliable=identity_reliable,
        )
    issues.extend(camera_issues)
    return NormalizationResult(
        normalized_info=normalized_info,
        robot_identity=identity_record,
        camera_mappings=camera_records,
        machine_mappings=(),
        issues=tuple(issues),
    )
