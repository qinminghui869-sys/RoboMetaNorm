"""Compile confirmed dataset mappings into the compact YAML descriptor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import re

from robometanorm.models import (
    CameraAssignment,
    DatasetEvidence,
    DatasetMapping,
    HardwareProfile,
    Issue,
    MachineAssignment,
    MachineComponent,
    MachineSlice,
)
from robometanorm.standard import render_camera_key


_JOINT_LABEL = re.compile(r"(?:[a-z]+_)*(?:joint|j)[_-]?\d+", re.IGNORECASE)
_SIDED_JOINT = re.compile(
    r"(?P<side>left|right)(?:_[a-z]+)*_(?:joint|j)[_-]?(?P<index>\d+)(?:_[a-z]+)?$",
    re.IGNORECASE,
)
_MAIN_FOLLOWER_JOINT = re.compile(
    r"main_follower_joint_(?P<index>[0-9]{1,18})$", re.IGNORECASE
)
_MAIN_FOLLOWER_LABEL = re.compile(r"main_follower_joint_[0-9]+$", re.IGNORECASE)
_GROUP_WEIGHTS = {"arm_motion": 0.3, "gripper": 0.45}


@dataclass(frozen=True)
class AnnotationResult:
    """A descriptor or structured reasons why it cannot be compiled."""

    document: dict[str, object] | None
    issues: tuple[Issue, ...]


def preflight_annotation(evidence: DatasetEvidence) -> tuple[Issue, ...]:
    """Block raw joint labels that carry no side or component meaning."""

    source_file = _relative_info_path(evidence)
    issues: list[Issue] = []
    for machine in evidence.machines:
        names = machine.schema.names
        indices = [
            index
            for index, name in enumerate(names)
            if type(name) is str
            and _JOINT_LABEL.fullmatch(name) is not None
            and not _has_side(name)
            and (
                machine.schema.source_key not in {"action", "observation.state"}
                or _MAIN_FOLLOWER_LABEL.fullmatch(name) is None
            )
        ]
        if indices:
            observed = [names[index] for index in indices]
            issues.append(
                Issue(
                    "ANNOTATION_JOINT_AMBIGUOUS",
                    "无方位的泛化关节名无法安全映射到语义通道",
                    "annotation",
                    {
                        "source_file": source_file,
                        "source_feature": machine.schema.source_key,
                        "source_indices": indices,
                        "observed_names": observed,
                        "hint": "请提供 left/right 方位和连续关节顺序，或更正 info.json。",
                    },
                    "block",
                )
            )
    issues.extend(_joint_layout_issues(evidence))
    return tuple(issues)


def has_main_follower_candidate(evidence: DatasetEvidence) -> bool:
    """Return whether raw vectors form a locally valid main-arm candidate."""

    return bool(_main_joint_layouts(evidence)) and _main_joint_layout_issue(evidence) is None


def _has_side(name: str) -> bool:
    return bool({"left", "right"} & set(re.split(r"[_-]", name.casefold())))


def _joint_layout_issues(evidence: DatasetEvidence) -> tuple[Issue, ...]:
    issues: list[Issue] = []
    layouts: dict[str, tuple[tuple[str, int, int], ...]] = {}
    for machine in evidence.machines:
        source_feature = machine.schema.source_key
        if source_feature not in {"action", "observation.state"}:
            continue
        layout = _sided_joint_layout(machine.schema.names)
        if layout is not None:
            layouts[source_feature] = layout
            if not _is_contiguous(layout):
                issues.append(
                    Issue(
                        "ANNOTATION_JOINT_AMBIGUOUS",
                        "带方位的关节编号不连续，无法安全推断通道切片",
                        "annotation",
                        {
                            "source_file": _relative_info_path(evidence),
                            "source_feature": source_feature,
                            "observed_names": list(machine.schema.names),
                            "hint": "请提供每侧连续且顺序一致的关节编号。",
                        },
                        "block",
                    )
                )
    action = layouts.get("action")
    qpos = layouts.get("observation.state")
    if action is not None and qpos is not None and action != qpos:
        issues.append(
            Issue(
                "ANNOTATION_JOINT_LAYOUT_MISMATCH",
                "action 与 observation.state 的关节侧别或顺序不一致",
                "annotation",
                {
                    "source_file": _relative_info_path(evidence),
                    "source_features": ["action", "observation.state"],
                    "action_layout": [list(item) for item in action],
                    "observation_state_layout": [list(item) for item in qpos],
                    "hint": "请使两个字段的 left/right 关节编号与顺序一致。",
                },
                "block",
            )
        )
    main_issue = _main_joint_layout_issue(evidence)
    if main_issue is not None:
        issues.append(main_issue)
    return tuple(issues)


def _sided_joint_layout(
    names: tuple[object, ...],
) -> tuple[tuple[str, int, int], ...] | None:
    layout: list[tuple[str, int, int]] = []
    for position, name in enumerate(names):
        if type(name) is not str:
            continue
        matched = _SIDED_JOINT.fullmatch(name)
        if matched is not None:
            layout.append(
                (matched["side"].casefold(), int(matched["index"]), position)
            )
    return tuple(layout) if layout else None


def _is_contiguous(layout: tuple[tuple[str, int, int], ...]) -> bool:
    by_side: dict[str, list[tuple[int, int]]] = {}
    for side, index, position in layout:
        by_side.setdefault(side, []).append((index, position))
    return all(
        indices == list(range(indices[0], indices[0] + len(indices)))
        and positions == list(range(positions[0], positions[0] + len(positions)))
        and indices[0] in {0, 1}
        for indices, positions in (
            ([index for index, _ in values], [position for _, position in values])
            for values in by_side.values()
        )
    )


def _main_joint_layouts(
    evidence: DatasetEvidence,
) -> dict[str, tuple[tuple[int, int], ...]]:
    layouts: dict[str, tuple[tuple[int, int], ...]] = {}
    for machine in evidence.machines:
        source_feature = machine.schema.source_key
        if source_feature not in {"action", "observation.state"}:
            continue
        layout = tuple(
            (int(matched["index"]), position)
            for position, name in enumerate(machine.schema.names)
            if type(name) is str
            and (matched := _MAIN_FOLLOWER_JOINT.fullmatch(name)) is not None
        )
        if layout:
            layouts[source_feature] = layout
    return layouts


def _main_joint_layout_issue(evidence: DatasetEvidence) -> Issue | None:
    for machine in evidence.machines:
        source_feature = machine.schema.source_key
        if source_feature not in {"action", "observation.state"}:
            continue
        indices = [
            position
            for position, name in enumerate(machine.schema.names)
            if type(name) is str
            and _MAIN_FOLLOWER_LABEL.fullmatch(name) is not None
            and _MAIN_FOLLOWER_JOINT.fullmatch(name) is None
        ]
        if indices:
            return _main_layout_issue(
                evidence,
                "main_follower 关节编号超出可安全解析范围",
                {
                    "source_feature": source_feature,
                    "source_indices": indices,
                    "observed_names": [machine.schema.names[index] for index in indices],
                },
            )
    layouts = _main_joint_layouts(evidence)
    required = {"action", "observation.state"}
    if not layouts:
        return None
    if set(layouts) != required:
        return _main_layout_issue(
            evidence,
            "main_follower 关节必须同时出现在 action 与 observation.state 中",
            {"source_features": sorted(layouts)},
        )
    for source_feature, layout in layouts.items():
        if not _is_main_contiguous(layout):
            return _main_layout_issue(
                evidence,
                "main_follower 关节编号或原始向量位置不连续",
                {
                    "source_feature": source_feature,
                    "source_indices": [position for _, position in layout],
                    "observed_names": _machine_names(evidence, source_feature),
                },
            )
    if layouts["action"] != layouts["observation.state"]:
        return _main_layout_issue(
            evidence,
            "action 与 observation.state 的 main_follower 关节编号或顺序不一致",
            {
                "source_features": ["action", "observation.state"],
                "action_layout": [list(item) for item in layouts["action"]],
                "observation_state_layout": [
                    list(item) for item in layouts["observation.state"]
                ],
            },
        )
    return None


def _is_main_contiguous(layout: tuple[tuple[int, int], ...]) -> bool:
    indices = [index for index, _ in layout]
    positions = [position for _, position in layout]
    return (
        bool(indices)
        and indices[0] in {0, 1}
        and indices == list(range(indices[0], indices[0] + len(indices)))
        and positions == list(range(positions[0], positions[0] + len(positions)))
    )


def _machine_names(evidence: DatasetEvidence, source_feature: str) -> list[object]:
    return [
        *next(
            (
                machine.schema.names
                for machine in evidence.machines
                if machine.schema.source_key == source_feature
            ),
            (),
        )
    ]


def _main_layout_issue(
    evidence: DatasetEvidence,
    message: str,
    details: dict[str, object],
) -> Issue:
    return Issue(
        "ANNOTATION_MAIN_ARM_LAYOUT_INVALID",
        message,
        "annotation",
        {"source_file": _relative_info_path(evidence), **details},
        "block",
    )


def compile_annotation(
    evidence: DatasetEvidence,
    profile: HardwareProfile | None,
    mapping: DatasetMapping | None,
    *,
    normalized_info: Mapping[str, object],
    confidence_threshold: float,
) -> AnnotationResult:
    """Compile only fully confirmed camera and machine assignments."""

    issues = preflight_annotation(evidence)
    if issues:
        return AnnotationResult(None, issues)
    main_layouts = _main_joint_layouts(evidence)
    if not isinstance(profile, HardwareProfile) or not isinstance(
        mapping, DatasetMapping
    ):
        if main_layouts:
            return _main_unconfirmed("缺少已确认的硬件画像或数据映射")
        return _unconfirmed("缺少已确认的硬件画像或数据映射")
    if not _valid_threshold(confidence_threshold):
        return _unconfirmed("确认置信度门槛无效")
    robot_type = normalized_info.get("robot_type")
    if not _safe_text(robot_type):
        return _unconfirmed("info.json 缺少可用的 robot_type")

    main_component: MachineComponent | None = None
    if main_layouts:
        main_component, main_issue = _confirmed_main_component(
            evidence, profile, confidence_threshold
        )
        if main_issue is not None:
            return AnnotationResult(None, (main_issue,))

    components = _unique_by_id(profile.components, "component_id")
    if components is None:
        if main_layouts:
            return _main_unconfirmed("硬件组件标识缺失或重复")
        return _unconfirmed("硬件组件标识缺失或重复")
    camera_document, camera_issue = _compile_cameras(
        evidence, profile, mapping, confidence_threshold
    )
    if camera_issue is not None:
        if main_layouts:
            return _main_unconfirmed(camera_issue.message, camera_issue.evidence)
        return AnnotationResult(None, (camera_issue,))
    state_layout, action_layout, layout_issue = _compile_machine_layouts(
        evidence, mapping, components, confidence_threshold
    )
    if layout_issue is not None:
        if main_layouts:
            return _main_unconfirmed(layout_issue.message, layout_issue.evidence)
        return AnnotationResult(None, (layout_issue,))
    if _layout_signature(state_layout) != _layout_signature(action_layout):
        if main_layouts:
            return _main_unconfirmed("action 与 observation.state 的已确认组件顺序不一致")
        return _unconfirmed("action 与 observation.state 的已确认组件顺序不一致")

    if main_component is not None and not _main_slices_match(
        state_layout, action_layout, main_layouts, main_component
    ):
        return _main_unconfirmed("main_follower 关节未被完整且精确地映射到单臂组件")

    channels, channel_issue = _compile_channels(
        state_layout, output_side="main" if main_component is not None else None
    )
    if channel_issue is not None:
        if main_component is not None:
            return _main_unconfirmed(channel_issue.message, channel_issue.evidence)
        return AnnotationResult(None, (channel_issue,))
    groups = {channel["group"] for channel in channels.values()}
    document: dict[str, object] = {
        "version": "dataset_annotation_config_v1",
        "robot_type": robot_type,
        "adapter": {
            "base_type": "LeRobot",
            "base": {"qpos": "observation.state", "action": "action"},
            "cameras": camera_document,
        },
        "robot_channel_schema": {
            "version": "channel_schema_v1",
            "robot_type": robot_type,
            "channels": channels,
            "group_weights": {
                name: weight for name, weight in _GROUP_WEIGHTS.items() if name in groups
            },
        },
    }
    return AnnotationResult(document, ())


def _relative_info_path(evidence: DatasetEvidence) -> str:
    try:
        return evidence.candidate.info_path.relative_to(
            evidence.candidate.source_path
        ).as_posix()
    except ValueError:
        return Path("meta").joinpath("info.json").as_posix()


def _safe_text(value: object) -> bool:
    return type(value) is str and bool(value) and value == value.strip()


def _valid_threshold(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1


def _confirmed(value: object, threshold: float) -> bool:
    return (
        type(getattr(value, "ambiguous", None)) is bool
        and not value.ambiguous
        and type(getattr(value, "confidence", None)) in {int, float}
        and math.isfinite(value.confidence)
        and 0 <= value.confidence <= 1
        and value.confidence >= threshold
        and _safe_text(getattr(value, "reason", None))
    )


def _unique_by_id(
    values: tuple[object, ...], attribute: str
) -> dict[str, object] | None:
    indexed: dict[str, object] = {}
    for value in values:
        identifier = getattr(value, attribute, None)
        if not _safe_text(identifier) or identifier in indexed:
            return None
        indexed[identifier] = value
    return indexed


def _unconfirmed(
    message: str,
    evidence: dict[str, object] | None = None,
) -> AnnotationResult:
    return AnnotationResult(
        None,
        (
            Issue(
                "ANNOTATION_MAPPING_UNCONFIRMED",
                message,
                "annotation",
                evidence or {},
            ),
        ),
    )


def _main_unconfirmed(
    message: str,
    evidence: dict[str, object] | None = None,
) -> AnnotationResult:
    return AnnotationResult(
        None,
        (
            Issue(
                "ANNOTATION_MAIN_ARM_UNCONFIRMED",
                message,
                "annotation",
                evidence or {},
            ),
        ),
    )


def _confirmed_main_component(
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    threshold: float,
) -> tuple[MachineComponent | None, Issue | None]:
    source_robot_type = evidence.source_info.get("robot_type")
    identity = profile.identity
    if (
        not _safe_text(source_robot_type)
        or not _confirmed(identity, threshold)
        or not _safe_text(identity.manufacturer)
        or not _safe_text(identity.model)
    ):
        return None, _main_unconfirmed("单臂硬件身份未被充分确认").issues[0]
    arm_components = [
        component
        for component in profile.components
        if isinstance(component, MachineComponent) and component.kind == "arm_joint"
    ]
    if len(arm_components) != 1 or not _confirmed(arm_components[0], threshold):
        return None, _main_unconfirmed("硬件画像未确认恰有一个机械臂").issues[0]
    return arm_components[0], None


def _main_slices_match(
    state_layout: tuple[tuple[MachineSlice, MachineComponent], ...],
    action_layout: tuple[tuple[MachineSlice, MachineComponent], ...],
    raw_layouts: dict[str, tuple[tuple[int, int], ...]],
    arm_component: MachineComponent,
) -> bool:
    return all(
        _main_slice_matches(
            layout,
            raw_layouts[source_feature],
            arm_component.component_id,
        )
        for source_feature, layout in (
            ("observation.state", state_layout),
            ("action", action_layout),
        )
    )


def _main_slice_matches(
    layout: tuple[tuple[MachineSlice, MachineComponent], ...],
    raw_layout: tuple[tuple[int, int], ...],
    component_id: str,
) -> bool:
    matches = [
        machine_slice
        for machine_slice, component in layout
        if component.component_id == component_id and component.kind == "arm_joint"
    ]
    return len(matches) == 1 and (
        matches[0].start,
        matches[0].end,
    ) == (raw_layout[0][1], raw_layout[-1][1] + 1)


def _compile_cameras(
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    mapping: DatasetMapping,
    threshold: float,
) -> tuple[dict[str, str], Issue | None]:
    sources = {camera.schema.source_key for camera in evidence.cameras}
    slots = _unique_by_id(profile.cameras, "camera_id")
    if slots is None:
        return {}, _unconfirmed("相机标识缺失或重复").issues[0]
    output: dict[str, str] = {}
    mapped_sources: set[str] = set()
    for assignment in mapping.cameras:
        if not isinstance(assignment, CameraAssignment) or not _confirmed(
            assignment, threshold
        ):
            return {}, _unconfirmed("相机映射未确认").issues[0]
        if (
            assignment.source_key not in sources
            or assignment.source_key in mapped_sources
        ):
            return {}, _unconfirmed(
                "相机源字段缺失或重复", {"source_feature": assignment.source_key}
            ).issues[0]
        slot = slots.get(assignment.camera_id)
        if slot is None or not _confirmed(slot, threshold):
            return {}, _unconfirmed(
                "相机组件未确认", {"source_feature": assignment.source_key}
            ).issues[0]
        target = render_camera_key(slot)
        if target is None or target in output:
            return {}, _unconfirmed(
                "相机标准名称无效或重复",
                {"source_feature": assignment.source_key},
            ).issues[0]
        output[target] = assignment.source_key
        mapped_sources.add(assignment.source_key)
    if mapped_sources != sources:
        return {}, _unconfirmed("存在未确认的相机源字段").issues[0]
    return output, None


def _compile_machine_layouts(
    evidence: DatasetEvidence,
    mapping: DatasetMapping,
    components: dict[str, object],
    threshold: float,
) -> tuple[
    tuple[tuple[MachineSlice, MachineComponent], ...],
    tuple[tuple[MachineSlice, MachineComponent], ...],
    Issue | None,
]:
    schemas = {
        machine.schema.source_key: machine.schema for machine in evidence.machines
    }
    assignments = _unique_by_id(mapping.machines, "source_feature")
    if assignments is None:
        return (), (), _unconfirmed("机器映射源字段缺失或重复").issues[0]
    layouts: list[tuple[tuple[MachineSlice, MachineComponent], ...]] = []
    for source_feature in ("observation.state", "action"):
        schema = schemas.get(source_feature)
        assignment = assignments.get(source_feature)
        if schema is None or not isinstance(assignment, MachineAssignment):
            return (), (), _unconfirmed("缺少 observation.state 或 action 的确认映射").issues[0]
        layout, issue = _machine_layout(
            schema.shape, assignment, components, threshold
        )
        if issue is not None:
            return (), (), issue
        layouts.append(layout)
    return layouts[0], layouts[1], None


def _machine_layout(
    shape: tuple[object, ...],
    assignment: MachineAssignment,
    components: dict[str, object],
    threshold: float,
) -> tuple[tuple[tuple[MachineSlice, MachineComponent], ...], Issue | None]:
    if (
        not _confirmed(assignment, threshold)
        or not shape
        or type(shape[0]) is not int
        or shape[0] <= 0
        or not assignment.slices
    ):
        return (), _unconfirmed("机器向量或映射未确认").issues[0]
    layout: list[tuple[MachineSlice, MachineComponent]] = []
    cursor = 0
    for machine_slice in assignment.slices:
        component = (
            components.get(machine_slice.component_id)
            if isinstance(machine_slice, MachineSlice)
            else None
        )
        if (
            not isinstance(machine_slice, MachineSlice)
            or not isinstance(component, MachineComponent)
            or not _confirmed(component, threshold)
            or machine_slice.start != cursor
            or machine_slice.end <= machine_slice.start
            or machine_slice.end > shape[0]
            or machine_slice.end - machine_slice.start != component.count
            or machine_slice.element_order != component.element_order
        ):
            return (), _unconfirmed("机器切片或组件未确认").issues[0]
        layout.append((machine_slice, component))
        cursor = machine_slice.end
    if cursor != shape[0]:
        return (), _unconfirmed("机器切片未完整覆盖向量").issues[0]
    return tuple(layout), None


def _layout_signature(
    layout: tuple[tuple[MachineSlice, MachineComponent], ...]
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            machine_slice.start,
            machine_slice.end,
            component.kind,
            component.side,
            component.count,
            component.representation,
            component.unit,
            component.element_order,
        )
        for machine_slice, component in layout
    )


def _compile_channels(
    layout: tuple[tuple[MachineSlice, MachineComponent], ...],
    *,
    output_side: str | None = None,
) -> tuple[dict[str, dict[str, object]], Issue | None]:
    grouped: dict[str, dict[str, tuple[MachineSlice, MachineComponent]]] = {}
    for machine_slice, component in layout:
        if component.side not in {"left", "right"}:
            return {}, _unconfirmed("机器组件缺少可确认的 left/right 方位").issues[0]
        side = grouped.setdefault(component.side, {})
        if component.kind in side:
            return {}, _unconfirmed("同一语义组件重复映射").issues[0]
        side[component.kind] = (machine_slice, component)

    if output_side is not None and len(grouped) != 1:
        return {}, _unconfirmed("单臂组件方位不一致").issues[0]

    channels: dict[str, dict[str, object]] = {}
    for side in sorted(grouped):
        parts = grouped[side]
        channel_side = output_side or side
        joint = parts.get("arm_joint")
        if joint is not None:
            machine_slice, component = joint
            if component.representation != "joint_vector" or component.unit != "rad":
                return {}, _unconfirmed("关节组件表示或单位未确认").issues[0]
            channels[f"arm.{channel_side}.joint"] = _channel(
                machine_slice, "arm_motion", component.unit
            )

        position = parts.get("eef_position")
        rotation = parts.get("eef_rotation")
        if (position is None) != (rotation is None):
            return {}, _unconfirmed("末端位姿必须同时确认位置与旋转").issues[0]
        if position is not None and rotation is not None:
            position_slice, position_component = position
            rotation_slice, rotation_component = rotation
            if (
                position_component.representation != "position_xyz"
                or position_component.unit != "m"
                or rotation_component.representation != "euler_xyz"
                or rotation_component.unit != "rad"
                or position_slice.end != rotation_slice.start
            ):
                return {}, _unconfirmed("末端位姿组件表示、单位或顺序未确认").issues[0]
            channels[f"arm.{channel_side}.eef"] = _channel(
                MachineSlice(
                    position_slice.start,
                    rotation_slice.end,
                    "eef",
                    position_slice.element_order + rotation_slice.element_order,
                ),
                "arm_motion",
                "mixed_pose",
            )

        gripper = parts.get("gripper_open") or parts.get("gripper_open_scale")
        if gripper is not None:
            machine_slice, component = gripper
            if component.representation != "scalar" or component.count != 1:
                return {}, _unconfirmed("夹爪组件表示未确认").issues[0]
            channels[f"gripper.{channel_side}"] = _channel(
                machine_slice, "gripper", component.unit
            )

        unsupported = set(parts) - {
            "arm_joint",
            "eef_position",
            "eef_rotation",
            "gripper_open",
            "gripper_open_scale",
        }
        if unsupported:
            return {}, _unconfirmed("存在不能安全写入标注的机器组件").issues[0]
    if not channels:
        return {}, _unconfirmed("没有可确认的机械通道").issues[0]
    return channels, None


def _channel(
    machine_slice: MachineSlice,
    group: str,
    unit: str,
) -> dict[str, object]:
    return {
        "source": "qpos",
        "field": "qpos",
        "slice": [machine_slice.start, machine_slice.end],
        "group": group,
        "unit": unit,
        "norm": "robust_mad",
        "weight": 1.0,
        "optional": False,
    }
