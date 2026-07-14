"""P2 机器字段的保守规范建议。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re

from robometanorm.machine.models import (
    GripperDirectionEvidence,
    GripperRangeInference,
    GripperTransformProposal,
    MachineNormalizationResult,
    MachineReviewItem,
    ParquetProfile,
    VectorProfile,
)
from robometanorm.machine.rules import (
    PARENT_MACHINE_FEATURES,
    action_equals_state,
    build_confirmed_machine_name,
    build_names_from_semantics,
    declared_names,
    declared_vector_length,
    discover_machine_features,
    gripper_direction_from_name,
    gripper_side_from_name,
    infer_gripper_range,
    is_out_of_scope_machine_field,
    resolve_child_slices,
    risk_categories,
    unknown_unit_indices,
)
from robometanorm.machine.vlm import (
    MachineSemantics,
    MachineVlmResolver,
    can_apply_semantics,
)
from robometanorm.robot_identity import RobotIdentity, robot_identity_payload


def normalize_machine_fields(
    source_info: Mapping[str, object],
    parquet_profile: ParquetProfile | None,
    *,
    robot_identity: RobotIdentity | None = None,
    vlm_resolver: MachineVlmResolver | None = None,
    dataset_name: str | None = None,
    gripper_directions: Mapping[str, GripperDirectionEvidence] | None = None,
) -> MachineNormalizationResult:
    """只为维度、表示形式和单位均可确认的字段生成名称建议。"""
    normalized_info = deepcopy(dict(source_info))
    robot_type_evidence: object = (
        robot_identity_payload(robot_identity)
        if robot_identity is not None
        else source_info.get("robot_type") or source_info.get("root_type")
    )
    source_features = discover_machine_features(source_info)
    out_of_scope = _find_out_of_scope_field(source_features)
    if out_of_scope is not None:
        feature_name, names = out_of_scope
        return MachineNormalizationResult(
            normalized_info,
            (
                _review(
                    feature_name,
                    "OUT_OF_SCOPE_MACHINE_FIELD",
                    tuple(names),
                    _required_action("OUT_OF_SCOPE_MACHINE_FIELD"),
                ),
            ),
        )
    normalized_features = normalized_info.get("features")
    if not isinstance(normalized_features, dict):
        return MachineNormalizationResult(normalized_info, ())

    review_items: list[MachineReviewItem] = []
    gripper_proposals: list[GripperTransformProposal] = []
    equal_action_state = action_equals_state(parquet_profile)
    state_child_slices = _resolve_state_child_slices(parquet_profile, source_features)
    child_normalized_names: dict[str, list[str] | None] = {}

    # 子字段是 VLM 的最小分析单元；父字段只继承已经确认的子字段结果。
    for feature_name, feature in source_features.items():
        if feature_name in PARENT_MACHINE_FEATURES:
            continue
        normalized_names, field_reviews, field_proposals = _normalize_feature_names(
            feature_name,
            feature,
            parquet_profile,
            source_slice=state_child_slices.get(feature_name),
            equal_action_state=equal_action_state,
            vlm_resolver=vlm_resolver,
            dataset_name=dataset_name,
            robot_type=robot_type_evidence,
            gripper_directions=gripper_directions,
        )
        child_normalized_names[feature_name] = normalized_names
        review_items.extend(field_reviews)
        gripper_proposals.extend(field_proposals)
        if normalized_names is not None:
            _replace_names(normalized_features, feature_name, normalized_names)

    state_feature = source_features.get("observation.state")
    state_names: list[str] | None = None
    if state_feature is not None and state_child_slices:
        state_names = _build_parent_names_from_children(
            state_feature,
            parquet_profile,
            state_child_slices,
            child_normalized_names,
        )
        if state_names is not None:
            _replace_names(normalized_features, "observation.state", state_names)
    elif state_feature is not None:
        state_names, field_reviews, field_proposals = _normalize_feature_names(
            "observation.state",
            state_feature,
            parquet_profile,
            source_slice=None,
            equal_action_state=equal_action_state,
            vlm_resolver=vlm_resolver,
            dataset_name=dataset_name,
            robot_type=robot_type_evidence,
            gripper_directions=gripper_directions,
        )
        review_items.extend(field_reviews)
        gripper_proposals.extend(field_proposals)
        if state_names is not None:
            _replace_names(normalized_features, "observation.state", state_names)

    action_feature = source_features.get("action")
    if action_feature is not None:
        if equal_action_state and state_feature is not None:
            # action/state 样本和值均相同，action 直接复用 state 的布局结果。
            if state_names is not None:
                _replace_names(normalized_features, "action", state_names)
            if parquet_profile is not None:
                action_declared_names = declared_names(action_feature)
                if action_declared_names is not None:
                    direct_names, field_reviews, field_proposals = _analyze_grippers(
                        "action",
                        action_declared_names,
                        parquet_profile,
                        source_slice=None,
                        direction_evidence=gripper_directions or {},
                    )
                    review_items.extend(field_reviews)
                    gripper_proposals.extend(field_proposals)
                    base_names = (
                        list(state_names)
                        if state_names is not None
                        and len(state_names) == len(action_declared_names)
                        else list(action_declared_names)
                    )
                    for index, source_name in enumerate(action_declared_names):
                        if "gripper" not in source_name.lower():
                            continue
                        base_names[index] = direct_names.get(index, source_name)
                    if direct_names or state_names is not None:
                        _replace_names(normalized_features, "action", base_names)
        else:
            action_names, field_reviews, field_proposals = _normalize_feature_names(
                "action",
                action_feature,
                parquet_profile,
                source_slice=None,
                equal_action_state=equal_action_state,
                vlm_resolver=vlm_resolver,
                dataset_name=dataset_name,
                robot_type=robot_type_evidence,
                gripper_directions=gripper_directions,
            )
            review_items.extend(field_reviews)
            gripper_proposals.extend(field_proposals)
            if action_names is not None:
                _replace_names(normalized_features, "action", action_names)

    return MachineNormalizationResult(
        normalized_info,
        tuple(_deduplicate_reviews(review_items)),
        tuple(gripper_proposals),
    )


def _find_out_of_scope_field(
    source_features: Mapping[str, Mapping[str, object]],
) -> tuple[str, list[str]] | None:
    """在任何分析前定位当前规范未覆盖的灵巧手或骨架字段。"""
    for feature_name, feature in source_features.items():
        names = declared_names(feature) or []
        if is_out_of_scope_machine_field(feature_name, names):
            return feature_name, names
    return None


def _build_parent_names_from_children(
    parent_feature: Mapping[str, object],
    profile: ParquetProfile | None,
    child_slices: Mapping[str, tuple[int, int]],
    child_normalized_names: Mapping[str, list[str] | None],
) -> list[str] | None:
    """按 Parquet 实际切片将已确认的子字段名称拼回父字段。"""
    if profile is None:
        return None
    parent_profile = profile.columns.get("observation.state")
    parent_length = parent_profile.vector_length if parent_profile else None
    declared = declared_names(parent_feature)
    if parent_length is None:
        return None
    if declared is not None and len(declared) == parent_length:
        assembled: list[str | None] = list(declared)
    else:
        assembled = [None] * parent_length

    changed = False
    for child_name, (start, end) in child_slices.items():
        child_names = child_normalized_names.get(child_name)
        if child_names is None or len(child_names) != end - start:
            continue
        assembled[start:end] = child_names
        changed = True
    if not changed or not all(isinstance(name, str) for name in assembled):
        return None
    return [name for name in assembled if isinstance(name, str)]


def _resolve_state_child_slices(
    profile: ParquetProfile | None,
    source_features: Mapping[str, Mapping[str, object]],
) -> dict[str, tuple[int, int]]:
    """只以实际 Parquet 样本恢复 observation.state 的连续子切片。"""
    if profile is None or "observation.state" not in profile.samples:
        return {}
    child_samples = {
        name: profile.samples[name]
        for name in source_features
        if name.startswith("observation.state.") and name in profile.samples
    }
    return resolve_child_slices(profile.samples["observation.state"], child_samples)


def _normalize_feature_names(
    feature_name: str,
    feature: Mapping[str, object],
    profile: ParquetProfile | None,
    *,
    source_slice: tuple[int, int] | None,
    equal_action_state: bool,
    vlm_resolver: MachineVlmResolver | None,
    dataset_name: str | None,
    robot_type: object,
    gripper_directions: Mapping[str, GripperDirectionEvidence] | None,
) -> tuple[
    list[str] | None,
    list[MachineReviewItem],
    list[GripperTransformProposal],
]:
    """先校验实际长度，再在规则与 VLM 双重门槛下生成建议。"""
    names = declared_names(feature)
    declared_length = declared_vector_length(feature)
    vector_profile = profile.columns.get(feature_name) if profile else None
    actual_length = vector_profile.vector_length if vector_profile else None
    if (
        names is None
        or declared_length is None
        or (actual_length is not None and actual_length != declared_length)
    ):
        return None, [
            _review(
                feature_name,
                "NAMES_ORDER_MISMATCH",
                tuple(names or ()),
                "声明 names、shape 与 Parquet 实际向量长度不一致。",
                source_slice=source_slice,
            )
        ], []
    if profile is None or actual_length is None:
        return None, [
            _review(
                feature_name,
                "PARQUET_PROFILE_UNAVAILABLE",
                tuple(names),
                "未获得 Parquet 实际向量长度，不能安全修改名称。",
                source_slice=source_slice,
            )
        ], []
    if len(names) > declared_length:
        return None, [
            _review(
                feature_name,
                "NAMES_ORDER_MISMATCH",
                tuple(names),
                "声明 names 数量超过实际向量维度，不能安全修改名称。",
                source_slice=source_slice,
            )
        ], []
    if len(names) == declared_length and _feature_layout_inconsistent(
        feature_name, profile
    ):
        return None, [
            _review(
                feature_name,
                "CROSS_EPISODE_LAYOUT_INCONSISTENT",
                tuple(names),
                "确认不同 Episode 的字段 schema 和向量长度一致后再规范化。",
                source_slice=source_slice,
            )
        ], []
    if len(names) < declared_length:
        normalized, reviews = _normalize_grouped_feature_names(
            feature_name,
            feature,
            names,
            declared_length,
            vector_profile,
            source_slice=source_slice,
            equal_action_state=equal_action_state,
            vlm_resolver=vlm_resolver,
            dataset_name=dataset_name,
            robot_type=robot_type,
        )
        return normalized, reviews, []

    normalized_names = [build_confirmed_machine_name(name) or name for name in names]
    gripper_indices = {
        index for index, name in enumerate(names) if "gripper" in name.lower()
    }
    direct_gripper_names, gripper_reviews, proposals = _analyze_grippers(
        feature_name,
        names,
        profile,
        source_slice=source_slice,
        direction_evidence=gripper_directions or {},
    )
    for index, target_name in direct_gripper_names.items():
        normalized_names[index] = target_name
    categories = risk_categories(names)
    names_are_confirmed = all(
        build_confirmed_machine_name(name) is not None or index in gripper_indices
        for index, name in enumerate(names)
    )
    semantics: MachineSemantics | None = None
    resolver_error: str | None = None
    if vlm_resolver is not None and (categories or not names_are_confirmed):
        semantics, resolver_error = _resolve_vlm_semantics(
            vlm_resolver,
            _build_vlm_evidence(
                dataset_name,
                robot_type,
                feature_name,
                source_slice,
                declared_length,
                names,
                vector_profile,
                equal_action_state,
            ),
        )

    target_candidates: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    if semantics is not None:
        categories.update(_semantics_review_categories(semantics))
        target_candidates = _target_names_from_semantics(semantics)
        candidates = _review_candidates(semantics, target_candidates)
        if _can_apply_vlm_names(semantics, feature, categories, target_candidates):
            normalized_names = list(target_candidates)
            categories.clear()
            names_are_confirmed = True
        elif not categories:
            categories.add("VLM_SEMANTICS_REVIEW")
    elif resolver_error is not None:
        categories.add("VLM_RESOLUTION_FAILED")

    if not names_are_confirmed and not categories:
        categories.add("UNCLASSIFIED_MACHINE_FIELD")
    vlm_result = _semantics_to_dict(semantics) if semantics else None
    reviews: list[MachineReviewItem] = list(gripper_reviews)
    for category in sorted(categories):
        review_names, review_slice = _review_scope(category, names, source_slice)
        reviews.append(
            _review(
                feature_name,
                category,
                review_names,
                _required_action(category),
                source_slice=review_slice,
                vlm_result=vlm_result,
                candidates=candidates,
                vlm_error=(
                    resolver_error
                    if category == "VLM_RESOLUTION_FAILED"
                    else None
                ),
            )
        )
    return normalized_names, reviews, proposals


def _analyze_grippers(
    feature_name: str,
    names: list[str],
    profile: ParquetProfile,
    *,
    source_slice: tuple[int, int] | None,
    direction_evidence: Mapping[str, GripperDirectionEvidence],
) -> tuple[
    dict[int, str],
    list[MachineReviewItem],
    list[GripperTransformProposal],
]:
    """逐维确认夹爪量程和方向，并生成非破坏性转换建议。"""
    direct_names: dict[int, str] = {}
    reviews: list[MachineReviewItem] = []
    proposals: list[GripperTransformProposal] = []
    parent_offset = source_slice[0] if source_slice is not None else 0
    range_inferences = _infer_feature_gripper_ranges(feature_name, names, profile)

    for index, source_name in enumerate(names):
        if "gripper" not in source_name.lower():
            continue
        item_slice = (parent_offset + index, parent_offset + index + 1)
        side = gripper_side_from_name(source_name)
        target_name = f"{side}_gripper_open" if side is not None else None
        scalar_profile = profile.gripper_profiles.get(f"{feature_name}:{index}")
        range_inference = range_inferences.get(index)
        if range_inference is None:
            reviews.append(
                _review(
                    feature_name,
                    "GRIPPER_RANGE_UNKNOWN",
                    (source_name,),
                    _required_action("GRIPPER_RANGE_UNKNOWN"),
                    source_slice=item_slice,
                    candidates=(target_name,) if target_name else (),
                )
            )

        direction = gripper_direction_from_name(source_name)
        direction_method = "declared_name"
        direction_confidence = 0.99
        if direction is None and side is not None:
            supplied = direction_evidence.get(
                f"{feature_name}:{index}", direction_evidence.get(side)
            )
            if (
                supplied is not None
                and supplied.direction
                in {"increasing_is_open", "decreasing_is_open"}
                and supplied.confidence >= 0.85
            ):
                direction = supplied.direction
                direction_method = supplied.method
                direction_confidence = supplied.confidence
        if direction is None:
            reviews.append(
                _review(
                    feature_name,
                    "GRIPPER_DIRECTION_UNKNOWN",
                    (source_name,),
                    _required_action("GRIPPER_DIRECTION_UNKNOWN"),
                    source_slice=item_slice,
                    candidates=(target_name,) if target_name else (),
                )
            )
        if side is None:
            reviews.append(
                _review(
                    feature_name,
                    "UNKNOWN_LEFT_RIGHT",
                    (source_name,),
                    _required_action("UNKNOWN_LEFT_RIGHT"),
                    source_slice=item_slice,
                )
            )

        if range_inference is None or direction is None or target_name is None:
            continue

        lower = range_inference.closed_value
        upper = range_inference.open_value
        if direction == "increasing_is_open":
            source_closed, source_open = lower, upper
        else:
            source_closed, source_open = upper, lower
        formula = _gripper_formula(source_closed, source_open)
        transform_required = bool(
            range_inference.clipping_required
            or source_closed != 0.0
            or source_open != 1.0
        )
        proposal = GripperTransformProposal(
            source_feature=feature_name,
            source_index=index,
            source_name=source_name,
            target_name=target_name,
            source_closed=source_closed,
            source_open=source_open,
            target_range=(0.0, 1.0),
            formula=formula,
            clipping_policy="clip_to_unit_interval",
            direction_evidence=direction_method,
            range_evidence=range_inference.evidence,
            confidence=min(range_inference.confidence, direction_confidence),
            transform_required=transform_required,
            observed_profile=_scalar_profile_payload(scalar_profile),
        )
        proposals.append(proposal)
        if not transform_required:
            direct_names[index] = target_name

    return direct_names, reviews, proposals


def _infer_feature_gripper_ranges(
    feature_name: str,
    names: list[str],
    profile: ParquetProfile,
) -> dict[int, GripperRangeInference]:
    """独立推断各维量程；常量维仅可继承同字段唯一且可靠的量程。"""
    gripper_indices = [
        index for index, name in enumerate(names) if "gripper" in name.lower()
    ]
    inferred: dict[int, GripperRangeInference] = {}
    for index in gripper_indices:
        scalar = profile.gripper_profiles.get(f"{feature_name}:{index}")
        if scalar is None:
            continue
        result = infer_gripper_range(scalar)
        if result is not None:
            inferred[index] = result

    scales = {result.open_value for result in inferred.values()}
    if len(scales) != 1:
        return inferred
    scale = next(iter(scales))
    for index in gripper_indices:
        if index in inferred:
            continue
        scalar = profile.gripper_profiles.get(f"{feature_name}:{index}")
        if scalar is None or scalar.p50 is None or scalar.unique_count != 1:
            continue
        if scalar.nan_ratio > 0.05 or scalar.inf_ratio > 0.0:
            continue
        boundary_distance = min(abs(scalar.p50), abs(scalar.p50 - scale))
        if boundary_distance > 0.1 * scale:
            continue
        minimum, maximum = scalar.min_value, scalar.max_value
        clipping_required = bool(
            (minimum is not None and minimum < 0.0)
            or (maximum is not None and maximum > scale)
        )
        inferred[index] = GripperRangeInference(
            closed_value=0.0,
            open_value=scale,
            confidence=0.9,
            clipping_required=clipping_required,
            evidence="sibling_parquet_scale",
        )
    return inferred


def _gripper_formula(source_closed: float, source_open: float) -> str:
    """生成可审计且不依赖样本极值的线性归一化公式。"""
    if source_closed == 0.0 and source_open == 1.0:
        return "clip(x, 0, 1)"
    if source_closed == 0.0:
        return f"clip(x / {source_open:g}, 0, 1)"
    if source_open == 0.0:
        return f"clip(1 - x / {source_closed:g}, 0, 1)"
    return (
        f"clip((x - {source_closed:g}) / "
        f"{source_open - source_closed:g}, 0, 1)"
    )


def _scalar_profile_payload(profile: object) -> dict[str, object]:
    """保留足以复核量程结论的分位数和边界。"""
    if profile is None:
        return {}
    return {
        "sample_count": getattr(profile, "sample_count", None),
        "min": getattr(profile, "min_value", None),
        "max": getattr(profile, "max_value", None),
        "p01": getattr(profile, "p01", None),
        "p05": getattr(profile, "p05", None),
        "p50": getattr(profile, "p50", None),
        "p95": getattr(profile, "p95", None),
        "p99": getattr(profile, "p99", None),
        "unique_count": getattr(profile, "unique_count", None),
    }


def _review_scope(
    category: str,
    names: list[str],
    source_slice: tuple[int, int] | None,
) -> tuple[tuple[str, ...], tuple[int, int] | None]:
    """将可定位的单位问题收敛到实际受影响维度。"""
    if category != "UNKNOWN_UNIT":
        return tuple(names), source_slice
    indices = unknown_unit_indices(names)
    if not indices:
        return tuple(names), source_slice
    start, end = min(indices), max(indices) + 1
    parent_offset = source_slice[0] if source_slice is not None else 0
    return tuple(names[start:end]), (parent_offset + start, parent_offset + end)


def _normalize_grouped_feature_names(
    feature_name: str,
    feature: Mapping[str, object],
    names: list[str],
    vector_length: int,
    vector_profile: VectorProfile,
    *,
    source_slice: tuple[int, int] | None,
    equal_action_state: bool,
    vlm_resolver: MachineVlmResolver | None,
    dataset_name: str | None,
    robot_type: object,
) -> tuple[list[str] | None, list[MachineReviewItem]]:
    """分组名称不假定逐维顺序，仅允许 VLM 加规则共同确认后展开。"""
    semantics: MachineSemantics | None = None
    resolver_error: str | None = None
    if vlm_resolver is not None:
        semantics, resolver_error = _resolve_vlm_semantics(
            vlm_resolver,
            _build_vlm_evidence(
                dataset_name,
                robot_type,
                feature_name,
                source_slice,
                vector_length,
                names,
                vector_profile,
                equal_action_state,
            ),
        )
    target_candidates = _target_names_from_semantics(semantics) if semantics else ()
    candidates = _review_candidates(semantics, target_candidates) if semantics else ()
    semantic_categories = _semantics_review_categories(semantics) if semantics else set()
    if semantics is not None and _can_apply_vlm_names(
        semantics, feature, semantic_categories, target_candidates
    ):
        return list(target_candidates), []

    categories = ["NAMES_ORDER_MISMATCH"]
    categories.extend(sorted(semantic_categories))
    if resolver_error is not None:
        categories.append("VLM_RESOLUTION_FAILED")
    vlm_result = _semantics_to_dict(semantics) if semantics else None
    return None, [
        _review(
            feature_name,
            category,
            tuple(names),
            _required_action(category),
            source_slice=source_slice,
            vlm_result=vlm_result,
            candidates=candidates,
            vlm_error=(
                resolver_error if category == "VLM_RESOLUTION_FAILED" else None
            ),
        )
        for category in categories
    ]


def _semantics_review_categories(semantics: MachineSemantics) -> set[str]:
    """将 VLM 给出的风险语义转为不可自动写入的复核类别。"""
    categories: set[str] = set()
    for segment in semantics.segments:
        if segment.declared_name_status == "misleading":
            categories.add("DECLARED_NAME_CONFLICT")
        if segment.required_transform == "quaternion_to_euler":
            categories.add("QUATERNION_REQUIRES_EULER_CONVERSION")
        if (
            segment.semantic_type
            in {
                "arm_joint",
                "gripper_open",
                "eef_position",
                "eef_rotation_euler",
            }
            and segment.side == "unknown"
        ):
            categories.add("UNKNOWN_LEFT_RIGHT")
    return categories


def _review_candidates(
    semantics: MachineSemantics, target_candidates: tuple[str, ...]
) -> tuple[str, ...]:
    """优先展示可生成的标准名，否则展示 VLM 的受限语义候选。"""
    if target_candidates:
        return target_candidates
    candidates: list[str] = []
    for segment in semantics.segments:
        candidates.append(segment.semantic_type)
        candidates.extend(
            alternative["semantic_type"]
            for alternative in segment.alternatives
            if isinstance(alternative.get("semantic_type"), str)
        )
    return tuple(dict.fromkeys(candidates))


def _target_names_from_semantics(
    semantics: MachineSemantics,
) -> tuple[str, ...]:
    """按局部切片顺序拼接每个可支持区段的目标名称。"""
    names: list[str] = []
    for segment in semantics.segments:
        start, end = segment.local_slice
        built = build_names_from_semantics(segment, end - start)
        if built is None:
            return ()
        names.extend(built)
    return tuple(names)


def _feature_layout_inconsistent(feature_name: str, profile: ParquetProfile) -> bool:
    """父字段在任一已知子字段布局不一致时同样禁止写入名称。"""
    if feature_name in profile.inconsistent_columns:
        return True
    return feature_name == "observation.state" and any(
        name.startswith("observation.state.") for name in profile.inconsistent_columns
    )


def _build_vlm_evidence(
    dataset_name: str | None,
    robot_type: object,
    feature_name: str,
    source_slice: tuple[int, int] | None,
    declared_length: int,
    names: list[str],
    vector_profile: VectorProfile,
    equal_action_state: bool,
) -> dict[str, object]:
    """提供有限样本画像和实际布局，不向 VLM 暴露目标字段名。"""
    numeric_profile = {
        "min": vector_profile.min_value,
        "max": vector_profile.max_value,
        "p01": vector_profile.p01,
        "p50": vector_profile.p50,
        "p99": vector_profile.p99,
        "mean": vector_profile.mean_value,
        "std": vector_profile.std_value,
        "nan_ratio": vector_profile.nan_ratio,
        "inf_ratio": vector_profile.inf_ratio,
        "mean_abs_diff": vector_profile.mean_abs_diff,
        "max_abs_diff": vector_profile.max_abs_diff,
        "adjacent_correlation": vector_profile.adjacent_correlation,
        "mean_vector_norm": vector_profile.mean_vector_norm,
        "triplet_grouping_possible": vector_profile.triplet_grouping_possible,
        "quaternion_norm_valid": vector_profile.quaternion_norm_valid,
    }
    identity = dict(robot_type) if isinstance(robot_type, Mapping) else None
    canonical_id = identity.get("canonical_id") if identity is not None else robot_type
    return {
        "dataset_name": dataset_name,
        "robot_type": canonical_id if isinstance(canonical_id, str) else None,
        "robot_identity": identity,
        "parent_feature": "observation.state" if feature_name.startswith("observation.state.") else feature_name,
        "source_feature": feature_name,
        "source_slice": list(source_slice) if source_slice else None,
        "shape": [declared_length],
        "declared_names": names,
        "numeric_profile": numeric_profile,
        "relations": {
            "is_parent_slice": source_slice is not None,
            "action_equals_state": equal_action_state,
        },
        "rule_candidates": [],
    }


def _resolve_vlm_semantics(
    resolver: MachineVlmResolver, evidence: Mapping[str, object]
) -> tuple[MachineSemantics | None, str | None]:
    """VLM 失败仅转为复核，不影响其余字段处理。"""
    try:
        return resolver.resolve(evidence), None
    except Exception as error:
        return None, _sanitize_vlm_error(error)


def _sanitize_vlm_error(error: Exception) -> str:
    """保留可审核的协议错误，但移除 Authorization 和 API Key。"""
    message = " ".join(str(error).split())
    message = re.sub(
        r"(?i)authorization\s*:\s*bearer\s+[^;\s]+;?\s*",
        "",
        message,
    )
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "sk-[REDACTED]", message)
    return (message or type(error).__name__)[:1000]


def _can_apply_vlm_names(
    semantics: MachineSemantics,
    feature: Mapping[str, object],
    categories: set[str],
    candidates: tuple[str, ...],
) -> bool:
    """落实 P2 自动写入门槛，禁止以 VLM 推断未知单位或 wrist 关系。"""
    if categories or not candidates or len(set(candidates)) != len(candidates):
        return False
    if not can_apply_semantics(semantics, len(candidates)):
        return False
    declared_unit = feature.get("unit", feature.get("units"))
    for segment in semantics.segments:
        if segment.semantic_type == "head_orientation_quaternion":
            continue
        # 有量纲字段必须由元数据显式确认单位，VLM 不能单独补全单位。
        if segment.unit in {"unknown", "none"} or declared_unit != segment.unit:
            return False
    return True


def _semantics_to_dict(semantics: MachineSemantics) -> dict[str, object]:
    """将受限 VLM 语义写入复核文件，保留审计证据。"""
    return {
        "segments": [
            {
                "local_slice": list(segment.local_slice),
                "semantic_type": segment.semantic_type,
                "side": segment.side,
                "body_part": segment.body_part,
                "representation": segment.representation,
                "unit": segment.unit,
                "declared_name_status": segment.declared_name_status,
                "standardizable": segment.standardizable,
                "required_transform": segment.required_transform,
                "confidence": segment.confidence,
                "alternatives": list(segment.alternatives),
                "need_human_review": segment.need_human_review,
                "reason": segment.reason,
            }
            for segment in semantics.segments
        ],
        "need_human_review": semantics.need_human_review,
        "reason": semantics.reason,
    }


def _replace_names(features: dict[str, object], feature_name: str, names: list[str]) -> None:
    """深拷贝结果中仅替换目标 feature 的 names。"""
    feature = features.get(feature_name)
    if isinstance(feature, Mapping):
        updated_feature = dict(feature)
        updated_feature["names"] = names
        features[feature_name] = updated_feature


def _review(
    source_feature: str,
    category: str,
    declared_names: tuple[str, ...],
    required_action: str,
    *,
    source_slice: tuple[int, int] | None = None,
    vlm_result: dict[str, object] | None = None,
    candidates: tuple[str, ...] = (),
    vlm_error: str | None = None,
) -> MachineReviewItem:
    """创建 P2 机器字段人工复核项。"""
    return MachineReviewItem(
        source_feature=source_feature,
        source_slice=source_slice,
        category=category,
        severity="confirmation",
        declared_names=declared_names,
        vlm_result=vlm_result,
        candidates=candidates,
        required_action=required_action,
        vlm_error=vlm_error,
    )


def _required_action(category: str) -> str:
    """提供与风险类别对应的人工确认动作。"""
    actions = {
        "WRIST_EEF_RELATION_UNKNOWN": "确认 wrist 字段是否为末端执行器位姿。",
        "GRIPPER_RANGE_UNKNOWN": "确认夹爪物理量程和目标量程。",
        "GRIPPER_DIRECTION_UNKNOWN": "确认夹爪开合方向。",
        "UNKNOWN_UNIT": "确认物理单位后再添加 _m 或 _rad。",
        "UNKNOWN_LEFT_RIGHT": "确认字段对应的左侧、右侧或双侧后再规范化。",
        "DECLARED_NAME_CONFLICT": "确认声明名称与实际字段语义是否一致。",
        "QUATERNION_REQUIRES_EULER_CONVERSION": "确认旋转表示转换后再写入欧拉角字段名。",
        "UNCLASSIFIED_MACHINE_FIELD": "确认字段语义、表示形式和单位后再规范化。",
        "OUT_OF_SCOPE_MACHINE_FIELD": (
            "灵巧手、手指、骨架或关键点字段不在当前规范范围，"
            "保留全部源字段并人工复核。"
        ),
        "VLM_SEMANTICS_REVIEW": "确认 VLM 语义及其单位、表示形式后再采纳。",
        "VLM_RESOLUTION_FAILED": "检查 VLM 服务或改为人工确认字段语义。",
        "CROSS_EPISODE_LAYOUT_INCONSISTENT": "确认不同 Episode 的字段 schema 和向量长度一致。",
    }
    return actions.get(category, "确认字段结构和实际向量顺序。")


def _deduplicate_reviews(items: list[MachineReviewItem]) -> list[MachineReviewItem]:
    """同一字段、切片与类别只保留一条复核项。"""
    unique: dict[
        tuple[str, tuple[int, int] | None, str], MachineReviewItem
    ] = {}
    for item in items:
        unique[(item.source_feature, item.source_slice, item.category)] = item
    return list(unique.values())
