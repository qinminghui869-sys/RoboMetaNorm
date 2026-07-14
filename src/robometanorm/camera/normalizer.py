"""P1 相机字段规范建议与保守复核。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import re
import tempfile

from robometanorm.camera.media import (
    MediaInfo,
    discover_camera_media_keys,
    discover_camera_features,
    extract_rgb_frames,
    find_camera_media,
    first_stage_ratios,
    probe_media,
    second_stage_ratios,
)
from robometanorm.camera.models import (
    CameraMount,
    CameraNameProposal,
    CameraNormalizationResult,
    CameraReviewCandidate,
    CameraReviewItem,
    RobotCameraTopology,
)
from robometanorm.camera.naming import (
    build_camera_key,
    find_colliding_sources,
    parse_standard_camera_key,
    propose_camera_name,
    propose_robot_camera_name,
)
from robometanorm.camera.topology import RobotCameraTopologyResolver
from robometanorm.camera.vlm import CameraSemantics, VlmClassifier, build_vlm_prompt
from robometanorm.domain.models import DatasetCandidate
from robometanorm.episode_sampling import select_representative_episodes
from robometanorm.robot_identity import RobotIdentity


@dataclass(frozen=True)
class _CameraEvidence:
    """单个源相机在联合约束前收集到的证据。"""

    source_key: str
    modality: str | None
    semantics: CameraSemantics | None
    issue: CameraReviewItem | None
    robot_alias: CameraNameProposal | None


def normalize_cameras(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    *,
    robot_identity: RobotIdentity | None = None,
    vlm_classifier: VlmClassifier | None = None,
    topology_resolver: RobotCameraTopologyResolver | None = None,
    confidence_threshold: float = 0.85,
) -> CameraNormalizationResult:
    """用机器人拓扑、本地画面和相机间关系联合生成命名建议。"""
    normalized_info = deepcopy(dict(source_info))
    camera_features = discover_camera_features(source_info)
    proposals: dict[str, CameraNameProposal] = {}
    review_items: list[CameraReviewItem] = []
    proposal_reviews: dict[str, CameraReviewItem] = {}
    robot_id = robot_identity.canonical_id if robot_identity is not None else None
    topology_lookup_requested = bool(
        robot_id is not None
        and topology_resolver is not None
        and any(
            propose_camera_name(source_key) is None
            for source_key in camera_features
        )
    )
    topology = (
        topology_resolver.resolve(robot_id)
        if topology_lookup_requested
        else None
    )
    pending: list[_CameraEvidence] = []
    occupied_mounts: set[CameraMount] = set()

    for source_key, feature in camera_features.items():
        standard_proposal = propose_camera_name(source_key)
        if standard_proposal is not None:
            proposals[source_key] = standard_proposal
            parsed = parse_standard_camera_key(source_key)
            if parsed is not None:
                occupied_mounts.add(parsed[0])
            continue
        pending.append(
            _collect_camera_evidence(
                candidate,
                source_info,
                source_key,
                feature,
                tuple(camera_features),
                robot_id,
                vlm_classifier,
                propose_robot_camera_name(robot_id, source_key),
            )
        )

    unresolved: list[_CameraEvidence] = []
    for evidence in pending:
        semantics = evidence.semantics
        if _semantics_is_decisive(
            semantics, confidence_threshold
        ) and _semantics_matches_source_category(evidence.source_key, semantics):
            mount = _mount_from_semantics(semantics)
            if mount is not None:
                confirmed = _topology_confirms_mount(
                    topology, mount, confidence_threshold
                )
                proposal = _proposal_from_semantics(
                    evidence.source_key,
                    semantics,
                    inference_level="CONFIRMED" if confirmed else "INFERRED",
                )
                if proposal is not None:
                    proposals[evidence.source_key] = proposal
                    if topology is not None and mount in topology.camera_mounts:
                        occupied_mounts.add(mount)
                    if not confirmed:
                        proposal_reviews[evidence.source_key] = _inferred_review(
                            evidence,
                            proposal,
                            robot_id,
                            topology,
                            reason=(
                                "external_local_evidence"
                                if mount.mount_type == "external"
                                else "local_evidence_without_matching_topology"
                            ),
                        )
                    continue
        if evidence.robot_alias is not None:
            alias = replace(evidence.robot_alias, inference_level="INFERRED")
            proposals[evidence.source_key] = alias
            parsed_alias = parse_standard_camera_key(alias.target_key)
            if parsed_alias is not None:
                occupied_mounts.add(parsed_alias[0])
            proposal_reviews[evidence.source_key] = _inferred_review(
                evidence,
                alias,
                robot_id,
                topology,
                reason="verified_robot_alias",
            )
            continue
        unresolved.append(evidence)

    _infer_unique_remaining_mounts(
        unresolved,
        topology,
        occupied_mounts,
        proposals,
        proposal_reviews,
        robot_id,
    )
    for evidence in unresolved:
        if evidence.source_key in proposals:
            continue
        review_items.append(
            _unresolved_review(evidence, robot_id, topology, topology_resolver)
        )

    colliding_sources = find_colliding_sources(proposals.values())
    for source_key in sorted(colliding_sources):
        proposal = proposals.pop(source_key)
        proposal_reviews.pop(source_key, None)
        review_items.append(
            _review(
                source_key,
                "TARGET_NAME_COLLISION",
                candidates=(CameraReviewCandidate(proposal.target_key, proposal.confidence),),
                evidence={
                    "conflicting_target_key": proposal.target_key,
                    "inference_level": "UNRESOLVED",
                },
            )
        )

    review_items.extend(proposal_reviews.values())
    review_items.extend(_validate_media(candidate, source_info, camera_features))
    _replace_feature_keys(normalized_info, proposals)
    confirmed_count = sum(
        proposal.inference_level == "CONFIRMED" for proposal in proposals.values()
    )
    inferred_count = sum(
        proposal.inference_level == "INFERRED" for proposal in proposals.values()
    )
    return CameraNormalizationResult(
        normalized_info=normalized_info,
        camera_review_items=tuple(_deduplicate_reviews(review_items)),
        confirmed_count=confirmed_count,
        inferred_count=inferred_count,
        unresolved_count=len(camera_features) - len(proposals),
        topology_error_count=int(
            topology_lookup_requested
            and topology_resolver is not None
            and bool(getattr(topology_resolver, "last_error_code", None))
        ),
    )


def _replace_feature_keys(
    normalized_info: dict[str, object], proposals: Mapping[str, CameraNameProposal]
) -> None:
    """仅替换无冲突的相机键，并写入目标编码建议。"""
    features = normalized_info.get("features")
    if not isinstance(features, Mapping):
        return
    normalized_features: dict[str, object] = {}
    for source_key, feature in features.items():
        proposal = proposals.get(str(source_key))
        if proposal is None:
            normalized_feature = (
                dict(feature) if isinstance(feature, Mapping) else feature
            )
            modality = (
                _infer_camera_modality(str(source_key), feature)
                if isinstance(feature, Mapping)
                else None
            )
            if isinstance(normalized_feature, dict) and modality is not None:
                normalized_feature["codec"] = (
                    "ffv1" if modality == "depth" else "av1"
                )
            normalized_features[str(source_key)] = normalized_feature
            continue
        normalized_feature = dict(feature) if isinstance(feature, Mapping) else feature
        if isinstance(normalized_feature, dict):
            normalized_feature["codec"] = "ffv1" if proposal.modality == "depth" else "av1"
        normalized_features[proposal.target_key] = normalized_feature
    normalized_info["features"] = normalized_features


def _infer_camera_modality(
    source_key: str, feature: Mapping[str, object]
) -> str | None:
    """在不判断安装位置的前提下保守推断相机模态。"""
    info = feature.get("info")
    video_info = info.get("video") if isinstance(info, Mapping) else None
    if isinstance(video_info, Mapping):
        is_depth_map = video_info.get("is_depth_map")
        if is_depth_map is True:
            return "depth"
        if is_depth_map is False:
            return "rgb"
    tokens = set(re.findall(r"[a-z0-9]+", source_key.lower()))
    if "depth" in tokens:
        return "depth"
    shape = feature.get("shape")
    if (
        isinstance(shape, Sequence)
        and not isinstance(shape, (str, bytes))
        and shape
        and shape[-1] == 3
    ):
        return "rgb"
    return None


def _collect_camera_evidence(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    source_key: str,
    feature: Mapping[str, object],
    other_camera_keys: Sequence[str],
    robot_id: str | None,
    vlm_classifier: VlmClassifier | None,
    robot_alias: CameraNameProposal | None,
) -> _CameraEvidence:
    """先收集证据，避免在看到同数据集其他相机前做最终决定。"""
    modality = _infer_camera_modality(source_key, feature)
    if vlm_classifier is None:
        return _CameraEvidence(
            source_key, modality, None, None, robot_alias
        )
    media_files = find_camera_media(candidate, source_key)
    if not media_files:
        return _CameraEvidence(
            source_key,
            modality,
            None,
            _missing_media_review(candidate, source_key),
            robot_alias,
        )
    try:
        media = probe_media(media_files[0])
        semantics = _classify_with_two_stages(
            candidate,
            source_info,
            source_key,
            feature,
            other_camera_keys,
            robot_id,
            media_files,
            media,
            vlm_classifier,
        )
    except ValueError as error:
        return _CameraEvidence(
            source_key,
            modality,
            None,
            _review(source_key, "MEDIA_UNREADABLE", evidence={"message": str(error)}),
            robot_alias,
        )
    issue: CameraReviewItem | None = None
    if semantics is None:
        error_code = getattr(vlm_classifier, "last_error_code", None)
        evidence = dict(getattr(vlm_classifier, "last_error_evidence", {}) or {})
        error_message = getattr(vlm_classifier, "last_error", None)
        if isinstance(error_message, str):
            evidence["message"] = error_message
        reason_code = (
            "VLM_SEMANTICS_INVALID"
            if error_code == "VLM_SEMANTICS_INVALID"
            else error_code or "VLM_UNAVAILABLE"
        )
        issue = _review(source_key, reason_code, evidence=evidence)
    return _CameraEvidence(
        source_key,
        modality,
        semantics,
        issue,
        robot_alias,
    )


def _classify_with_two_stages(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    source_key: str,
    feature: Mapping[str, object],
    other_camera_keys: Sequence[str],
    robot_id: str | None,
    media_files: Sequence[Path],
    media: MediaInfo,
    vlm_classifier: VlmClassifier,
) -> CameraSemantics | None:
    """先少量抽帧，结果不足时按 P1 第二阶段加密抽帧。"""
    system_prompt, user_prompt = build_vlm_prompt(
        dataset_name=candidate.dataset_name,
        robot_type=(
            robot_id
            or _string_or_none(source_info.get("robot_type"))
            or _string_or_none(source_info.get("root_type"))
        ),
        source_key=source_key,
        feature=feature,
        declared_fps=source_info.get("fps"),
        media=media,
        other_camera_keys=other_camera_keys,
    )
    with tempfile.TemporaryDirectory(prefix="robometanorm-camera-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        first_frames = extract_rgb_frames(
            media_files[0], first_stage_ratios(), temporary_path / "stage1", media
        )
        semantics = vlm_classifier.classify(system_prompt, user_prompt, first_frames)
        if _semantics_is_decisive(semantics):
            return semantics
        if getattr(vlm_classifier, "last_error_code", None) == "VLM_SEMANTICS_INVALID":
            return None

        selected_media = _select_second_stage_episodes(media_files)
        second_frames: list[Path] = []
        for index, media_path in enumerate(selected_media):
            second_frames.extend(
                extract_rgb_frames(
                    media_path,
                    second_stage_ratios(len(media_files)),
                    temporary_path / f"stage2_{index:02d}",
                )
            )
        return vlm_classifier.classify(system_prompt, user_prompt, tuple(second_frames))


def _semantics_is_decisive(
    semantics: CameraSemantics | None, confidence_threshold: float = 0.85
) -> bool:
    """判断第一阶段是否仍需进入增强抽帧。"""
    return bool(
        semantics
        and semantics.confidence >= confidence_threshold
        and not semantics.ambiguous
        and not semantics.need_human_review
        and build_camera_key(
            semantics.mount_type or "",
            semantics.direction_tokens,
            semantics.body_part,
            semantics.modality,
        )
        is not None
    )


def _select_second_stage_episodes(media_files: Sequence[Path]) -> tuple[Path, ...]:
    """按稳定顺序最多选择首、末两个 Episode。"""
    return select_representative_episodes(media_files)


def _proposal_from_semantics(
    source_key: str,
    semantics: CameraSemantics,
    *,
    inference_level: str,
) -> CameraNameProposal | None:
    """将已通过决策门槛的语义用内置规则构造目标名。"""
    target_key = build_camera_key(
        semantics.mount_type or "",
        semantics.direction_tokens,
        semantics.body_part,
        semantics.modality,
    )
    if target_key is None:
        return None
    return CameraNameProposal(
        source_key=source_key,
        target_key=target_key,
        modality=semantics.modality,
        method="vlm",
        confidence=semantics.confidence,
        inference_level=inference_level,
    )


def _mount_from_semantics(semantics: CameraSemantics) -> CameraMount | None:
    """把 VLM 语义转换为严格安装槽位。"""
    if semantics.mount_type is None:
        return None
    mount = CameraMount(
        semantics.mount_type, semantics.direction_tokens, semantics.body_part
    )
    return (
        mount
        if build_camera_key(
            mount.mount_type,
            mount.direction_tokens,
            mount.body_part,
            semantics.modality,
        )
        is not None
        else None
    )


def _topology_confirms_mount(
    topology: RobotCameraTopology | None,
    mount: CameraMount,
    confidence_threshold: float,
) -> bool:
    """联网拓扑和本地画面必须一致，才能标记确认。"""
    return bool(
        topology
        and not topology.partial
        and not topology.ambiguous
        and topology.confidence >= confidence_threshold
        and mount in topology.camera_mounts
    )


def _infer_unique_remaining_mounts(
    unresolved: Sequence[_CameraEvidence],
    topology: RobotCameraTopology | None,
    occupied_mounts: set[CameraMount],
    proposals: dict[str, CameraNameProposal],
    proposal_reviews: dict[str, CameraReviewItem],
    robot_id: str | None,
) -> None:
    """用已占槽位和弱提示传播唯一剩余的兼容解。"""
    if topology is None or topology.ambiguous or topology.confidence < 0.85:
        return
    available = [
        mount for mount in topology.camera_mounts if mount not in occupied_mounts
    ]
    made_progress = True
    while made_progress:
        made_progress = False
        remaining_sources = [
            evidence for evidence in unresolved if evidence.source_key not in proposals
        ]
        candidate_sets: list[tuple[_CameraEvidence, list[CameraMount]]] = []
        for evidence in remaining_sources:
            if evidence.modality not in {"rgb", "depth"}:
                continue
            candidates = _supported_remaining_mounts(
                evidence, available, occupied_mounts, len(remaining_sources)
            )
            candidate_sets.append((evidence, candidates))
        for evidence, candidates in candidate_sets:
            if len(candidates) != 1 or sum(
                candidates[0] in other_candidates
                for _, other_candidates in candidate_sets
            ) != 1:
                continue
            mount = candidates[0]
            target_key = build_camera_key(
                mount.mount_type,
                mount.direction_tokens,
                mount.body_part,
                evidence.modality,
            )
            if target_key is None:
                continue
            proposal = CameraNameProposal(
                evidence.source_key,
                target_key,
                evidence.modality,
                "constraint",
                topology.confidence,
                "INFERRED",
            )
            proposals[evidence.source_key] = proposal
            proposal_reviews[evidence.source_key] = _inferred_review(
                evidence,
                proposal,
                robot_id,
                topology,
                reason="unique_remaining_topology_slot",
            )
            occupied_mounts.add(mount)
            available.remove(mount)
            made_progress = True
            break


def _supported_remaining_mounts(
    evidence: _CameraEvidence,
    available: Sequence[CameraMount],
    occupied_mounts: set[CameraMount],
    remaining_source_count: int,
) -> list[CameraMount]:
    """先要求本地画面支持安装类别，再用槽位关系形成唯一解。"""
    semantics = evidence.semantics
    if (
        semantics is None
        or semantics.mount_type is None
        or semantics.confidence < 0.85
        or semantics.ambiguous
        or semantics.need_human_review
    ):
        return []
    expected_mount_type = _source_mount_type_constraint(evidence.source_key)
    compatible = [
        mount
        for mount in available
        if mount.mount_type == semantics.mount_type
        and (
            expected_mount_type is None
            or mount.mount_type == expected_mount_type
        )
    ]
    source_tokens = set(re.findall(r"[a-z0-9]+", evidence.source_key.lower()))
    hinted = [
        mount
        for mount in compatible
        if source_tokens.intersection(
            {*mount.direction_tokens, *([mount.body_part] if mount.body_part else [])}
        )
    ]
    if len(hinted) == 1:
        return hinted
    if (
        len(compatible) == 1
        and remaining_source_count == 1
        and occupied_mounts
    ):
        return compatible
    return []


def _inferred_review(
    evidence: _CameraEvidence,
    proposal: CameraNameProposal,
    robot_id: str | None,
    topology: RobotCameraTopology | None,
    *,
    reason: str,
) -> CameraReviewItem:
    """记录已生成兼容名称但仍需人工确认的推理。"""
    review_evidence = _camera_evidence_payload(evidence, robot_id, topology)
    review_evidence.update({"inference_level": "INFERRED", "reason": reason})
    return _review(
        evidence.source_key,
        "CAMERA_NAME_INFERRED",
        candidates=(
            CameraReviewCandidate(proposal.target_key, proposal.confidence),
        ),
        evidence=review_evidence,
    )


def _unresolved_review(
    evidence: _CameraEvidence,
    robot_id: str | None,
    topology: RobotCameraTopology | None,
    topology_resolver: RobotCameraTopologyResolver | None,
) -> CameraReviewItem:
    """只有无法形成唯一兼容解时保留源名。"""
    review_evidence = _camera_evidence_payload(evidence, robot_id, topology)
    review_evidence["inference_level"] = "UNRESOLVED"
    if topology is None and topology_resolver is not None:
        topology_error = getattr(topology_resolver, "last_error", None)
        topology_error_code = getattr(topology_resolver, "last_error_code", None)
        topology_error_evidence = dict(
            getattr(topology_resolver, "last_error_evidence", {}) or {}
        )
        if topology_error_code:
            review_evidence["topology_error_code"] = topology_error_code
        if topology_error:
            review_evidence["topology_error"] = topology_error
        if topology_error_evidence:
            review_evidence["topology_error_evidence"] = topology_error_evidence

    candidates: tuple[CameraReviewCandidate, ...] = ()
    source_category_conflict = bool(
        evidence.semantics is not None
        and not _semantics_matches_source_category(
            evidence.source_key, evidence.semantics
        )
    )
    if evidence.semantics is not None and not source_category_conflict:
        target_key = build_camera_key(
            evidence.semantics.mount_type or "",
            evidence.semantics.direction_tokens,
            evidence.semantics.body_part,
            evidence.semantics.modality,
        )
        if target_key is not None:
            candidates = (
                CameraReviewCandidate(target_key, evidence.semantics.confidence),
            )
    if evidence.issue is not None:
        review_evidence = {**evidence.issue.evidence, **review_evidence}
        return _review(
            evidence.source_key,
            evidence.issue.reason_code,
            candidates=evidence.issue.candidates or candidates,
            evidence=review_evidence,
        )
    topology_error_code = review_evidence.get("topology_error_code")
    if source_category_conflict:
        reason_code = "SOURCE_CAMERA_CATEGORY_CONFLICT"
    elif isinstance(topology_error_code, str):
        reason_code = topology_error_code
    elif evidence.semantics is None:
        reason_code = "CAMERA_NAME_UNRESOLVED"
    elif (
        evidence.semantics.confidence >= 0.85
        and not evidence.semantics.ambiguous
        and not evidence.semantics.need_human_review
    ):
        reason_code = "CAMERA_SEMANTICS_INCOMPLETE"
    else:
        reason_code = "VLM_LOW_CONFIDENCE_OR_AMBIGUOUS"
    return _review(
        evidence.source_key,
        reason_code,
        candidates=candidates,
        evidence=review_evidence,
    )


def _semantics_matches_source_category(
    source_key: str, semantics: CameraSemantics
) -> bool:
    """执行少量具有明确类别含义的源字段约束。"""
    expected_mount_type = _source_mount_type_constraint(source_key)
    return expected_mount_type is None or semantics.mount_type == expected_mount_type


def _source_mount_type_constraint(source_key: str) -> str | None:
    """third-view 类字段明确表示外部相机，但不提供具体方位。"""
    tokens = set(re.findall(r"[a-z0-9]+", source_key.lower()))
    if "third" in tokens and ({"view", "person"} & tokens):
        return "external"
    return None


def _camera_evidence_payload(
    evidence: _CameraEvidence,
    robot_id: str | None,
    topology: RobotCameraTopology | None,
) -> dict[str, object]:
    """生成不含敏感传输内容的可审计证据。"""
    return {
        "robot_id": robot_id,
        "robot_topology": _topology_payload(topology),
        "local_semantics": _semantics_payload(evidence.semantics),
        "modality": evidence.modality,
    }


def _topology_payload(topology: RobotCameraTopology | None) -> object:
    if topology is None:
        return None
    return {
        "robot_id": topology.robot_id,
        "camera_mounts": [
            {
                "mount_type": mount.mount_type,
                "direction_tokens": list(mount.direction_tokens),
                "body_part": mount.body_part,
            }
            for mount in topology.camera_mounts
        ],
        "confidence": topology.confidence,
        "ambiguous": topology.ambiguous,
        "partial": topology.partial,
        "rejected_mounts": [
            {
                "field": rejection.field,
                "value": rejection.value,
                "reason": rejection.reason,
            }
            for rejection in topology.rejected_mounts
        ],
    }


def _semantics_payload(semantics: CameraSemantics | None) -> object:
    if semantics is None:
        return None
    return {
        "mount_type": semantics.mount_type,
        "direction_tokens": list(semantics.direction_tokens),
        "body_part": semantics.body_part,
        "modality": semantics.modality,
        "confidence": semantics.confidence,
        "ambiguous": semantics.ambiguous,
        "need_human_review": semantics.need_human_review,
    }


def _validate_media(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    camera_features: Mapping[str, Mapping[str, object]],
) -> list[CameraReviewItem]:
    """检查实际媒体的可读性、FPS 和 shape；无媒体目录时跳过。"""
    if candidate.video_path is None and candidate.depth_path is None:
        return []
    review_items: list[CameraReviewItem] = []
    for source_key, feature in camera_features.items():
        media_files = find_camera_media(candidate, source_key)
        if not media_files:
            review_items.append(_missing_media_review(candidate, source_key))
            continue
        try:
            media = probe_media(media_files[0])
        except ValueError as error:
            review_items.append(
                _review(source_key, "MEDIA_UNREADABLE", evidence={"message": str(error)})
            )
            continue
        declared_fps = source_info.get("fps")
        if isinstance(declared_fps, (int, float)) and media.fps is not None and abs(media.fps - declared_fps) > 0.01:
            review_items.append(
                _review(
                    source_key,
                    "FPS_MISMATCH",
                    evidence={"declared_fps": declared_fps, "actual_fps": media.fps},
                )
            )
        shape = feature.get("shape")
        if (
            isinstance(shape, Sequence)
            and len(shape) >= 2
            and isinstance(shape[0], int)
            and isinstance(shape[1], int)
            and media.height is not None
            and media.width is not None
            and (shape[0], shape[1]) != (media.height, media.width)
        ):
            review_items.append(
                _review(
                    source_key,
                    "SHAPE_MISMATCH",
                    evidence={"declared_shape": list(shape), "actual_shape": [media.height, media.width]},
                )
            )
    return review_items


def _review(
    source_key: str,
    reason_code: str,
    *,
    candidates: tuple[CameraReviewCandidate, ...] = (),
    evidence: dict[str, object] | None = None,
) -> CameraReviewItem:
    """创建统一的 P1 相机复核项。"""
    return CameraReviewItem(source_key, reason_code, candidates, evidence or {})


def _missing_media_review(
    candidate: DatasetCandidate, source_key: str
) -> CameraReviewItem:
    """区分媒体缺失与元数据、目录字段键不一致。"""
    available_keys = discover_camera_media_keys(candidate)
    if available_keys:
        return _review(
            source_key,
            "MEDIA_KEY_MISMATCH",
            evidence={"available_media_keys": list(available_keys)},
        )
    return _review(source_key, "MEDIA_NOT_FOUND")


def _deduplicate_reviews(items: Sequence[CameraReviewItem]) -> list[CameraReviewItem]:
    """同一相机字段的同类复核只保留首条。"""
    unique: dict[tuple[str, str], CameraReviewItem] = {}
    for item in items:
        unique.setdefault((item.source_key, item.reason_code), item)
    return list(unique.values())


def _string_or_none(value: object) -> str | None:
    """读取可选字符串元数据。"""
    return value if isinstance(value, str) else None
