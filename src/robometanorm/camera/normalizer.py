"""P1 相机字段规范建议与保守复核。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
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
    CameraNameProposal,
    CameraNormalizationResult,
    CameraReviewCandidate,
    CameraReviewItem,
)
from robometanorm.camera.naming import (
    build_camera_key,
    find_colliding_sources,
    propose_camera_name,
    propose_robot_camera_name,
)
from robometanorm.camera.vlm import CameraSemantics, VlmClassifier, build_vlm_prompt
from robometanorm.domain.models import DatasetCandidate
from robometanorm.episode_sampling import select_representative_episodes
from robometanorm.robot_identity import RobotIdentity


def normalize_cameras(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    *,
    robot_identity: RobotIdentity | None = None,
    vlm_classifier: VlmClassifier | None = None,
    confidence_threshold: float = 0.85,
) -> CameraNormalizationResult:
    """生成确定性相机建议；未知字段仅在足够证据下自动采纳。"""
    normalized_info = deepcopy(dict(source_info))
    camera_features = discover_camera_features(source_info)
    proposals: dict[str, CameraNameProposal] = {}
    review_items: list[CameraReviewItem] = []
    robot_id = robot_identity.canonical_id if robot_identity is not None else None

    for source_key, feature in camera_features.items():
        proposal = propose_robot_camera_name(robot_id, source_key)
        if proposal is None:
            proposal = propose_camera_name(source_key)
        if proposal is None:
            proposal, review_item = _resolve_unknown_camera(
                candidate,
                source_info,
                source_key,
                feature,
                tuple(camera_features),
                robot_id,
                vlm_classifier,
                confidence_threshold,
            )
            if review_item is not None:
                review_items.append(review_item)
        if proposal is not None:
            proposals[source_key] = proposal

    colliding_sources = find_colliding_sources(proposals.values())
    for source_key in sorted(colliding_sources):
        proposal = proposals.pop(source_key)
        review_items.append(
            _review(
                source_key,
                "TARGET_NAME_COLLISION",
                candidates=(CameraReviewCandidate(proposal.target_key, proposal.confidence),),
                evidence={"conflicting_target_key": proposal.target_key},
            )
        )

    review_items.extend(_validate_media(candidate, source_info, camera_features))
    _replace_feature_keys(normalized_info, proposals)
    return CameraNormalizationResult(
        normalized_info, tuple(_deduplicate_reviews(review_items))
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


def _resolve_unknown_camera(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    source_key: str,
    feature: Mapping[str, object],
    other_camera_keys: Sequence[str],
    robot_id: str | None,
    vlm_classifier: VlmClassifier | None,
    confidence_threshold: float,
) -> tuple[CameraNameProposal | None, CameraReviewItem | None]:
    """对未知字段按需抽帧请求 VLM；默认只写入复核。"""
    robot_identity_known = robot_id is not None
    if vlm_classifier is None:
        if robot_identity_known:
            return None, _review(
                source_key,
                "ROBOT_CAMERA_MAPPING_UNKNOWN",
                evidence={"robot_id": robot_id},
            )
        return None, _review(source_key, "UNKNOWN_CAMERA_NAME")
    media_files = find_camera_media(candidate, source_key)
    if not media_files:
        return None, _missing_media_review(candidate, source_key)
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
        return None, _review(source_key, "MEDIA_UNREADABLE", evidence={"message": str(error)})
    proposal = _proposal_from_semantics(source_key, semantics, confidence_threshold)
    if proposal is not None:
        if robot_identity_known:
            return None, _review(
                source_key,
                "ROBOT_CAMERA_MAPPING_UNKNOWN",
                candidates=(
                    CameraReviewCandidate(proposal.target_key, proposal.confidence),
                ),
                evidence={"robot_id": robot_id},
            )
        return proposal, None
    candidates: tuple[CameraReviewCandidate, ...] = ()
    if semantics is not None:
        target_key = build_camera_key(
            semantics.direction_tokens, semantics.body_part, semantics.modality
        )
        if target_key is not None:
            candidates = (CameraReviewCandidate(target_key, semantics.confidence),)
    if semantics is None:
        error_code = getattr(vlm_classifier, "last_error_code", None)
        evidence = dict(getattr(vlm_classifier, "last_error_evidence", {}) or {})
        error_message = getattr(vlm_classifier, "last_error", None)
        if isinstance(error_message, str):
            evidence["message"] = error_message
        reason_code = (
            "VLM_SEMANTICS_INVALID"
            if error_code == "VLM_SEMANTICS_INVALID"
            else "VLM_UNAVAILABLE"
        )
    else:
        reason_code = (
            "VLM_SEMANTICS_INSUFFICIENT"
            if not candidates
            else "VLM_LOW_CONFIDENCE_OR_AMBIGUOUS"
        )
        evidence = {}
    return None, _review(
        source_key,
        reason_code,
        candidates=candidates,
        evidence=evidence,
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


def _semantics_is_decisive(semantics: CameraSemantics | None) -> bool:
    """判断第一阶段是否仍需进入增强抽帧。"""
    return bool(
        semantics
        and semantics.confidence >= 0.85
        and not semantics.ambiguous
        and not semantics.need_human_review
        and build_camera_key(
            semantics.direction_tokens, semantics.body_part, semantics.modality
        )
        is not None
    )


def _select_second_stage_episodes(media_files: Sequence[Path]) -> tuple[Path, ...]:
    """按稳定顺序最多选择首、末两个 Episode。"""
    return select_representative_episodes(media_files)


def _proposal_from_semantics(
    source_key: str, semantics: CameraSemantics | None, confidence_threshold: float
) -> CameraNameProposal | None:
    """将足够置信的语义用规则构造目标名。"""
    if (
        semantics is None
        or semantics.modality == "unknown"
        or semantics.confidence < confidence_threshold
        or semantics.ambiguous
        or semantics.need_human_review
    ):
        return None
    target_key = build_camera_key(
        semantics.direction_tokens, semantics.body_part, semantics.modality
    )
    if target_key is None:
        return None
    return CameraNameProposal(
        source_key=source_key,
        target_key=target_key,
        modality=semantics.modality,
        method="vlm",
        confidence=semantics.confidence,
    )


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
