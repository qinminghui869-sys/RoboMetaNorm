"""P1 相机字段规范建议与保守复核。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import tempfile

from robometanorm.camera.collision_checker import find_colliding_sources
from robometanorm.camera.discovery import discover_camera_features, find_camera_media
from robometanorm.camera.frame_sampler import (
    extract_rgb_frames,
    first_stage_ratios,
    second_stage_ratios,
)
from robometanorm.camera.media_probe import MediaInfo, probe_media
from robometanorm.camera.models import (
    CameraNameProposal,
    CameraNormalizationResult,
    CameraReviewCandidate,
    CameraReviewItem,
)
from robometanorm.camera.name_builder import build_camera_key
from robometanorm.camera.name_parser import propose_camera_name
from robometanorm.camera.prompt_builder import build_vlm_prompt
from robometanorm.camera.vlm_classifier import CameraSemantics, VlmClassifier
from robometanorm.domain.models import DatasetCandidate
from robometanorm.episode_sampling import select_representative_episodes


def normalize_cameras(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    *,
    vlm_classifier: VlmClassifier | None = None,
    confidence_threshold: float = 0.85,
) -> CameraNormalizationResult:
    """生成确定性相机建议；未知字段仅在足够证据下自动采纳。"""
    normalized_info = deepcopy(dict(source_info))
    camera_features = discover_camera_features(source_info)
    proposals: dict[str, CameraNameProposal] = {}
    review_items: list[CameraReviewItem] = []

    for source_key, feature in camera_features.items():
        proposal = propose_camera_name(source_key)
        if proposal is None:
            proposal, review_item = _resolve_unknown_camera(
                candidate,
                source_info,
                source_key,
                feature,
                tuple(camera_features),
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
    return CameraNormalizationResult(normalized_info, tuple(review_items))


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
            normalized_features[str(source_key)] = feature
            continue
        normalized_feature = dict(feature) if isinstance(feature, Mapping) else feature
        if isinstance(normalized_feature, dict):
            normalized_feature["codec"] = "ffv1" if proposal.modality == "depth" else "av1"
        normalized_features[proposal.target_key] = normalized_feature
    normalized_info["features"] = normalized_features


def _resolve_unknown_camera(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    source_key: str,
    feature: Mapping[str, object],
    other_camera_keys: Sequence[str],
    vlm_classifier: VlmClassifier | None,
    confidence_threshold: float,
) -> tuple[CameraNameProposal | None, CameraReviewItem | None]:
    """对未知字段按需抽帧请求 VLM；默认只写入复核。"""
    if vlm_classifier is None:
        return None, _review(source_key, "UNKNOWN_CAMERA_NAME")
    media_files = find_camera_media(candidate, source_key)
    if not media_files:
        return None, _review(source_key, "MEDIA_NOT_FOUND")
    try:
        media = probe_media(media_files[0])
        semantics = _classify_with_two_stages(
            candidate,
            source_info,
            source_key,
            feature,
            other_camera_keys,
            media_files,
            media,
            vlm_classifier,
        )
    except ValueError as error:
        return None, _review(source_key, "MEDIA_UNREADABLE", evidence={"message": str(error)})
    proposal = _proposal_from_semantics(source_key, semantics, confidence_threshold)
    if proposal is not None:
        return proposal, None
    candidates: tuple[CameraReviewCandidate, ...] = ()
    if semantics is not None:
        target_key = build_camera_key(
            semantics.direction_tokens, semantics.body_part, semantics.modality
        )
        if target_key is not None:
            candidates = (CameraReviewCandidate(target_key, semantics.confidence),)
    return None, _review(
        source_key,
        "VLM_LOW_CONFIDENCE_OR_AMBIGUOUS" if semantics else "VLM_UNAVAILABLE",
        candidates=candidates,
    )


def _classify_with_two_stages(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    source_key: str,
    feature: Mapping[str, object],
    other_camera_keys: Sequence[str],
    media_files: Sequence[Path],
    media: MediaInfo,
    vlm_classifier: VlmClassifier,
) -> CameraSemantics | None:
    """先少量抽帧，结果不足时按 P1 第二阶段加密抽帧。"""
    system_prompt, user_prompt = build_vlm_prompt(
        dataset_name=candidate.dataset_name,
        robot_type=_string_or_none(source_info.get("robot_type")),
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
            review_items.append(_review(source_key, "MEDIA_NOT_FOUND"))
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


def _string_or_none(value: object) -> str | None:
    """读取可选字符串元数据。"""
    return value if isinstance(value, str) else None
