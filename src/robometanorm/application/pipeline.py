"""P0 扫描、检查和基础输出编排。"""

from __future__ import annotations

from collections.abc import Callable
import json
from dataclasses import replace
from pathlib import Path
import re
import tempfile

from robometanorm.adapters.filesystem import discover_datasets
from robometanorm.application.preconditions import check_preconditions
from robometanorm.camera.normalizer import normalize_cameras
from robometanorm.camera.media import extract_rgb_frame_at, find_camera_media
from robometanorm.camera.vlm import VlmClassifier
from robometanorm.domain.models import (
    DatasetCandidate,
    DatasetResult,
    DatasetStatus,
    LayoutType,
    PreconditionReport,
    ReviewItem,
)
from robometanorm.episode_sampling import select_representative_episodes
from robometanorm.machine.normalizer import normalize_machine_fields
from robometanorm.machine.models import (
    GripperDirectionEvidence,
    ParquetProfile,
    ProfileProgress,
)
from robometanorm.machine.profiling import (
    load_or_profile_parquets,
    sample_gripper_extremes,
)
from robometanorm.machine.rules import (
    gripper_direction_from_name,
    gripper_side_from_name,
    infer_gripper_range,
    is_out_of_scope_machine_field,
)
from robometanorm.machine.vlm import GripperDirectionResolver, MachineVlmResolver
from robometanorm.robot_identity import (
    RobotIdentity,
    resolve_robot_identity,
    robot_identity_payload,
)
from robometanorm.writers.json_writer import write_normalization_files


def scan_datasets(root: Path, layout: LayoutType = LayoutType.AUTO) -> list[DatasetResult]:
    """发现数据集、读取 info.json 并执行 P0 前置检查。"""
    results: list[DatasetResult] = []
    for candidate in discover_datasets(root, layout):
        try:
            source_info = _read_info(candidate)
            report = check_preconditions(candidate, source_info)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            results.append(_error_result(candidate, error))
            continue
        results.append(
            DatasetResult(
                candidate=candidate,
                status=report.status,
                review_items=report.review_items,
                camera_count=report.camera_count,
                machine_field_count=report.machine_field_count,
                source_info=source_info,
            )
        )
    return results


def normalize_datasets(
    root: Path,
    layout: LayoutType = LayoutType.AUTO,
    *,
    vlm_classifier: VlmClassifier | None = None,
    machine_vlm_resolver: MachineVlmResolver | None = None,
    gripper_direction_resolver: GripperDirectionResolver | None = None,
    confidence_threshold: float = 0.85,
    profile_progress: Callable[[ProfileProgress], None] | None = None,
) -> list[DatasetResult]:
    """执行 P1/P2 规范建议并生成两个输出文件。"""
    results = scan_datasets(root, layout)
    for index, result in enumerate(results):
        if result.source_info is None:
            continue
        try:
            robot_identity = resolve_robot_identity(
                result.candidate.info_path.parent, result.source_info
            )
            identity_reviews = _robot_identity_reviews(robot_identity)
            camera_result = normalize_cameras(
                result.candidate,
                result.source_info,
                robot_identity=robot_identity,
                vlm_classifier=vlm_classifier,
                confidence_threshold=confidence_threshold,
            )
            parquet_paths = _representative_parquet_paths(result.candidate)
            parquet_profile = _profile_dataset_parquets(
                result.candidate,
                result.source_info,
                profile_progress,
                parquet_paths=parquet_paths,
            )
            gripper_directions = (
                _resolve_gripper_directions(
                    result.candidate,
                    result.source_info,
                    parquet_paths,
                    parquet_profile,
                    gripper_direction_resolver,
                )
                if parquet_profile is not None
                and gripper_direction_resolver is not None
                else {}
            )
            machine_result = normalize_machine_fields(
                camera_result.normalized_info,
                parquet_profile,
                robot_identity=robot_identity,
                vlm_resolver=machine_vlm_resolver,
                dataset_name=result.candidate.dataset_name,
                gripper_directions=gripper_directions,
            )
            status = _status_after_reviews(
                result.status,
                camera_result.camera_review_items,
                machine_result.machine_review_items,
                identity_reviews,
            )
            report = PreconditionReport(
                status=status,
                review_items=(*result.review_items, *identity_reviews),
                camera_count=result.camera_count,
                machine_field_count=result.machine_field_count,
            )
            write_normalization_files(
                result.candidate,
                machine_result.normalized_info,
                report,
                camera_review_items=camera_result.camera_review_items,
                machine_review_items=machine_result.machine_review_items,
                gripper_transform_proposals=(
                    machine_result.gripper_transform_proposals
                ),
                robot_identity=robot_identity,
                phase="P2",
            )
            results[index] = replace(
                result,
                status=status,
                review_items=report.review_items,
                camera_review_count=len(camera_result.camera_review_items),
                machine_review_count=len(machine_result.machine_review_items),
            )
        except (OSError, RuntimeError, ValueError) as error:
            results[index] = _error_result(result.candidate, error)
    return results


def _profile_dataset_parquets(
    candidate: DatasetCandidate,
    source_info: dict[str, object],
    progress: Callable[[ProfileProgress], None] | None = None,
    *,
    parquet_paths: tuple[Path, ...] | None = None,
):
    """读取受限样本并比较各 Episode 布局，不改写任何源数据。"""
    if candidate.data_path is None:
        return None
    parquet_paths = parquet_paths or _representative_parquet_paths(candidate)
    if not parquet_paths:
        return None
    return load_or_profile_parquets(
        parquet_paths,
        candidate.info_path.parent / ".robometanorm_cache",
        progress=progress,
        gripper_indices=_declared_gripper_indices(source_info),
    )


def _representative_parquet_paths(candidate: DatasetCandidate) -> tuple[Path, ...]:
    """只选择首、末两个 Episode 的 Parquet。"""
    if candidate.data_path is None:
        return ()
    return select_representative_episodes(
        tuple(candidate.data_path.rglob("*.parquet"))
    )


def _resolve_gripper_directions(
    candidate: DatasetCandidate,
    source_info: dict[str, object],
    parquet_paths: tuple[Path, ...],
    profile: ParquetProfile,
    resolver: GripperDirectionResolver,
) -> dict[str, GripperDirectionEvidence]:
    """用同侧相机的同步低值/高值帧补充无法从名称确定的方向。"""
    requests = _unresolved_grippers(source_info, profile)
    if not requests or not parquet_paths:
        return {}
    fps_value = source_info.get("fps")
    fps = (
        float(fps_value)
        if isinstance(fps_value, (int, float))
        and not isinstance(fps_value, bool)
        and fps_value > 0
        else None
    )
    resolved: dict[str, GripperDirectionEvidence] = {}
    for profile_key, (side, feature_name, index, source_name) in requests.items():
        camera_key = _select_gripper_camera_key(source_info, side)
        if camera_key is None:
            continue
        media_paths = find_camera_media(candidate, camera_key)
        if not media_paths:
            continue
        samples = sample_gripper_extremes(
            parquet_paths, feature_name, index, fps=fps
        )
        if samples is None:
            continue
        low, high = samples
        low_media = _match_episode_media(low.parquet_path, parquet_paths, media_paths)
        high_media = _match_episode_media(high.parquet_path, parquet_paths, media_paths)
        if low_media is None or high_media is None:
            continue
        evidence: dict[str, object] = {
            "side": side,
            "source_feature": feature_name,
            "source_index": index,
            "source_name": source_name,
            "camera_key": camera_key,
            "low_value": low.value,
            "high_value": high.value,
            "low_timestamp_seconds": low.timestamp_seconds,
            "high_timestamp_seconds": high.timestamp_seconds,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="robometanorm_gripper_") as temp:
                temp_path = Path(temp)
                image_paths = (
                    extract_rgb_frame_at(
                        low_media, low.timestamp_seconds, temp_path / "low.jpg"
                    ),
                    extract_rgb_frame_at(
                        high_media, high.timestamp_seconds, temp_path / "high.jpg"
                    ),
                )
                direction = resolver.resolve(evidence, image_paths)
        except (OSError, RuntimeError, ValueError):
            continue
        if direction is not None:
            resolved[profile_key] = direction
    return resolved


def _unresolved_grippers(
    source_info: dict[str, object], profile: ParquetProfile
) -> dict[str, tuple[str, str, int, str]]:
    """选择量程有效且名称未说明开合方向的夹爪维度。"""
    features = source_info.get("features")
    if not isinstance(features, dict):
        return {}
    if _contains_out_of_scope_machine_fields(features):
        return {}
    requests: dict[str, tuple[str, str, int, str]] = {}
    for feature_name in ("action", "observation.state"):
        feature = features.get(feature_name)
        if not isinstance(feature, dict):
            continue
        names = feature.get("names")
        if not isinstance(names, list):
            continue
        for index, source_name in enumerate(names):
            if not isinstance(source_name, str) or "gripper" not in source_name.lower():
                continue
            side = gripper_side_from_name(source_name)
            profile_key = f"{feature_name}:{index}"
            scalar_profile = profile.gripper_profiles.get(profile_key)
            if (
                side is None
                or gripper_direction_from_name(source_name) is not None
                or scalar_profile is None
                or infer_gripper_range(scalar_profile) is None
            ):
                continue
            requests[profile_key] = (side, feature_name, index, source_name)
    return requests


def _select_gripper_camera_key(
    source_info: dict[str, object], side: str
) -> str | None:
    """优先选择同侧腕部相机，再选择其他同侧 RGB 相机。"""
    features = source_info.get("features")
    if not isinstance(features, dict):
        return None
    side_pattern = re.compile(rf"(?:^|[._-]){re.escape(side)}(?:$|[._-])")
    candidates = [
        str(key)
        for key, feature in features.items()
        if isinstance(feature, dict)
        and str(key).startswith("observation.images.")
        and side_pattern.search(str(key).lower()) is not None
        and _is_rgb_camera_feature(str(key), feature)
    ]
    if not candidates:
        return None

    def score(key: str) -> tuple[int, str]:
        lowered = key.lower()
        priority = 2 if "wrist" in lowered else 1 if "arm" in lowered else 0
        return (-priority, key)

    return sorted(candidates, key=score)[0]


def _match_episode_media(
    parquet_path: Path,
    parquet_paths: tuple[Path, ...],
    media_paths: tuple[Path, ...],
) -> Path | None:
    """按 Episode 编号匹配 Parquet 和视频，必要时按代表样本顺序回退。"""
    episode = _episode_number(parquet_path)
    if episode is not None:
        exact = [path for path in media_paths if _episode_number(path) == episode]
        if len(exact) == 1:
            return exact[0]
    if len(parquet_paths) == len(media_paths) and parquet_path in parquet_paths:
        return media_paths[parquet_paths.index(parquet_path)]
    if len(media_paths) == 1 and len(parquet_paths) == 1:
        return media_paths[0]
    return None


def _episode_number(path: Path) -> int | None:
    match = re.search(r"episode[_-]?(\d+)", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _declared_gripper_indices(
    source_info: dict[str, object],
) -> dict[str, tuple[int, ...]]:
    """从逐维 names 中提取 action/state 的夹爪维度。"""
    features = source_info.get("features")
    if not isinstance(features, dict):
        return {}
    if _contains_out_of_scope_machine_fields(features):
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for feature_name in ("action", "observation.state"):
        feature = features.get(feature_name)
        if not isinstance(feature, dict):
            continue
        names = feature.get("names")
        if not isinstance(names, list):
            continue
        indices = tuple(
            index
            for index, name in enumerate(names)
            if isinstance(name, str) and "gripper" in name.lower()
        )
        if indices:
            result[feature_name] = indices
    return result


def _contains_out_of_scope_machine_fields(
    features: dict[str, object],
) -> bool:
    """灵巧手、手指和骨架数据不进入夹爪画像或方向判定。"""
    for feature_name, feature in features.items():
        if not isinstance(feature, dict) or not (
            feature_name in {"action", "observation.state"}
            or feature_name.startswith("observation.state.")
        ):
            continue
        names = feature.get("names")
        declared = (
            [name for name in names if isinstance(name, str)]
            if isinstance(names, list)
            else []
        )
        if is_out_of_scope_machine_field(feature_name, declared):
            return True
    return False


def _is_rgb_camera_feature(key: str, feature: dict[str, object]) -> bool:
    """排除明确声明为 Depth 的同侧媒体。"""
    lowered_key = key.lower()
    dtype = feature.get("dtype")
    if "depth" in lowered_key or (
        isinstance(dtype, str) and "depth" in dtype.lower()
    ):
        return False
    shape = feature.get("shape")
    if isinstance(shape, list) and shape and shape[-1] == 1:
        return False
    return True


def _status_after_reviews(
    status: DatasetStatus,
    camera_review_items: tuple[object, ...],
    machine_review_items: tuple[object, ...],
    identity_review_items: tuple[ReviewItem, ...] = (),
) -> DatasetStatus:
    """规范化复核只会将 PASS 提升为 REVIEW。"""
    if (
        camera_review_items or machine_review_items or identity_review_items
    ) and status is DatasetStatus.PASS:
        return DatasetStatus.REVIEW
    return status


def _robot_identity_reviews(
    identity: RobotIdentity,
) -> tuple[ReviewItem, ...]:
    """将机器人身份冲突转换为通用人工复核项。"""
    if not identity.conflicts:
        return ()
    return (
        ReviewItem(
            review_id="robot_identity_conflict",
            category="ROBOT_IDENTITY_CONFLICT",
            severity="confirmation",
            reason="机器人身份元数据来源不一致，已按证据优先级选择。",
            evidence=robot_identity_payload(identity),
            required_action="核对机器人型号并确认采用的身份是否正确。",
        ),
    )


def _read_info(candidate: DatasetCandidate) -> dict[str, object]:
    """读取且限制 info.json 的顶层为 JSON 对象。"""
    with candidate.info_path.open("r", encoding="utf-8") as file_handle:
        source_info = json.load(file_handle)
    if not isinstance(source_info, dict):
        raise ValueError("info.json 顶层必须是 JSON 对象")
    return source_info


def _error_result(candidate: DatasetCandidate, error: Exception) -> DatasetResult:
    """将单数据集异常转换为可继续汇总的 ERROR 结果。"""
    review_item = ReviewItem(
        review_id="info_read_error",
        category="info_read_error",
        severity="block",
        reason=f"读取或写入元数据失败: {error}",
        evidence={"error_type": type(error).__name__},
        required_action="修复该数据集的元数据或文件权限后重新运行。",
    )
    return DatasetResult(
        candidate=candidate,
        status=DatasetStatus.ERROR,
        review_items=(review_item,),
        camera_count=0,
        machine_field_count=0,
        source_info=None,
    )
