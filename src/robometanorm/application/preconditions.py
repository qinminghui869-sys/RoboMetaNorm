"""P0 阶段的转换前置条件检查。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from robometanorm.domain.models import (
    DatasetCandidate,
    DatasetStatus,
    PreconditionReport,
    ReviewItem,
)


MEDIA_SUFFIXES = {".avi", ".bmp", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".png", ".webp"}
SCRIPT_SUFFIXES = {".py", ".sh"}


def check_preconditions(
    candidate: DatasetCandidate, info: Mapping[str, object]
) -> PreconditionReport:
    """检查 P0 必需证据，缺失项写入人工复核。"""
    features = _features_from(info)
    review_items: list[ReviewItem] = []

    has_action = "action" in features
    has_observation = any(name.startswith("observation.") for name in features)
    camera_count = sum(1 for name, value in features.items() if _is_camera(name, value))
    has_rgb_media = _has_media_file(candidate)

    if not has_rgb_media:
        review_items.append(_review("missing_rgb", "block", "未发现 RGB 视频或图片文件。"))
    if not has_action:
        review_items.append(_review("missing_action", "block", "元数据缺少 action 字段。"))
    if not has_observation:
        review_items.append(
            _review("missing_observation", "block", "元数据缺少 observation 字段。")
        )
    if camera_count == 0:
        review_items.append(
            _review("camera_primary", "block", "未发现可作为主摄像头的视觉特征。")
        )
    if not _contains_file(candidate.source_path, {".urdf"}):
        review_items.append(_review("missing_urdf", "block", "未发现机器人 URDF 文件。"))
    if not _contains_named_script(candidate.source_path, ("collect", "record", "capture")):
        review_items.append(
            _review("missing_collection_program", "warning", "未发现数据采集落盘程序。")
        )
    if not _contains_named_script(candidate.source_path, ("lerobot", "convert")):
        review_items.append(
            _review("missing_conversion_script", "warning", "未发现已有 LeRobot 转换脚本。")
        )

    return PreconditionReport(
        status=_status_from(review_items),
        review_items=tuple(review_items),
        camera_count=camera_count,
        machine_field_count=_machine_field_count(features),
    )


def _features_from(info: Mapping[str, object]) -> Mapping[str, object]:
    """容错读取 features，异常结构按空集合处理。"""
    features = info.get("features", {})
    if not isinstance(features, Mapping):
        return {}
    return {str(name): value for name, value in features.items()}


def _is_camera(name: str, value: object) -> bool:
    """P0 仅以视觉特征声明判定相机候选。"""
    if name.startswith("observation.images."):
        return True
    return isinstance(value, Mapping) and value.get("dtype") in {"video", "image"}


def _has_media_file(candidate: DatasetCandidate) -> bool:
    """检查视频目录或图片目录中是否已有可见媒体。"""
    roots = [path for path in (candidate.video_path, candidate.source_path / "images") if path]
    return any(_contains_file(path, MEDIA_SUFFIXES) for path in roots if path.is_dir())


def _contains_file(root: Path, suffixes: set[str]) -> bool:
    """找到第一个指定扩展名文件即停止扫描。"""
    return any(
        path.is_file() and path.suffix.lower() in suffixes for path in root.rglob("*")
    )


def _contains_named_script(root: Path, keywords: tuple[str, ...]) -> bool:
    """以脚本名提供的采集或转换证据作为 P0 依据。"""
    return any(
        path.is_file()
        and path.suffix.lower() in SCRIPT_SUFFIXES
        and any(keyword in path.name.lower() for keyword in keywords)
        for path in root.rglob("*")
    )


def _review(category: str, severity: str, reason: str) -> ReviewItem:
    """生成统一结构的基础复核项。"""
    return ReviewItem(
        review_id=category,
        category=category,
        severity=severity,
        reason=reason,
        evidence={},
        required_action="补充证据或确认该数据集是否可继续转换。",
    )


def _status_from(review_items: Sequence[ReviewItem]) -> DatasetStatus:
    """按 P0 状态优先级汇总复核项。"""
    if any(item.severity == "block" for item in review_items):
        return DatasetStatus.BLOCKED
    if review_items:
        return DatasetStatus.REVIEW
    return DatasetStatus.PASS


def _machine_field_count(features: Mapping[str, object]) -> int:
    """优先使用 action.names，缺失时回退到第一维 shape。"""
    action = features.get("action")
    if not isinstance(action, Mapping):
        return 0
    names = action.get("names")
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return len(names)
    shape = action.get("shape")
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and shape:
        first_dimension = shape[0]
        if isinstance(first_dimension, int):
            return first_dimension
    return 0
