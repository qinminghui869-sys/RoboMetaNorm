"""从元数据和目录中发现相机字段及媒体文件。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from robometanorm.domain.models import DatasetCandidate


MEDIA_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def discover_camera_features(info: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """返回 observation.images 前缀的相机特征定义。"""
    features = info.get("features")
    if not isinstance(features, Mapping):
        return {}
    return {
        str(key): value
        for key, value in features.items()
        if str(key).startswith("observation.images.") and isinstance(value, Mapping)
    }


def find_camera_media(candidate: DatasetCandidate, source_key: str) -> tuple[Path, ...]:
    """按视频目录名匹配源字段，返回排序后的 episode 文件。"""
    roots = [path for path in (candidate.video_path, candidate.depth_path) if path is not None]
    files = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in MEDIA_SUFFIXES
        and source_key in path.parts
    ]
    return tuple(sorted(files))
