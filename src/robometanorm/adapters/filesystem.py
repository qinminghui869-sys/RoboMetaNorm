"""文件系统中的数据集发现。"""

from __future__ import annotations

import os
from pathlib import Path

from robometanorm.models import DatasetCandidate, LayoutType


EXCLUDED_DIRECTORY_NAMES = {".git", ".cache", "__pycache__"}


def discover_datasets(root: Path, layout: LayoutType = LayoutType.AUTO) -> list[DatasetCandidate]:
    """递归发现以 ``meta/info.json`` 标识的数据集。"""
    # The explicit CLI root is the user's trust anchor. Resolve it before
    # traversal; links discovered inside that root remain untrusted.
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"输入根目录不存在或不是目录: {root}")

    candidates: list[DatasetCandidate] = []
    for directory, child_directories, filenames in os.walk(root):
        # 原地裁剪，避免进入工具目录。
        child_directories[:] = [
            name for name in child_directories if name not in EXCLUDED_DIRECTORY_NAMES
        ]
        meta_path = Path(directory)
        if meta_path.name != "meta" or "info.json" not in filenames:
            continue

        candidate = _build_candidate(root, meta_path.parent)
        if candidate is not None and _matches_requested_layout(candidate, layout):
            candidates.append(candidate)

    return sorted(candidates, key=lambda candidate: candidate.source_path.as_posix())


def _build_candidate(root: Path, dataset_path: Path) -> DatasetCandidate | None:
    """将目录转换为候选对象，过滤聚合目录。"""
    data_path = dataset_path / "data"
    video_path = dataset_path / "videos"
    depth_path = dataset_path / "depth"
    if not data_path.is_dir() and not video_path.is_dir():
        return None

    relative_path = dataset_path.relative_to(root)
    if len(relative_path.parts) <= 1:
        layout_type = LayoutType.FLAT
        task_name = None
    else:
        layout_type = LayoutType.TASK_GROUPED
        task_name = relative_path.parts[0]

    return DatasetCandidate(
        dataset_name=dataset_path.name,
        task_name=task_name,
        source_path=dataset_path,
        layout_type=layout_type,
        info_path=dataset_path / "meta" / "info.json",
        data_path=data_path if data_path.is_dir() else None,
        video_path=video_path if video_path.is_dir() else None,
        depth_path=depth_path if depth_path.is_dir() else None,
    )


def _matches_requested_layout(candidate: DatasetCandidate, layout: LayoutType) -> bool:
    """在显式布局模式下仅保留对应候选。"""
    return layout == LayoutType.AUTO or candidate.layout_type == layout
