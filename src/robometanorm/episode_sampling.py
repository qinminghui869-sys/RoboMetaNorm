"""为不同数据读取链路提供统一的 Episode 路径采样。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def select_representative_episodes(paths: Sequence[Path]) -> tuple[Path, ...]:
    """按路径排序并最多返回首、末两个 Episode。"""
    ordered = tuple(sorted(paths))
    if len(ordered) <= 2:
        return ordered
    return ordered[0], ordered[-1]
