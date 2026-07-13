"""根据 Parquet 样本恢复父向量与子字段切片。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def resolve_child_slices(
    parent_samples: np.ndarray, child_samples: Mapping[str, np.ndarray]
) -> dict[str, tuple[int, int]]:
    """查找每个子字段在父向量中的连续切片。"""
    if parent_samples.ndim != 2:
        raise ValueError("父字段样本必须是二维向量")
    resolved: dict[str, tuple[int, int]] = {}
    for child_name, samples in child_samples.items():
        if samples.ndim != 2 or samples.shape[0] != parent_samples.shape[0]:
            continue
        width = samples.shape[1]
        for start in range(parent_samples.shape[1] - width + 1):
            end = start + width
            if np.allclose(parent_samples[:, start:end], samples, equal_nan=True):
                resolved[child_name] = (start, end)
                break
    return resolved
