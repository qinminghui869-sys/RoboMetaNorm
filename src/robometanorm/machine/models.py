"""机器字段发现、画像与复核数据对象。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VectorProfile:
    """Parquet 向量列的有限样本数值画像。"""

    column_name: str
    vector_length: int | None
    min_value: float | None
    max_value: float | None
    p01: float | None
    p50: float | None
    p99: float | None
    mean_value: float | None
    std_value: float | None
    nan_ratio: float
    inf_ratio: float
    mean_abs_diff: float | None = None
    max_abs_diff: float | None = None
    adjacent_correlation: float | None = None
    mean_vector_norm: float | None = None
    triplet_grouping_possible: bool = False
    quaternion_norm_valid: bool = False


@dataclass(frozen=True)
class ParquetProfile:
    """一个 episode Parquet 的 schema、受限样本和数值画像。"""

    row_count: int
    row_group_count: int
    schema_columns: tuple[str, ...]
    columns: dict[str, VectorProfile]
    samples: dict[str, np.ndarray]
    episode_count: int = 1
    inconsistent_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class MachineReviewItem:
    """P2 机器字段的人工复核项。"""

    source_feature: str
    source_slice: tuple[int, int] | None
    category: str
    severity: str
    declared_names: tuple[str, ...]
    vlm_result: dict[str, object] | None
    candidates: tuple[str, ...]
    required_action: str


@dataclass(frozen=True)
class MachineNormalizationResult:
    """机器字段规范建议和机器复核结果。"""

    normalized_info: dict[str, object]
    machine_review_items: tuple[MachineReviewItem, ...]
