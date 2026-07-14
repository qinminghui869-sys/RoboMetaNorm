"""机器字段发现、画像与复核数据对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProfileProgress:
    """Parquet 画像阶段向应用层报告的结构化进度。"""

    kind: str
    current: int
    total: int
    path: Path | None = None
    message: str | None = None


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
class ScalarProfile:
    """单个机器向量维度在代表性 Episode 中的稳健数值画像。"""

    sample_count: int
    min_value: float | None
    max_value: float | None
    p01: float | None
    p05: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    mean_value: float | None
    std_value: float | None
    nan_ratio: float
    inf_ratio: float
    unique_count: int


@dataclass(frozen=True)
class GripperRangeInference:
    """从代表性 Parquet 数据推断出的夹爪标称量程。"""

    closed_value: float
    open_value: float
    confidence: float
    clipping_required: bool
    evidence: str = "parquet_percentiles"


@dataclass(frozen=True)
class GripperDirectionEvidence:
    """夹爪数值方向及其证据来源。"""

    direction: str
    confidence: float
    method: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GripperTransformProposal:
    """不改写源数据的夹爪归一化转换建议。"""

    source_feature: str
    source_index: int
    source_name: str
    target_name: str
    source_closed: float
    source_open: float
    target_range: tuple[float, float]
    formula: str
    clipping_policy: str
    direction_evidence: str
    range_evidence: str
    confidence: float
    transform_required: bool
    observed_profile: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GripperFrameSample:
    """夹爪值与同一 Episode 视频时刻的关联样本。"""

    parquet_path: Path
    row_index: int
    timestamp_seconds: float
    value: float


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
    gripper_profiles: dict[str, ScalarProfile] = field(default_factory=dict)


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
    vlm_error: str | None = None


@dataclass(frozen=True)
class MachineNormalizationResult:
    """机器字段规范建议和机器复核结果。"""

    normalized_info: dict[str, object]
    machine_review_items: tuple[MachineReviewItem, ...]
    gripper_transform_proposals: tuple[GripperTransformProposal, ...] = ()
