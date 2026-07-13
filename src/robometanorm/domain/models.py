"""P0 阶段使用的领域对象。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class LayoutType(str, Enum):
    """支持的数据集目录布局。"""

    AUTO = "auto"
    FLAT = "flat"
    TASK_GROUPED = "task_grouped"


class DatasetStatus(str, Enum):
    """命令行和复核文件使用的处理状态。"""

    PASS = "PASS"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DatasetCandidate:
    """扫描后统一表示的一个数据集。"""

    dataset_name: str
    task_name: str | None
    source_path: Path
    layout_type: LayoutType
    info_path: Path
    data_path: Path | None
    video_path: Path | None
    depth_path: Path | None


@dataclass(frozen=True)
class ReviewItem:
    """需要人工处理的一条证据化复核项。"""

    review_id: str
    category: str
    severity: str
    reason: str
    evidence: dict[str, object]
    required_action: str


@dataclass(frozen=True)
class PreconditionReport:
    """数据集前置条件检查的汇总结果。"""

    status: DatasetStatus
    review_items: tuple[ReviewItem, ...]
    camera_count: int
    machine_field_count: int


@dataclass(frozen=True)
class DatasetResult:
    """一次扫描或规范建议处理后的数据集结果。"""

    candidate: DatasetCandidate
    status: DatasetStatus
    review_items: tuple[ReviewItem, ...]
    camera_count: int
    machine_field_count: int
    source_info: dict[str, object] | None
    camera_review_count: int = 0
    machine_review_count: int = 0
