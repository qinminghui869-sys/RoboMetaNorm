"""相机规范化的轻量领域对象。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraNameProposal:
    """由确定性规则得到的相机目标名称建议。"""

    source_key: str
    target_key: str
    modality: str
    method: str
    confidence: float = 1.0


@dataclass(frozen=True)
class CameraReviewCandidate:
    """供人工选择的目标字段候选。"""

    target_key: str
    confidence: float


@dataclass(frozen=True)
class CameraReviewItem:
    """P1 相机字段的人工复核项。"""

    source_key: str
    reason_code: str
    candidates: tuple[CameraReviewCandidate, ...]
    evidence: dict[str, object]


@dataclass(frozen=True)
class CameraNormalizationResult:
    """单数据集的相机规范建议及复核结果。"""

    normalized_info: dict[str, object]
    camera_review_items: tuple[CameraReviewItem, ...]
