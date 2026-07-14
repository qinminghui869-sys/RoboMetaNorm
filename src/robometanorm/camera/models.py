"""相机规范化的轻量领域对象。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraMount:
    """相机安装类别、方位与本体部位。"""

    mount_type: str
    direction_tokens: tuple[str, ...]
    body_part: str | None


@dataclass(frozen=True)
class RobotCameraTopology:
    """联网查询并通过严格 schema 校验的机器人本体相机拓扑。"""

    robot_id: str
    camera_mounts: tuple[CameraMount, ...]
    confidence: float
    ambiguous: bool


@dataclass(frozen=True)
class CameraNameProposal:
    """由确定性规则得到的相机目标名称建议。"""

    source_key: str
    target_key: str
    modality: str
    method: str
    confidence: float = 1.0
    inference_level: str = "CONFIRMED"


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
