"""Shared domain models for dataset evidence and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LayoutType(str, Enum):
    """Supported dataset directory layouts."""

    AUTO = "auto"
    FLAT = "flat"
    TASK_GROUPED = "task_grouped"


class DatasetStatus(str, Enum):
    """Dataset processing outcomes."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_name: str
    task_name: str | None
    source_path: Path
    layout_type: LayoutType
    info_path: Path
    data_path: Path | None
    video_path: Path | None
    depth_path: Path | None


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    scope: str
    evidence: dict[str, object] = field(default_factory=dict)
    severity: str = "review"


@dataclass(frozen=True)
class IdentityEvidence:
    info_robot_type_state: str
    info_robot_type: object | None
    common_record_state: str
    common_record: object | None
    tasks_state: str
    tasks: tuple[object, ...]
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class MediaSample:
    relative_path: str
    media_type: str
    codec: str | None
    fps: float | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    pixel_format: str | None
    frame_path: Path | None


@dataclass(frozen=True)
class FeatureSchema:
    source_key: str
    dtype: object | None
    shape: tuple[object, ...]
    names: tuple[object, ...]
    fps: object | None
    codec: object | None


@dataclass(frozen=True)
class CameraEvidence:
    schema: FeatureSchema
    samples: tuple[MediaSample, ...]


@dataclass(frozen=True)
class GripperRange:
    index: int
    minimum: float | None
    maximum: float | None
    finite_count: int
    nonfinite_count: int


@dataclass(frozen=True)
class ParquetEpisodeEvidence:
    relative_path: str
    schema_columns: tuple[str, ...]
    vector_lengths: dict[str, int | None]


@dataclass(frozen=True)
class MachineEvidence:
    schema: FeatureSchema
    episodes: tuple[ParquetEpisodeEvidence, ...]
    episode_lengths: tuple[int, ...]
    gripper_ranges: tuple[GripperRange, ...] = ()


@dataclass(frozen=True)
class DatasetEvidence:
    candidate: DatasetCandidate
    source_info: dict[str, object]
    identity: IdentityEvidence
    cameras: tuple[CameraEvidence, ...]
    machines: tuple[MachineEvidence, ...]
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    title: str
    url: str
    kind: str


@dataclass(frozen=True)
class IdentityAssessment:
    local_source: str
    relation: str
    explanation: str


@dataclass(frozen=True)
class RobotIdentityFact:
    manufacturer: str | None
    model: str | None
    confidence: float
    ambiguous: bool
    reason: str
    local_evidence_status: str
    source_ids: tuple[str, ...]
    assessments: tuple[IdentityAssessment, ...]


@dataclass(frozen=True)
class CameraSlot:
    camera_id: str
    interface_name: str | None
    mount_type: str
    direction_tokens: tuple[str, ...]
    body_part: str | None
    modality: str
    confidence: float
    ambiguous: bool
    reason: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class MachineComponent:
    component_id: str
    kind: str
    side: str | None
    count: int
    element_order: tuple[str, ...]
    representation: str
    unit: str
    open_range: tuple[float, float] | None
    open_direction: str | None
    confidence: float
    ambiguous: bool
    reason: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class HardwareProfile:
    identity: RobotIdentityFact
    sources: tuple[SourceReference, ...]
    cameras: tuple[CameraSlot, ...]
    components: tuple[MachineComponent, ...]


@dataclass(frozen=True)
class CameraAssignment:
    source_key: str
    camera_id: str | None
    confidence: float
    ambiguous: bool
    reason: str


@dataclass(frozen=True)
class MachineSlice:
    start: int
    end: int
    component_id: str
    element_order: tuple[str, ...]


@dataclass(frozen=True)
class MachineAssignment:
    source_feature: str
    slices: tuple[MachineSlice, ...]
    confidence: float
    ambiguous: bool
    reason: str


@dataclass(frozen=True)
class DatasetMapping:
    cameras: tuple[CameraAssignment, ...]
    machines: tuple[MachineAssignment, ...]


@dataclass(frozen=True)
class DatasetAnalysis:
    """One locally scoped hardware structure and its dataset assignments."""

    profile: HardwareProfile
    mapping: DatasetMapping


@dataclass(frozen=True)
class MappingRecord:
    source_address: str
    source: object
    output: object
    candidate: object | None
    changed: bool
    vlm_semantics: dict[str, object]
    citations: tuple[dict[str, object], ...]
    decision: str
    reason: str


@dataclass(frozen=True)
class NormalizationResult:
    normalized_info: dict[str, object]
    robot_identity: MappingRecord
    camera_mappings: tuple[MappingRecord, ...]
    machine_mappings: tuple[MappingRecord, ...]
    issues: tuple[Issue, ...]


@dataclass(frozen=True)
class DatasetResult:
    candidate: DatasetCandidate
    status: DatasetStatus
    camera_count: int
    machine_field_count: int
    changed_field_count: int
    issue_count: int
    source_info: dict[str, object] | None
