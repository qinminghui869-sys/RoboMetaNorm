"""Reusable, deterministic fixtures for the mini-pipeline tests.

Everything in this module is test-only.  Helpers build data and test doubles but
do not reproduce production decisions.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from robometanorm.models import (
    CameraAssignment,
    CameraSlot,
    DatasetCandidate,
    DatasetEvidence,
    DatasetMapping,
    FeatureSchema,
    HardwareProfile,
    IdentityAssessment,
    IdentityEvidence,
    Issue,
    LayoutType,
    MachineAssignment,
    MachineComponent,
    MachineSlice,
    RobotIdentityFact,
    SourceReference,
)

@dataclass(frozen=True)
class DatasetFixture:
    """Paths and source metadata for one temporary fictional dataset."""

    root: Path
    candidate: DatasetCandidate
    info: dict[str, object]

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        dataset_name: str = "acme_pick_001",
        task_name: str | None = None,
        info: dict[str, object] | None = None,
        with_data: bool = True,
        with_videos: bool = False,
        with_depth: bool = False,
    ) -> "DatasetFixture":
        """Create a minimal on-disk dataset without invoking production code."""
        dataset_path = (
            root / dataset_name
            if task_name is None
            else root / task_name / dataset_name
        )
        meta_path = dataset_path / "meta"
        meta_path.mkdir(parents=True, exist_ok=True)

        source_info = deepcopy(info if info is not None else cls.default_info())
        serialized_info = deepcopy(source_info)
        info_path = meta_path / "info.json"
        info_path.write_text(
            json.dumps(serialized_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        data_path = dataset_path / "data" if with_data else None
        video_path = dataset_path / "videos" if with_videos else None
        depth_path = dataset_path / "depth" if with_depth else None
        for path in (data_path, video_path, depth_path):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

        candidate = DatasetCandidate(
            dataset_name=dataset_name,
            task_name=task_name,
            source_path=dataset_path,
            layout_type=(
                LayoutType.FLAT if task_name is None else LayoutType.TASK_GROUPED
            ),
            info_path=info_path,
            data_path=data_path,
            video_path=video_path,
            depth_path=depth_path,
        )
        return cls(root=root, candidate=candidate, info=source_info)

    @staticmethod
    def default_info() -> dict[str, object]:
        """Return fresh metadata for the fictional Acme TestBot dataset."""
        return {
            "robot_type": "acme_testbot",
            "fps": 30,
            "features": {
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "fps": 30,
                    "codec": "h264",
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [
                        "joint_1",
                        "joint_2",
                        "joint_3",
                        "joint_4",
                        "joint_5",
                        "joint_6",
                    ],
                },
            },
        }

    def write_parquet(
        self,
        rows: dict[str, object],
        *,
        relative_path: str = "chunk-000/episode_000000.parquet",
    ) -> Path:
        """Write deterministic columns with PyArrow for evidence tests."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.candidate.data_path is None:
            raise ValueError("DatasetFixture was created without a data directory")
        output_path = self.candidate.data_path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(rows), output_path)
        return output_path

    def write_media_placeholder(
        self,
        *,
        relative_path: str = "chunk-000/observation.images.wrist/episode_000000.mp4",
        payload: bytes = b"fictional-acme-media",
    ) -> Path:
        """Create a deterministic media path for tests that stub probing."""
        if self.candidate.video_path is None:
            raise ValueError("DatasetFixture was created without a videos directory")
        output_path = self.candidate.video_path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        return output_path


@dataclass(frozen=True)
class PipelineFixture:
    """Canonical fictional hardware profile and mapping builders."""

    manufacturer: str = "Acme Robotics"
    model: str = "TestBot One"

    def hardware_profile(self) -> HardwareProfile:
        source = SourceReference(
            source_id="acme-manual",
            title="Acme TestBot One Test Manual",
            url="https://fixtures.invalid/acme/testbot-one",
            kind="official_manual",
        )
        identity = RobotIdentityFact(
            manufacturer=self.manufacturer,
            model=self.model,
            confidence=1.0,
            ambiguous=False,
            reason="Official fictional Acme test documentation",
            local_evidence_status="consistent",
            source_ids=(source.source_id,),
            assessments=(
                IdentityAssessment(
                    local_source="info_robot_type",
                    relation="supports",
                    explanation="The fictional local identifier names TestBot One.",
                ),
                IdentityAssessment(
                    local_source="common_record",
                    relation="missing",
                    explanation="No fictional common-record identity was supplied.",
                ),
                IdentityAssessment(
                    local_source="tasks",
                    relation="missing",
                    explanation="No fictional task identity was supplied.",
                ),
            ),
        )
        camera = CameraSlot(
            camera_id="wrist_rgb",
            interface_name="observation.images.wrist",
            mount_type="on_robot",
            direction_tokens=("front",),
            body_part="wrist",
            modality="rgb",
            confidence=1.0,
            ambiguous=False,
            reason="Official fictional camera table",
            source_ids=(source.source_id,),
        )
        component = MachineComponent(
            component_id="arm",
            kind="arm_joint",
            side="left",
            count=6,
            element_order=(
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ),
            representation="joint_vector",
            unit="rad",
            open_range=None,
            open_direction=None,
            confidence=1.0,
            ambiguous=False,
            reason="Official fictional actuator table",
            source_ids=(source.source_id,),
        )
        return HardwareProfile(
            identity=identity,
            sources=(source,),
            cameras=(camera,),
            components=(component,),
        )

    def dataset_mapping(self) -> DatasetMapping:
        return DatasetMapping(
            cameras=(
                CameraAssignment(
                    source_key="observation.images.wrist",
                    camera_id="wrist_rgb",
                    confidence=1.0,
                    ambiguous=False,
                    reason="Exact fictional interface match",
                ),
            ),
            machines=(
                MachineAssignment(
                    source_feature="observation.state",
                    slices=(
                        MachineSlice(
                            start=0,
                            end=6,
                            component_id="arm",
                            element_order=(
                                "joint_1",
                                "joint_2",
                                "joint_3",
                                "joint_4",
                                "joint_5",
                                "joint_6",
                            ),
                        ),
                    ),
                    confidence=1.0,
                    ambiguous=False,
                    reason="Exact fictional schema match",
                ),
            ),
        )

    @staticmethod
    def camera_schema(source_key: str = "observation.images.wrist") -> FeatureSchema:
        return FeatureSchema(
            source_key=source_key,
            dtype="video",
            shape=(480, 640, 3),
            names=("height", "width", "channel"),
            fps=30,
            codec="h264",
        )

    @staticmethod
    def machine_schema(source_key: str = "observation.state") -> FeatureSchema:
        return FeatureSchema(
            source_key=source_key,
            dtype="float32",
            shape=(6,),
            names=(
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ),
            fps=None,
            codec=None,
        )

    @staticmethod
    def media_probe_payload() -> dict[str, object]:
        return {
            "media_type": "video",
            "codec": "h264",
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "duration_seconds": 1.0,
            "pixel_format": "yuv420p",
        }

@dataclass(frozen=True)
class VlmFixture:
    """Strict JSON payload builders used by fake VLM transports."""

    @staticmethod
    def valid_hardware_payload() -> dict[str, object]:
        payload: dict[str, object] = json.loads(
            json.dumps(asdict(PipelineFixture().hardware_profile()))
        )
        return payload

    @staticmethod
    def valid_mapping_payload() -> dict[str, object]:
        payload: dict[str, object] = json.loads(
            json.dumps(asdict(PipelineFixture().dataset_mapping()))
        )
        return payload

    @staticmethod
    def json_text(payload: dict[str, object]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def create_openai_service(*args: object, **kwargs: object) -> object:
        """Instantiate the future production service without an eager import."""
        from robometanorm.vlm import OpenAICompatibleDatasetVlm

        return OpenAICompatibleDatasetVlm(*args, **kwargs)


def _missing_vlm_issue() -> Issue:
    return Issue(
        code="VLM_CONFIG_MISSING",
        message="The fictional test VLM is not configured.",
        scope="vlm",
    )


def _missing_hardware_result() -> tuple[HardwareProfile | None, Issue | None]:
    return None, _missing_vlm_issue()


def _missing_mapping_result() -> tuple[DatasetMapping | None, Issue | None]:
    return None, _missing_vlm_issue()


@dataclass
class FakeVlm:
    """Fixed-result dataset VLM fake with inspectable calls."""

    research_result: tuple[HardwareProfile | None, Issue | None] = field(
        default_factory=_missing_hardware_result
    )
    mapping_result: tuple[DatasetMapping | None, Issue | None] = field(
        default_factory=_missing_mapping_result
    )
    research_calls: int = field(default=0, init=False)
    mapping_calls: int = field(default=0, init=False)

    def research_hardware(
        self, identity: IdentityEvidence
    ) -> tuple[HardwareProfile | None, Issue | None]:
        self.research_calls += 1
        return self.research_result

    def map_dataset(
        self, evidence: DatasetEvidence, profile: HardwareProfile
    ) -> tuple[DatasetMapping | None, Issue | None]:
        self.mapping_calls += 1
        return self.mapping_result


@dataclass
class StubTransport:
    """Fixed-result web and chat transport with inspectable attempts."""

    web_payload: dict[str, object] | None = None
    web_issue: Issue | None = None
    chat_payload: dict[str, object] | None = None
    chat_issue: Issue | None = None
    web_attempts: int = field(default=0, init=False)
    chat_attempts: int = field(default=0, init=False)

    def request_web_json(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[dict[str, object] | None, Issue | None]:
        self.web_attempts += 1
        return self.web_payload, self.web_issue

    def request_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: tuple[Path, ...],
    ) -> tuple[dict[str, object] | None, Issue | None]:
        self.chat_attempts += 1
        return self.chat_payload, self.chat_issue
