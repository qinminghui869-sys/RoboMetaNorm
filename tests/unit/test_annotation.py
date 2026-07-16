"""Contract tests for the YAML annotation compiler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.annotation import (
    compile_annotation,
    preflight_annotation,
)
from robometanorm.models import (
    CameraAssignment,
    CameraEvidence,
    CameraSlot,
    DatasetCandidate,
    DatasetEvidence,
    DatasetMapping,
    FeatureSchema,
    HardwareProfile,
    IdentityEvidence,
    LayoutType,
    MachineAssignment,
    MachineComponent,
    MachineEvidence,
    MachineSlice,
    RobotIdentityFact,
)


class AnnotationFixture:
    """Small confirmed mappings with side-qualified raw names."""

    @staticmethod
    def evidence(*, sides: tuple[str, ...] = ("left", "right")) -> DatasetEvidence:
        names = tuple(
            name
            for side in sides
            for name in (
                f"{side}_joint_1",
                f"{side}_joint_2",
                f"{side}_pose_x",
                f"{side}_pose_y",
                f"{side}_pose_z",
                f"{side}_pose_rx",
                f"{side}_pose_ry",
                f"{side}_pose_rz",
                f"{side}_gripper",
            )
        )
        root = Path("/tmp/annotation_fixture")
        candidate = DatasetCandidate(
            dataset_name="fixture",
            task_name=None,
            source_path=root,
            layout_type=LayoutType.FLAT,
            info_path=root / "meta" / "info.json",
            data_path=root / "data",
            video_path=root / "videos",
            depth_path=None,
        )
        machine_schemas = tuple(
            MachineEvidence(
                schema=FeatureSchema(
                    source_key=source_key,
                    dtype="float32",
                    shape=(len(names),),
                    names=names,
                    fps=None,
                    codec=None,
                ),
                episodes=(),
                episode_lengths=(),
            )
            for source_key in ("observation.state", "action")
        )
        camera = CameraEvidence(
            schema=FeatureSchema(
                source_key="observation.images.image_left",
                dtype="video",
                shape=(480, 640, 3),
                names=("height", "width", "channel"),
                fps=30,
                codec="av1",
            ),
            samples=(),
        )
        return DatasetEvidence(
            candidate=candidate,
            source_info={
                "robot_type": "fixture_dual_arm",
                "features": {
                    "observation.images.image_left": {},
                    "observation.state": {"names": list(names)},
                    "action": {"names": list(names)},
                },
            },
            identity=IdentityEvidence(
                info_robot_type_state="present",
                info_robot_type="fixture_dual_arm",
                common_record_state="missing",
                common_record=None,
                tasks_state="missing",
                tasks=(),
            ),
            cameras=(camera,),
            machines=machine_schemas,
        )

    @staticmethod
    def profile(*, sides: tuple[str, ...] = ("left", "right")) -> HardwareProfile:
        components = tuple(
            component
            for side in sides
            for component in (
                MachineComponent(
                    component_id=f"{side}_joint",
                    kind="arm_joint",
                    side=side,
                    count=2,
                    element_order=("joint_1", "joint_2"),
                    representation="joint_vector",
                    unit="rad",
                    open_range=None,
                    open_direction=None,
                    confidence=1.0,
                    ambiguous=False,
                    reason="confirmed joints",
                    source_ids=(),
                ),
                MachineComponent(
                    component_id=f"{side}_position",
                    kind="eef_position",
                    side=side,
                    count=3,
                    element_order=("x", "y", "z"),
                    representation="position_xyz",
                    unit="m",
                    open_range=None,
                    open_direction=None,
                    confidence=1.0,
                    ambiguous=False,
                    reason="confirmed position",
                    source_ids=(),
                ),
                MachineComponent(
                    component_id=f"{side}_rotation",
                    kind="eef_rotation",
                    side=side,
                    count=3,
                    element_order=("x", "y", "z"),
                    representation="euler_xyz",
                    unit="rad",
                    open_range=None,
                    open_direction=None,
                    confidence=1.0,
                    ambiguous=False,
                    reason="confirmed rotation",
                    source_ids=(),
                ),
                MachineComponent(
                    component_id=f"{side}_gripper",
                    kind="gripper_open",
                    side=side,
                    count=1,
                    element_order=("open",),
                    representation="scalar",
                    unit="unitless",
                    open_range=(0.0, 1.0),
                    open_direction="increasing",
                    confidence=1.0,
                    ambiguous=False,
                    reason="confirmed gripper",
                    source_ids=(),
                ),
            )
        )
        return HardwareProfile(
            identity=RobotIdentityFact(
                manufacturer="Fixture",
                model="Robot",
                confidence=1.0,
                ambiguous=False,
                reason="confirmed",
                local_evidence_status="consistent",
                source_ids=(),
                assessments=(),
            ),
            sources=(),
            cameras=(
                CameraSlot(
                    camera_id="left_wrist",
                    interface_name="observation.images.image_left",
                    mount_type="on_robot",
                    direction_tokens=("left",),
                    body_part="wrist",
                    modality="rgb",
                    confidence=1.0,
                    ambiguous=False,
                    reason="confirmed camera",
                    source_ids=(),
                ),
            ),
            components=components,
        )

    @staticmethod
    def mapping(*, sides: tuple[str, ...] = ("left", "right")) -> DatasetMapping:
        slices = tuple(
            machine_slice
            for side_index, side in enumerate(sides)
            for machine_slice in (
                MachineSlice(
                    start=side_index * 9,
                    end=side_index * 9 + 2,
                    component_id=f"{side}_joint",
                    element_order=("joint_1", "joint_2"),
                ),
                MachineSlice(
                    start=side_index * 9 + 2,
                    end=side_index * 9 + 5,
                    component_id=f"{side}_position",
                    element_order=("x", "y", "z"),
                ),
                MachineSlice(
                    start=side_index * 9 + 5,
                    end=side_index * 9 + 8,
                    component_id=f"{side}_rotation",
                    element_order=("x", "y", "z"),
                ),
                MachineSlice(
                    start=side_index * 9 + 8,
                    end=side_index * 9 + 9,
                    component_id=f"{side}_gripper",
                    element_order=("open",),
                ),
            )
        )
        return DatasetMapping(
            cameras=(
                CameraAssignment(
                    source_key="observation.images.image_left",
                    camera_id="left_wrist",
                    confidence=1.0,
                    ambiguous=False,
                    reason="confirmed camera",
                ),
            ),
            machines=tuple(
                MachineAssignment(
                    source_feature=source_key,
                    slices=slices,
                    confidence=1.0,
                    ambiguous=False,
                    reason="confirmed machine",
                )
                for source_key in ("observation.state", "action")
            ),
        )

    @staticmethod
    def normalized_info() -> dict[str, object]:
        return {"robot_type": "fixture_final"}


class AnnotationCompilerTest(unittest.TestCase):
    def test_preflight_blocks_generic_joint_names_with_source_file(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        generic = MachineEvidence(
            schema=FeatureSchema(
                source_key="observation.state",
                dtype="float32",
                shape=(3,),
                names=("joint1", "joint_2", "j3"),
                fps=None,
                codec=None,
            ),
            episodes=(),
            episode_lengths=(),
        )
        evidence = DatasetEvidence(
            **{**evidence.__dict__, "machines": (generic,)}
        )

        issues = preflight_annotation(evidence)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "ANNOTATION_JOINT_AMBIGUOUS")
        self.assertEqual(issues[0].severity, "block")
        self.assertEqual(issues[0].evidence["source_file"], "meta/info.json")
        self.assertEqual(issues[0].evidence["source_feature"], "observation.state")
        self.assertEqual(issues[0].evidence["source_indices"], [0, 1, 2])
        self.assertEqual(issues[0].evidence["observed_names"], ["joint1", "joint_2", "j3"])
        self.assertIn("hint", issues[0].evidence)

    def test_compiles_confirmed_dual_arm_channels_and_preserves_camera_source(self) -> None:
        result = compile_annotation(
            AnnotationFixture.evidence(),
            AnnotationFixture.profile(),
            AnnotationFixture.mapping(),
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertEqual(result.issues, ())
        self.assertEqual(
            list(result.document),
            ["version", "robot_type", "adapter", "robot_channel_schema"],
        )
        self.assertEqual(result.document["robot_type"], "fixture_final")
        self.assertEqual(
            result.document["robot_channel_schema"]["robot_type"],
            "fixture_final",
        )
        self.assertEqual(
            result.document["adapter"]["cameras"],
            {"observation.images.cam_left_wrist_rgb": "observation.images.image_left"},
        )
        channels = result.document["robot_channel_schema"]["channels"]
        self.assertEqual(
            list(channels),
            [
                "arm.left.joint",
                "arm.left.eef",
                "gripper.left",
                "arm.right.joint",
                "arm.right.eef",
                "gripper.right",
            ],
        )
        self.assertEqual(channels["arm.left.eef"]["slice"], [2, 8])
        self.assertEqual(channels["gripper.right"]["slice"], [17, 18])
        self.assertEqual(
            yaml.safe_load(yaml.safe_dump(result.document, sort_keys=False)),
            result.document,
        )

    def test_compiles_only_confirmed_side_for_single_arm(self) -> None:
        result = compile_annotation(
            AnnotationFixture.evidence(sides=("left",)),
            AnnotationFixture.profile(sides=("left",)),
            AnnotationFixture.mapping(sides=("left",)),
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertEqual(result.issues, ())
        channels = result.document["robot_channel_schema"]["channels"]
        self.assertEqual(
            list(channels),
            ["arm.left.joint", "arm.left.eef", "gripper.left"],
        )
        self.assertEqual(
            result.document["robot_channel_schema"]["group_weights"],
            {"arm_motion": 0.3, "gripper": 0.45},
        )

    def test_does_not_compile_when_a_mapped_component_is_unconfirmed(self) -> None:
        profile = AnnotationFixture.profile(sides=("left",))
        incomplete = HardwareProfile(
            identity=profile.identity,
            sources=profile.sources,
            cameras=profile.cameras,
            components=profile.components[:-1],
        )

        result = compile_annotation(
            AnnotationFixture.evidence(sides=("left",)),
            incomplete,
            AnnotationFixture.mapping(sides=("left",)),
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertIsNone(result.document)
        self.assertEqual(result.issues[0].code, "ANNOTATION_MAPPING_UNCONFIRMED")
        self.assertEqual(result.issues[0].severity, "review")


if __name__ == "__main__":
    unittest.main()
