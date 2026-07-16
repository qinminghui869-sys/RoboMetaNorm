"""Contract tests for the YAML annotation compiler."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast

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
    Issue,
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
    def main_follower_evidence(
        *,
        state_names: tuple[str, ...] | None = None,
        action_names: tuple[str, ...] | None = None,
    ) -> DatasetEvidence:
        """A single raw arm whose side is confirmed only by the mapping."""

        evidence = AnnotationFixture.evidence(sides=("left",))
        names = (
            "main_follower_joint_1",
            "main_follower_joint_2",
            "main_follower_pose_x",
            "main_follower_pose_y",
            "main_follower_pose_z",
            "main_follower_pose_rx",
            "main_follower_pose_ry",
            "main_follower_pose_rz",
            "main_follower_gripper",
        )
        replacements = {
            "observation.state": state_names or names,
            "action": action_names or names,
        }
        return replace(
            evidence,
            source_info={
                **evidence.source_info,
                "features": {
                    **evidence.source_info["features"],
                    **{
                        source_key: {"names": list(source_names)}
                        for source_key, source_names in replacements.items()
                    },
                },
            },
            machines=tuple(
                replace(
                    machine,
                    schema=replace(
                        machine.schema,
                        names=replacements[machine.schema.source_key],
                    ),
                )
                for machine in evidence.machines
            ),
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
                shape=(4,),
                names=("joint1", "joint_2", "j3", "raw_joint_4"),
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
        self.assertEqual(issues[0].evidence["source_indices"], [0, 1, 2, 3])
        self.assertEqual(
            issues[0].evidence["observed_names"],
            ["joint1", "joint_2", "j3", "raw_joint_4"],
        )
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
            ["version", "robot_type", "adapter", "robot_channel_schema", "review"],
        )
        self.assertEqual(result.document["review"], {"required": False, "issues": []})
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

    def test_compiles_confirmed_main_follower_single_arm_channels(self) -> None:
        result = compile_annotation(
            AnnotationFixture.main_follower_evidence(),
            AnnotationFixture.profile(sides=("left",)),
            AnnotationFixture.mapping(sides=("left",)),
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertEqual(result.issues, ())
        channels = result.document["robot_channel_schema"]["channels"]
        self.assertEqual(
            list(channels),
            ["arm.main.joint", "arm.main.eef", "gripper.main"],
        )
        self.assertEqual(channels["arm.main.joint"]["slice"], [0, 2])
        self.assertEqual(channels["arm.main.eef"]["slice"], [2, 8])
        self.assertEqual(channels["gripper.main"]["slice"], [8, 9])

    def test_rejects_main_follower_when_profile_has_two_arms(self) -> None:
        result = compile_annotation(
            AnnotationFixture.main_follower_evidence(),
            AnnotationFixture.profile(),
            AnnotationFixture.mapping(sides=("left",)),
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.issues[0].code, "ANNOTATION_MAIN_ARM_UNCONFIRMED")
        self.assertEqual(result.issues[0].severity, "review")

    def test_rejects_main_follower_when_mapping_is_missing(self) -> None:
        result = compile_annotation(
            AnnotationFixture.main_follower_evidence(),
            AnnotationFixture.profile(sides=("left",)),
            None,
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.issues[0].code, "ANNOTATION_MAIN_ARM_UNCONFIRMED")
        self.assertEqual(result.issues[0].severity, "review")

    def test_rejects_main_follower_when_camera_mapping_is_unconfirmed(self) -> None:
        mapping = AnnotationFixture.mapping(sides=("left",))
        unconfirmed_mapping = replace(
            mapping,
            cameras=(replace(mapping.cameras[0], ambiguous=True),),
        )

        result = compile_annotation(
            AnnotationFixture.main_follower_evidence(),
            AnnotationFixture.profile(sides=("left",)),
            unconfirmed_mapping,
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.issues[0].code, "ANNOTATION_MAIN_ARM_UNCONFIRMED")
        self.assertEqual(result.issues[0].severity, "review")

    def test_rejects_main_follower_when_machine_layout_is_missing(self) -> None:
        mapping = AnnotationFixture.mapping(sides=("left",))
        incomplete_mapping = DatasetMapping(
            cameras=mapping.cameras,
            machines=(),
        )

        result = compile_annotation(
            AnnotationFixture.main_follower_evidence(),
            AnnotationFixture.profile(sides=("left",)),
            incomplete_mapping,
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
        )

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.issues[0].code, "ANNOTATION_MAIN_ARM_UNCONFIRMED")
        self.assertEqual(result.issues[0].severity, "review")

    def test_preflight_blocks_gapped_main_follower_joints_with_source_file(self) -> None:
        state = list(AnnotationFixture.main_follower_evidence().machines[0].schema.names)
        state[1] = "main_follower_joint_3"

        issues = preflight_annotation(
            AnnotationFixture.main_follower_evidence(state_names=tuple(state))
        )

        self.assertEqual(issues[0].code, "ANNOTATION_MAIN_ARM_LAYOUT_INVALID")
        self.assertEqual(issues[0].severity, "block")
        self.assertEqual(issues[0].evidence["source_file"], "meta/info.json")
        self.assertEqual(issues[0].evidence["source_feature"], "observation.state")

    def test_preflight_blocks_main_follower_with_oversized_joint_index(self) -> None:
        state = list(AnnotationFixture.main_follower_evidence().machines[0].schema.names)
        state[0] = f"main_follower_joint_{'9' * 5000}"

        issues = preflight_annotation(
            AnnotationFixture.main_follower_evidence(state_names=tuple(state))
        )

        issue = next(
            item
            for item in issues
            if item.code == "ANNOTATION_MAIN_ARM_LAYOUT_INVALID"
        )
        self.assertEqual(issue.severity, "block")
        self.assertEqual(issue.evidence["source_file"], "meta/info.json")
        self.assertEqual(issue.evidence["source_feature"], "observation.state")

    def test_preflight_blocks_mismatched_main_follower_layouts_with_source_file(self) -> None:
        action = list(AnnotationFixture.main_follower_evidence().machines[0].schema.names)
        action[0] = "main_follower_joint_0"
        action[1] = "main_follower_joint_1"

        issues = preflight_annotation(
            AnnotationFixture.main_follower_evidence(action_names=tuple(action))
        )

        self.assertEqual(issues[0].code, "ANNOTATION_MAIN_ARM_LAYOUT_INVALID")
        self.assertEqual(issues[0].severity, "block")
        self.assertEqual(issues[0].evidence["source_file"], "meta/info.json")
        self.assertEqual(
            issues[0].evidence["source_features"],
            ["action", "observation.state"],
        )

    def test_preflight_blocks_main_follower_outside_machine_vectors(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        auxiliary = MachineEvidence(
            schema=FeatureSchema(
                source_key="observation.auxiliary",
                dtype="float32",
                shape=(1,),
                names=("main_follower_joint_1",),
                fps=None,
                codec=None,
            ),
            episodes=(),
            episode_lengths=(),
        )

        issues = preflight_annotation(
            replace(evidence, machines=(*evidence.machines, auxiliary))
        )

        self.assertEqual(issues[-1].code, "ANNOTATION_JOINT_AMBIGUOUS")
        self.assertEqual(issues[-1].severity, "block")
        self.assertEqual(issues[-1].evidence["source_feature"], "observation.auxiliary")

    def test_preflight_blocks_inconsistent_sided_action_and_qpos_layouts(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        machines = tuple(
            replace(
                machine,
                schema=replace(
                    machine.schema,
                    names=("right_joint_1", "right_joint_2", *machine.schema.names[2:]),
                ),
            )
            if machine.schema.source_key == "action"
            else machine
            for machine in evidence.machines
        )

        issues = preflight_annotation(replace(evidence, machines=machines))

        self.assertEqual(issues[0].code, "ANNOTATION_JOINT_LAYOUT_MISMATCH")
        self.assertEqual(issues[0].severity, "block")
        self.assertEqual(issues[0].evidence["source_file"], "meta/info.json")

    def test_preflight_blocks_interleaved_sided_joint_vectors(self) -> None:
        evidence = AnnotationFixture.evidence()
        interleaved = (
            "left_joint_1",
            "right_joint_1",
            "left_joint_2",
            "right_joint_2",
            *(
                name
                for name in evidence.machines[0].schema.names
                if "_joint_" not in name
            ),
        )
        machines = tuple(
            replace(machine, schema=replace(machine.schema, names=interleaved))
            for machine in evidence.machines
        )

        issues = preflight_annotation(replace(evidence, machines=machines))

        self.assertEqual(issues[0].code, "ANNOTATION_JOINT_AMBIGUOUS")
        self.assertEqual(issues[0].severity, "block")

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

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.issues[0].code, "ANNOTATION_MAPPING_UNCONFIRMED")
        self.assertEqual(result.issues[0].severity, "review")

    def test_projects_existing_issues_on_a_confirmed_document(self) -> None:
        issue = Issue("VLM_UNCERTAIN", "请人工确认", "vlm", {"raw": "kept out"})

        result = compile_annotation(
            AnnotationFixture.evidence(),
            AnnotationFixture.profile(),
            AnnotationFixture.mapping(),
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
            existing_issues=(issue,),
        )

        self.assertEqual(result.issues, ())
        self.assertEqual(
            result.document["review"],
            {"required": True, "issues": [{"code": "VLM_UNCERTAIN", "message": "请人工确认"}]},
        )

    def test_fallback_preserves_local_identity_and_safe_sided_joint_slices(self) -> None:
        issue = Issue("VLM_UNCERTAIN", "请人工确认", "vlm", {"full": "evidence"})

        result = compile_annotation(
            AnnotationFixture.evidence(),
            None,
            None,
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
            existing_issues=(issue,),
        )

        self.assertEqual(result.issues[0].code, "ANNOTATION_MAPPING_UNCONFIRMED")
        self.assertEqual(result.document["robot_type"], "fixture_final")
        self.assertEqual(result.document["adapter"]["base"], {"qpos": "observation.state", "action": "action"})
        self.assertEqual(
            result.document["robot_channel_schema"]["channels"],
            {
                "arm.left.joint": {
                    "source": "qpos", "field": "qpos", "slice": [0, 2], "group": "arm_motion",
                    "unit": "unknown", "norm": "robust_mad", "weight": 1.0, "optional": False,
                },
                "arm.right.joint": {
                    "source": "qpos", "field": "qpos", "slice": [9, 11], "group": "arm_motion",
                    "unit": "unknown", "norm": "robust_mad", "weight": 1.0, "optional": False,
                },
            },
        )
        self.assertEqual(
            result.document["review"]["issues"],
            [
                {"code": "VLM_UNCERTAIN", "message": "请人工确认"},
                {"code": "ANNOTATION_MAPPING_UNCONFIRMED", "message": "缺少已确认的硬件画像或数据映射"},
            ],
        )

    def test_fallback_does_not_return_annotation_issue_already_seen_by_pipeline(self) -> None:
        existing = Issue(
            "ANNOTATION_MAPPING_UNCONFIRMED",
            "缺少已确认的硬件画像或数据映射",
            "annotation",
            {"already": "reported"},
        )

        result = compile_annotation(
            AnnotationFixture.evidence(),
            None,
            None,
            normalized_info=AnnotationFixture.normalized_info(),
            confidence_threshold=0.85,
            existing_issues=(existing,),
        )

        self.assertEqual(result.issues, ())
        self.assertEqual(
            result.document["review"]["issues"],
            [{"code": "ANNOTATION_MAPPING_UNCONFIRMED", "message": "缺少已确认的硬件画像或数据映射"}],
        )

    def test_fallback_ambiguous_or_invalid_identity_is_reviewed_without_channels(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        generic = tuple("joint_" + str(index) for index in range(1, 3))
        evidence = replace(
            evidence,
            machines=tuple(
                replace(machine, schema=replace(machine.schema, names=generic))
                for machine in evidence.machines
            ),
        )

        result = compile_annotation(
            evidence, None, None, normalized_info={"robot_type": " bad "}, confidence_threshold=0.85
        )

        self.assertIsNone(result.document["robot_type"])
        self.assertEqual(result.document["robot_channel_schema"]["channels"], {})
        self.assertEqual(result.document["review"]["issues"][0]["code"], "ANNOTATION_JOINT_AMBIGUOUS")

    def test_fallback_reuses_only_canonical_camera_keys_and_conditional_base_fields(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        canonical = replace(
            evidence.cameras[0],
            schema=replace(evidence.cameras[0].schema, source_key="observation.images.cam_left_wrist_rgb"),
        )
        evidence = replace(evidence, cameras=(canonical, evidence.cameras[0]))
        evidence = replace(
            evidence,
            machines=tuple(machine for machine in evidence.machines if machine.schema.source_key == "action"),
        )

        result = compile_annotation(evidence, None, None, normalized_info=AnnotationFixture.normalized_info(), confidence_threshold=0.85)

        self.assertEqual(result.document["adapter"]["base"], {"action": "action"})
        self.assertEqual(result.document["adapter"]["cameras"], {"observation.images.cam_left_wrist_rgb": "observation.images.cam_left_wrist_rgb"})

    def test_fallback_main_follower_requires_matching_contiguous_layouts(self) -> None:
        result = compile_annotation(
            AnnotationFixture.main_follower_evidence(), None, None,
            normalized_info=AnnotationFixture.normalized_info(), confidence_threshold=0.85,
        )
        self.assertEqual(list(result.document["robot_channel_schema"]["channels"]), ["arm.main.joint"])

        gapped = list(AnnotationFixture.main_follower_evidence().machines[0].schema.names)
        gapped[1] = "main_follower_joint_3"
        invalid = compile_annotation(
            AnnotationFixture.main_follower_evidence(state_names=tuple(gapped)), None, None,
            normalized_info=AnnotationFixture.normalized_info(), confidence_threshold=0.85,
        )
        self.assertEqual(invalid.document["robot_channel_schema"]["channels"], {})

    def test_fallback_rejects_oversized_sided_joint_indices_without_raising(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        names = list(evidence.machines[0].schema.names)
        names[0] = f"left_joint_{'9' * 5000}"
        evidence = replace(
            evidence,
            machines=tuple(
                replace(machine, schema=replace(machine.schema, names=tuple(names)))
                for machine in evidence.machines
            ),
        )

        result = compile_annotation(
            evidence, None, None,
            normalized_info=AnnotationFixture.normalized_info(), confidence_threshold=0.85,
        )

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.document["robot_channel_schema"]["channels"], {})
        self.assertEqual(result.issues[0].code, "ANNOTATION_JOINT_AMBIGUOUS")

    def test_fallback_omits_malformed_machine_and_camera_source_keys(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        invalid_machine = replace(
            evidence.machines[0],
            schema=replace(evidence.machines[0].schema, source_key=cast(str, [])),
        )
        invalid_camera = replace(
            evidence.cameras[0],
            schema=replace(evidence.cameras[0].schema, source_key=cast(str, 1)),
        )
        evidence = replace(
            evidence,
            machines=(invalid_machine, evidence.machines[1]),
            cameras=(invalid_camera,),
        )

        result = compile_annotation(
            evidence, None, None,
            normalized_info=AnnotationFixture.normalized_info(), confidence_threshold=0.85,
        )

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.document["adapter"]["cameras"], {})
        self.assertEqual(result.document["robot_channel_schema"]["channels"], {})

    def test_fallback_rejects_duplicate_machine_sources_without_choosing_one(self) -> None:
        evidence = AnnotationFixture.evidence(sides=("left",))
        action = next(
            machine
            for machine in evidence.machines
            if machine.schema.source_key == "action"
        )
        conflicting_action = replace(
            action,
            schema=replace(
                action.schema,
                names=("right_joint_1", "right_joint_2", *action.schema.names[2:]),
            ),
        )
        evidence = replace(
            evidence,
            machines=(evidence.machines[0], conflicting_action, action),
        )

        result = compile_annotation(
            evidence, None, None,
            normalized_info=AnnotationFixture.normalized_info(), confidence_threshold=0.85,
        )

        self.assertTrue(result.document["review"]["required"])
        self.assertEqual(result.document["robot_channel_schema"]["channels"], {})

    def test_review_issues_are_ordered_and_deduplicated(self) -> None:
        first = Issue("FIRST", "same", "one", {"a": 1})
        duplicate = Issue("FIRST", "same", "two", {"b": 2})
        second = Issue("SECOND", "later", "one", {})

        result = compile_annotation(
            AnnotationFixture.evidence(), AnnotationFixture.profile(), AnnotationFixture.mapping(),
            normalized_info=AnnotationFixture.normalized_info(), confidence_threshold=0.85,
            existing_issues=(first, duplicate, second),
        )

        self.assertEqual(
            result.document["review"]["issues"],
            [{"code": "FIRST", "message": "same"}, {"code": "SECOND", "message": "later"}],
        )


if __name__ == "__main__":
    unittest.main()
