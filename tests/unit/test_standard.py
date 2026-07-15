"""Strict camera-key standard contract tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import sys
import unittest
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.models import (
    CameraAssignment,
    CameraEvidence,
    CameraSlot,
    DatasetCandidate,
    DatasetEvidence,
    DatasetMapping,
    FeatureSchema,
    GripperRange,
    HardwareProfile,
    IdentityAssessment,
    IdentityEvidence,
    Issue,
    LayoutType,
    MachineAssignment,
    MachineComponent,
    MachineEvidence,
    MachineSlice,
    MediaSample,
    NormalizationResult,
    ParquetEpisodeEvidence,
    RobotIdentityFact,
    SourceReference,
)
from robometanorm.standard import (
    BODY_PARTS,
    CAMERA_PREFIX,
    CONFLICT_GROUPS,
    DIRECTION_ORDER,
    EXTERNAL_DIRECTIONS,
    FIXED_COMPONENTS,
    INDEXED_COMPONENTS,
    JOINT_COMPONENTS,
    ON_ROBOT_DIRECTIONS,
    SIDED_COMPONENTS,
    apply_standard,
    are_standard_machine_names,
    check_preconditions,
    is_standard_machine_name,
    parse_standard_camera_key,
    render_camera_key,
    render_component_names,
)


class CameraStandardTest(unittest.TestCase):
    """Verify camera keys are rendered and parsed using one strict grammar."""

    @staticmethod
    def slot(
        *,
        mount_type: str = "on_robot",
        direction_tokens: tuple[str, ...] = (),
        body_part: str | None = "wrist",
        modality: str = "rgb",
    ) -> CameraSlot:
        return CameraSlot(
            camera_id="fixture-camera",
            interface_name=None,
            mount_type=mount_type,
            direction_tokens=direction_tokens,
            body_part=body_part,
            modality=modality,
            confidence=1.0,
            ambiguous=False,
            reason="camera standard fixture",
            source_ids=(),
        )

    def test_exposes_exact_frozen_camera_vocabulary(self) -> None:
        self.assertEqual(CAMERA_PREFIX, "observation.images.cam_")
        self.assertIsInstance(BODY_PARTS, frozenset)
        self.assertEqual(
            BODY_PARTS,
            frozenset({"wrist", "head", "chest", "arm", "leg", "torso", "fisheye"}),
        )
        self.assertIsInstance(ON_ROBOT_DIRECTIONS, frozenset)
        self.assertEqual(
            ON_ROBOT_DIRECTIONS,
            frozenset({"front", "rear", "left", "right", "upper", "lower", "middle"}),
        )
        self.assertIsInstance(EXTERNAL_DIRECTIONS, frozenset)
        self.assertEqual(
            EXTERNAL_DIRECTIONS,
            frozenset(
                {
                    "front",
                    "rear",
                    "left",
                    "right",
                    "upper",
                    "lower",
                    "middle",
                    "top",
                    "side",
                    "global",
                    "env",
                }
            ),
        )
        self.assertEqual(
            DIRECTION_ORDER,
            (
                "front",
                "rear",
                "upper",
                "lower",
                "middle",
                "top",
                "left",
                "right",
                "side",
                "global",
                "env",
            ),
        )
        self.assertIsInstance(CONFLICT_GROUPS, tuple)
        self.assertEqual(
            CONFLICT_GROUPS,
            (
                frozenset({"front", "rear"}),
                frozenset({"upper", "lower", "middle", "top"}),
                frozenset({"left", "right", "side"}),
                frozenset({"global", "env"}),
            ),
        )

    def test_renders_on_robot_compound_in_canonical_order(self) -> None:
        slot = self.slot(direction_tokens=("left", "front"), body_part="head")

        self.assertEqual(
            render_camera_key(slot),
            "observation.images.cam_front_left_head_rgb",
        )

    def test_renders_external_compound_depth_in_canonical_order(self) -> None:
        slot = self.slot(
            mount_type="external",
            direction_tokens=("left", "front"),
            body_part=None,
            modality="depth",
        )

        self.assertEqual(
            render_camera_key(slot),
            "observation.images.cam_front_left_depth",
        )

    def test_renders_external_top_side(self) -> None:
        slot = self.slot(
            mount_type="external",
            direction_tokens=("side", "top"),
            body_part=None,
        )

        self.assertEqual(
            render_camera_key(slot),
            "observation.images.cam_top_side_rgb",
        )

    def test_renders_standalone_and_body_only_forms(self) -> None:
        cases = (
            (
                self.slot(direction_tokens=("ego",), body_part=None),
                "observation.images.cam_ego_rgb",
            ),
            (
                self.slot(
                    mount_type="external",
                    direction_tokens=("global",),
                    body_part=None,
                    modality="depth",
                ),
                "observation.images.cam_global_depth",
            ),
            (
                self.slot(
                    mount_type="external",
                    direction_tokens=("env",),
                    body_part=None,
                ),
                "observation.images.cam_env_rgb",
            ),
            (self.slot(body_part="wrist"), "observation.images.cam_wrist_rgb"),
        )

        for slot, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(render_camera_key(slot), expected)
                self.assertEqual(parse_standard_camera_key(expected), slot.modality)

    def test_rejects_conflicting_or_duplicate_render_directions(self) -> None:
        invalid_directions = (
            ("left", "right"),
            ("front", "rear"),
            ("upper", "middle"),
            ("front", "front"),
        )

        for direction_tokens in invalid_directions:
            with self.subTest(direction_tokens=direction_tokens):
                self.assertIsNone(
                    render_camera_key(
                        self.slot(
                            mount_type="external",
                            direction_tokens=direction_tokens,
                            body_part=None,
                        )
                    )
                )

    def test_rejects_invalid_render_mount_body_and_direction_combinations(self) -> None:
        invalid_slots = (
            self.slot(
                mount_type="external",
                direction_tokens=("left",),
                body_part="wrist",
            ),
            self.slot(direction_tokens=("top",), body_part="head"),
            self.slot(direction_tokens=("front",), body_part=None),
            self.slot(direction_tokens=("ego", "front"), body_part=None),
            self.slot(direction_tokens=("ego",), body_part="head"),
            self.slot(
                mount_type="external",
                direction_tokens=("ego",),
                body_part=None,
            ),
            self.slot(
                mount_type="external",
                direction_tokens=(),
                body_part=None,
            ),
            self.slot(
                mount_type="external",
                direction_tokens=("global", "left"),
                body_part=None,
            ),
            self.slot(
                mount_type="external",
                direction_tokens=("env", "top"),
                body_part=None,
            ),
        )

        for slot in invalid_slots:
            with self.subTest(slot=slot):
                self.assertIsNone(render_camera_key(slot))

    def test_rejects_unknown_render_values(self) -> None:
        invalid_slots = (
            self.slot(modality="infrared"),
            self.slot(mount_type="ceiling", direction_tokens=("front",)),
            self.slot(direction_tokens=("diagonal",)),
            self.slot(body_part="camera"),
        )

        for slot in invalid_slots:
            with self.subTest(slot=slot):
                self.assertIsNone(render_camera_key(slot))

    def test_parses_valid_on_robot_keys(self) -> None:
        valid_keys = (
            "observation.images.cam_left_wrist_rgb",
            "observation.images.cam_front_left_head_rgb",
            "observation.images.cam_wrist_rgb",
        )

        for key in valid_keys:
            with self.subTest(key=key):
                self.assertEqual(parse_standard_camera_key(key), "rgb")

    def test_parses_valid_external_keys(self) -> None:
        cases = (
            ("observation.images.cam_front_left_depth", "depth"),
            ("observation.images.cam_top_side_rgb", "rgb"),
            ("observation.images.cam_global_depth", "depth"),
            ("observation.images.cam_env_rgb", "rgb"),
        )

        for key, modality in cases:
            with self.subTest(key=key):
                self.assertEqual(parse_standard_camera_key(key), modality)

    def test_parses_standalone_ego(self) -> None:
        self.assertEqual(
            parse_standard_camera_key("observation.images.cam_ego_rgb"),
            "rgb",
        )

    def test_parser_rejects_wrong_prefix_or_extra_suffix(self) -> None:
        invalid_keys = (
            "observation.images.image_left_rgb",
            "observation.images.cam_left_wrist_rgb_extra",
            "observation.images.cam_left_wrist",
        )

        for key in invalid_keys:
            with self.subTest(key=key):
                self.assertIsNone(parse_standard_camera_key(key))


    def test_parser_rejects_conflicting_directions(self) -> None:
        invalid_keys = (
            "observation.images.cam_left_right_rgb",
            "observation.images.cam_upper_top_rgb",
            "observation.images.cam_front_rear_head_depth",
        )

        for key in invalid_keys:
            with self.subTest(key=key):
                self.assertIsNone(parse_standard_camera_key(key))

    def test_parser_rejects_noncanonical_direction_order(self) -> None:
        invalid_keys = (
            "observation.images.cam_left_front_rgb",
            "observation.images.cam_side_top_depth",
            "observation.images.cam_left_front_head_rgb",
        )

        for key in invalid_keys:
            with self.subTest(key=key):
                self.assertIsNone(parse_standard_camera_key(key))

    def test_parser_rejects_ego_and_global_combinations(self) -> None:
        invalid_keys = (
            "observation.images.cam_front_ego_rgb",
            "observation.images.cam_ego_front_rgb",
            "observation.images.cam_ego_head_rgb",
            "observation.images.cam_global_left_rgb",
            "observation.images.cam_env_top_depth",
        )

        for key in invalid_keys:
            with self.subTest(key=key):
                self.assertIsNone(parse_standard_camera_key(key))

    def test_parser_rejects_empty_tokens(self) -> None:
        invalid_keys = (
            "observation.images.cam__rgb",
            "observation.images.cam_front__rgb",
            "observation.images.cam_rgb",
        )

        for key in invalid_keys:
            with self.subTest(key=key):
                self.assertIsNone(parse_standard_camera_key(key))

    def test_parser_rejects_unknown_modality_body_or_direction(self) -> None:
        invalid_keys = (
            "observation.images.cam_left_wrist_ir",
            "observation.images.cam_left_camera_rgb",
            "observation.images.cam_diagonal_rgb",
            "observation.images.cam_front_diagonal_head_depth",
        )

        for key in invalid_keys:
            with self.subTest(key=key):
                self.assertIsNone(parse_standard_camera_key(key))


class MachineStandardTest(unittest.TestCase):
    """Verify machine names use the exact published canonical grammar."""

    @staticmethod
    def component(
        *,
        kind: str,
        side: str | None,
        count: int,
        element_order: tuple[str, ...],
        representation: str,
        unit: str,
    ) -> MachineComponent:
        return MachineComponent(
            component_id="fixture-component",
            kind=kind,
            side=side,
            count=count,
            element_order=element_order,
            representation=representation,
            unit=unit,
            open_range=None,
            open_direction=None,
            confidence=1.0,
            ambiguous=False,
            reason="machine standard fixture",
            source_ids=(),
        )

    def test_exposes_exact_frozen_machine_vocabulary(self) -> None:
        self.assertEqual(
            FIXED_COMPONENTS,
            {
                "eef_position": ("position_xyz", "m", 3),
                "eef_rotation": ("euler_xyz", "rad", 3),
                "head_rotation": ("euler_xyz", "rad", 3),
                "head_orientation": ("quaternion_xyzw", "unitless", 4),
                "base_position": ("position_xyz", "m", 3),
                "base_rotation": ("euler_xyz", "rad", 3),
            },
        )
        self.assertEqual(
            SIDED_COMPONENTS,
            frozenset(
                {
                    "arm_joint",
                    "hand_joint",
                    "gripper_open",
                    "gripper_open_scale",
                    "eef_position",
                    "eef_rotation",
                }
            ),
        )
        self.assertEqual(
            JOINT_COMPONENTS,
            frozenset(
                {
                    "arm_joint",
                    "hand_joint",
                    "head_joint",
                    "torso_joint",
                    "neck_joint",
                }
            ),
        )
        self.assertEqual(INDEXED_COMPONENTS, frozenset({"head_position"}))

    def test_renders_every_pdf_machine_family(self) -> None:
        cases = (
            (
                self.component(
                    kind="arm_joint",
                    side="left",
                    count=2,
                    element_order=("shoulder", "elbow"),
                    representation="joint_vector",
                    unit="rad",
                ),
                ("left_arm_joint_0_rad", "left_arm_joint_1_rad"),
            ),
            (
                self.component(
                    kind="hand_joint",
                    side="right",
                    count=2,
                    element_order=("thumb", "index"),
                    representation="joint_vector",
                    unit="rad",
                ),
                ("right_hand_joint_0_rad", "right_hand_joint_1_rad"),
            ),
            (
                self.component(
                    kind="gripper_open",
                    side="left",
                    count=1,
                    element_order=("opening",),
                    representation="scalar",
                    unit="unitless",
                ),
                ("left_gripper_open",),
            ),
            (
                self.component(
                    kind="gripper_open_scale",
                    side="right",
                    count=1,
                    element_order=("scale",),
                    representation="scalar",
                    unit="unitless",
                ),
                ("right_gripper_open_scale",),
            ),
            (
                self.component(
                    kind="eef_position",
                    side="left",
                    count=3,
                    element_order=("x", "y", "z"),
                    representation="position_xyz",
                    unit="m",
                ),
                ("left_eef_pos_x_m", "left_eef_pos_y_m", "left_eef_pos_z_m"),
            ),
            (
                self.component(
                    kind="eef_rotation",
                    side="right",
                    count=3,
                    element_order=("x", "y", "z"),
                    representation="euler_xyz",
                    unit="rad",
                ),
                (
                    "right_eef_rot_euler_x_rad",
                    "right_eef_rot_euler_y_rad",
                    "right_eef_rot_euler_z_rad",
                ),
            ),
            (
                self.component(
                    kind="head_joint",
                    side=None,
                    count=2,
                    element_order=("pan", "tilt"),
                    representation="joint_vector",
                    unit="rad",
                ),
                ("head_joint_0_rad", "head_joint_1_rad"),
            ),
            (
                self.component(
                    kind="head_position",
                    side=None,
                    count=2,
                    element_order=("lift", "reach"),
                    representation="position_vector",
                    unit="m",
                ),
                ("head_pos_0_m", "head_pos_1_m"),
            ),
            (
                self.component(
                    kind="head_rotation",
                    side=None,
                    count=3,
                    element_order=("x", "y", "z"),
                    representation="euler_xyz",
                    unit="rad",
                ),
                (
                    "head_rot_euler_x_rad",
                    "head_rot_euler_y_rad",
                    "head_rot_euler_z_rad",
                ),
            ),
            (
                self.component(
                    kind="head_orientation",
                    side=None,
                    count=4,
                    element_order=("x", "y", "z", "w"),
                    representation="quaternion_xyzw",
                    unit="unitless",
                ),
                (
                    "head_orient_quat_x",
                    "head_orient_quat_y",
                    "head_orient_quat_z",
                    "head_orient_quat_w",
                ),
            ),
            (
                self.component(
                    kind="torso_joint",
                    side=None,
                    count=2,
                    element_order=("lift", "bend"),
                    representation="joint_vector",
                    unit="rad",
                ),
                ("torso_joint_0_rad", "torso_joint_1_rad"),
            ),
            (
                self.component(
                    kind="neck_joint",
                    side=None,
                    count=2,
                    element_order=("yaw", "pitch"),
                    representation="joint_vector",
                    unit="rad",
                ),
                ("neck_joint_0_rad", "neck_joint_1_rad"),
            ),
            (
                self.component(
                    kind="base_position",
                    side=None,
                    count=3,
                    element_order=("x", "y", "z"),
                    representation="position_xyz",
                    unit="m",
                ),
                ("base_pos_x_m", "base_pos_y_m", "base_pos_z_m"),
            ),
            (
                self.component(
                    kind="base_rotation",
                    side=None,
                    count=3,
                    element_order=("x", "y", "z"),
                    representation="euler_xyz",
                    unit="rad",
                ),
                (
                    "base_rot_euler_x_rad",
                    "base_rot_euler_y_rad",
                    "base_rot_euler_z_rad",
                ),
            ),
        )

        for component, expected in cases:
            with self.subTest(kind=component.kind):
                rendered = render_component_names(component)
                self.assertEqual(rendered, expected)
                self.assertTrue(
                    all(is_standard_machine_name(name) for name in expected)
                )

    def test_render_ignores_open_range_and_direction(self) -> None:
        arm = self.component(
            kind="arm_joint",
            side="left",
            count=2,
            element_order=("shoulder", "elbow"),
            representation="joint_vector",
            unit="rad",
        )
        gripper = self.component(
            kind="gripper_open",
            side="right",
            count=1,
            element_order=("opening",),
            representation="scalar",
            unit="unitless",
        )
        cases = (
            (
                replace(arm, open_range=(-10.0, 10.0)),
                ("left_arm_joint_0_rad", "left_arm_joint_1_rad"),
            ),
            (
                replace(arm, open_direction="research-only-direction"),
                ("left_arm_joint_0_rad", "left_arm_joint_1_rad"),
            ),
            (
                replace(
                    gripper,
                    open_range=(0.0, 1.0),
                    open_direction="increasing",
                ),
                ("right_gripper_open",),
            ),
        )

        for component, expected in cases:
            with self.subTest(component=component):
                self.assertEqual(render_component_names(component), expected)

    def test_render_rejects_wrong_shape_or_vocabulary(self) -> None:
        valid_arm = self.component(
            kind="arm_joint",
            side="left",
            count=1,
            element_order=("joint",),
            representation="joint_vector",
            unit="rad",
        )
        invalid_components = (
            self.component(
                kind="arm_joint",
                side=None,
                count=1,
                element_order=("joint",),
                representation="joint_vector",
                unit="rad",
            ),
            self.component(
                kind="head_joint",
                side="left",
                count=1,
                element_order=("joint",),
                representation="joint_vector",
                unit="rad",
            ),
            self.component(
                kind="gripper_open",
                side="middle",
                count=1,
                element_order=("opening",),
                representation="scalar",
                unit="unitless",
            ),
            replace(valid_arm, count=0, element_order=()),
            replace(valid_arm, count=True),
            replace(valid_arm, count=-1),
            replace(valid_arm, count=cast(int, 1.0)),
            replace(valid_arm, count=cast(int, "1")),
            self.component(
                kind="wheel_joint",
                side=None,
                count=1,
                element_order=("wheel",),
                representation="joint_vector",
                unit="rad",
            ),
        )

        for component in invalid_components:
            with self.subTest(component=component):
                self.assertIsNone(render_component_names(component))

    def test_render_rejects_category_specific_metadata_mismatches(self) -> None:
        joint = self.component(
            kind="arm_joint",
            side="left",
            count=1,
            element_order=("joint",),
            representation="joint_vector",
            unit="rad",
        )
        gripper = self.component(
            kind="gripper_open",
            side="right",
            count=1,
            element_order=("opening",),
            representation="scalar",
            unit="unitless",
        )
        fixed_xyz = self.component(
            kind="base_position",
            side=None,
            count=3,
            element_order=("x", "y", "z"),
            representation="position_xyz",
            unit="m",
        )
        fixed_quaternion = self.component(
            kind="head_orientation",
            side=None,
            count=4,
            element_order=("x", "y", "z", "w"),
            representation="quaternion_xyzw",
            unit="unitless",
        )
        indexed = self.component(
            kind="head_position",
            side=None,
            count=2,
            element_order=("lift", "reach"),
            representation="position_vector",
            unit="m",
        )
        invalid_components = (
            ("joint representation", replace(joint, representation="scalar")),
            ("joint unit", replace(joint, unit="degree")),
            ("joint count", replace(joint, count=0, element_order=())),
            (
                "gripper representation",
                replace(gripper, representation="joint_vector"),
            ),
            ("gripper unit", replace(gripper, unit="m")),
            (
                "gripper count",
                replace(gripper, count=2, element_order=("a", "b")),
            ),
            (
                "fixed xyz representation",
                replace(fixed_xyz, representation="position_vector"),
            ),
            ("fixed xyz unit", replace(fixed_xyz, unit="cm")),
            (
                "fixed xyz count",
                replace(fixed_xyz, count=2, element_order=("x", "y")),
            ),
            (
                "fixed quaternion representation",
                replace(fixed_quaternion, representation="euler_xyz"),
            ),
            (
                "fixed quaternion unit",
                replace(fixed_quaternion, unit="rad"),
            ),
            (
                "fixed quaternion count",
                replace(
                    fixed_quaternion,
                    count=3,
                    element_order=("x", "y", "z"),
                ),
            ),
            (
                "indexed representation",
                replace(indexed, representation="position_xyz"),
            ),
            ("indexed unit", replace(indexed, unit="rad")),
            (
                "indexed count",
                replace(indexed, count=0, element_order=()),
            ),
        )

        for case, component in invalid_components:
            with self.subTest(case=case):
                self.assertIsNone(render_component_names(component))

    def test_render_validates_element_order_without_normalizing_it(self) -> None:
        head_position = self.component(
            kind="head_position",
            side=None,
            count=2,
            element_order=("vertical", "forward"),
            representation="position_vector",
            unit="m",
        )
        invalid_components = (
            self.component(
                kind="head_position",
                side=None,
                count=2,
                element_order=("lift",),
                representation="position_vector",
                unit="m",
            ),
            self.component(
                kind="head_position",
                side=None,
                count=2,
                element_order=("lift", "lift"),
                representation="position_vector",
                unit="m",
            ),
            self.component(
                kind="head_position",
                side=None,
                count=2,
                element_order=("lift", "   "),
                representation="position_vector",
                unit="m",
            ),
            self.component(
                kind="base_position",
                side=None,
                count=3,
                element_order=("z", "y", "x"),
                representation="position_xyz",
                unit="m",
            ),
            self.component(
                kind="head_orientation",
                side=None,
                count=4,
                element_order=("w", "x", "y", "z"),
                representation="quaternion_xyzw",
                unit="unitless",
            ),
            replace(
                head_position,
                element_order=cast(tuple[str, ...], ["vertical", "forward"]),
            ),
            replace(
                head_position,
                element_order=cast(tuple[str, ...], ("vertical", 2)),
            ),
        )

        for component in invalid_components:
            with self.subTest(component=component):
                self.assertIsNone(render_component_names(component))

        self.assertEqual(
            render_component_names(head_position),
            ("head_pos_0_m", "head_pos_1_m"),
        )

    def test_individual_validator_accepts_all_machine_families(self) -> None:
        valid_names = (
            "left_arm_joint_0_rad",
            "right_hand_joint_12_rad",
            "left_gripper_open",
            "right_gripper_open_scale",
            "left_eef_pos_x_m",
            "right_eef_rot_euler_z_rad",
            "head_joint_0_rad",
            "head_pos_2_m",
            "head_rot_euler_y_rad",
            "head_orient_quat_w",
            "torso_joint_3_rad",
            "neck_joint_4_rad",
            "base_pos_z_m",
            "base_rot_euler_x_rad",
        )

        for name in valid_names:
            with self.subTest(name=name):
                self.assertTrue(is_standard_machine_name(name))

    def test_individual_validator_rejects_noncanonical_names(self) -> None:
        invalid_names = (
            "prefix_left_arm_joint_0_rad",
            "left_arm_joint_0_rad_suffix",
            "left_arm_joint_00_rad",
            "left_arm_joint_01_rad",
            "left_arm_joint_-1_rad",
            "left_arm_joint_0_degree",
            "left_eef_pos_w_m",
            "left_eef_pos_x_mm",
            "LEFT_arm_joint_0_rad",
            "wheel_joint_0_rad",
            "head_orient_quat_q",
            "base_rot_euler_x",
            "",
            42,
        )

        for name in invalid_names:
            with self.subTest(name=name):
                self.assertFalse(is_standard_machine_name(name))

    def test_array_validator_enforces_numbered_family_sequences(self) -> None:
        self.assertTrue(
            are_standard_machine_names(
                (
                    "left_arm_joint_0_rad",
                    "left_arm_joint_1_rad",
                    "right_arm_joint_0_rad",
                    "head_pos_0_m",
                    "head_pos_1_m",
                )
            )
        )
        invalid_sequences = (
            ("left_arm_joint_0_rad", "left_arm_joint_2_rad"),
            ("right_arm_joint_0_rad", "right_arm_joint_2_rad"),
            ("left_hand_joint_1_rad",),
            ("right_hand_joint_1_rad",),
            ("head_joint_1_rad", "head_joint_0_rad"),
            ("head_pos_0_m", "head_pos_2_m"),
            ("torso_joint_1_rad",),
            ("neck_joint_1_rad", "neck_joint_0_rad"),
        )

        for names in invalid_sequences:
            with self.subTest(names=names):
                self.assertFalse(are_standard_machine_names(names))

    def test_array_validator_rejects_very_long_index_without_raising(self) -> None:
        very_long_index = "9" * 5000

        self.assertFalse(
            are_standard_machine_names(
                (f"head_joint_{very_long_index}_rad",)
            )
        )

    def test_array_validator_enforces_complete_ordered_fixed_axes(self) -> None:
        self.assertTrue(
            are_standard_machine_names(
                (
                    "left_eef_pos_x_m",
                    "left_eef_pos_y_m",
                    "left_eef_pos_z_m",
                    "right_eef_pos_x_m",
                    "right_eef_pos_y_m",
                    "right_eef_pos_z_m",
                    "head_orient_quat_x",
                    "head_orient_quat_y",
                    "head_orient_quat_z",
                    "head_orient_quat_w",
                )
            )
        )
        invalid_sequences = (
            ("left_eef_pos_x_m", "left_eef_pos_y_m"),
            ("right_eef_pos_x_m", "right_eef_pos_y_m"),
            (
                "left_eef_rot_euler_x_rad",
                "left_eef_rot_euler_z_rad",
                "left_eef_rot_euler_y_rad",
            ),
            (
                "right_eef_rot_euler_x_rad",
                "right_eef_rot_euler_z_rad",
                "right_eef_rot_euler_y_rad",
            ),
            ("head_rot_euler_x_rad", "head_rot_euler_y_rad"),
            (
                "head_orient_quat_x",
                "head_orient_quat_y",
                "head_orient_quat_z",
            ),
            ("base_pos_x_m", "base_pos_z_m", "base_pos_y_m"),
            (
                "base_rot_euler_x_rad",
                "base_rot_euler_y_rad",
            ),
        )

        for names in invalid_sequences:
            with self.subTest(names=names):
                self.assertFalse(are_standard_machine_names(names))

    def test_array_validator_rejects_empty_or_duplicate_and_combines_families(
        self,
    ) -> None:
        self.assertFalse(are_standard_machine_names(()))
        self.assertFalse(
            are_standard_machine_names(
                ("left_gripper_open", "left_gripper_open")
            )
        )
        self.assertFalse(
            are_standard_machine_names(
                ("left_gripper_open", "wheel_joint_0_rad")
            )
        )
        self.assertTrue(
            are_standard_machine_names(
                (
                    "left_gripper_open",
                    "right_hand_joint_0_rad",
                    "right_hand_joint_1_rad",
                    "base_rot_euler_x_rad",
                    "base_rot_euler_y_rad",
                    "base_rot_euler_z_rad",
                    "neck_joint_0_rad",
                )
            )
        )


class _ExplodingCopyDict(dict[str, object]):
    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise MemoryError("copy allocation failed")


class _FloatSubclass(float):
    pass


class _StringSubclass(str):
    pass


class _ExplodingEquality:
    def __deepcopy__(self, memo: dict[int, object]) -> object:
        return self

    def __eq__(self, other: object) -> bool:
        raise MemoryError("comparison allocation failed")


class StandardApplicationTest(unittest.TestCase):
    """Verify conservative application of researched identity and camera facts."""

    SOURCE_KEY = "observation.images.source_camera"
    SECOND_SOURCE_KEY = "observation.images.second_camera"
    TARGET_KEY = "observation.images.cam_front_head_rgb"
    SECOND_TARGET_KEY = "observation.images.cam_left_wrist_rgb"

    @staticmethod
    def _candidate() -> DatasetCandidate:
        root = Path("/fixture/dataset")
        return DatasetCandidate(
            dataset_name="dataset",
            task_name=None,
            source_path=root,
            layout_type=LayoutType.FLAT,
            info_path=root / "meta/info.json",
            data_path=root / "data",
            video_path=root / "videos",
            depth_path=root / "depth",
        )

    @staticmethod
    def _identity_evidence(
        *,
        info_state: str = "present",
        info_value: object | None = "raw-model",
        common_state: str = "missing",
        common_value: object | None = None,
        tasks_state: str = "missing",
        tasks: tuple[object, ...] = (),
        issues: tuple[Issue, ...] = (),
    ) -> IdentityEvidence:
        return IdentityEvidence(
            info_robot_type_state=info_state,
            info_robot_type=info_value,
            common_record_state=common_state,
            common_record=common_value,
            tasks_state=tasks_state,
            tasks=tasks,
            issues=issues,
        )

    @staticmethod
    def _schema(
        source_key: str,
        *,
        dtype: object = "video",
        shape: tuple[object, ...] = (480, 640, 3),
        fps: object = 20,
        codec: object = "h264",
    ) -> FeatureSchema:
        return FeatureSchema(source_key, dtype, shape, (), fps, codec)

    @staticmethod
    def _sample(
        source_key: str,
        *,
        media_type: str = "video",
        codec: str | None = "av1",
        fps: float | None = 20.0,
        width: int | None = 640,
        height: int | None = 480,
        frame_path: Path | None = Path("/fixture/frame.jpg"),
    ) -> MediaSample:
        return MediaSample(
            relative_path=f"videos/{source_key}/episode_000000.mp4",
            media_type=media_type,
            codec=codec,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=1.0,
            pixel_format="yuv420p",
            frame_path=frame_path,
        )

    @staticmethod
    def _machine(source_key: str) -> MachineEvidence:
        schema = FeatureSchema(source_key, "float32", (2,), ("raw_0", "raw_1"), None, None)
        return MachineEvidence(schema, (), ())

    def _evidence(
        self,
        *,
        source_key: str | None = None,
        dtype: object = "video",
        shape: tuple[object, ...] = (480, 640, 3),
        fps: object = 20,
        source_codec: object = "h264",
        sample: MediaSample | None = None,
        samples: tuple[MediaSample, ...] | None = None,
        identity: IdentityEvidence | None = None,
        include_robot_type: bool = True,
        issues: tuple[Issue, ...] = (),
    ) -> DatasetEvidence:
        key = source_key or self.SOURCE_KEY
        schema = self._schema(key, dtype=dtype, shape=shape, fps=fps, codec=source_codec)
        if samples is None:
            actual_sample = sample or self._sample(
                key,
                media_type=dtype if isinstance(dtype, str) else "video",
                fps=fps if type(fps) in (int, float) else cast(float, fps),
                width=shape[-2] if len(shape) >= 2 and type(shape[-2]) is int else 640,
                height=shape[-3] if len(shape) >= 3 and type(shape[-3]) is int else 480,
            )
            samples = (actual_sample,)
        feature = {
            "dtype": dtype,
            "shape": list(shape),
            "fps": fps,
            "codec": source_codec,
            "custom": {"preserve": True},
        }
        source_info: dict[str, object] = {
            "fps": 20,
            "untouched": {"nested": [1, 2, 3]},
            "features": {
                key: feature,
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["raw_action_0", "raw_action_1"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["raw_state_0", "raw_state_1"],
                },
            },
        }
        if include_robot_type:
            source_info["robot_type"] = "raw-model"
        identity_value = identity or self._identity_evidence(
            info_state="present" if include_robot_type else "missing",
            info_value="raw-model" if include_robot_type else None,
        )
        return DatasetEvidence(
            candidate=self._candidate(),
            source_info=source_info,
            identity=identity_value,
            cameras=(CameraEvidence(schema, samples),),
            machines=(self._machine("action"), self._machine("observation.state")),
            issues=issues,
        )

    @staticmethod
    def _source(
        source_id: str = "official-product",
        *,
        kind: str = "official_product",
    ) -> SourceReference:
        return SourceReference(
            source_id,
            f"Fixture source {source_id}",
            f"https://fixtures.invalid/{source_id}",
            kind,
        )

    @staticmethod
    def _assessments() -> tuple[IdentityAssessment, ...]:
        return (
            IdentityAssessment("info_robot_type", "supports", "local model token agrees"),
            IdentityAssessment("common_record", "missing", "file is absent"),
            IdentityAssessment("tasks", "missing", "file is absent"),
        )

    def _identity_fact(
        self,
        *,
        manufacturer: str | None = "Acme Robotics",
        model: str | None = "XR-7",
        confidence: float = 0.95,
        ambiguous: bool = False,
        status: str = "consistent",
        source_ids: tuple[str, ...] = ("official-product",),
        assessments: tuple[IdentityAssessment, ...] | None = None,
    ) -> RobotIdentityFact:
        return RobotIdentityFact(
            manufacturer=manufacturer,
            model=model,
            confidence=confidence,
            ambiguous=ambiguous,
            reason="local evidence agrees with the official product page",
            local_evidence_status=status,
            source_ids=source_ids,
            assessments=assessments if assessments is not None else self._assessments(),
        )

    @staticmethod
    def _slot(
        *,
        camera_id: str = "head-rgb",
        direction_tokens: tuple[str, ...] = ("front",),
        body_part: str | None = "head",
        modality: str = "rgb",
        confidence: float = 0.96,
        ambiguous: bool = False,
        source_ids: tuple[str, ...] = ("official-product",),
    ) -> CameraSlot:
        return CameraSlot(
            camera_id=camera_id,
            interface_name="fixture interface",
            mount_type="on_robot",
            direction_tokens=direction_tokens,
            body_part=body_part,
            modality=modality,
            confidence=confidence,
            ambiguous=ambiguous,
            reason="official camera specification",
            source_ids=source_ids,
        )

    def _profile(
        self,
        *,
        identity: RobotIdentityFact | None = None,
        sources: tuple[SourceReference, ...] | None = None,
        cameras: tuple[CameraSlot, ...] | None = None,
    ) -> HardwareProfile:
        return HardwareProfile(
            identity=identity or self._identity_fact(),
            sources=sources if sources is not None else (self._source(),),
            cameras=cameras if cameras is not None else (self._slot(),),
            components=(),
        )

    def _mapping(
        self,
        *,
        source_key: str | None = None,
        camera_id: str | None = "head-rgb",
        confidence: float = 0.94,
        ambiguous: bool = False,
        cameras: tuple[CameraAssignment, ...] | None = None,
        machines: tuple[MachineAssignment, ...] = (),
    ) -> DatasetMapping:
        assignment = CameraAssignment(
            source_key or self.SOURCE_KEY,
            camera_id,
            confidence,
            ambiguous,
            "representative frames match the official slot",
        )
        return DatasetMapping(cameras or (assignment,), machines)

    def _with_second_camera(self, evidence: DatasetEvidence) -> DatasetEvidence:
        source_info = deepcopy(evidence.source_info)
        features = cast(dict[str, object], source_info["features"])
        features[self.SECOND_SOURCE_KEY] = deepcopy(features[self.SOURCE_KEY])
        first = evidence.cameras[0]
        second_schema = replace(first.schema, source_key=self.SECOND_SOURCE_KEY)
        second_sample = replace(
            first.samples[0],
            relative_path=f"videos/{self.SECOND_SOURCE_KEY}/episode_000000.mp4",
            frame_path=Path("/fixture/second-frame.jpg"),
        )
        return replace(
            evidence,
            source_info=source_info,
            cameras=(*evidence.cameras, CameraEvidence(second_schema, (second_sample,))),
        )

    def assert_camera_kept(
        self,
        result: NormalizationResult,
        source_key: str | None = None,
    ) -> None:
        features = cast(dict[str, object], result.normalized_info["features"])
        self.assertIn(source_key or self.SOURCE_KEY, features)

    def assert_no_surrogate_text(self, value: object) -> None:
        if isinstance(value, str):
            self.assertFalse(
                any(0xD800 <= ord(character) <= 0xDFFF for character in value)
            )
        elif isinstance(value, dict):
            for key, item in value.items():
                self.assert_no_surrogate_text(key)
                self.assert_no_surrogate_text(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.assert_no_surrogate_text(item)
        elif hasattr(value, "__dataclass_fields__"):
            for field_name in value.__dataclass_fields__:
                self.assert_no_surrogate_text(getattr(value, field_name))

    def test_preconditions_return_only_exact_ordered_blocking_codes(self) -> None:
        complete = self._evidence()
        self.assertEqual(check_preconditions(complete), ())

        invalid_action = replace(
            complete.machines[0],
            schema=replace(complete.machines[0].schema, source_key="action.extra"),
        )
        invalid_observation = replace(
            complete.machines[1],
            schema=replace(complete.machines[1].schema, source_key="observation.stateful"),
        )
        evidence = replace(
            complete,
            cameras=(),
            machines=(invalid_action, invalid_observation),
        )

        issues = check_preconditions(evidence)

        self.assertEqual(
            [item.code for item in issues],
            ["MISSING_PRIMARY_CAMERA", "MISSING_ACTION", "MISSING_OBSERVATION"],
        )
        self.assertTrue(all(item.severity == "block" for item in issues))
        dotted_observation = replace(
            complete.machines[1],
            schema=replace(complete.machines[1].schema, source_key="observation.state.arm"),
        )
        self.assertEqual(
            check_preconditions(replace(complete, machines=(complete.machines[0], dotted_observation))),
            (),
        )

    def test_precondition_rgb_requires_strict_schema_and_every_frame(self) -> None:
        base = self._evidence()
        camera = base.cameras[0]
        invalid_cameras = (
            replace(camera, schema=replace(camera.schema, dtype="Video")),
            replace(camera, schema=replace(camera.schema, dtype=cast(object, ["video"]))),
            replace(camera, schema=replace(camera.schema, shape=(640, 3))),
            replace(camera, schema=replace(camera.schema, shape=(480, 640, 1))),
            replace(camera, schema=replace(camera.schema, shape=(480, 640, True))),
            replace(camera, schema=replace(camera.schema, shape=(480, 640, 3.0))),
            replace(camera, samples=()),
            replace(camera, samples=(replace(camera.samples[0], frame_path=None),)),
            replace(
                camera,
                samples=(camera.samples[0], replace(camera.samples[0], frame_path=None)),
            ),
        )
        for invalid in invalid_cameras:
            with self.subTest(invalid=invalid):
                self.assertEqual(
                    [item.code for item in check_preconditions(replace(base, cameras=(invalid,)))],
                    ["MISSING_PRIMARY_CAMERA"],
                )

        image = replace(
            camera,
            schema=replace(camera.schema, dtype="image", shape=(480, 640, 4)),
            samples=(replace(camera.samples[0], media_type="image"),),
        )
        self.assertEqual(check_preconditions(replace(base, cameras=(image,))), ())

    def test_missing_profile_or_mapping_returns_independent_exact_source_copy(self) -> None:
        evidence = self._evidence(source_key=self.TARGET_KEY, source_codec=None)
        source_feature = cast(dict[str, object], evidence.source_info["features"])[self.TARGET_KEY]
        cast(dict[str, object], source_feature).pop("codec")
        evidence = replace(
            evidence,
            cameras=(replace(evidence.cameras[0], schema=replace(evidence.cameras[0].schema, codec=None)),),
            issues=(Issue("EVIDENCE_FIRST", "evidence", "fixture"),),
        )
        extra = (Issue("EXTRA_SECOND", "extra", "fixture"),)
        cases = ((None, self._mapping(source_key=self.TARGET_KEY)), (self._profile(), None))

        for profile, mapping in cases:
            with self.subTest(profile=profile is not None, mapping=mapping is not None):
                result = apply_standard(
                    evidence,
                    profile,
                    mapping,
                    confidence_threshold=0.85,
                    extra_issues=extra,
                )
                self.assertEqual(result.normalized_info, evidence.source_info)
                self.assertIsNot(result.normalized_info, evidence.source_info)
                self.assertIsNot(result.normalized_info["features"], evidence.source_info["features"])
                self.assertNotIn(
                    "codec",
                    cast(dict[str, dict[str, object]], result.normalized_info["features"])[self.TARGET_KEY],
                )
                self.assertEqual([item.code for item in result.issues[:2]], ["EVIDENCE_FIRST", "EXTRA_SECOND"])
                self.assertEqual(
                    [record.source_address for record in result.machine_mappings],
                    ["features.action.names", "features.observation.state.names"],
                )
                self.assertEqual(result.robot_identity.output, "raw-model")

    def test_threshold_is_explicit_finite_builtin_and_equality_passes(self) -> None:
        threshold = 0.85
        profile = self._profile(
            identity=self._identity_fact(confidence=threshold),
            cameras=(self._slot(confidence=threshold),),
        )
        mapping = self._mapping(confidence=threshold)
        result = apply_standard(self._evidence(), profile, mapping, confidence_threshold=threshold)
        self.assertEqual(result.normalized_info["robot_type"], "acme_robotics_xr_7")
        self.assertIn(self.TARGET_KEY, result.normalized_info["features"])

        invalid_values = (True, False, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, _FloatSubclass(0.85))
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    apply_standard(self._evidence(), profile, mapping, confidence_threshold=cast(float, invalid))

    def test_memory_error_from_source_copy_propagates(self) -> None:
        evidence = replace(self._evidence(), source_info=_ExplodingCopyDict(self._evidence().source_info))
        with self.assertRaises(MemoryError):
            apply_standard(evidence, self._profile(), self._mapping(), confidence_threshold=0.85)

    def test_applies_sourced_identity_with_scoped_citations_and_generic_slug(self) -> None:
        third = self._source("community", kind="third_party")
        profile = self._profile(sources=(self._source(), third))

        result = apply_standard(self._evidence(), profile, self._mapping(), confidence_threshold=0.85)

        self.assertEqual(result.normalized_info["robot_type"], "acme_robotics_xr_7")
        record = result.robot_identity
        self.assertEqual((record.source, record.output, record.candidate), ("raw-model", "acme_robotics_xr_7", "acme_robotics_xr_7"))
        self.assertTrue(record.changed)
        self.assertNotEqual(record.decision, "review")
        self.assertEqual([citation["source_id"] for citation in record.citations], ["official-product"])
        self.assertNotIn("community", {citation["source_id"] for citation in record.citations})

    def test_identity_slug_requires_renderable_tokens_from_manufacturer_and_model(self) -> None:
        evidence = self._evidence()
        unrenderable_manufacturer = self._identity_fact(
            manufacturer="宇树科技",
            model="G1",
        )

        result = apply_standard(
            evidence,
            self._profile(identity=unrenderable_manufacturer),
            self._mapping(),
            confidence_threshold=0.85,
        )

        self.assertEqual(result.normalized_info["robot_type"], "raw-model")
        self.assertNotEqual(result.robot_identity.candidate, "g1")
        self.assertEqual(result.robot_identity.decision, "review")
        self.assert_camera_kept(result)

        normal = apply_standard(
            evidence,
            self._profile(),
            self._mapping(),
            confidence_threshold=0.85,
        )
        self.assertEqual(normal.normalized_info["robot_type"], "acme_robotics_xr_7")

    def test_identity_rejects_third_party_or_unreferenced_official_sources(self) -> None:
        third = self._source("community", kind="third_party")
        cases = (
            self._profile(sources=(replace(self._source(), kind="third_party"),)),
            self._profile(
                identity=self._identity_fact(source_ids=("community",)),
                sources=(self._source(), third),
            ),
        )
        for profile in cases:
            with self.subTest(profile=profile):
                result = apply_standard(self._evidence(), profile, self._mapping(), confidence_threshold=0.85)
                self.assertEqual(result.normalized_info["robot_type"], "raw-model")
                self.assertEqual(result.robot_identity.output, "raw-model")
                self.assertEqual(result.robot_identity.candidate, "acme_robotics_xr_7")
                self.assertEqual(result.robot_identity.decision, "review")
                self.assertIn("ROBOT_IDENTITY_UNRESOLVED", {item.code for item in result.issues})

    def test_missing_source_robot_type_is_never_created(self) -> None:
        evidence = self._evidence(include_robot_type=False)
        result = apply_standard(evidence, self._profile(), self._mapping(), confidence_threshold=0.85)
        self.assertNotIn("robot_type", result.normalized_info)
        self.assertIsNone(result.robot_identity.source)
        self.assertIsNone(result.robot_identity.output)
        self.assertEqual(result.robot_identity.candidate, "acme_robotics_xr_7")
        self.assertEqual(result.robot_identity.decision, "review")

    def test_missing_robot_type_does_not_block_identity_confirmed_by_other_local_source(self) -> None:
        identity_evidence = self._identity_evidence(
            info_state="missing",
            info_value=None,
            common_state="present",
            common_value={"robot": "XR-7"},
        )
        evidence = self._evidence(include_robot_type=False, identity=identity_evidence)
        fact = self._identity_fact(
            assessments=(
                IdentityAssessment("info_robot_type", "missing", "field is absent"),
                IdentityAssessment("common_record", "supports", "record identifies XR-7"),
                IdentityAssessment("tasks", "missing", "file is absent"),
            )
        )

        result = apply_standard(
            evidence,
            self._profile(identity=fact),
            self._mapping(),
            confidence_threshold=0.85,
        )

        self.assertNotIn("robot_type", result.normalized_info)
        self.assertNotIn(self.SOURCE_KEY, result.normalized_info["features"])
        self.assertIn(self.TARGET_KEY, result.normalized_info["features"])

    def test_identity_local_states_require_self_consistent_values_and_exact_tuples(self) -> None:
        base = self._evidence()
        invalid_identities = (
            replace(base.identity, common_record={"unexpected": True}),
            replace(base.identity, tasks=({"unexpected": True},)),
            replace(
                base.identity,
                tasks=cast(tuple[object, ...], [{"not": "an exact tuple"}]),
            ),
        )
        for identity in invalid_identities:
            with self.subTest(identity=identity):
                evidence = replace(base, identity=identity)
                result = apply_standard(
                    evidence,
                    self._profile(),
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assertEqual(result.normalized_info["robot_type"], "raw-model")
                self.assert_camera_kept(result)

    def test_present_null_common_and_invalid_tasks_with_valid_records_are_consistent(self) -> None:
        cases = (
            (
                replace(
                    self._evidence().identity,
                    common_record_state="present",
                    common_record=None,
                ),
                (
                    IdentityAssessment("info_robot_type", "supports", "model agrees"),
                    IdentityAssessment("common_record", "supports", "JSON null is explicit evidence"),
                    IdentityAssessment("tasks", "missing", "file is absent"),
                ),
            ),
            (
                replace(
                    self._evidence().identity,
                    tasks_state="invalid",
                    tasks=({"task": "valid retained line"},),
                ),
                (
                    IdentityAssessment("info_robot_type", "supports", "model agrees"),
                    IdentityAssessment("common_record", "missing", "file is absent"),
                    IdentityAssessment("tasks", "invalid", "other lines were invalid"),
                ),
            ),
        )
        for identity, assessments in cases:
            with self.subTest(identity=identity):
                result = apply_standard(
                    replace(self._evidence(), identity=identity),
                    self._profile(identity=self._identity_fact(assessments=assessments)),
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assertEqual(result.normalized_info["robot_type"], "acme_robotics_xr_7")
                self.assertIn(self.TARGET_KEY, result.normalized_info["features"])

    def test_identity_assessments_must_match_local_states_and_declared_status(self) -> None:
        base_evidence = self._evidence()
        invalid_issue = Issue("INFO_ROBOT_TYPE_INVALID", "invalid local value", "identity")
        invalid_cases: tuple[tuple[DatasetEvidence, RobotIdentityFact], ...] = (
            (
                base_evidence,
                self._identity_fact(assessments=(
                    IdentityAssessment("info_robot_type", "supports", "ok"),
                    IdentityAssessment("info_robot_type", "missing", "duplicate"),
                    IdentityAssessment("tasks", "missing", "absent"),
                )),
            ),
            (
                base_evidence,
                self._identity_fact(assessments=(
                    IdentityAssessment("info_robot_type", "supports", "ok"),
                    IdentityAssessment("common_record", "supports", "wrong for missing"),
                    IdentityAssessment("tasks", "missing", "absent"),
                )),
            ),
            (
                replace(base_evidence, identity=replace(base_evidence.identity, common_record_state="invalid")),
                self._identity_fact(),
            ),
            (
                replace(base_evidence, identity=replace(base_evidence.identity, issues=(invalid_issue,))),
                self._identity_fact(),
            ),
            (
                base_evidence,
                self._identity_fact(
                    status="conflicts_explained",
                    assessments=(
                        IdentityAssessment("info_robot_type", "supports", "ok"),
                        IdentityAssessment("common_record", "missing", "absent"),
                        IdentityAssessment("tasks", "missing", "absent"),
                    ),
                ),
            ),
            (
                base_evidence,
                self._identity_fact(
                    status="conflicts_explained",
                    assessments=(
                        IdentityAssessment("info_robot_type", "supports", "ok"),
                        IdentityAssessment("common_record", "conflicts", ""),
                        IdentityAssessment("tasks", "missing", "absent"),
                    ),
                ),
            ),
        )
        for evidence, identity in invalid_cases:
            with self.subTest(identity=identity, local=evidence.identity):
                result = apply_standard(
                    evidence,
                    self._profile(identity=identity),
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assertEqual(result.normalized_info.get("robot_type"), "raw-model")
                self.assertEqual(result.robot_identity.decision, "review")

    def test_explained_identity_conflict_is_accepted_when_support_and_explanation_exist(self) -> None:
        evidence = self._evidence(
            identity=self._identity_evidence(
                common_state="present",
                common_value={"legacy_model": "XR-6"},
            )
        )
        fact = self._identity_fact(
            status="conflicts_explained",
            assessments=(
                IdentityAssessment("info_robot_type", "supports", "source names XR-7"),
                IdentityAssessment(
                    "common_record",
                    "conflicts",
                    "record is a documented stale XR-6 export",
                ),
                IdentityAssessment("tasks", "missing", "file is absent"),
            ),
        )

        result = apply_standard(
            evidence,
            self._profile(identity=fact),
            self._mapping(),
            confidence_threshold=0.85,
        )

        self.assertEqual(result.normalized_info["robot_type"], "acme_robotics_xr_7")
        self.assertIn(self.TARGET_KEY, result.normalized_info["features"])

    def test_identity_rejects_unsafe_ambiguous_low_or_malformed_facts(self) -> None:
        facts = (
            self._identity_fact(ambiguous=True),
            self._identity_fact(confidence=0.84),
            self._identity_fact(confidence=cast(float, True)),
            self._identity_fact(confidence=float("nan")),
            self._identity_fact(manufacturer="Acme\nRobotics"),
            self._identity_fact(model=" XR-7"),
            self._identity_fact(source_ids=("official-product", "official-product")),
        )
        for fact in facts:
            with self.subTest(fact=fact):
                result = apply_standard(
                    self._evidence(),
                    self._profile(identity=fact),
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assertEqual(result.normalized_info["robot_type"], "raw-model")
                self.assertEqual(result.robot_identity.decision, "review")

    def test_surrogate_identity_and_source_text_fails_closed_without_reaching_review(self) -> None:
        surrogate = "\ud800"
        base_fact = self._identity_fact()
        profiles = (
            self._profile(identity=replace(base_fact, manufacturer=f"Acme{surrogate}")),
            self._profile(identity=replace(base_fact, reason=f"unsafe{surrogate}reason")),
            self._profile(sources=(replace(self._source(), title=f"unsafe{surrogate}title"),)),
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                result = apply_standard(
                    self._evidence(),
                    profile,
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assertEqual(result.normalized_info["robot_type"], "raw-model")
                self.assert_camera_kept(result)
                self.assert_no_surrogate_text(
                    (result.robot_identity, result.camera_mappings, result.issues)
                )

    def test_camera_requires_reliable_identity_assignment_slot_and_own_official_source(self) -> None:
        third = self._source("community", kind="third_party")
        low_identity = self._profile(identity=self._identity_fact(confidence=0.2))
        slot_third_party = self._profile(
            sources=(self._source(), third),
            cameras=(self._slot(source_ids=("community",)),),
        )
        cases = (
            (low_identity, self._mapping()),
            (self._profile(), self._mapping(ambiguous=True)),
            (self._profile(), self._mapping(confidence=0.2)),
            (self._profile(cameras=(self._slot(ambiguous=True),)), self._mapping()),
            (self._profile(cameras=(self._slot(confidence=0.2),)), self._mapping()),
            (slot_third_party, self._mapping()),
            (self._profile(), self._mapping(camera_id=None, ambiguous=True)),
        )
        for profile, mapping in cases:
            with self.subTest(profile=profile, mapping=mapping):
                result = apply_standard(self._evidence(), profile, mapping, confidence_threshold=0.85)
                self.assert_camera_kept(result)
                self.assertEqual(result.camera_mappings[0].decision, "review")
                self.assertTrue(any(item.scope == f"features.{self.SOURCE_KEY}" for item in result.issues))

    def test_camera_slot_renderer_requires_exact_safe_builtin_semantics(self) -> None:
        valid = self._slot()
        malformed_slots = (
            replace(valid, mount_type=_StringSubclass("on_robot")),
            replace(valid, modality=_StringSubclass("rgb")),
            replace(valid, body_part=_StringSubclass("head")),
            replace(valid, direction_tokens=cast(tuple[str, ...], ["front"])),
            replace(valid, direction_tokens=(_StringSubclass("front"),)),
            replace(valid, direction_tokens=cast(tuple[str, ...], (["front"],))),
        )
        for slot in malformed_slots:
            with self.subTest(slot=slot):
                result = apply_standard(
                    self._evidence(),
                    self._profile(cameras=(slot,)),
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assert_camera_kept(result)
                self.assertEqual(result.camera_mappings[0].decision, "review")

    def test_identity_states_and_assessment_source_ids_require_exact_builtin_strings(self) -> None:
        base = self._evidence()
        state_subclass = replace(
            base.identity,
            info_robot_type_state=_StringSubclass("present"),
        )
        source_subclass = self._identity_fact(
            assessments=(
                IdentityAssessment(_StringSubclass("info_robot_type"), "supports", "agrees"),
                IdentityAssessment("common_record", "missing", "absent"),
                IdentityAssessment("tasks", "missing", "absent"),
            )
        )
        cases = (
            (replace(base, identity=state_subclass), self._identity_fact()),
            (base, source_subclass),
        )
        for evidence, fact in cases:
            with self.subTest(evidence=evidence, fact=fact):
                result = apply_standard(
                    evidence,
                    self._profile(identity=fact),
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assertEqual(result.normalized_info["robot_type"], "raw-model")
                self.assert_camera_kept(result)

    def test_camera_rejects_missing_sources_and_handcrafted_duplicate_identifiers(self) -> None:
        duplicate_slot = replace(self._slot(direction_tokens=("left",), body_part="wrist"), camera_id="head-rgb")
        duplicate_profile = self._profile(cameras=(self._slot(), duplicate_slot))
        duplicate_mapping = self._mapping(
            cameras=(
                CameraAssignment(self.SOURCE_KEY, "head-rgb", 0.95, False, "first"),
                CameraAssignment(self.SOURCE_KEY, "head-rgb", 0.95, False, "duplicate"),
            )
        )
        cases = (
            (duplicate_profile, self._mapping()),
            (self._profile(), duplicate_mapping),
            (self._profile(), self._mapping(source_key="observation.images.unknown")),
        )
        for profile, mapping in cases:
            with self.subTest(profile=profile, mapping=mapping):
                result = apply_standard(self._evidence(), profile, mapping, confidence_threshold=0.85)
                self.assert_camera_kept(result)
                self.assertEqual(result.camera_mappings[0].decision, "review")

    def test_camera_rejects_schema_frame_media_and_numeric_mismatches(self) -> None:
        base = self._evidence()
        camera = base.cameras[0]
        sample = camera.samples[0]
        invalid_evidence = (
            replace(base, cameras=(replace(camera, schema=replace(camera.schema, dtype="array")),)),
            replace(base, cameras=(replace(camera, schema=replace(camera.schema, shape=(480, 640, 1))),)),
            replace(base, cameras=(replace(camera, schema=replace(camera.schema, shape=(480, 640, True))),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, frame_path=None),)),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, media_type="image"),)),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, fps=19.0),)),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, fps=float("nan")),)),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, fps=cast(float, True)),)),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, width=641),)),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, width=cast(int, True)),)),)),
            replace(base, cameras=(replace(camera, samples=(replace(sample, height=479),)),)),
        )
        source_mismatch = deepcopy(base.source_info)
        cast(dict[str, dict[str, object]], source_mismatch["features"])[self.SOURCE_KEY]["shape"] = [720, 1280, 3]
        bool_fps = deepcopy(base.source_info)
        cast(dict[str, dict[str, object]], bool_fps["features"])[self.SOURCE_KEY]["fps"] = True
        invalid_evidence = (
            *invalid_evidence,
            replace(base, source_info=source_mismatch),
            replace(
                self._evidence(fps=1, sample=self._sample(self.SOURCE_KEY, fps=1.0)),
                source_info=bool_fps,
            ),
        )

        for evidence in invalid_evidence:
            with self.subTest(evidence=evidence):
                result = apply_standard(evidence, self._profile(), self._mapping(), confidence_threshold=0.85)
                self.assert_camera_kept(result)
                self.assertIn("CAMERA_MEDIA_MISMATCH", {item.code for item in result.issues})

    def test_extreme_builtin_media_number_fails_closed_without_overflow(self) -> None:
        huge = 10 ** 10000
        evidence = self._evidence(
            fps=huge,
            sample=self._sample(self.SOURCE_KEY, fps=cast(float, huge)),
        )

        result = apply_standard(
            evidence,
            self._profile(),
            self._mapping(),
            confidence_threshold=0.85,
        )

        self.assert_camera_kept(result)
        self.assertIn("CAMERA_MEDIA_MISMATCH", {item.code for item in result.issues})

    def test_every_sample_must_match_and_static_images_need_complete_probe_data(self) -> None:
        base = self._evidence()
        first = base.cameras[0].samples[0]
        invalid_second = replace(first, relative_path="videos/second.mp4", height=479, frame_path=Path("/fixture/two.jpg"))
        multi = replace(base, cameras=(replace(base.cameras[0], samples=(first, invalid_second)),))
        result = apply_standard(multi, self._profile(), self._mapping(), confidence_threshold=0.85)
        self.assert_camera_kept(result)
        self.assertIn("CAMERA_MEDIA_MISMATCH", {item.code for item in result.issues})

        image = self._evidence(dtype="image", sample=self._sample(self.SOURCE_KEY, media_type="image", fps=None))
        image_result = apply_standard(image, self._profile(), self._mapping(), confidence_threshold=0.85)
        self.assert_camera_kept(image_result)
        self.assertIn("CAMERA_MEDIA_MISMATCH", {item.code for item in image_result.issues})

    def test_camera_application_needs_at_least_one_frame_not_one_per_sample(self) -> None:
        base = self._evidence()
        first = base.cameras[0].samples[0]
        second_without_frame = replace(
            first,
            relative_path="videos/second.mp4",
            frame_path=None,
        )
        evidence = replace(
            base,
            cameras=(replace(base.cameras[0], samples=(first, second_without_frame)),),
        )

        result = apply_standard(
            evidence,
            self._profile(),
            self._mapping(),
            confidence_threshold=0.85,
        )

        self.assertNotIn(self.SOURCE_KEY, result.normalized_info["features"])
        self.assertIn(self.TARGET_KEY, result.normalized_info["features"])

        no_frames = replace(
            base,
            cameras=(
                replace(
                    base.cameras[0],
                    samples=(replace(first, frame_path=None), second_without_frame),
                ),
            ),
        )
        rejected = apply_standard(
            no_frames,
            self._profile(),
            self._mapping(),
            confidence_threshold=0.85,
        )
        self.assert_camera_kept(rejected)
        self.assertIn("CAMERA_MEDIA_MISMATCH", {item.code for item in rejected.issues})

    def test_rgb_and_depth_apply_pdf_codecs_without_changing_declared_schema(self) -> None:
        rgb = self._evidence(shape=(480, 640, 4))
        rgb_result = apply_standard(rgb, self._profile(), self._mapping(), confidence_threshold=0.85)
        rgb_output = cast(dict[str, dict[str, object]], rgb_result.normalized_info["features"])[self.TARGET_KEY]
        self.assertEqual((rgb_output["dtype"], rgb_output["shape"], rgb_output["fps"], rgb_output["codec"]), ("video", [480, 640, 4], 20, "av1"))

        depth_slot = self._slot(modality="depth")
        depth_sample = self._sample(self.SOURCE_KEY, codec="ffv1")
        depth = self._evidence(shape=(480, 640, 1), sample=depth_sample)
        depth_result = apply_standard(
            depth,
            self._profile(cameras=(depth_slot,)),
            self._mapping(),
            confidence_threshold=0.85,
        )
        depth_key = "observation.images.cam_front_head_depth"
        depth_output = cast(dict[str, dict[str, object]], depth_result.normalized_info["features"])[depth_key]
        self.assertEqual(depth_output["codec"], "ffv1")
        self.assertNotIn("MEDIA_TRANSCODE_REQUIRED", {item.code for item in depth_result.issues})

    def test_different_or_unknown_actual_codec_applies_metadata_and_requires_review(self) -> None:
        for codec in ("h264", None):
            with self.subTest(codec=codec):
                evidence = self._evidence(sample=self._sample(self.SOURCE_KEY, codec=codec))
                result = apply_standard(evidence, self._profile(), self._mapping(), confidence_threshold=0.85)
                features = cast(dict[str, dict[str, object]], result.normalized_info["features"])
                self.assertNotIn(self.SOURCE_KEY, features)
                self.assertEqual(features[self.TARGET_KEY]["codec"], "av1")
                self.assertIn("MEDIA_TRANSCODE_REQUIRED", {item.code for item in result.issues})
                self.assertEqual(result.camera_mappings[0].decision, "review")

    def test_duplicate_render_targets_keep_every_involved_source(self) -> None:
        evidence = self._with_second_camera(self._evidence())
        slots = (
            self._slot(camera_id="head-a"),
            self._slot(camera_id="head-b"),
        )
        mapping = self._mapping(
            cameras=(
                CameraAssignment(self.SOURCE_KEY, "head-a", 0.95, False, "first"),
                CameraAssignment(self.SECOND_SOURCE_KEY, "head-b", 0.95, False, "second"),
            )
        )

        result = apply_standard(evidence, self._profile(cameras=slots), mapping, confidence_threshold=0.85)

        features = cast(dict[str, object], result.normalized_info["features"])
        self.assertIn(self.SOURCE_KEY, features)
        self.assertIn(self.SECOND_SOURCE_KEY, features)
        self.assertNotIn(self.TARGET_KEY, features)
        self.assertEqual([record.decision for record in result.camera_mappings], ["review", "review"])
        self.assertEqual([item.code for item in result.issues].count("CAMERA_NAME_COLLISION"), 2)

    def test_render_collision_includes_low_ambiguous_and_media_rejected_candidates(self) -> None:
        base = self._with_second_camera(self._evidence())
        slots = (
            self._slot(camera_id="head-a"),
            self._slot(camera_id="head-b"),
        )
        second = CameraAssignment(
            self.SECOND_SOURCE_KEY,
            "head-b",
            0.95,
            False,
            "second",
        )
        cases = (
            (
                base,
                replace(second, confidence=0.2),
            ),
            (
                base,
                replace(second, ambiguous=True),
            ),
            (
                replace(
                    base,
                    cameras=(
                        base.cameras[0],
                        replace(
                            base.cameras[1],
                            samples=(replace(base.cameras[1].samples[0], width=641),),
                        ),
                    ),
                ),
                second,
            ),
        )
        for evidence, unsafe_second in cases:
            with self.subTest(unsafe_second=unsafe_second):
                mapping = self._mapping(
                    cameras=(
                        CameraAssignment(self.SOURCE_KEY, "head-a", 0.95, False, "first"),
                        unsafe_second,
                    )
                )
                result = apply_standard(
                    evidence,
                    self._profile(cameras=slots),
                    mapping,
                    confidence_threshold=0.85,
                )
                features = cast(dict[str, object], result.normalized_info["features"])
                self.assertIn(self.SOURCE_KEY, features)
                self.assertIn(self.SECOND_SOURCE_KEY, features)
                self.assertNotIn(self.TARGET_KEY, features)
                self.assertEqual(
                    [item.code for item in result.issues].count("CAMERA_NAME_COLLISION"),
                    2,
                )

    def test_unresolved_assignment_without_render_target_does_not_block_safe_target(self) -> None:
        evidence = self._with_second_camera(self._evidence())
        mapping = self._mapping(
            cameras=(
                CameraAssignment(self.SOURCE_KEY, "head-rgb", 0.95, False, "safe"),
                CameraAssignment(self.SECOND_SOURCE_KEY, None, 0.2, True, "unresolved"),
            )
        )

        result = apply_standard(
            evidence,
            self._profile(),
            mapping,
            confidence_threshold=0.85,
        )

        features = cast(dict[str, object], result.normalized_info["features"])
        self.assertNotIn(self.SOURCE_KEY, features)
        self.assertIn(self.TARGET_KEY, features)
        self.assertIn(self.SECOND_SOURCE_KEY, features)
        self.assertNotIn("CAMERA_NAME_COLLISION", {item.code for item in result.issues})

    def test_occupied_target_blocks_only_that_plan_and_safe_camera_still_applies(self) -> None:
        evidence = self._with_second_camera(self._evidence())
        source_info = deepcopy(evidence.source_info)
        cast(dict[str, object], source_info["features"])[self.TARGET_KEY] = {"dtype": "float32", "shape": [1]}
        evidence = replace(evidence, source_info=source_info)
        slots = (
            self._slot(camera_id="head"),
            self._slot(camera_id="wrist", direction_tokens=("left",), body_part="wrist"),
        )
        mapping = self._mapping(
            cameras=(
                CameraAssignment(self.SOURCE_KEY, "head", 0.95, False, "occupied"),
                CameraAssignment(self.SECOND_SOURCE_KEY, "wrist", 0.95, False, "safe"),
            )
        )

        result = apply_standard(evidence, self._profile(cameras=slots), mapping, confidence_threshold=0.85)

        features = cast(dict[str, object], result.normalized_info["features"])
        self.assertIn(self.SOURCE_KEY, features)
        self.assertEqual(features[self.TARGET_KEY], {"dtype": "float32", "shape": [1]})
        self.assertNotIn(self.SECOND_SOURCE_KEY, features)
        self.assertIn(self.SECOND_TARGET_KEY, features)
        self.assertEqual([record.decision for record in result.camera_mappings], ["review", "apply"])
        self.assertIn("CAMERA_NAME_COLLISION", {item.code for item in result.issues})

    def test_standard_source_key_changes_only_codec_and_reports_actual_schema_objects(self) -> None:
        evidence = self._evidence(source_key=self.TARGET_KEY)
        mapping = self._mapping(source_key=self.TARGET_KEY)

        result = apply_standard(evidence, self._profile(), mapping, confidence_threshold=0.85)

        record = result.camera_mappings[0]
        self.assertEqual(record.source_address, f"features.{self.TARGET_KEY}")
        self.assertEqual(record.source["codec"], "h264")
        self.assertEqual(record.output["codec"], "av1")
        self.assertEqual(record.candidate, self.TARGET_KEY)
        self.assertTrue(record.changed)
        self.assertEqual(record.decision, "apply")
        self.assertEqual(tuple(result.normalized_info["features"]), tuple(evidence.source_info["features"]))

    def test_rejected_camera_keeps_actual_output_and_candidate_separate(self) -> None:
        result = apply_standard(
            self._evidence(),
            self._profile(),
            self._mapping(ambiguous=True),
            confidence_threshold=0.85,
        )
        source_feature = cast(dict[str, dict[str, object]], self._evidence().source_info["features"])[self.SOURCE_KEY]
        record = result.camera_mappings[0]
        self.assertEqual(record.source_address, f"features.{self.SOURCE_KEY}")
        self.assertEqual(record.source, source_feature)
        self.assertEqual(record.output, source_feature)
        self.assertEqual(record.candidate, self.TARGET_KEY)
        self.assertFalse(record.changed)
        self.assertEqual(record.decision, "review")
        self.assertTrue(any(item.scope == record.source_address for item in result.issues))

    def test_preserves_unrelated_metadata_and_never_changes_machine_fields(self) -> None:
        evidence = self._evidence()
        before_action = deepcopy(cast(dict[str, object], evidence.source_info["features"])["action"])
        result = apply_standard(
            evidence,
            self._profile(),
            self._mapping(
                machines=(MachineAssignment("action", (), 1.0, False, "ignored until machine phase"),)
            ),
            confidence_threshold=0.85,
        )
        self.assertEqual(result.normalized_info["untouched"], {"nested": [1, 2, 3]})
        self.assertEqual(cast(dict[str, object], result.normalized_info["features"])["action"], before_action)
        self.assertEqual(
            [record.source_address for record in result.machine_mappings],
            ["features.action.names", "features.observation.state.names"],
        )

    def test_manual_nan_bool_and_duplicate_values_fail_closed_without_crashing(self) -> None:
        profiles_and_mappings = (
            (self._profile(identity=self._identity_fact(confidence=float("nan"))), self._mapping()),
            (self._profile(cameras=(self._slot(confidence=float("nan")),)), self._mapping()),
            (self._profile(cameras=(self._slot(confidence=cast(float, True)),)), self._mapping()),
            (self._profile(), self._mapping(confidence=float("nan"))),
            (self._profile(), self._mapping(confidence=cast(float, True))),
        )
        for profile, mapping in profiles_and_mappings:
            with self.subTest(profile=profile, mapping=mapping):
                result = apply_standard(self._evidence(), profile, mapping, confidence_threshold=0.85)
                self.assertEqual(result.normalized_info["robot_type"], "raw-model" if profile.identity.confidence != 0.95 else "acme_robotics_xr_7")
                self.assert_camera_kept(result)
                self.assertEqual(result.camera_mappings[0].decision, "review")

    def test_malformed_official_url_cannot_authorize_identity_or_camera_changes(self) -> None:
        malformed = replace(self._source(), url="https://?model=xr7")
        result = apply_standard(
            self._evidence(),
            self._profile(sources=(malformed,)),
            self._mapping(),
            confidence_threshold=0.85,
        )
        self.assertEqual(result.normalized_info["robot_type"], "raw-model")
        self.assert_camera_kept(result)

    def test_identity_comparison_memory_error_propagates(self) -> None:
        exploding = _ExplodingEquality()
        evidence = self._evidence(
            identity=self._identity_evidence(info_value=exploding)
        )
        source_info = deepcopy(evidence.source_info)
        source_info["robot_type"] = exploding
        evidence = replace(evidence, source_info=source_info)

        with self.assertRaises(MemoryError):
            apply_standard(
                evidence,
                self._profile(),
                self._mapping(),
                confidence_threshold=0.85,
            )

    def test_issue_order_is_evidence_then_extra_then_identity_then_camera(self) -> None:
        evidence = self._evidence(issues=(Issue("EVIDENCE", "evidence", "fixture"),))
        result = apply_standard(
            evidence,
            self._profile(identity=self._identity_fact(ambiguous=True)),
            self._mapping(ambiguous=True),
            confidence_threshold=0.85,
            extra_issues=(Issue("EXTRA", "extra", "fixture"),),
        )
        codes = [item.code for item in result.issues]
        self.assertEqual(codes[:2], ["EVIDENCE", "EXTRA"])
        self.assertLess(codes.index("ROBOT_IDENTITY_UNRESOLVED"), codes.index("CAMERA_MAPPING_UNRESOLVED"))


class MachineApplicationTest(unittest.TestCase):
    """Verify feature-atomic machine naming and conservative gripper handling."""

    @staticmethod
    def _candidate() -> DatasetCandidate:
        root = Path("/fixture/machine-dataset")
        return DatasetCandidate(
            "machine-dataset",
            None,
            root,
            LayoutType.FLAT,
            root / "meta/info.json",
            root / "data",
            None,
            None,
        )

    @staticmethod
    def _identity_evidence() -> IdentityEvidence:
        return IdentityEvidence(
            "present",
            "raw-model",
            "missing",
            None,
            "missing",
            (),
        )

    @staticmethod
    def _identity_fact() -> RobotIdentityFact:
        return RobotIdentityFact(
            "Fixture Robotics",
            "Model One",
            0.95,
            False,
            "official identity",
            "consistent",
            ("official-identity",),
            (
                IdentityAssessment("info_robot_type", "supports", "agrees"),
                IdentityAssessment("common_record", "missing", "absent"),
                IdentityAssessment("tasks", "missing", "absent"),
            ),
        )

    @staticmethod
    def _source(
        source_id: str,
        *,
        kind: str = "official_product",
    ) -> SourceReference:
        return SourceReference(
            source_id,
            f"Fixture source {source_id}",
            f"https://fixtures.invalid/{source_id}",
            kind,
        )

    @classmethod
    def _machine(
        cls,
        source_feature: str = "action",
        *,
        names: tuple[object, ...] = ("raw_0", "raw_1"),
        shape: tuple[object, ...] = (2,),
        episode_lengths: tuple[object, ...] = (2, 2),
        gripper_ranges: tuple[GripperRange, ...] = (),
    ) -> MachineEvidence:
        episodes = tuple(
            ParquetEpisodeEvidence(
                f"data/chunk-000/episode_{index:06d}.parquet",
                (source_feature,),
                {source_feature: length if type(length) is int else None},
            )
            for index, length in enumerate(episode_lengths)
        )
        return MachineEvidence(
            FeatureSchema(source_feature, "float32", shape, names, None, None),
            episodes,
            cast(tuple[int, ...], episode_lengths),
            gripper_ranges,
        )

    @classmethod
    def _evidence(
        cls,
        machines: tuple[MachineEvidence, ...] | None = None,
    ) -> DatasetEvidence:
        actual_machines = machines or (cls._machine(),)
        features: dict[str, object] = {}
        for machine in actual_machines:
            features[machine.schema.source_key] = {
                "dtype": machine.schema.dtype,
                "shape": list(machine.schema.shape),
                "names": list(machine.schema.names),
                "values": [0.25, 0.75],
                "custom": {"keep": True},
            }
        return DatasetEvidence(
            cls._candidate(),
            {
                "robot_type": "raw-model",
                "features": features,
                "statistics": {"minimum": -7.0, "maximum": 9.0},
            },
            cls._identity_evidence(),
            (),
            actual_machines,
        )

    @classmethod
    def _component(
        cls,
        component_id: str = "left-arm",
        *,
        kind: str = "arm_joint",
        side: str | None = "left",
        count: object = 2,
        element_order: object = ("shoulder", "elbow"),
        representation: str = "joint_vector",
        unit: str = "rad",
        open_range: object = None,
        open_direction: object = None,
        confidence: object = 0.85,
        ambiguous: object = False,
        reason: object = "official component order",
        source_ids: object = ("official-component",),
    ) -> MachineComponent:
        return MachineComponent(
            component_id,
            kind,
            side,
            cast(int, count),
            cast(tuple[str, ...], element_order),
            representation,
            unit,
            cast(tuple[float, float] | None, open_range),
            cast(str | None, open_direction),
            cast(float, confidence),
            cast(bool, ambiguous),
            cast(str, reason),
            cast(tuple[str, ...], source_ids),
        )

    @classmethod
    def _gripper_component(
        cls,
        *,
        component_id: str = "left-gripper",
        open_range: object = (0.0, 100.0),
        open_direction: object = "increasing",
        confidence: object = 0.85,
        ambiguous: object = False,
        source_ids: object = ("official-gripper",),
    ) -> MachineComponent:
        return cls._component(
            component_id,
            kind="gripper_open",
            count=1,
            element_order=("opening",),
            representation="scalar",
            unit="unitless",
            open_range=open_range,
            open_direction=open_direction,
            confidence=confidence,
            ambiguous=ambiguous,
            source_ids=source_ids,
        )

    @classmethod
    def _profile(
        cls,
        components: tuple[MachineComponent, ...],
        *,
        component_sources: tuple[SourceReference, ...] | None = None,
    ) -> HardwareProfile:
        sources = component_sources or (
            cls._source("official-identity"),
            cls._source("official-component"),
            cls._source("official-gripper", kind="official_manual"),
            cls._source("unused-third-party", kind="third_party"),
        )
        return HardwareProfile(cls._identity_fact(), sources, (), components)

    @staticmethod
    def _assignment(
        source_feature: str = "action",
        *,
        slices: object = (MachineSlice(0, 2, "left-arm", ("shoulder", "elbow")),),
        confidence: object = 0.85,
        ambiguous: object = False,
        reason: object = "whole feature mapped",
    ) -> MachineAssignment:
        return MachineAssignment(
            source_feature,
            cast(tuple[MachineSlice, ...], slices),
            cast(float, confidence),
            cast(bool, ambiguous),
            cast(str, reason),
        )

    @staticmethod
    def _mapping(*assignments: MachineAssignment) -> DatasetMapping:
        return DatasetMapping((), assignments)

    @staticmethod
    def _names(result: NormalizationResult, source_feature: str = "action") -> list[object]:
        features = cast(dict[str, dict[str, object]], result.normalized_info["features"])
        return cast(list[object], features[source_feature]["names"])

    @staticmethod
    def _codes(result: NormalizationResult) -> set[str]:
        return {issue.code for issue in result.issues}

    def test_replaces_names_only_for_complete_sourced_order_at_threshold(self) -> None:
        evidence = self._evidence()
        source_snapshot = deepcopy(evidence.source_info)
        result = apply_standard(
            evidence,
            self._profile((self._component(),)),
            self._mapping(self._assignment()),
            confidence_threshold=0.85,
        )

        self.assertEqual(
            self._names(result),
            ["left_arm_joint_0_rad", "left_arm_joint_1_rad"],
        )
        self.assertEqual(evidence.source_info, source_snapshot)
        self.assertEqual(result.normalized_info["statistics"], source_snapshot["statistics"])
        output_feature = cast(dict[str, dict[str, object]], result.normalized_info["features"])["action"]
        self.assertEqual(output_feature["values"], [0.25, 0.75])
        self.assertEqual(output_feature["custom"], {"keep": True})
        self.assertEqual(len(result.machine_mappings), 1)
        record = result.machine_mappings[0]
        self.assertEqual(record.source_address, "features.action.names")
        self.assertEqual(record.source, ["raw_0", "raw_1"])
        self.assertEqual(record.output, ["left_arm_joint_0_rad", "left_arm_joint_1_rad"])
        self.assertEqual(record.candidate, ["left_arm_joint_0_rad", "left_arm_joint_1_rad"])
        self.assertTrue(record.changed)
        self.assertEqual(record.decision, "apply")
        self.assertTrue(record.reason)
        self.assertEqual(
            [citation["source_id"] for citation in record.citations],
            ["official-component"],
        )
        self.assertNotIn("unused-third-party", str(record.citations))

    def test_keeps_already_standard_names_verbatim_over_alternate_candidate(self) -> None:
        standard = ("right_arm_joint_0_rad", "right_arm_joint_1_rad")
        evidence = self._evidence((self._machine(names=standard),))

        result = apply_standard(
            evidence,
            self._profile((self._component(),)),
            self._mapping(self._assignment()),
            confidence_threshold=0.85,
        )

        self.assertEqual(self._names(result), list(standard))
        self.assertEqual(result.machine_mappings[0].output, list(standard))
        self.assertEqual(result.machine_mappings[0].decision, "keep")
        self.assertNotIn("MACHINE_MAPPING_INVALID", self._codes(result))

    def test_keeps_standard_non_gripper_names_without_valid_vlm_mapping(self) -> None:
        standard = ("neck_joint_0_rad", "neck_joint_1_rad")
        evidence = self._evidence((self._machine(names=standard),))
        invalid = self._assignment(ambiguous=True)

        result = apply_standard(
            evidence,
            self._profile((self._component(),)),
            self._mapping(invalid),
            confidence_threshold=0.85,
        )

        self.assertEqual(self._names(result), list(standard))
        self.assertEqual(result.machine_mappings[0].decision, "keep")
        self.assertNotIn("MACHINE_MAPPING_INVALID", self._codes(result))

    def test_missing_or_empty_hardware_still_records_every_machine_in_order(self) -> None:
        action = self._machine()
        state = self._machine(
            "observation.state",
            names=("neck_joint_0_rad", "neck_joint_1_rad"),
        )
        evidence = self._evidence((action, state))
        cases = (
            (None, None),
            (self._profile(()), self._mapping()),
        )
        for profile, mapping in cases:
            with self.subTest(profile=profile, mapping=mapping):
                result = apply_standard(
                    evidence,
                    profile,
                    mapping,
                    confidence_threshold=0.85,
                )
                self.assertEqual(
                    [record.source_address for record in result.machine_mappings],
                    ["features.action.names", "features.observation.state.names"],
                )
                self.assertEqual(
                    [record.decision for record in result.machine_mappings],
                    ["review", "keep"],
                )
                machine_issues = [
                    issue
                    for issue in result.issues
                    if issue.scope.endswith(".names")
                ]
                self.assertEqual(
                    [(issue.code, issue.scope) for issue in machine_issues],
                    [("MACHINE_MAPPING_INVALID", "features.action.names")],
                )

    def test_standard_gripper_without_hardware_keeps_name_and_records_unconfirmed_range(self) -> None:
        evidence = self._evidence(
            (
                self._machine(
                    names=("left_gripper_open",),
                    shape=(1,),
                    episode_lengths=(1, 1),
                ),
            )
        )

        result = apply_standard(
            evidence,
            None,
            None,
            confidence_threshold=0.85,
        )

        self.assertEqual(self._names(result), ["left_gripper_open"])
        self.assertEqual(result.machine_mappings[0].decision, "review")
        self.assertIn("GRIPPER_RANGE_UNCONFIRMED", self._codes(result))

    def test_malformed_machine_schema_is_recorded_instead_of_silently_skipped(self) -> None:
        malformed = replace(
            self._machine(),
            schema=cast(FeatureSchema, None),
        )
        evidence = replace(self._evidence(), machines=(malformed,))

        result = apply_standard(
            evidence,
            self._profile((self._component(),)),
            self._mapping(self._assignment()),
            confidence_threshold=0.85,
        )

        self.assertEqual(len(result.machine_mappings), 1)
        self.assertEqual(result.machine_mappings[0].decision, "review")
        self.assertIn("MACHINE_MAPPING_INVALID", self._codes(result))

    def test_invalid_length_shape_and_source_schema_fail_atomically(self) -> None:
        base = self._machine()
        wrong_source = self._evidence((base,))
        cast(dict[str, dict[str, object]], wrong_source.source_info["features"])["action"]["shape"] = [3]
        cases = (
            self._evidence((replace(base, episode_lengths=()),)),
            self._evidence((replace(base, episode_lengths=(2, 3)),)),
            self._evidence((replace(base, episode_lengths=(2, True)),)),
            self._evidence((replace(base, schema=replace(base.schema, shape=())),)),
            self._evidence((replace(base, schema=replace(base.schema, shape=(True,))),)),
            self._evidence((replace(base, schema=replace(base.schema, shape=(3,))),)),
            wrong_source,
        )
        for evidence in cases:
            with self.subTest(evidence=evidence):
                result = apply_standard(
                    evidence,
                    self._profile((self._component(),)),
                    self._mapping(self._assignment()),
                    confidence_threshold=0.85,
                )
                self.assertEqual(self._names(result), ["raw_0", "raw_1"])
                self.assertIn("MACHINE_MAPPING_INVALID", self._codes(result))

    def test_gap_overlap_bounds_coverage_count_and_order_fail_atomically(self) -> None:
        first = self._component(
            "joint-a",
            count=1,
            element_order=("a",),
            source_ids=("official-component",),
        )
        second = self._component(
            "joint-b",
            count=1,
            element_order=("b",),
            source_ids=("official-component",),
        )
        valid_slices = (
            MachineSlice(0, 1, "joint-a", ("a",)),
            MachineSlice(1, 2, "joint-b", ("b",)),
        )
        invalid_slices = (
            (MachineSlice(1, 2, "joint-a", ("a",)),),
            (MachineSlice(0, 1, "joint-a", ("a",)), MachineSlice(2, 3, "joint-b", ("b",))),
            (MachineSlice(0, 2, "joint-a", ("a",)), MachineSlice(1, 2, "joint-b", ("b",))),
            (MachineSlice(0, 1, "joint-a", ("a",)),),
            (MachineSlice(0, 3, "joint-a", ("a",)),),
            (MachineSlice(0, 1, "joint-a", ("wrong",)), MachineSlice(1, 2, "joint-b", ("b",))),
            (MachineSlice(0, 1, "joint-a", cast(tuple[str, ...], ["a"])), MachineSlice(1, 2, "joint-b", ("b",))),
            (MachineSlice(cast(int, True), 1, "joint-a", ("a",)), MachineSlice(1, 2, "joint-b", ("b",))),
            (MachineSlice(0, cast(int, True), "joint-a", ("a",)), MachineSlice(1, 2, "joint-b", ("b",))),
        )
        for slices in invalid_slices:
            with self.subTest(slices=slices):
                result = apply_standard(
                    self._evidence(),
                    self._profile((first, second)),
                    self._mapping(self._assignment(slices=slices)),
                    confidence_threshold=0.85,
                )
                self.assertEqual(self._names(result), ["raw_0", "raw_1"])
                self.assertIn("MACHINE_MAPPING_INVALID", self._codes(result))

        valid = apply_standard(
            self._evidence(),
            self._profile((first, second)),
            self._mapping(self._assignment(slices=valid_slices)),
            confidence_threshold=0.85,
        )
        self.assertEqual(self._names(valid), ["raw_0", "raw_1"])
        self.assertIn("MACHINE_MAPPING_INVALID", self._codes(valid))

    def test_duplicate_missing_unknown_and_unsourced_inputs_fail_closed(self) -> None:
        action = self._machine()
        action_assignment = self._assignment()
        duplicate_evidence = self._evidence((action, replace(action, schema=replace(action.schema))))
        missing_feature = self._evidence((action,))
        cast(dict[str, object], missing_feature.source_info["features"]).pop("action")
        third_party_profile = self._profile(
            (replace(self._component(), source_ids=("community",)),),
            component_sources=(
                self._source("official-identity"),
                self._source("community", kind="third_party"),
            ),
        )
        cases = (
            (duplicate_evidence, self._profile((self._component(),)), self._mapping(action_assignment, action_assignment)),
            (self._evidence((action,)), self._profile((self._component(),)), self._mapping(action_assignment, action_assignment)),
            (self._evidence((action,)), self._profile((self._component(), self._component())), self._mapping(action_assignment)),
            (self._evidence((action,)), self._profile((self._component(),)), self._mapping(replace(action_assignment, source_feature="unknown"))),
            (missing_feature, self._profile((self._component(),)), self._mapping(action_assignment)),
            (self._evidence((action,)), third_party_profile, self._mapping(action_assignment)),
            (
                self._evidence((action,)),
                self._profile((self._component(),)),
                self._mapping(
                    replace(
                        action_assignment,
                        slices=(
                            MachineSlice(
                                0,
                                2,
                                "missing",
                                ("shoulder", "elbow"),
                            ),
                        ),
                    )
                ),
            ),
        )
        for evidence, profile, mapping in cases:
            with self.subTest(evidence=evidence, profile=profile, mapping=mapping):
                result = apply_standard(evidence, profile, mapping, confidence_threshold=0.85)
                self.assertTrue(result.machine_mappings)
                self.assertTrue(all(record.decision == "review" for record in result.machine_mappings))
                self.assertIn("MACHINE_MAPPING_INVALID", self._codes(result))

    def test_assignment_and_component_trust_fields_are_strict(self) -> None:
        bad_assignments = (
            self._assignment(confidence=0.849),
            self._assignment(confidence=True),
            self._assignment(confidence=float("nan")),
            self._assignment(confidence=float("inf")),
            self._assignment(ambiguous=True),
            self._assignment(ambiguous=1),
            self._assignment(reason=""),
            self._assignment(reason=" padded "),
        )
        for assignment in bad_assignments:
            with self.subTest(assignment=assignment):
                result = apply_standard(
                    self._evidence(),
                    self._profile((self._component(),)),
                    self._mapping(assignment),
                    confidence_threshold=0.85,
                )
                self.assertIn("MACHINE_MAPPING_INVALID", self._codes(result))

        bad_components = (
            self._component(confidence=0.849),
            self._component(confidence=True),
            self._component(confidence=float("nan")),
            self._component(confidence=float("inf")),
            self._component(ambiguous=True),
            self._component(ambiguous=1),
            self._component(reason=""),
            self._component(source_ids=[]),
        )
        for component in bad_components:
            with self.subTest(component=component):
                result = apply_standard(
                    self._evidence(),
                    self._profile((component,)),
                    self._mapping(self._assignment()),
                    confidence_threshold=0.85,
                )
                self.assertIn("MACHINE_MAPPING_INVALID", self._codes(result))

    def test_one_invalid_component_keeps_entire_feature_and_candidate_is_never_output(self) -> None:
        arm = self._component(
            count=1,
            element_order=("shoulder",),
        )
        gripper = self._gripper_component(open_direction="decreasing")
        mapping = self._mapping(
            self._assignment(
                slices=(
                    MachineSlice(0, 1, "left-arm", ("shoulder",)),
                    MachineSlice(1, 2, "left-gripper", ("opening",)),
                )
            )
        )
        evidence = self._evidence(
            (
                self._machine(
                    gripper_ranges=(GripperRange(1, 0.0, 100.0, 8, 0),),
                ),
            )
        )

        result = apply_standard(
            evidence,
            self._profile((arm, gripper)),
            mapping,
            confidence_threshold=0.85,
        )

        record = result.machine_mappings[0]
        self.assertEqual(self._names(result), ["raw_0", "raw_1"])
        self.assertEqual(
            record.candidate,
            ["left_arm_joint_0_rad", "left_gripper_open"],
        )
        self.assertEqual(record.output, ["raw_0", "raw_1"])
        self.assertEqual(record.decision, "review")
        self.assertIn("GRIPPER_TRANSFORM_REQUIRED", self._codes(result))

    def test_independent_feature_success_and_component_reuse_are_allowed(self) -> None:
        action = self._machine()
        state = self._machine("observation.state")
        mappings = (
            self._assignment(ambiguous=True),
            self._assignment("observation.state"),
        )
        result = apply_standard(
            self._evidence((action, state)),
            self._profile((self._component(),)),
            self._mapping(*mappings),
            confidence_threshold=0.85,
        )

        self.assertEqual(self._names(result, "action"), ["raw_0", "raw_1"])
        self.assertEqual(
            self._names(result, "observation.state"),
            ["left_arm_joint_0_rad", "left_arm_joint_1_rad"],
        )
        self.assertEqual(
            [record.source_address for record in result.machine_mappings],
            ["features.action.names", "features.observation.state.names"],
        )
        self.assertEqual(
            [record.decision for record in result.machine_mappings],
            ["review", "apply"],
        )

    def test_missing_or_duplicate_assignment_blocks_only_its_source_feature(self) -> None:
        action = self._machine()
        state = self._machine("observation.state")
        state_assignment = self._assignment("observation.state")
        cases = (
            self._mapping(state_assignment),
            self._mapping(self._assignment(), self._assignment(), state_assignment),
        )
        for mapping in cases:
            with self.subTest(mapping=mapping):
                result = apply_standard(
                    self._evidence((action, state)),
                    self._profile((self._component(),)),
                    mapping,
                    confidence_threshold=0.85,
                )
                self.assertEqual(self._names(result, "action"), ["raw_0", "raw_1"])
                self.assertEqual(
                    self._names(result, "observation.state"),
                    ["left_arm_joint_0_rad", "left_arm_joint_1_rad"],
                )
                self.assertEqual(
                    [record.decision for record in result.machine_mappings],
                    ["review", "apply"],
                )

    def test_duplicate_component_id_blocks_only_features_that_reference_it(self) -> None:
        action = self._machine()
        state = self._machine("observation.state")
        duplicate = self._component()
        state_component = self._component(
            "right-arm",
            side="right",
            source_ids=("official-component",),
        )
        result = apply_standard(
            self._evidence((action, state)),
            self._profile((duplicate, duplicate, state_component)),
            self._mapping(
                self._assignment(),
                self._assignment(
                    "observation.state",
                    slices=(
                        MachineSlice(
                            0,
                            2,
                            "right-arm",
                            ("shoulder", "elbow"),
                        ),
                    ),
                ),
            ),
            confidence_threshold=0.85,
        )

        self.assertEqual(self._names(result, "action"), ["raw_0", "raw_1"])
        self.assertEqual(
            self._names(result, "observation.state"),
            ["right_arm_joint_0_rad", "right_arm_joint_1_rad"],
        )
        self.assertEqual(
            [record.decision for record in result.machine_mappings],
            ["review", "apply"],
        )

    def test_schema_mismatch_record_uses_actual_info_names_for_source_and_output(self) -> None:
        evidence = self._evidence()
        source_info = deepcopy(evidence.source_info)
        cast(dict[str, dict[str, object]], source_info["features"])["action"]["shape"] = [3]
        evidence = replace(evidence, source_info=source_info)

        result = apply_standard(
            evidence,
            self._profile((self._component(),)),
            self._mapping(self._assignment()),
            confidence_threshold=0.85,
        )

        record = result.machine_mappings[0]
        self.assertEqual(record.source, ["raw_0", "raw_1"])
        self.assertEqual(record.output, ["raw_0", "raw_1"])
        self.assertEqual(self._names(result), ["raw_0", "raw_1"])

    def test_gripper_ranges_are_checked_only_for_actual_or_standard_grippers(self) -> None:
        malformed_ranges = cast(
            tuple[GripperRange, ...],
            [GripperRange(0, 0.0, 100.0, 8, 0)],
        )
        ordinary = self._evidence(
            (
                self._machine(
                    names=("neck_joint_0_rad", "neck_joint_1_rad"),
                    gripper_ranges=malformed_ranges,
                ),
            )
        )
        ordinary_result = apply_standard(
            ordinary,
            self._profile(()),
            self._mapping(),
            confidence_threshold=0.85,
        )
        self.assertEqual(
            self._names(ordinary_result),
            ["neck_joint_0_rad", "neck_joint_1_rad"],
        )
        self.assertEqual(ordinary_result.machine_mappings[0].decision, "keep")
        self.assertNotIn("MACHINE_MAPPING_INVALID", self._codes(ordinary_result))

        gripper = self._evidence(
            (
                self._machine(
                    names=("left_gripper_open",),
                    shape=(1,),
                    episode_lengths=(1, 1),
                    gripper_ranges=malformed_ranges,
                ),
            )
        )
        gripper_result = apply_standard(
            gripper,
            self._profile((self._gripper_component(),)),
            self._mapping(
                self._assignment(
                    slices=(
                        MachineSlice(0, 1, "left-gripper", ("opening",)),
                    )
                )
            ),
            confidence_threshold=0.85,
        )
        self.assertEqual(self._names(gripper_result), ["left_gripper_open"])
        self.assertIn("GRIPPER_RANGE_UNCONFIRMED", self._codes(gripper_result))

    def test_standard_gripper_ordinary_mapping_failures_are_machine_invalid(self) -> None:
        evidence = self._evidence(
            (
                self._machine(
                    names=("left_gripper_open",),
                    shape=(1,),
                    episode_lengths=(1, 1),
                    gripper_ranges=(GripperRange(0, 0.0, 100.0, 8, 0),),
                ),
            )
        )
        valid_slice = (MachineSlice(0, 1, "left-gripper", ("opening",)),)
        cases = (
            (
                self._profile((self._gripper_component(),)),
                self._mapping(self._assignment(slices=valid_slice, ambiguous=True)),
            ),
            (
                self._profile((self._gripper_component(),)),
                self._mapping(
                    self._assignment(
                        slices=(MachineSlice(1, 2, "left-gripper", ("opening",)),)
                    )
                ),
            ),
            (
                self._profile((self._gripper_component(),)),
                self._mapping(
                    self._assignment(
                        slices=(MachineSlice(0, 1, "missing", ("opening",)),)
                    )
                ),
            ),
            (
                self._profile(
                    (self._gripper_component(), self._gripper_component())
                ),
                self._mapping(self._assignment(slices=valid_slice)),
            ),
            (
                self._profile((self._gripper_component(confidence=0.84),)),
                self._mapping(self._assignment(slices=valid_slice)),
            ),
        )
        for profile, mapping in cases:
            with self.subTest(profile=profile, mapping=mapping):
                result = apply_standard(
                    evidence,
                    profile,
                    mapping,
                    confidence_threshold=0.85,
                )
                machine_codes = {
                    issue.code
                    for issue in result.issues
                    if issue.scope == "features.action.names"
                }
                self.assertEqual(machine_codes, {"MACHINE_MAPPING_INVALID"})
                self.assertEqual(self._names(result), ["left_gripper_open"])

    def test_partial_render_failure_records_no_candidate(self) -> None:
        first = self._component(
            count=1,
            element_order=("shoulder",),
        )
        result = apply_standard(
            self._evidence(),
            self._profile((first,)),
            self._mapping(
                self._assignment(
                    slices=(
                        MachineSlice(0, 1, "left-arm", ("shoulder",)),
                        MachineSlice(1, 2, "missing", ("elbow",)),
                    )
                )
            ),
            confidence_threshold=0.85,
        )

        self.assertIsNone(result.machine_mappings[0].candidate)
        self.assertEqual(result.machine_mappings[0].output, ["raw_0", "raw_1"])

    def test_standard_names_require_trustworthy_episode_lengths(self) -> None:
        standard = ("neck_joint_0_rad", "neck_joint_1_rad")
        machines = (
            self._machine(names=standard, episode_lengths=()),
            self._machine(names=standard, episode_lengths=(2, True)),
            self._machine(names=standard, episode_lengths=(2, 3)),
            self._machine(names=standard, episode_lengths=(3, 3)),
        )
        for machine in machines:
            with self.subTest(lengths=machine.episode_lengths):
                result = apply_standard(
                    self._evidence((machine,)),
                    self._profile(()),
                    self._mapping(),
                    confidence_threshold=0.85,
                )
                self.assertEqual(self._names(result), list(standard))
                self.assertEqual(result.machine_mappings[0].decision, "review")
                self.assertIn("MACHINE_MAPPING_INVALID", self._codes(result))

    def test_accepts_all_four_gripper_ranges_and_observed_subranges(self) -> None:
        for maximum in (1.0, 10.0, 100.0, 1000.0):
            for observed in ((0.0, maximum), (maximum * 0.25, maximum * 0.75)):
                with self.subTest(maximum=maximum, observed=observed):
                    evidence = self._evidence(
                        (
                            self._machine(
                                names=("raw_gripper",),
                                shape=(1,),
                                episode_lengths=(1, 1),
                                gripper_ranges=(
                                    GripperRange(0, observed[0], observed[1], 8, 0),
                                ),
                            ),
                        )
                    )
                    result = apply_standard(
                        evidence,
                        self._profile((self._gripper_component(open_range=(0.0, maximum)),)),
                        self._mapping(
                            self._assignment(
                                slices=(MachineSlice(0, 1, "left-gripper", ("opening",)),)
                            )
                        ),
                        confidence_threshold=0.85,
                    )
                    self.assertEqual(self._names(result), ["left_gripper_open"])
                    self.assertNotIn("GRIPPER_TRANSFORM_REQUIRED", self._codes(result))
                    self.assertNotIn("GRIPPER_RANGE_UNCONFIRMED", self._codes(result))

    def test_gripper_transform_cases_keep_whole_source_feature(self) -> None:
        cases = (
            ((0.0, 0.1), "increasing", (0.0, 0.1), 0),
            ((0.0, 10000.0), "increasing", (0.0, 10000.0), 0),
            ((0.0, 100.0), "decreasing", (0.0, 100.0), 0),
            ((0.0, 100.0), "unknown", (0.0, 100.0), 0),
            ((0.0, 100.0), "increasing", (-1.0, 100.0), 0),
            ((0.0, 100.0), "increasing", (0.0, 101.0), 0),
            ((0.0, 100.0), "increasing", (0.0, 100.0), 1),
        )
        for nominal, direction, observed, nonfinite in cases:
            with self.subTest(case=(nominal, direction, observed, nonfinite)):
                evidence = self._evidence(
                    (
                        self._machine(
                            names=("raw_gripper",),
                            shape=(1,),
                            episode_lengths=(1, 1),
                            gripper_ranges=(GripperRange(0, *observed, 8, nonfinite),),
                        ),
                    )
                )
                result = apply_standard(
                    evidence,
                    self._profile((self._gripper_component(open_range=nominal, open_direction=direction),)),
                    self._mapping(
                        self._assignment(
                            slices=(MachineSlice(0, 1, "left-gripper", ("opening",)),)
                        )
                    ),
                    confidence_threshold=0.85,
                )
                self.assertEqual(self._names(result), ["raw_gripper"])
                self.assertIn("GRIPPER_TRANSFORM_REQUIRED", self._codes(result))

    def test_missing_duplicate_or_untrustworthy_gripper_range_is_unconfirmed(self) -> None:
        malformed_ranges = (
            (),
            (GripperRange(0, 0.0, 100.0, 8, 0), GripperRange(0, 0.0, 100.0, 8, 0)),
            (GripperRange(cast(int, True), 0.0, 100.0, 8, 0),),
            (GripperRange(0, None, 100.0, 8, 0),),
            (GripperRange(0, 100.0, 0.0, 8, 0),),
            (GripperRange(0, 0.0, 100.0, cast(int, True), 0),),
            (GripperRange(0, 0.0, 100.0, 0, 0),),
            (GripperRange(0, 0.0, 100.0, 8, cast(int, True)),),
        )
        for ranges in malformed_ranges:
            with self.subTest(ranges=ranges):
                evidence = self._evidence(
                    (
                        self._machine(
                            names=("left_gripper_open",),
                            shape=(1,),
                            episode_lengths=(1, 1),
                            gripper_ranges=ranges,
                        ),
                    )
                )
                result = apply_standard(
                    evidence,
                    self._profile((self._gripper_component(),)),
                    self._mapping(
                        self._assignment(
                            slices=(MachineSlice(0, 1, "left-gripper", ("opening",)),)
                        )
                    ),
                    confidence_threshold=0.85,
                )
                self.assertEqual(self._names(result), ["left_gripper_open"])
                self.assertIn("GRIPPER_RANGE_UNCONFIRMED", self._codes(result))

    def test_issue_evidence_and_records_are_json_safe_and_minimal(self) -> None:
        result = apply_standard(
            self._evidence(),
            self._profile((self._component(),)),
            self._mapping(self._assignment(ambiguous=True)),
            confidence_threshold=0.85,
        )

        encoded = json.dumps(
            {
                "issues": [issue.evidence for issue in result.issues],
                "records": [
                    {
                        "source": record.source,
                        "output": record.output,
                        "candidate": record.candidate,
                        "semantics": record.vlm_semantics,
                        "citations": record.citations,
                    }
                    for record in result.machine_mappings
                ],
            },
            allow_nan=False,
        )
        self.assertNotIn("Machine", encoded)
        machine_issue = next(issue for issue in result.issues if issue.code == "MACHINE_MAPPING_INVALID")
        self.assertLessEqual(set(machine_issue.evidence), {"candidate_names"})

    def test_irrelevant_gripper_ranges_are_ignored_and_memory_error_propagates(self) -> None:
        malformed = replace(
            self._machine(),
            gripper_ranges=cast(tuple[GripperRange, ...], [GripperRange(0, 0.0, 1.0, 1, 0)]),
        )
        result = apply_standard(
            self._evidence((malformed,)),
            self._profile((self._component(),)),
            self._mapping(self._assignment()),
            confidence_threshold=0.85,
        )
        self.assertEqual(
            self._names(result),
            ["left_arm_joint_0_rad", "left_arm_joint_1_rad"],
        )
        self.assertNotIn("MACHINE_MAPPING_INVALID", self._codes(result))

        exploding = _ExplodingEquality()
        evidence = self._evidence((self._machine(names=(exploding, "raw_1")),))
        with self.assertRaises(MemoryError):
            apply_standard(
                evidence,
                self._profile((self._component(),)),
                self._mapping(self._assignment()),
                confidence_threshold=0.85,
            )


if __name__ == "__main__":
    unittest.main()
