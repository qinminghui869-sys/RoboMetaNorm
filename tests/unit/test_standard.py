"""Strict camera-key standard contract tests."""

from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.models import CameraSlot, MachineComponent
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
    are_standard_machine_names,
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


if __name__ == "__main__":
    unittest.main()
