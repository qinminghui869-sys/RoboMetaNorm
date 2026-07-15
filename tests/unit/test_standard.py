"""Strict camera-key standard contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.models import CameraSlot
from robometanorm.standard import (
    BODY_PARTS,
    CAMERA_PREFIX,
    CONFLICT_GROUPS,
    DIRECTION_ORDER,
    EXTERNAL_DIRECTIONS,
    ON_ROBOT_DIRECTIONS,
    parse_standard_camera_key,
    render_camera_key,
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


if __name__ == "__main__":
    unittest.main()
