"""相机确定性命名和冲突检测测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.naming import (
    build_camera_key,
    find_colliding_sources,
    parse_standard_camera_key,
    propose_camera_name,
    propose_robot_camera_name,
)
from robometanorm.camera.models import CameraNameProposal


class CameraNamingTest(unittest.TestCase):
    """验证 P1 的标准词表、顺序和冲突保护。"""

    def test_builds_on_robot_camera_names_in_standard_order(self) -> None:
        self.assertEqual(
            build_camera_key("on_robot", ("left", "front"), "head", "rgb"),
            "observation.images.cam_front_left_head_rgb",
        )
        self.assertEqual(
            build_camera_key("on_robot", (), "head", "rgb"),
            "observation.images.cam_head_rgb",
        )
        self.assertEqual(
            build_camera_key("on_robot", ("ego",), None, "rgb"),
            "observation.images.cam_ego_rgb",
        )

    def test_builds_external_camera_names_from_the_external_vocabulary(self) -> None:
        self.assertEqual(
            build_camera_key("external", ("global",), None, "rgb"),
            "observation.images.cam_global_rgb",
        )
        self.assertEqual(
            build_camera_key("external", ("env",), None, "depth"),
            "observation.images.cam_env_depth",
        )

    def test_rejects_compound_external_camera_directions(self) -> None:
        self.assertIsNone(
            build_camera_key("external", ("top", "side"), None, "rgb")
        )
        self.assertIsNone(
            parse_standard_camera_key(
                "observation.images.cam_top_side_rgb"
            )
        )

    def test_rejects_tokens_that_do_not_match_the_mount_type(self) -> None:
        self.assertIsNone(build_camera_key("external", ("left",), "wrist", "rgb"))
        self.assertIsNone(build_camera_key("on_robot", ("env",), "head", "rgb"))
        self.assertIsNone(build_camera_key("on_robot", (), None, "rgb"))

    def test_parses_only_already_standard_camera_keys(self) -> None:
        wrist_mount, wrist_modality = parse_standard_camera_key(
            "observation.images.cam_left_wrist_rgb"
        )
        external_mount, external_modality = parse_standard_camera_key(
            "observation.images.cam_left_depth"
        )

        self.assertEqual(wrist_mount.mount_type, "on_robot")
        self.assertEqual(wrist_mount.direction_tokens, ("left",))
        self.assertEqual(wrist_mount.body_part, "wrist")
        self.assertEqual(wrist_modality, "rgb")
        self.assertEqual(external_mount.mount_type, "external")
        self.assertEqual(external_modality, "depth")
        self.assertIsNone(parse_standard_camera_key("observation.images.cam_high_rgb"))

    def test_does_not_map_ambiguous_source_tokens_without_evidence(self) -> None:
        self.assertIsNone(propose_camera_name("observation.images.image_left"))
        self.assertIsNone(propose_camera_name("observation.images.image_top_depth"))

    def test_uses_verified_robot_camera_mapping_before_global_rules(self) -> None:
        proposal = propose_robot_camera_name(
            "airbot_mmk2", "observation.images.cam_left_wrist_rgb"
        )

        self.assertEqual(
            proposal.target_key, "observation.images.cam_left_wrist_rgb"
        )
        self.assertEqual(proposal.method, "robot")

    def test_does_not_invent_unverified_airbot_camera_mapping(self) -> None:
        proposal = propose_robot_camera_name(
            "airbot_mmk2", "observation.images.cam_third_view"
        )

        self.assertIsNone(proposal)

    def test_marks_all_sources_that_resolve_to_same_target_as_collision(self) -> None:
        left_image = CameraNameProposal(
            "observation.images.image_left",
            "observation.images.cam_left_rgb",
            "rgb",
            "vlm",
        )
        left_camera = CameraNameProposal(
            "observation.images.camera_left",
            "observation.images.cam_left_rgb",
            "rgb",
            "vlm",
        )

        collisions = find_colliding_sources([left_image, left_camera])

        self.assertEqual(
            collisions,
            {
                "observation.images.image_left",
                "observation.images.camera_left",
            },
        )


if __name__ == "__main__":
    unittest.main()
