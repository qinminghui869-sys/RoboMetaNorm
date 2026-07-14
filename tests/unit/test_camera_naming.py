"""相机确定性命名和冲突检测测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.naming import find_colliding_sources, propose_camera_name


class CameraNamingTest(unittest.TestCase):
    """验证 P1 的标准词表、顺序和冲突保护。"""

    def test_uses_exact_mapping_for_known_rgb_and_depth_fields(self) -> None:
        rgb = propose_camera_name("observation.images.image_left")
        depth = propose_camera_name("observation.images.image_top_depth")

        self.assertEqual(rgb.target_key, "observation.images.cam_left_rgb")
        self.assertEqual(rgb.method, "exact")
        self.assertEqual(depth.target_key, "observation.images.cam_top_depth")
        self.assertEqual(depth.modality, "depth")

    def test_builds_tokens_in_standard_order(self) -> None:
        proposal = propose_camera_name("observation.images.image_left_front_head")

        self.assertEqual(
            proposal.target_key,
            "observation.images.cam_front_left_head_rgb",
        )
        self.assertEqual(proposal.method, "regex")

    def test_marks_all_sources_that_resolve_to_same_target_as_collision(self) -> None:
        left_image = propose_camera_name("observation.images.image_left")
        left_camera = propose_camera_name("observation.images.camera_left")

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
