"""P1 相机规范建议测试。"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.normalizer import normalize_cameras
from robometanorm.domain.models import DatasetCandidate, LayoutType


class CameraNormalizerTest(unittest.TestCase):
    """验证确定性相机改名与保守复核策略。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset_001"
        (self.dataset_path / "meta").mkdir(parents=True)
        (self.dataset_path / "data").mkdir()
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.info_path.write_text(json.dumps({}), encoding="utf-8")
        self.candidate = DatasetCandidate(
            dataset_name="dataset_001",
            task_name=None,
            source_path=self.dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=self.info_path,
            data_path=self.dataset_path / "data",
            video_path=None,
            depth_path=None,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_replaces_deterministic_camera_keys_and_preserves_feature_values(self) -> None:
        source_info = self._info(
            {
                "observation.images.image_left": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "info": {"source": "left"},
                },
                "observation.images.image_top_depth": {
                    "dtype": "video",
                    "shape": [480, 640, 1],
                },
            }
        )
        original_info = copy.deepcopy(source_info)

        result = normalize_cameras(self.candidate, source_info)

        features = result.normalized_info["features"]
        self.assertNotIn("observation.images.image_left", features)
        self.assertEqual(
            features["observation.images.cam_left_rgb"]["shape"], [480, 640, 3]
        )
        self.assertEqual(features["observation.images.cam_left_rgb"]["codec"], "av1")
        self.assertEqual(
            features["observation.images.cam_top_depth"]["codec"], "ffv1"
        )
        self.assertEqual(result.normalized_info["fps"], 30)
        self.assertEqual(result.camera_review_items, ())
        self.assertEqual(source_info, original_info)

    def test_keeps_unknown_camera_and_creates_review_item(self) -> None:
        result = normalize_cameras(
            self.candidate,
            self._info({"observation.images.camera_1": {"dtype": "video"}}),
        )

        self.assertIn("observation.images.camera_1", result.normalized_info["features"])
        self.assertEqual(len(result.camera_review_items), 1)
        self.assertEqual(result.camera_review_items[0].reason_code, "UNKNOWN_CAMERA_NAME")
        self.assertEqual(result.camera_review_items[0].source_key, "observation.images.camera_1")

    def test_keeps_all_sources_when_target_names_collide(self) -> None:
        result = normalize_cameras(
            self.candidate,
            self._info(
                {
                    "observation.images.image_left": {"dtype": "video"},
                    "observation.images.camera_left": {"dtype": "video"},
                }
            ),
        )

        features = result.normalized_info["features"]
        self.assertIn("observation.images.image_left", features)
        self.assertIn("observation.images.camera_left", features)
        self.assertEqual(
            {item.reason_code for item in result.camera_review_items},
            {"TARGET_NAME_COLLISION"},
        )

    @staticmethod
    def _info(features: dict[str, object]) -> dict[str, object]:
        return {"fps": 30, "features": copy.deepcopy(features)}


if __name__ == "__main__":
    unittest.main()
