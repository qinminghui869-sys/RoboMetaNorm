"""转换前置条件检查测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.application.preconditions import check_preconditions
from robometanorm.domain.models import DatasetCandidate, DatasetStatus, LayoutType


class PreconditionsTest(unittest.TestCase):
    """验证 P0 的基础数据可用性检查。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset_001"
        (self.dataset_path / "meta").mkdir(parents=True)
        (self.dataset_path / "data").mkdir()
        (self.dataset_path / "videos" / "front").mkdir(parents=True)
        (self.dataset_path / "videos" / "front" / "episode_000000.mp4").touch()
        (self.dataset_path / "robot.urdf").touch()
        (self.dataset_path / "collector.py").touch()
        (self.dataset_path / "convert_to_lerobot.py").touch()
        self.candidate = DatasetCandidate(
            dataset_name="dataset_001",
            task_name=None,
            source_path=self.dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=self.dataset_path / "meta" / "info.json",
            data_path=self.dataset_path / "data",
            video_path=self.dataset_path / "videos",
            depth_path=None,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_passes_when_required_evidence_is_present(self) -> None:
        report = check_preconditions(self.candidate, self._complete_info())

        self.assertEqual(report.status, DatasetStatus.PASS)
        self.assertEqual(report.camera_count, 1)
        self.assertEqual(report.review_items, ())

    def test_blocks_when_action_feature_is_missing(self) -> None:
        info = self._complete_info()
        del info["features"]["action"]

        report = check_preconditions(self.candidate, info)

        self.assertEqual(report.status, DatasetStatus.BLOCKED)
        self.assertEqual(
            [item.category for item in report.review_items], ["missing_action"]
        )
        self.assertEqual(report.review_items[0].severity, "block")

    @staticmethod
    def _complete_info() -> dict[str, object]:
        return {
            "features": {
                "action": {"dtype": "float32", "shape": [2]},
                "observation.state": {"dtype": "float32", "shape": [2]},
                "observation.images.cam_front_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
            }
        }


if __name__ == "__main__":
    unittest.main()
