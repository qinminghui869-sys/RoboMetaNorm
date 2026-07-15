"""数据集发现行为测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.adapters.filesystem import discover_datasets
from robometanorm.models import LayoutType


class DatasetDiscoveryTest(unittest.TestCase):
    """验证两种输入目录和排除规则。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_dataset(self, relative_path: str) -> Path:
        dataset_path = self.root / relative_path
        (dataset_path / "meta").mkdir(parents=True)
        (dataset_path / "data").mkdir()
        (dataset_path / "meta" / "info.json").write_text(
            json.dumps({"features": {}}), encoding="utf-8"
        )
        return dataset_path

    def test_discovers_flat_and_task_grouped_datasets_and_ignores_excluded_paths(self) -> None:
        flat_path = self._create_dataset("flat_dataset")
        grouped_path = self._create_dataset("pick_task/pick_task_001")
        self._create_dataset(".git/ignored_dataset")

        candidates = discover_datasets(self.root)

        self.assertEqual([candidate.dataset_name for candidate in candidates], [
            "flat_dataset",
            "pick_task_001",
        ])
        self.assertEqual(candidates[0].source_path, flat_path)
        self.assertEqual(candidates[0].layout_type, LayoutType.FLAT)
        self.assertIsNone(candidates[0].task_name)
        self.assertEqual(candidates[1].source_path, grouped_path)
        self.assertEqual(candidates[1].layout_type, LayoutType.TASK_GROUPED)
        self.assertEqual(candidates[1].task_name, "pick_task")


if __name__ == "__main__":
    unittest.main()
