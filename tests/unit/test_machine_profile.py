"""P2 Parquet 数值画像与字段布局测试。"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.machine.layout_resolver import resolve_child_slices
from robometanorm.machine.parquet_profiler import profile_parquet, profile_parquets


class MachineParquetProfileTest(unittest.TestCase):
    """验证 P2 只读取受限样本即可恢复字段事实。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temp_dir.name) / "episode.parquet"
        table = pa.table(
            {
                "action": [[10.0, 11.0, 12.0, 20.0, 21.0], [13.0, 14.0, 15.0, 22.0, 23.0]],
                "observation.state": [
                    [10.0, 11.0, 12.0, 20.0, 21.0],
                    [13.0, 14.0, 15.0, 22.0, 23.0],
                ],
                "observation.state.head": [[10.0, 11.0], [13.0, 14.0]],
                "observation.state.finger": [[12.0, 20.0, 21.0], [15.0, 22.0, 23.0]],
                "diagnostic": [[1.0, math.nan, 3.0], [4.0, 5.0, 6.0]],
            }
        )
        pq.write_table(table, self.parquet_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_profiles_vector_columns_and_recovers_contiguous_child_slices(self) -> None:
        profile = profile_parquet(self.parquet_path, sample_rows=16)
        slices = resolve_child_slices(
            profile.samples["observation.state"],
            {
                "observation.state.head": profile.samples["observation.state.head"],
                "observation.state.finger": profile.samples["observation.state.finger"],
            },
        )

        action = profile.columns["action"]
        finger = profile.columns["diagnostic"]
        self.assertEqual(profile.row_count, 2)
        self.assertEqual(action.vector_length, 5)
        self.assertEqual(action.min_value, 10.0)
        self.assertEqual(action.max_value, 23.0)
        self.assertAlmostEqual(action.mean_abs_diff, 2.6)
        self.assertFalse(action.triplet_grouping_possible)
        self.assertEqual(finger.vector_length, 3)
        self.assertAlmostEqual(finger.nan_ratio, 1 / 6)
        self.assertEqual(slices["observation.state.head"], (0, 2))
        self.assertEqual(slices["observation.state.finger"], (2, 5))

    def test_marks_columns_with_inconsistent_episode_layouts(self) -> None:
        second_path = Path(self.temp_dir.name) / "episode_001.parquet"
        pq.write_table(
            pa.table(
                {
                    "action": [[1.0, 2.0, 3.0, 4.0]],
                    "observation.state": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                }
            ),
            second_path,
        )

        profile = profile_parquets((self.parquet_path, second_path), sample_rows=16)

        self.assertEqual(profile.episode_count, 2)
        self.assertIn("action", profile.inconsistent_columns)


if __name__ == "__main__":
    unittest.main()
