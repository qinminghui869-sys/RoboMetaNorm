"""Episode 代表样本选择测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.episode_sampling import select_representative_episodes


class EpisodeSamplingTest(unittest.TestCase):
    """验证 episode 路径选择规则稳定且最多返回两个文件。"""

    def test_returns_empty_and_single_inputs_unchanged(self) -> None:
        episode = Path("episode_000004.parquet")

        self.assertEqual(select_representative_episodes(()), ())
        self.assertEqual(select_representative_episodes((episode,)), (episode,))

    def test_selects_first_and_last_after_sorting(self) -> None:
        paths = tuple(
            Path(f"episode_{index:06d}.parquet") for index in (2, 0, 3, 1)
        )

        self.assertEqual(
            select_representative_episodes(paths),
            (Path("episode_000000.parquet"), Path("episode_000003.parquet")),
        )


if __name__ == "__main__":
    unittest.main()
