"""Parquet 画像持久化缓存测试。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.machine.models import ParquetProfile, ProfileProgress
from robometanorm.machine.parquet_profiler import profile_parquets
from robometanorm.machine.profile_cache import load_or_profile_parquets


class ProfileCacheTest(unittest.TestCase):
    """验证画像缓存命中、失效与损坏回退。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data" / "chunk-000"
        self.data_dir.mkdir(parents=True)
        self.cache_dir = self.root / "meta" / ".robometanorm_cache"
        self.paths = tuple(
            self._write_parquet(index) for index in range(2)
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_second_call_loads_cache_without_profiling_again(self) -> None:
        events: list[ProfileProgress] = []
        first = self._load(events)

        with patch(
            "robometanorm.machine.profile_cache.profile_parquets"
        ) as profiler:
            second = self._load(events)

        profiler.assert_not_called()
        self.assertEqual(second.episode_count, first.episode_count)
        self.assertEqual(second.schema_columns, first.schema_columns)
        np.testing.assert_array_equal(
            second.samples["action"], first.samples["action"]
        )
        self.assertTrue(any(event.kind == "cache_hit" for event in events))
        self.assertTrue((self.cache_dir / "parquet_profile_v2.json").is_file())
        self.assertTrue((self.cache_dir / "parquet_samples_v2.npz").is_file())

    def test_mtime_change_invalidates_cache(self) -> None:
        self._load()
        changed = self.paths[1]
        stat = changed.stat()
        os.utime(
            changed,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
        )

        with patch(
            "robometanorm.machine.profile_cache.profile_parquets",
            wraps=profile_parquets,
        ) as profiler:
            self._load()

        profiler.assert_called_once()

    def test_corrupt_metadata_is_recomputed(self) -> None:
        self._load()
        metadata_path = self.cache_dir / "parquet_profile_v2.json"
        metadata_path.write_text("not json", encoding="utf-8")

        with patch(
            "robometanorm.machine.profile_cache.profile_parquets",
            wraps=profile_parquets,
        ) as profiler:
            result = self._load()

        profiler.assert_called_once()
        self.assertEqual(result.episode_count, 2)
        self.assertEqual(json.loads(metadata_path.read_text())["cache_version"], 2)

    def test_cache_write_failure_keeps_profile_and_emits_warning(self) -> None:
        events: list[ProfileProgress] = []

        with patch(
            "robometanorm.machine.profile_cache._write",
            side_effect=RuntimeError("cache serializer failed"),
        ):
            result = self._load(events)

        self.assertEqual(result.episode_count, 2)
        warning = next(
            event for event in events if event.kind == "cache_write_warning"
        )
        self.assertIn("cache serializer failed", warning.message or "")

    def test_mismatched_npz_fingerprint_is_recomputed(self) -> None:
        self._load()
        samples_path = self.cache_dir / "parquet_samples_v2.npz"
        with samples_path.open("wb") as file_handle:
            np.savez_compressed(
                file_handle,
                __fingerprint__=np.asarray("wrong"),
                sample_0000=np.zeros((1, 1)),
            )

        with patch(
            "robometanorm.machine.profile_cache.profile_parquets",
            wraps=profile_parquets,
        ) as profiler:
            result = self._load()

        profiler.assert_called_once()
        self.assertEqual(result.episode_count, 2)

    def test_truncated_npz_is_recomputed(self) -> None:
        self._load()
        samples_path = self.cache_dir / "parquet_samples_v2.npz"
        samples_path.write_bytes(b"")

        with patch(
            "robometanorm.machine.profile_cache.profile_parquets",
            wraps=profile_parquets,
        ) as profiler:
            result = self._load()

        profiler.assert_called_once()
        self.assertEqual(result.episode_count, 2)

    def _load(
        self, events: list[ProfileProgress] | None = None
    ) -> ParquetProfile:
        return load_or_profile_parquets(
            self.paths,
            self.cache_dir,
            sample_rows=16,
            progress=events.append if events is not None else None,
        )

    def _write_parquet(self, index: int) -> Path:
        path = self.data_dir / f"episode_{index:06d}.parquet"
        pq.write_table(
            pa.table(
                {
                    "action": [
                        [float(index), 1.0, 2.0],
                        [float(index + 1), 2.0, 3.0],
                    ],
                    "observation.state": [
                        [float(index), 1.0, 2.0],
                        [float(index + 1), 2.0, 3.0],
                    ],
                }
            ),
            path,
        )
        return path


if __name__ == "__main__":
    unittest.main()
