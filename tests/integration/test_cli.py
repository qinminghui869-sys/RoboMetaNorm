"""P0 命令行端到端测试。"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.cli.main import main
from robometanorm.camera.vlm_classifier import OpenAICompatibleVlmClassifier


class CliIntegrationTest(unittest.TestCase):
    """验证命令边界和基础输出。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "collect_data"
        self.dataset_path = self.root / "dataset_001"
        (self.dataset_path / "meta").mkdir(parents=True)
        (self.dataset_path / "data").mkdir()
        (self.dataset_path / "videos" / "front").mkdir(parents=True)
        (self.dataset_path / "videos" / "front" / "episode_000000.mp4").touch()
        (self.dataset_path / "robot.urdf").touch()
        (self.dataset_path / "collector.py").touch()
        (self.dataset_path / "convert_to_lerobot.py").touch()
        info = {
            "fps": 20,
            "features": {
                "action": {"dtype": "float32", "shape": [2]},
                "observation.state": {"dtype": "float32", "shape": [2]},
                "observation.images.image_left": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
            },
        }
        (self.dataset_path / "meta" / "info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )
        self.source_info = info

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_scan_prints_summary_without_creating_output_files(self) -> None:
        output = self._run("scan", "--root", str(self.root))

        self.assertIn("Dataset", output)
        self.assertIn("dataset_001", output)
        self.assertIn("PASS", output)
        self.assertFalse((self.dataset_path / "meta" / "info_norm.json").exists())
        self.assertFalse((self.dataset_path / "meta" / "info_norm_review.json").exists())

    def test_normalize_writes_two_p0_files_and_prints_summary(self) -> None:
        output = self._run("normalize", "--root", str(self.root))

        self.assertIn("dataset_001", output)
        self.assertTrue((self.dataset_path / "meta" / "info_norm.json").is_file())
        self.assertTrue((self.dataset_path / "meta" / "info_norm_review.json").is_file())
        normalized = json.loads(
            (self.dataset_path / "meta" / "info_norm.json").read_text(encoding="utf-8")
        )
        source = json.loads(
            (self.dataset_path / "meta" / "info.json").read_text(encoding="utf-8")
        )
        normalized_feature = normalized["features"]["observation.images.cam_left_rgb"]
        self.assertEqual(normalized_feature["codec"], "av1")
        self.assertNotIn("observation.images.image_left", normalized["features"])
        self.assertEqual(source, self.source_info)

    def test_normalize_creates_vlm_classifier_only_when_endpoint_is_configured(self) -> None:
        with patch.dict(os.environ, {"P1_TEST_VLM_KEY": "test-key"}):
            with patch("robometanorm.cli.main.normalize_datasets", return_value=[]) as normalize:
                self._run(
                    "normalize",
                    "--root",
                    str(self.root),
                    "--vlm-endpoint",
                    "http://127.0.0.1:8002/v1",
                    "--vlm-model",
                    "test-vlm",
                    "--vlm-api-key-env",
                    "P1_TEST_VLM_KEY",
                    "--confidence-threshold",
                    "0.9",
                    "--vlm-timeout-seconds",
                    "90",
                    "--vlm-max-retries",
                    "3",
                    "--vlm-retry-backoff-seconds",
                    "0.5",
                    "--vlm-max-tokens",
                    "2048",
                )

        classifier = normalize.call_args.kwargs["vlm_classifier"]
        self.assertIsInstance(classifier, OpenAICompatibleVlmClassifier)
        self.assertEqual(normalize.call_args.kwargs["confidence_threshold"], 0.9)
        self.assertEqual(classifier.timeout_seconds, 90)
        self.assertEqual(classifier.max_retries, 3)
        self.assertEqual(classifier.retry_backoff_seconds, 0.5)
        self.assertEqual(classifier.max_tokens, 2048)

    def test_normalize_applies_safe_p2_machine_names_from_parquet(self) -> None:
        info = {
            "fps": 20,
            "features": {
                "action": self._head_quaternion_feature(),
                "observation.state": self._head_quaternion_feature(),
                "observation.images.image_left": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
            },
        }
        (self.dataset_path / "meta" / "info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )
        pq.write_table(
            pa.table(
                {
                    "action": [[0.0, 0.0, 0.0, 1.0]],
                    "observation.state": [[0.0, 0.0, 0.0, 1.0]],
                }
            ),
            self.dataset_path / "data" / "episode_000000.parquet",
        )

        self._run("normalize", "--root", str(self.root))

        normalized = json.loads(
            (self.dataset_path / "meta" / "info_norm.json").read_text(encoding="utf-8")
        )
        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        expected_names = [
            "head_orient_quat_x",
            "head_orient_quat_y",
            "head_orient_quat_z",
            "head_orient_quat_w",
        ]
        self.assertEqual(normalized["features"]["action"]["names"], expected_names)
        self.assertEqual(
            normalized["features"]["observation.state"]["names"], expected_names
        )
        self.assertEqual(review["generator"]["phase"], "P2")
        self.assertIn("machine_review_items", review)
        self.assertEqual(
            json.loads((self.dataset_path / "meta" / "info.json").read_text(encoding="utf-8")),
            info,
        )

    @staticmethod
    def _head_quaternion_feature() -> dict[str, object]:
        return {
            "dtype": "float32",
            "shape": [4],
            "names": [
                "head_rotation_quat_x",
                "head_rotation_quat_y",
                "head_rotation_quat_z",
                "head_rotation_quat_w",
            ],
        }

    @staticmethod
    def _run(*arguments: str) -> str:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(list(arguments))
        if exit_code != 0:
            raise AssertionError(f"命令返回非零状态: {exit_code}")
        return output.getvalue()


if __name__ == "__main__":
    unittest.main()
