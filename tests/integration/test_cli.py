"""P0 命令行端到端测试。"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
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
from robometanorm.camera.vlm import OpenAICompatibleVlmClassifier


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
        normalized_feature = normalized["features"]["observation.images.image_left"]
        self.assertEqual(normalized_feature["codec"], "av1")
        self.assertNotIn("observation.images.cam_left_rgb", normalized["features"])
        self.assertEqual(source, self.source_info)

    def test_airbot_keeps_unverified_camera_names_and_records_identity(self) -> None:
        info = {
            "robot_type": "Airbot_MMK2",
            "fps": 20,
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["left_hand_joint_1_rad", "right_hand_joint_1_rad"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["left_hand_joint_1_rad", "right_hand_joint_1_rad"],
                },
                "observation.images.cam_high_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
                "observation.images.cam_left_wrist_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
                "observation.images.cam_right_wrist_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
                "observation.images.cam_third_view": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
            },
        }
        (self.dataset_path / "meta" / "info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )
        (self.dataset_path / "meta" / "common_record.json").write_text(
            json.dumps({"machine_id": "sample_galbot_g1"}), encoding="utf-8"
        )

        self._run("normalize", "--root", str(self.root))

        normalized = json.loads(
            (self.dataset_path / "meta" / "info_norm.json").read_text(
                encoding="utf-8"
            )
        )
        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        features = normalized["features"]
        self.assertIn("observation.images.cam_high_rgb", features)
        self.assertIn("observation.images.cam_third_view", features)
        self.assertNotIn("observation.images.cam_front_top_rgb", features)
        self.assertNotIn("observation.images.cam_top_side_rgb", features)
        self.assertEqual(features["observation.images.cam_high_rgb"]["codec"], "av1")
        self.assertEqual(features["observation.images.cam_third_view"]["codec"], "av1")
        self.assertEqual(review["robot_identity"]["canonical_id"], "airbot_mmk2")
        self.assertIn(
            "ROBOT_IDENTITY_CONFLICT",
            {item["category"] for item in review["review_items"]},
        )
        unresolved = {
            item["source_key"]
            for item in review["camera_review_items"]
            if item["evidence"].get("inference_level") == "UNRESOLVED"
        }
        self.assertEqual(
            unresolved,
            {
                "observation.images.cam_high_rgb",
                "observation.images.cam_third_view",
            },
        )

    def test_normalize_uses_default_vlm_and_topology_resolver(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("robometanorm.cli.main.normalize_datasets", return_value=[]) as normalize:
                self._run("normalize", "--root", str(self.root))

        classifier = normalize.call_args.kwargs["vlm_classifier"]
        self.assertIsInstance(classifier, OpenAICompatibleVlmClassifier)
        self.assertEqual(
            classifier.endpoint,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(classifier.model, "qwen3.7-plus")
        self.assertEqual(classifier.api_key_env, "DASHSCOPE_API_KEY")
        self.assertIsNotNone(normalize.call_args.kwargs["camera_topology_resolver"])

    def test_normalize_allows_explicit_vlm_overrides(self) -> None:
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
        self.assertIsNotNone(normalize.call_args.kwargs["camera_topology_resolver"])

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

    def test_normalize_profiles_first_and_last_episode_then_hits_cache(self) -> None:
        for index in range(4):
            self._write_two_dimensional_parquet(
                f"episode_{index:06d}.parquet", float(index)
            )

        _, first_stderr = self._run_captured(
            "normalize", "--root", str(self.root)
        )
        middle_path = self.dataset_path / "data" / "episode_000001.parquet"
        stat = middle_path.stat()
        os.utime(
            middle_path,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
        )
        _, second_stderr = self._run_captured(
            "normalize", "--root", str(self.root)
        )

        self.assertIn(
            "正在分析 episode 1/2: episode_000000.parquet", first_stderr
        )
        self.assertIn(
            "正在分析 episode 2/2: episode_000003.parquet", first_stderr
        )
        self.assertNotIn("episode_000001.parquet", first_stderr)
        self.assertNotIn("episode_000002.parquet", first_stderr)
        self.assertIn("已加载 Parquet 画像缓存，共 2 episodes", second_stderr)

    def test_normalize_writes_non_destructive_gripper_transform_proposal(self) -> None:
        source_name = "leader_left_gripper_degree_mm.pos"
        info = {
            "fps": 20,
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": [source_name],
                },
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
            pa.table({"action": [[0.0], [25.0], [50.0], [75.0], [100.0]]}),
            self.dataset_path / "data" / "episode_000000.parquet",
        )

        self._run("normalize", "--root", str(self.root))

        normalized = json.loads(
            (self.dataset_path / "meta" / "info_norm.json").read_text(
                encoding="utf-8"
            )
        )
        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(normalized["features"]["action"]["names"], [source_name])
        proposal = review["gripper_transform_proposals"][0]
        self.assertEqual(proposal["target_name"], "left_gripper_open")
        self.assertEqual(proposal["formula"], "clip(x / 100, 0, 1)")
        self.assertTrue(proposal["transform_required"])
        self.assertEqual(
            json.loads(
                (self.dataset_path / "meta" / "info.json").read_text(
                    encoding="utf-8"
                )
            ),
            info,
        )

    def _write_two_dimensional_parquet(self, filename: str, offset: float) -> None:
        pq.write_table(
            pa.table(
                {
                    "action": [[offset, offset + 1.0]],
                    "observation.state": [[offset, offset + 1.0]],
                }
            ),
            self.dataset_path / "data" / filename,
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
        stdout, _ = CliIntegrationTest._run_captured(*arguments)
        return stdout

    @staticmethod
    def _run_captured(*arguments: str) -> tuple[str, str]:
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = main(list(arguments))
        if exit_code != 0:
            raise AssertionError(f"命令返回非零状态: {exit_code}")
        return output.getvalue(), errors.getvalue()


if __name__ == "__main__":
    unittest.main()
