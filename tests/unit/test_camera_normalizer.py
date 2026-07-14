"""P1 相机规范建议测试。"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.media import MediaInfo
from robometanorm.camera.normalizer import normalize_cameras
from robometanorm.camera.vlm import CameraSemantics
from robometanorm.domain.models import DatasetCandidate, LayoutType
from robometanorm.robot_identity import RobotIdentity


class _InvalidSemanticsClassifier:
    """模拟业务 schema 不合法的相机 VLM 返回。"""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_error = "相机 VLM 语义不合法: VLM 方位词不合法"
        self.last_error_code = "VLM_SEMANTICS_INVALID"
        self.last_error_evidence = {
            "field": "direction_tokens",
            "value": ["high"],
        }

    def classify(self, system_prompt: str, user_prompt: str, image_paths: object) -> None:
        self.call_count += 1
        return None


class _ConfidentClassifier:
    """模拟给出高置信位置但没有机器人拓扑依据的 VLM。"""

    def __init__(self) -> None:
        self.call_count = 0

    def classify(
        self, system_prompt: str, user_prompt: str, image_paths: object
    ) -> CameraSemantics:
        self.call_count += 1
        return CameraSemantics(
            modality="rgb",
            mount_type=None,
            direction_tokens=("top", "side"),
            body_part=None,
            is_primary=False,
            confidence=0.99,
            ambiguous=False,
            alternatives=(),
            need_human_review=False,
        )


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

    def test_keeps_unverified_airbot_camera_despite_confident_vlm(self) -> None:
        source_key = "observation.images.cam_third_view"
        candidate = self._candidate_with_media(source_key)
        classifier = _ConfidentClassifier()
        identity = RobotIdentity(
            "airbot_mmk2", "info.robot_type", "Airbot_MMK2"
        )
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch(
                "robometanorm.camera.normalizer.extract_rgb_frames",
                return_value=(),
            ),
        ):
            result = normalize_cameras(
                candidate,
                self._info(
                    {source_key: {"dtype": "video", "shape": [480, 640, 3]}}
                ),
                robot_identity=identity,
                vlm_classifier=classifier,
            )

        feature = result.normalized_info["features"][source_key]
        self.assertEqual(feature["codec"], "av1")
        self.assertNotIn(
            "observation.images.cam_top_side_rgb",
            result.normalized_info["features"],
        )
        review = next(
            item
            for item in result.camera_review_items
            if item.reason_code == "ROBOT_CAMERA_MAPPING_UNKNOWN"
        )
        self.assertEqual(review.source_key, source_key)
        self.assertEqual(
            review.candidates[0].target_key,
            "observation.images.cam_top_side_rgb",
        )
        self.assertEqual(review.evidence["robot_id"], "airbot_mmk2")

    def test_keeps_unknown_camera_for_identified_robot_without_registry(self) -> None:
        source_key = "observation.images.external_camera"
        candidate = self._candidate_with_media(source_key)
        classifier = _ConfidentClassifier()
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch(
                "robometanorm.camera.normalizer.extract_rgb_frames",
                return_value=(),
            ),
        ):
            result = normalize_cameras(
                candidate,
                self._info(
                    {source_key: {"dtype": "video", "shape": [480, 640, 3]}}
                ),
                robot_identity=RobotIdentity(
                    "franka", "common_record.machine_id", "CFvFyRtw8T1_franka"
                ),
                vlm_classifier=classifier,
            )

        self.assertIn(source_key, result.normalized_info["features"])
        self.assertNotIn(
            "observation.images.cam_top_side_rgb",
            result.normalized_info["features"],
        )
        review = next(
            item
            for item in result.camera_review_items
            if item.reason_code == "ROBOT_CAMERA_MAPPING_UNKNOWN"
        )
        self.assertEqual(review.evidence["robot_id"], "franka")

    def test_applies_codec_to_unresolved_rgb_without_vlm(self) -> None:
        source_key = "observation.images.cam_high_rgb"

        result = normalize_cameras(
            self.candidate,
            self._info(
                {source_key: {"dtype": "video", "shape": [480, 640, 3]}}
            ),
            robot_identity=RobotIdentity(
                "airbot_mmk2", "info.robot_type", "Airbot_MMK2"
            ),
        )

        self.assertIn(source_key, result.normalized_info["features"])
        self.assertEqual(
            result.normalized_info["features"][source_key]["codec"], "av1"
        )
        self.assertEqual(
            result.camera_review_items[0].reason_code,
            "ROBOT_CAMERA_MAPPING_UNKNOWN",
        )

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

    def test_schema_invalid_vlm_response_skips_second_stage_and_records_evidence(self) -> None:
        source_key = "observation.images.camera_1"
        candidate = self._candidate_with_media(source_key)
        classifier = _InvalidSemanticsClassifier()
        media = MediaInfo("h264", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info({source_key: {"dtype": "video", "shape": [480, 640, 3]}}),
                vlm_classifier=classifier,
            )

        self.assertEqual(classifier.call_count, 1)
        review = next(
            item
            for item in result.camera_review_items
            if item.reason_code == "VLM_SEMANTICS_INVALID"
        )
        self.assertEqual(review.evidence["field"], "direction_tokens")
        self.assertEqual(review.evidence["value"], ["high"])
        self.assertNotIn(
            "VLM_UNAVAILABLE", {item.reason_code for item in result.camera_review_items}
        )

    def test_reports_unresolved_media_key_mismatch_once(self) -> None:
        source_key = "observation.images.external_image"
        candidate = self._candidate_with_media("observation.images.image_top")
        classifier = _InvalidSemanticsClassifier()

        result = normalize_cameras(
            candidate,
            self._info({source_key: {"dtype": "video", "shape": [480, 640, 3]}}),
            vlm_classifier=classifier,
        )

        self.assertEqual(classifier.call_count, 0)
        self.assertEqual(len(result.camera_review_items), 1)
        review = result.camera_review_items[0]
        self.assertEqual(review.reason_code, "MEDIA_KEY_MISMATCH")
        self.assertEqual(
            review.evidence["available_media_keys"],
            ["observation.images.image_top"],
        )

    def _candidate_with_media(self, media_key: str) -> DatasetCandidate:
        media_directory = self.dataset_path / "videos" / media_key
        media_directory.mkdir(parents=True, exist_ok=True)
        (media_directory / "episode_000000.mp4").touch()
        return DatasetCandidate(
            dataset_name="dataset_001",
            task_name=None,
            source_path=self.dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=self.info_path,
            data_path=self.dataset_path / "data",
            video_path=self.dataset_path / "videos",
            depth_path=None,
        )

    @staticmethod
    def _info(features: dict[str, object]) -> dict[str, object]:
        return {"fps": 30, "features": copy.deepcopy(features)}


if __name__ == "__main__":
    unittest.main()
