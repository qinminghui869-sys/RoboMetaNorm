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
from robometanorm.camera.models import (
    CameraMount,
    RobotCameraTopology,
    TopologyRejection,
)
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


class _FixedClassifier:
    """按提示词中的源字段返回固定本地画面语义。"""

    def __init__(self, semantics: dict[str, CameraSemantics]) -> None:
        self.semantics = semantics
        self.call_count = 0
        self.last_error = None
        self.last_error_code = None
        self.last_error_evidence: dict[str, object] = {}

    def classify(
        self, system_prompt: str, user_prompt: str, image_paths: object
    ) -> CameraSemantics | None:
        self.call_count += 1
        source_line = next(
            line for line in user_prompt.splitlines() if line.startswith("source_key: ")
        )
        return self.semantics.get(source_line.removeprefix("source_key: "))


class _FixedTopologyResolver:
    """返回固定拓扑并记录解析次数。"""

    def __init__(
        self,
        topology: RobotCameraTopology | None,
        *,
        last_error: str | None = None,
        last_error_code: str | None = None,
        last_error_evidence: dict[str, object] | None = None,
    ) -> None:
        self.topology = topology
        self.call_count = 0
        self.last_error = last_error
        self.last_error_code = last_error_code
        self.last_error_evidence = last_error_evidence or {}

    def resolve(self, robot_id: str) -> RobotCameraTopology | None:
        self.call_count += 1
        return self.topology


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

    def test_does_not_rename_ambiguous_source_tokens_without_evidence(self) -> None:
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
        self.assertIn("observation.images.image_left", features)
        self.assertIn("observation.images.image_top_depth", features)
        self.assertEqual(features["observation.images.image_left"]["codec"], "av1")
        self.assertEqual(features["observation.images.image_top_depth"]["codec"], "ffv1")
        self.assertEqual(result.normalized_info["fps"], 30)
        self.assertEqual(
            {item.evidence["inference_level"] for item in result.camera_review_items},
            {"UNRESOLVED"},
        )
        self.assertEqual(source_info, original_info)

    def test_keeps_unknown_camera_and_creates_review_item(self) -> None:
        result = normalize_cameras(
            self.candidate,
            self._info({"observation.images.camera_1": {"dtype": "video"}}),
        )

        self.assertIn("observation.images.camera_1", result.normalized_info["features"])
        self.assertEqual(len(result.camera_review_items), 1)
        self.assertEqual(result.camera_review_items[0].reason_code, "CAMERA_NAME_UNRESOLVED")
        self.assertEqual(result.camera_review_items[0].evidence["inference_level"], "UNRESOLVED")
        self.assertEqual(result.camera_review_items[0].source_key, "observation.images.camera_1")
        self.assertEqual(result.confirmed_count, 0)
        self.assertEqual(result.inferred_count, 0)
        self.assertEqual(result.unresolved_count, 1)
        self.assertEqual(result.topology_error_count, 0)

    def test_confirms_airbot_head_camera_from_topology_and_local_frames(self) -> None:
        source_key = "observation.images.cam_high_rgb"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier({source_key: self._head_semantics()})
        resolver = _FixedTopologyResolver(self._airbot_topology())
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
                topology_resolver=resolver,
            )

        features = result.normalized_info["features"]
        self.assertNotIn(source_key, features)
        feature = features["observation.images.cam_head_rgb"]
        self.assertEqual(feature["codec"], "av1")
        self.assertNotIn(
            "CAMERA_NAME_INFERRED",
            {item.reason_code for item in result.camera_review_items},
        )
        self.assertEqual(result.confirmed_count, 1)
        self.assertEqual(result.inferred_count, 0)
        self.assertEqual(result.unresolved_count, 0)
        self.assertEqual(resolver.call_count, 1)

    def test_infers_airbot_high_as_head_from_partial_topology_and_occupied_wrists(self) -> None:
        source_key = "observation.images.cam_high_rgb"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier(
            {
                source_key: CameraSemantics(
                    "rgb", "on_robot", ("upper",), None, True, 0.95, False, (), False
                )
            }
        )
        topology = RobotCameraTopology(
            "airbot_mmk2",
            self._airbot_topology().camera_mounts,
            0.96,
            False,
            (
                TopologyRejection(
                    "camera_mounts[3]",
                    {
                        "mount_type": "on_robot",
                        "direction_tokens": ["top"],
                        "body_part": None,
                    },
                    "相机槽位不符合内置命名规范",
                ),
            ),
        )
        features = {
            source_key: {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.cam_left_wrist_rgb": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.images.cam_right_wrist_rgb": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
        }
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info(features),
                robot_identity=RobotIdentity(
                    "airbot_mmk2", "info.robot_type", "Airbot_MMK2"
                ),
                vlm_classifier=classifier,
                topology_resolver=_FixedTopologyResolver(
                    topology,
                    last_error="机器人相机拓扑仅部分有效",
                    last_error_code="ROBOT_TOPOLOGY_PARTIAL",
                    last_error_evidence={
                        "rejected_mounts": [
                            {
                                "field": "camera_mounts[3]",
                                "value": topology.rejected_mounts[0].value,
                                "reason": topology.rejected_mounts[0].reason,
                            }
                        ]
                    },
                ),
            )

        self.assertIn(
            "observation.images.cam_head_rgb", result.normalized_info["features"]
        )
        review = next(
            item
            for item in result.camera_review_items
            if item.reason_code == "CAMERA_NAME_INFERRED"
        )
        self.assertEqual(review.evidence["reason"], "unique_remaining_topology_slot")
        self.assertTrue(review.evidence["robot_topology"]["partial"])
        self.assertEqual(
            review.evidence["robot_topology"]["rejected_mounts"][0]["field"],
            "camera_mounts[3]",
        )
        self.assertEqual(result.confirmed_count, 2)
        self.assertEqual(result.inferred_count, 1)
        self.assertEqual(result.unresolved_count, 0)
        self.assertEqual(result.topology_error_count, 1)

    def test_confirms_agilex_high_as_standard_external_top_camera(self) -> None:
        source_key = "observation.images.cam_high_rgb"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier(
            {
                source_key: CameraSemantics(
                    "rgb", "external", ("top",), None, True, 0.96, False, (), False
                )
            }
        )
        topology = RobotCameraTopology(
            "agilex_cobot_magic",
            (
                CameraMount("on_robot", ("left",), "wrist"),
                CameraMount("on_robot", ("right",), "wrist"),
                CameraMount("external", ("top",), None),
            ),
            0.96,
            False,
        )
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info(
                    {source_key: {"dtype": "video", "shape": [480, 640, 3]}}
                ),
                robot_identity=RobotIdentity(
                    "agilex_cobot_magic", "info.robot_type", "Agilex_Cobot_Magic"
                ),
                vlm_classifier=classifier,
                topology_resolver=_FixedTopologyResolver(topology),
            )

        self.assertIn(
            "observation.images.cam_top_rgb", result.normalized_info["features"]
        )
        self.assertNotIn(
            "CAMERA_NAME_INFERRED",
            {item.reason_code for item in result.camera_review_items},
        )

    def test_reports_topology_schema_error_and_exact_evidence_for_incomplete_semantics(self) -> None:
        source_key = "observation.images.cam_high_rgb"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier(
            {
                source_key: CameraSemantics(
                    "rgb", "on_robot", ("upper",), None, True, 0.95, False, (), False
                )
            }
        )
        invalid_value = {
            "mount_type": "on_robot",
            "direction_tokens": ["top"],
            "body_part": None,
        }
        resolver = _FixedTopologyResolver(
            None,
            last_error="机器人相机拓扑不合法: 相机槽位不符合内置命名规范",
            last_error_code="ROBOT_TOPOLOGY_INVALID",
            last_error_evidence={"field": "camera_mounts[3]", "value": invalid_value},
        )
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info(
                    {source_key: {"dtype": "video", "shape": [480, 640, 3]}}
                ),
                robot_identity=RobotIdentity(
                    "airbot_mmk2", "info.robot_type", "Airbot_MMK2"
                ),
                vlm_classifier=classifier,
                topology_resolver=resolver,
            )

        review = result.camera_review_items[0]
        self.assertEqual(review.reason_code, "ROBOT_TOPOLOGY_INVALID")
        self.assertEqual(
            review.evidence["topology_error_evidence"],
            {"field": "camera_mounts[3]", "value": invalid_value},
        )
        self.assertEqual(result.unresolved_count, 1)
        self.assertEqual(result.topology_error_count, 1)

    def test_image_left_can_be_inferred_as_an_external_camera(self) -> None:
        source_key = "observation.images.image_left"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier(
            {source_key: self._external_left_semantics()}
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
                vlm_classifier=classifier,
            )

        self.assertIn(
            "observation.images.cam_left_rgb", result.normalized_info["features"]
        )
        review = next(
            item
            for item in result.camera_review_items
            if item.reason_code == "CAMERA_NAME_INFERRED"
        )
        self.assertEqual(review.evidence["inference_level"], "INFERRED")
        self.assertEqual(review.evidence["local_semantics"]["mount_type"], "external")

    def test_image_left_can_be_confirmed_as_a_left_wrist_camera(self) -> None:
        source_key = "observation.images.image_left"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier({source_key: self._left_wrist_semantics()})
        topology = RobotCameraTopology(
            "test_robot",
            (CameraMount("on_robot", ("left",), "wrist"),),
            0.97,
            False,
        )
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info({source_key: {"dtype": "video", "shape": [480, 640, 3]}}),
                robot_identity=RobotIdentity("test_robot", "info.robot_type", "test_robot"),
                vlm_classifier=classifier,
                topology_resolver=_FixedTopologyResolver(topology),
            )

        self.assertIn(
            "observation.images.cam_left_wrist_rgb",
            result.normalized_info["features"],
        )
        self.assertNotIn(
            "CAMERA_NAME_INFERRED",
            {item.reason_code for item in result.camera_review_items},
        )

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
            result.camera_review_items[0].reason_code, "CAMERA_NAME_UNRESOLVED"
        )

    def test_topology_alone_does_not_rename_a_camera(self) -> None:
        source_key = "observation.images.cam_high_rgb"

        result = normalize_cameras(
            self.candidate,
            self._info({source_key: {"dtype": "video", "shape": [480, 640, 3]}}),
            robot_identity=RobotIdentity("airbot_mmk2", "info.robot_type", "Airbot_MMK2"),
            topology_resolver=_FixedTopologyResolver(self._airbot_topology()),
        )

        self.assertIn(source_key, result.normalized_info["features"])
        self.assertEqual(result.camera_review_items[0].evidence["inference_level"], "UNRESOLVED")

    def test_skips_topology_lookup_when_all_camera_names_are_standard(self) -> None:
        resolver = _FixedTopologyResolver(self._airbot_topology())

        result = normalize_cameras(
            self.candidate,
            self._info(
                {
                    "observation.images.cam_left_wrist_rgb": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    }
                }
            ),
            robot_identity=RobotIdentity(
                "airbot_mmk2", "info.robot_type", "Airbot_MMK2"
            ),
            topology_resolver=resolver,
        )

        self.assertEqual(resolver.call_count, 0)
        self.assertIn(
            "observation.images.cam_left_wrist_rgb",
            result.normalized_info["features"],
        )

    def test_does_not_break_a_tie_between_identical_weak_hints(self) -> None:
        source_keys = (
            "observation.images.image_left",
            "observation.images.camera_left",
        )
        topology = RobotCameraTopology(
            "test_robot",
            (
                CameraMount("on_robot", ("left",), "wrist"),
                CameraMount("on_robot", ("right",), "wrist"),
            ),
            0.96,
            False,
        )

        result = normalize_cameras(
            self.candidate,
            self._info(
                {
                    source_key: {"dtype": "video", "shape": [480, 640, 3]}
                    for source_key in source_keys
                }
            ),
            robot_identity=RobotIdentity(
                "test_robot", "info.robot_type", "test_robot"
            ),
            topology_resolver=_FixedTopologyResolver(topology),
        )

        self.assertTrue(
            set(source_keys).issubset(result.normalized_info["features"])
        )
        self.assertEqual(
            {
                item.evidence["inference_level"]
                for item in result.camera_review_items
            },
            {"UNRESOLVED"},
        )

    def test_topology_and_occupied_slots_without_local_evidence_do_not_rename(self) -> None:
        source_key = "observation.images.cam_high_rgb"
        features = {
            source_key: {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.cam_left_wrist_rgb": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.cam_right_wrist_rgb": {"dtype": "video", "shape": [480, 640, 3]},
        }

        result = normalize_cameras(
            self.candidate,
            self._info(features),
            robot_identity=RobotIdentity("airbot_mmk2", "info.robot_type", "Airbot_MMK2"),
            topology_resolver=_FixedTopologyResolver(self._airbot_topology()),
        )

        self.assertIn(source_key, result.normalized_info["features"])
        self.assertNotIn(
            "CAMERA_NAME_INFERRED",
            {item.reason_code for item in result.camera_review_items},
        )

    def test_third_view_uses_local_external_direction(self) -> None:
        source_key = "observation.images.cam_third_view"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier(
            {
                source_key: CameraSemantics(
                    "rgb", "external", ("side",), None, False, 0.96, False, (), False
                )
            }
        )
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info(
                    {source_key: {"dtype": "video", "shape": [480, 640, 3]}}
                ),
                vlm_classifier=classifier,
            )

        self.assertIn(
            "observation.images.cam_side_rgb", result.normalized_info["features"]
        )

    def test_third_view_cannot_be_renamed_as_an_on_robot_camera(self) -> None:
        source_key = "observation.images.cam_third_view"
        candidate = self._candidate_with_media(source_key)
        classifier = _FixedClassifier({source_key: self._head_semantics()})
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")

        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info(
                    {source_key: {"dtype": "video", "shape": [480, 640, 3]}}
                ),
                vlm_classifier=classifier,
            )

        self.assertIn(source_key, result.normalized_info["features"])
        self.assertEqual(
            result.camera_review_items[0].reason_code,
            "SOURCE_CAMERA_CATEGORY_CONFLICT",
        )
        self.assertEqual(result.camera_review_items[0].candidates, ())

    def test_keeps_all_sources_when_target_names_collide(self) -> None:
        source_keys = (
            "observation.images.image_left",
            "observation.images.camera_left",
        )
        candidate = self._candidate_with_media(*source_keys)
        classifier = _FixedClassifier(
            {source_key: self._external_left_semantics() for source_key in source_keys}
        )
        media = MediaInfo("av1", 30.0, 640, 480, 2.0, 60, "yuv420p")
        with (
            patch("robometanorm.camera.normalizer.probe_media", return_value=media),
            patch("robometanorm.camera.normalizer.extract_rgb_frames", return_value=()),
        ):
            result = normalize_cameras(
                candidate,
                self._info({source_key: {"dtype": "video"} for source_key in source_keys}),
                vlm_classifier=classifier,
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

    def _candidate_with_media(self, *media_keys: str) -> DatasetCandidate:
        for media_key in media_keys:
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

    @staticmethod
    def _head_semantics() -> CameraSemantics:
        return CameraSemantics(
            "rgb", "on_robot", (), "head", True, 0.97, False, (), False
        )

    @staticmethod
    def _left_wrist_semantics() -> CameraSemantics:
        return CameraSemantics(
            "rgb", "on_robot", ("left",), "wrist", False, 0.97, False, (), False
        )

    @staticmethod
    def _external_left_semantics() -> CameraSemantics:
        return CameraSemantics(
            "rgb", "external", ("left",), None, False, 0.96, False, (), False
        )

    @staticmethod
    def _airbot_topology() -> RobotCameraTopology:
        return RobotCameraTopology(
            "airbot_mmk2",
            (
                CameraMount("on_robot", (), "head"),
                CameraMount("on_robot", ("left",), "wrist"),
                CameraMount("on_robot", ("right",), "wrist"),
            ),
            0.96,
            False,
        )


if __name__ == "__main__":
    unittest.main()
