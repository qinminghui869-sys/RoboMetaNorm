"""Mini pipeline orchestration and conservative degradation tests."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.models import (
    DatasetMapping,
    DatasetResult,
    DatasetStatus,
    Issue,
    MachineAssignment,
    MediaSample,
)
from tests.mini_fixtures import DatasetFixture, FakeVlm, PipelineFixture


class _RaisingVlm:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def research_hardware(self, identity: object) -> object:
        raise self.error

    def map_dataset(self, evidence: object, profile: object) -> object:
        raise AssertionError("mapping must not be called")


class MiniPipelineTest(unittest.TestCase):
    """Exercise the real evidence/standard/writer chain with a fixed fake VLM."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.fixture = self._create_dataset("dataset-a")
        self.pipeline_fixture = PipelineFixture()
        self.profile = self.pipeline_fixture.hardware_profile()
        state_assignment = self.pipeline_fixture.dataset_mapping().machines[0]
        self.mapping = DatasetMapping(
            cameras=self.pipeline_fixture.dataset_mapping().cameras,
            machines=(
                replace(state_assignment, source_feature="action"),
                state_assignment,
            ),
        )

    @staticmethod
    def _source_info() -> dict[str, object]:
        joint_names = [f"left_joint_{index}" for index in range(1, 7)]
        return {
            "robot_type": "acme_testbot",
            "fps": 30,
            "features": {
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "fps": 30,
                    "codec": "av1",
                },
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": list(joint_names),
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": list(joint_names),
                },
            },
        }

    def _create_dataset(self, name: str) -> DatasetFixture:
        fixture = DatasetFixture.create(
            self.root,
            dataset_name=name,
            info=self._source_info(),
            with_data=True,
            with_videos=True,
        )
        rows = {
            "action": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]],
            "observation.state": [[0.5, 0.4, 0.3, 0.2, 0.1, 0.0]],
        }
        fixture.write_parquet(rows, relative_path="chunk-000/episode_000000.parquet")
        fixture.write_parquet(rows, relative_path="chunk-001/episode_000001.parquet")
        fixture.write_media_placeholder(
            relative_path=(
                "chunk-000/observation.images.wrist/episode_000000.mp4"
            ),
            payload=b"source-video-bytes",
        )
        return fixture

    @staticmethod
    def _probe_sample(*args: object, **kwargs: object) -> MediaSample:
        return MediaSample(
            relative_path="",
            media_type="video",
            codec="av1",
            fps=30.0,
            width=640,
            height=480,
            duration_seconds=1.0,
            pixel_format="yuv420p",
            frame_path=None,
        )

    @staticmethod
    def _extract_frame(
        media_path: Path,
        output_path: Path,
        *,
        duration_seconds: float | None = None,
    ) -> Path:
        output_path.write_bytes(b"temporary-frame")
        return output_path

    @contextmanager
    def _media_stubs(self):
        with (
            patch(
                "robometanorm.evidence.probe_media",
                side_effect=self._probe_sample,
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._extract_frame,
            ),
        ):
            yield

    def _normalize(self, vlm: object):
        from robometanorm.pipeline import normalize_datasets

        with self._media_stubs():
            return normalize_datasets(
                self.root,
                vlm=vlm,
                confidence_threshold=0.85,
            )

    def _scan(self):
        from robometanorm.pipeline import scan_datasets

        with self._media_stubs():
            return scan_datasets(self.root)

    def _read_output(self, fixture: DatasetFixture, name: str) -> dict[str, object]:
        return json.loads(
            (fixture.candidate.info_path.parent / name).read_text(encoding="utf-8")
        )

    def _success_vlm(self) -> FakeVlm:
        return FakeVlm(
            research_result=(self.profile, None),
            mapping_result=(self.mapping, None),
        )

    def test_status_priority_is_error_block_review_pass_and_unknown_is_error(self) -> None:
        from robometanorm.pipeline import status_from_issues

        review = Issue("R", "review", "fixture")
        block = Issue("B", "block", "fixture", severity="block")
        error = Issue("E", "error", "fixture", severity="error")
        unknown = Issue("U", "unknown", "fixture", severity="warning")
        malformed = Issue(
            "M", "malformed", "fixture", severity=[]  # type: ignore[arg-type]
        )
        self.assertEqual(status_from_issues(()), DatasetStatus.PASS)
        self.assertEqual(status_from_issues((review,)), DatasetStatus.REVIEW)
        self.assertEqual(status_from_issues((review, block)), DatasetStatus.BLOCKED)
        self.assertEqual(status_from_issues((block, error)), DatasetStatus.ERROR)
        self.assertEqual(status_from_issues((unknown,)), DatasetStatus.ERROR)
        self.assertEqual(status_from_issues((malformed,)), DatasetStatus.ERROR)

    def test_normalize_threshold_is_explicit_and_rejects_nonfinite_or_bool(self) -> None:
        from robometanorm.pipeline import normalize_datasets

        parameter = inspect.signature(normalize_datasets).parameters[
            "confidence_threshold"
        ]
        self.assertIs(parameter.default, inspect.Parameter.empty)
        for invalid in (True, False, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalize_datasets(
                        self.root,
                        vlm=self._success_vlm(),
                        confidence_threshold=invalid,
                    )

    def test_scan_is_read_only_and_reports_real_counts_without_apply_or_writer(self) -> None:
        with (
            patch("robometanorm.pipeline.apply_standard") as apply,
            patch("robometanorm.pipeline.write_outputs") as writer,
        ):
            results = self._scan()

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, DatasetStatus.PASS)
        self.assertEqual((result.camera_count, result.machine_field_count), (1, 2))
        self.assertEqual((result.changed_field_count, result.issue_count), (0, 0))
        self.assertEqual(result.source_info, self._source_info())
        apply.assert_not_called()
        writer.assert_not_called()
        self.assertFalse((self.fixture.candidate.info_path.parent / "info_norm.json").exists())

    def test_scan_invalid_info_is_error_and_does_not_stop_next_dataset(self) -> None:
        second = self._create_dataset("dataset-b")
        self.fixture.candidate.info_path.write_text("[", encoding="utf-8")

        results = self._scan()

        self.assertEqual(
            [result.status for result in results],
            [DatasetStatus.ERROR, DatasetStatus.PASS],
        )
        self.assertIsNone(results[0].source_info)
        self.assertEqual(results[0].changed_field_count, 0)
        self.assertEqual(results[1].source_info, second.info)

    def test_scan_reports_each_completed_dataset_once_in_discovery_order(self) -> None:
        self._create_dataset("dataset-b")
        self.fixture.candidate.info_path.write_text("[", encoding="utf-8")
        completed: list[tuple[int, int, str]] = []

        def on_complete(index: int, total: int, result: DatasetResult) -> None:
            completed.append((index, total, result.candidate.dataset_name))

        from robometanorm.pipeline import scan_datasets

        with self._media_stubs():
            results = scan_datasets(self.root, progress=on_complete)

        self.assertEqual(
            [result.status for result in results],
            [DatasetStatus.ERROR, DatasetStatus.PASS],
        )
        self.assertEqual(
            [result.candidate.dataset_name for result in results],
            ["dataset-a", "dataset-b"],
        )
        self.assertEqual(completed, [(1, 2, "dataset-a"), (2, 2, "dataset-b")])

    def test_normalize_reports_each_completed_dataset_once_in_discovery_order(self) -> None:
        self._create_dataset("dataset-b")
        self.fixture.candidate.info_path.write_text("[", encoding="utf-8")
        completed: list[tuple[int, int, str]] = []

        def on_complete(index: int, total: int, result: DatasetResult) -> None:
            completed.append((index, total, result.candidate.dataset_name))

        from robometanorm.pipeline import normalize_datasets

        with self._media_stubs():
            results = normalize_datasets(
                self.root,
                vlm=self._success_vlm(),
                confidence_threshold=0.85,
                progress=on_complete,
            )

        self.assertEqual(
            [result.status for result in results],
            [DatasetStatus.ERROR, DatasetStatus.PASS],
        )
        self.assertEqual(
            [result.candidate.dataset_name for result in results],
            ["dataset-a", "dataset-b"],
        )
        self.assertEqual(completed, [(1, 2, "dataset-a"), (2, 2, "dataset-b")])

    def test_scan_ignores_progress_callback_errors(self) -> None:
        self._create_dataset("dataset-b")
        attempted: list[tuple[int, int, str]] = []

        def failing_progress(index: int, total: int, result: DatasetResult) -> None:
            attempted.append((index, total, result.candidate.dataset_name))
            raise OSError("terminal unavailable")

        from robometanorm.pipeline import scan_datasets

        with self._media_stubs():
            results = scan_datasets(self.root, progress=failing_progress)

        self.assertEqual(
            [result.candidate.dataset_name for result in results],
            ["dataset-a", "dataset-b"],
        )
        self.assertEqual(attempted, [(1, 2, "dataset-a"), (2, 2, "dataset-b")])

    def test_normalize_ignores_progress_callback_errors(self) -> None:
        self._create_dataset("dataset-b")
        attempted: list[tuple[int, int, str]] = []

        def failing_progress(index: int, total: int, result: DatasetResult) -> None:
            attempted.append((index, total, result.candidate.dataset_name))
            raise OSError("terminal unavailable")

        from robometanorm.pipeline import normalize_datasets

        with self._media_stubs():
            results = normalize_datasets(
                self.root,
                vlm=self._success_vlm(),
                confidence_threshold=0.85,
                progress=failing_progress,
            )

        self.assertEqual(
            [result.candidate.dataset_name for result in results],
            ["dataset-a", "dataset-b"],
        )
        self.assertEqual(attempted, [(1, 2, "dataset-a"), (2, 2, "dataset-b")])

    def test_success_uses_one_research_map_and_range_then_writes_real_changes(self) -> None:
        vlm = self._success_vlm()
        from robometanorm.evidence import collect_mapped_gripper_ranges as real_range

        with patch(
            "robometanorm.pipeline.collect_mapped_gripper_ranges",
            wraps=real_range,
        ) as ranges:
            results = self._normalize(vlm)

        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (1, 1))
        ranges.assert_called_once()
        result = results[0]
        self.assertEqual(result.status, DatasetStatus.PASS)
        self.assertEqual((result.camera_count, result.machine_field_count), (1, 2))
        self.assertEqual((result.changed_field_count, result.issue_count), (4, 0))
        info_norm = self._read_output(self.fixture, "info_norm.json")
        self.assertEqual(info_norm["robot_type"], "acme_robotics_testbot_one")
        features = info_norm["features"]
        self.assertIn("observation.images.cam_front_wrist_rgb", features)
        self.assertNotIn("observation.images.wrist", features)
        expected_names = [f"left_arm_joint_{index}_rad" for index in range(6)]
        self.assertEqual(features["action"]["names"], expected_names)
        self.assertEqual(features["observation.state"]["names"], expected_names)
        annotation = self.fixture.candidate.info_path.parent / "robo_annotation.yaml"
        self.assertTrue(annotation.is_file())
        self.assertIn("arm.left.joint", annotation.read_text(encoding="utf-8"))

    def test_ambiguous_joint_preflight_skips_vlm_and_records_source_file(self) -> None:
        source = self._source_info()
        names = [f"joint_{index}" for index in range(1, 7)]
        source["features"]["action"]["names"] = names
        source["features"]["observation.state"]["names"] = names
        self.fixture.candidate.info_path.write_text(json.dumps(source), encoding="utf-8")
        vlm = self._success_vlm()

        result = self._normalize(vlm)[0]

        self.assertEqual(result.status, DatasetStatus.BLOCKED)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
        review = self._read_output(self.fixture, "info_norm_review.json")
        issue = next(
            item
            for item in review["issues"]
            if item["code"] == "ANNOTATION_JOINT_AMBIGUOUS"
        )
        self.assertEqual(issue["evidence"]["source_file"], "meta/info.json")
        self.assertEqual(issue["evidence"]["source_indices"], list(range(6)))
        self.assertFalse(
            (self.fixture.candidate.info_path.parent / "robo_annotation.yaml").exists()
        )

    def test_main_follower_single_arm_emits_main_annotation(self) -> None:
        source = self._source_info()
        names = [f"main_follower_joint_{index}" for index in range(1, 7)]
        source["features"]["action"]["names"] = names
        source["features"]["observation.state"]["names"] = names
        self.fixture.candidate.info_path.write_text(json.dumps(source), encoding="utf-8")

        result = self._normalize(self._success_vlm())[0]

        self.assertEqual(result.status, DatasetStatus.PASS)
        annotation = yaml.safe_load(
            (
                self.fixture.candidate.info_path.parent / "robo_annotation.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertIn("arm.main.joint", annotation["robot_channel_schema"]["channels"])

    def test_main_follower_mapping_failure_records_main_confirmation_issue(self) -> None:
        source = self._source_info()
        names = [f"main_follower_joint_{index}" for index in range(1, 7)]
        source["features"]["action"]["names"] = names
        source["features"]["observation.state"]["names"] = names
        self.fixture.candidate.info_path.write_text(json.dumps(source), encoding="utf-8")
        vlm = FakeVlm(
            research_result=(self.profile, None),
            mapping_result=(None, Issue("DATASET_MAPPING_INVALID", "bad mapping", "vlm")),
        )

        result = self._normalize(vlm)[0]

        self.assertEqual(result.status, DatasetStatus.REVIEW)
        review = self._read_output(self.fixture, "info_norm_review.json")
        self.assertTrue(
            {"DATASET_MAPPING_INVALID", "ANNOTATION_MAIN_ARM_UNCONFIRMED"}
            <= {item["code"] for item in review["issues"]}
        )
        self.assertFalse(
            (self.fixture.candidate.info_path.parent / "robo_annotation.yaml").exists()
        )

    def test_main_follower_vlm_confirmation_failures_are_reviewed(self) -> None:
        source = self._source_info()
        names = [f"main_follower_joint_{index}" for index in range(1, 7)]
        source["features"]["action"]["names"] = names
        source["features"]["observation.state"]["names"] = names
        self.fixture.candidate.info_path.write_text(json.dumps(source), encoding="utf-8")
        unresolved_profile = replace(
            self.profile,
            identity=replace(self.profile.identity, manufacturer=" "),
        )
        unconfirmed_camera_mapping = replace(
            self.mapping,
            cameras=(replace(self.mapping.cameras[0], ambiguous=True),),
        )
        cases = (
            (
                FakeVlm(
                    research_result=(
                        None,
                        Issue("VLM_UNAVAILABLE", "offline", "vlm"),
                    )
                ),
                "VLM_UNAVAILABLE",
            ),
            (FakeVlm(research_result=(unresolved_profile, None)), "HARDWARE_IDENTITY_UNRESOLVED"),
            (
                FakeVlm(
                    research_result=(self.profile, None),
                    mapping_result=(unconfirmed_camera_mapping, None),
                ),
                "CAMERA_MAPPING_UNRESOLVED",
            ),
        )

        for vlm, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = self._normalize(vlm)[0]

                self.assertEqual(result.status, DatasetStatus.REVIEW)
                review = self._read_output(self.fixture, "info_norm_review.json")
                codes = {item["code"] for item in review["issues"]}
                self.assertIn(expected_code, codes)
                self.assertIn("ANNOTATION_MAIN_ARM_UNCONFIRMED", codes)
                self.assertFalse(
                    (self.fixture.candidate.info_path.parent / "robo_annotation.yaml").exists()
                )

    def test_invalid_main_follower_layout_blocks_before_vlm(self) -> None:
        source = self._source_info()
        names = [
            "main_follower_joint_1",
            "main_follower_joint_3",
            "main_follower_joint_4",
            "main_follower_joint_5",
            "main_follower_joint_6",
            "main_follower_joint_7",
        ]
        source["features"]["action"]["names"] = names
        source["features"]["observation.state"]["names"] = names
        self.fixture.candidate.info_path.write_text(json.dumps(source), encoding="utf-8")
        vlm = self._success_vlm()

        result = self._normalize(vlm)[0]

        self.assertEqual(result.status, DatasetStatus.BLOCKED)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
        review = self._read_output(self.fixture, "info_norm_review.json")
        self.assertIn(
            "ANNOTATION_MAIN_ARM_LAYOUT_INVALID",
            {item["code"] for item in review["issues"]},
        )
        self.assertFalse(
            (self.fixture.candidate.info_path.parent / "robo_annotation.yaml").exists()
        )

    def test_blocked_writes_source_copy_and_skips_all_vlm_and_range_work(self) -> None:
        source = self._source_info()
        del source["features"]["action"]
        self.fixture.candidate.info_path.write_text(json.dumps(source), encoding="utf-8")
        vlm = self._success_vlm()
        with patch("robometanorm.pipeline.collect_mapped_gripper_ranges") as ranges:
            result = self._normalize(vlm)[0]

        self.assertEqual(result.status, DatasetStatus.BLOCKED)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
        ranges.assert_not_called()
        self.assertEqual(self._read_output(self.fixture, "info_norm.json"), source)
        self.assertTrue(
            (self.fixture.candidate.info_path.parent / "info_norm_review.json").is_file()
        )

    def test_research_failure_or_conflicting_object_is_discarded_fail_closed(self) -> None:
        issue = Issue("WEB_SEARCH_UNAVAILABLE", "offline", "vlm")
        cases = ((None, issue), (self.profile, issue), (None, None))
        for research_result in cases:
            with self.subTest(research_result=research_result):
                vlm = FakeVlm(research_result=research_result)
                with patch("robometanorm.pipeline.collect_mapped_gripper_ranges") as ranges:
                    result = self._normalize(vlm)[0]
                self.assertEqual((vlm.research_calls, vlm.mapping_calls), (1, 0))
                ranges.assert_not_called()
                self.assertEqual(result.status, DatasetStatus.REVIEW)
                self.assertEqual(
                    self._read_output(self.fixture, "info_norm.json"),
                    self._source_info(),
                )
                codes = {
                    issue["code"]
                    for issue in self._read_output(
                        self.fixture, "info_norm_review.json"
                    )["issues"]
                }
                expected = (
                    "HARDWARE_RESEARCH_INVALID"
                    if research_result == (None, None)
                    else "WEB_SEARCH_UNAVAILABLE"
                )
                self.assertIn(expected, codes)

    def test_unresolved_hardware_identity_skips_map_and_preserves_source(self) -> None:
        unresolved = replace(
            self.profile,
            identity=replace(self.profile.identity, manufacturer=" "),
        )
        vlm = FakeVlm(research_result=(unresolved, None))
        with patch("robometanorm.pipeline.collect_mapped_gripper_ranges") as ranges:
            result = self._normalize(vlm)[0]

        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (1, 0))
        ranges.assert_not_called()
        self.assertEqual(result.status, DatasetStatus.REVIEW)
        self.assertEqual(self._read_output(self.fixture, "info_norm.json"), self._source_info())
        codes = {
            issue["code"]
            for issue in self._read_output(self.fixture, "info_norm_review.json")[
                "issues"
            ]
        }
        self.assertIn("HARDWARE_IDENTITY_UNRESOLVED", codes)

    def test_mapping_failure_none_or_conflicting_object_never_applies_identity(self) -> None:
        issue = Issue("DATASET_MAPPING_INVALID", "bad mapping", "vlm")
        cases = ((None, issue), (self.mapping, issue), (None, None))
        for mapping_result in cases:
            with self.subTest(mapping_result=mapping_result):
                vlm = FakeVlm(
                    research_result=(self.profile, None),
                    mapping_result=mapping_result,
                )
                with patch("robometanorm.pipeline.collect_mapped_gripper_ranges") as ranges:
                    result = self._normalize(vlm)[0]
                self.assertEqual((vlm.research_calls, vlm.mapping_calls), (1, 1))
                ranges.assert_not_called()
                self.assertEqual(result.status, DatasetStatus.REVIEW)
                self.assertEqual(
                    self._read_output(self.fixture, "info_norm.json"),
                    self._source_info(),
                )
                codes = {
                    item["code"]
                    for item in self._read_output(
                        self.fixture, "info_norm_review.json"
                    )["issues"]
                }
                self.assertIn("DATASET_MAPPING_INVALID", codes)

    def test_invalid_info_is_error_without_vlm_range_or_fabricated_outputs(self) -> None:
        self.fixture.candidate.info_path.write_text("[", encoding="utf-8")
        vlm = self._success_vlm()
        with patch("robometanorm.pipeline.collect_mapped_gripper_ranges") as ranges:
            result = self._normalize(vlm)[0]

        self.assertEqual(result.status, DatasetStatus.ERROR)
        self.assertIsNone(result.source_info)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
        ranges.assert_not_called()
        meta = self.fixture.candidate.info_path.parent
        self.assertFalse((meta / "info_norm.json").exists())
        self.assertFalse((meta / "info_norm_review.json").exists())

    def test_linked_info_is_error_with_zero_outputs_and_no_vlm_content_leak(self) -> None:
        sentinel = "OUTSIDE-INFO-SENTINEL"
        outside_info = self.root / "outside-info.json"
        outside_info.write_text(
            json.dumps({"robot_type": sentinel, "features": {}}),
            encoding="utf-8",
        )
        self.fixture.candidate.info_path.unlink()
        self.fixture.candidate.info_path.symlink_to(outside_info)
        vlm = self._success_vlm()

        result = self._normalize(vlm)[0]

        self.assertEqual(result.status, DatasetStatus.ERROR)
        self.assertIsNone(result.source_info)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
        self.assertNotIn(sentinel, repr(result))
        meta = self.fixture.candidate.info_path.parent
        self.assertFalse((meta / "info_norm.json").exists())
        self.assertFalse((meta / "info_norm_review.json").exists())

    def test_explicit_linked_root_is_resolved_as_the_user_trust_anchor(self) -> None:
        linked_root = self.root.parent / f"{self.root.name}-linked"
        linked_root.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(linked_root.unlink, missing_ok=True)
        vlm = self._success_vlm()
        from robometanorm.pipeline import normalize_datasets

        with self._media_stubs():
            result = normalize_datasets(
                linked_root,
                vlm=vlm,
                confidence_threshold=0.85,
            )[0]

        self.assertEqual(result.status, DatasetStatus.PASS)
        self.assertEqual(result.candidate.source_path, self.fixture.candidate.source_path)
        self.assertIsNotNone(result.source_info)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (1, 1))
        meta = self.fixture.candidate.info_path.parent
        self.assertTrue((meta / "info_norm.json").is_file())
        self.assertTrue((meta / "info_norm_review.json").is_file())

    def test_linked_optional_identity_is_review_with_two_safe_outputs(self) -> None:
        sentinel = "OUTSIDE-OPTIONAL-IDENTITY-SENTINEL"
        outside_common = self.root / "outside-common.json"
        outside_common.write_text(json.dumps({"secret": sentinel}), encoding="utf-8")
        common_path = self.fixture.candidate.info_path.parent / "common_record.json"
        common_path.symlink_to(outside_common)
        vlm = self._success_vlm()

        with patch.object(
            vlm,
            "research_hardware",
            wraps=vlm.research_hardware,
        ) as research:
            result = self._normalize(vlm)[0]

        self.assertEqual(result.status, DatasetStatus.REVIEW)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (1, 1))
        self.assertNotIn(sentinel, repr(research.call_args.args[0]))
        normalized = self._read_output(self.fixture, "info_norm.json")
        review = self._read_output(self.fixture, "info_norm_review.json")
        self.assertIsInstance(normalized, dict)
        self.assertIn("COMMON_RECORD_UNREADABLE", {item["code"] for item in review["issues"]})
        self.assertNotIn(sentinel, json.dumps(review, ensure_ascii=False))

    def test_optional_identity_invalid_forms_are_review_with_exact_two_outputs(self) -> None:
        meta = self.fixture.candidate.info_path.parent
        common_path = meta / "common_record.json"
        tasks_path = meta / "tasks.jsonl"
        outside_common = self.root / "outside-common-matrix.json"
        outside_tasks = self.root / "outside-tasks-matrix.jsonl"
        sentinel = "OUTSIDE-OPTIONAL-MATRIX-SENTINEL"
        outside_common.write_text(json.dumps({"secret": sentinel}), encoding="utf-8")
        outside_tasks.write_text(json.dumps({"task": sentinel}) + "\n", encoding="utf-8")
        local_limit = 4096
        oversized = b"x" * (local_limit + 1)

        def write_common(payload: bytes) -> None:
            common_path.write_bytes(payload)

        def write_tasks(payload: bytes) -> None:
            tasks_path.write_bytes(payload)

        cases = (
            ("common_nan", lambda: write_common(b'{"value":NaN}'), "COMMON_RECORD_INVALID"),
            ("tasks_nan", lambda: write_tasks(b'{"task":"ok"}\n{"value":Infinity}\n'), "TASKS_INVALID"),
            ("common_oversize", lambda: write_common(oversized), "COMMON_RECORD_INVALID"),
            ("tasks_oversize", lambda: write_tasks(oversized), "TASKS_INVALID"),
            ("common_link", lambda: common_path.symlink_to(outside_common), "COMMON_RECORD_UNREADABLE"),
            ("tasks_link", lambda: tasks_path.symlink_to(outside_tasks), "TASKS_UNREADABLE"),
        )

        for label, prepare, expected_code in cases:
            with self.subTest(label=label):
                for path in (common_path, tasks_path, meta / "info_norm.json", meta / "info_norm_review.json"):
                    path.unlink(missing_ok=True)
                prepare()
                vlm = self._success_vlm()
                with (
                    patch(
                        "robometanorm.evidence._LOCAL_JSON_BYTE_LIMIT",
                        local_limit,
                    ),
                    patch.object(
                        vlm,
                        "research_hardware",
                        wraps=vlm.research_hardware,
                    ) as research,
                ):
                    result = self._normalize(vlm)[0]

                self.assertEqual(result.status, DatasetStatus.REVIEW)
                self.assertTrue((meta / "info_norm.json").is_file())
                self.assertTrue((meta / "info_norm_review.json").is_file())
                review = self._read_output(self.fixture, "info_norm_review.json")
                self.assertIn(expected_code, {item["code"] for item in review["issues"]})
                self.assertNotIn(sentinel, repr(research.call_args.args[0]))
                self.assertNotIn(sentinel, json.dumps(review, ensure_ascii=False))

    def test_strict_info_failures_are_error_with_zero_outputs(self) -> None:
        meta = self.fixture.candidate.info_path.parent
        payloads = (
            b'{"robot_type":NaN}',
            b'{"robot_type":"\\ud800"}',
            b"x" * 129,
        )
        for payload in payloads:
            with self.subTest(payload=payload[:24]):
                self.fixture.candidate.info_path.unlink(missing_ok=True)
                self.fixture.candidate.info_path.write_bytes(payload)
                for name in ("info_norm.json", "info_norm_review.json"):
                    (meta / name).unlink(missing_ok=True)
                vlm = self._success_vlm()

                with patch("robometanorm.evidence._LOCAL_JSON_BYTE_LIMIT", 128):
                    result = self._normalize(vlm)[0]

                self.assertEqual(result.status, DatasetStatus.ERROR)
                self.assertIsNone(result.source_info)
                self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
                self.assertFalse((meta / "info_norm.json").exists())
                self.assertFalse((meta / "info_norm_review.json").exists())

    def test_linked_data_root_never_reaches_pyarrow_vlm_or_changes_names(self) -> None:
        data_path = self.fixture.candidate.data_path
        assert data_path is not None
        outside_data = self.root / "outside-data"
        data_path.rename(outside_data)
        data_path.symlink_to(outside_data, target_is_directory=True)
        vlm = self._success_vlm()

        with patch(
            "robometanorm.evidence.pq.ParquetFile",
            side_effect=AssertionError("linked data reached PyArrow"),
        ) as parquet_file:
            result = self._normalize(vlm)[0]

        parquet_file.assert_not_called()
        self.assertEqual(result.status, DatasetStatus.BLOCKED)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
        self.assertEqual(
            self._read_output(self.fixture, "info_norm.json"),
            self._source_info(),
        )
        review = self._read_output(self.fixture, "info_norm_review.json")
        self.assertIn("PARQUET_PATH_UNSAFE", {item["code"] for item in review["issues"]})

    def test_linked_video_root_never_reaches_tools_vlm_or_changes_names(self) -> None:
        video_path = self.fixture.candidate.video_path
        assert video_path is not None
        outside_videos = self.root / "outside-videos"
        video_path.rename(outside_videos)
        video_path.symlink_to(outside_videos, target_is_directory=True)
        vlm = self._success_vlm()
        from robometanorm.pipeline import normalize_datasets

        with (
            patch(
                "robometanorm.evidence.probe_media",
                side_effect=AssertionError("linked video reached ffprobe"),
            ) as probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=AssertionError("linked video reached ffmpeg"),
            ) as extract,
        ):
            result = normalize_datasets(
                self.root,
                vlm=vlm,
                confidence_threshold=0.85,
            )[0]

        probe.assert_not_called()
        extract.assert_not_called()
        self.assertEqual(result.status, DatasetStatus.BLOCKED)
        self.assertEqual((vlm.research_calls, vlm.mapping_calls), (0, 0))
        self.assertEqual(
            self._read_output(self.fixture, "info_norm.json"),
            self._source_info(),
        )
        review = self._read_output(self.fixture, "info_norm_review.json")
        self.assertIn("MEDIA_PATH_UNSAFE", {item["code"] for item in review["issues"]})

    def test_writer_failure_is_not_retried_and_next_dataset_still_completes(self) -> None:
        second = self._create_dataset("dataset-b")
        vlm = self._success_vlm()
        from robometanorm.writer import write_outputs as real_write

        calls = 0

        def fail_first(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("secret disk path")
            return real_write(*args, **kwargs)

        with patch("robometanorm.pipeline.write_outputs", side_effect=fail_first):
            results = self._normalize(vlm)

        self.assertEqual(calls, 2)
        self.assertEqual(
            [result.status for result in results],
            [DatasetStatus.ERROR, DatasetStatus.PASS],
        )
        self.assertIsNotNone(results[0].source_info)
        self.assertTrue(
            (second.candidate.info_path.parent / "info_norm_review.json").is_file()
        )

    def test_research_error_keeps_evidence_issue_in_error_count(self) -> None:
        meta = self.fixture.candidate.info_path.parent
        (meta / "common_record.json").write_text("{", encoding="utf-8")

        result = self._normalize(_RaisingVlm(OSError("offline")))[0]

        self.assertEqual(result.status, DatasetStatus.ERROR)
        self.assertEqual((result.changed_field_count, result.issue_count), (0, 2))
        self.assertFalse((meta / "info_norm.json").exists())
        self.assertFalse((meta / "info_norm_review.json").exists())

    def test_scan_precondition_error_keeps_evidence_issue_in_error_count(self) -> None:
        meta = self.fixture.candidate.info_path.parent
        (meta / "common_record.json").write_text("{", encoding="utf-8")

        with patch(
            "robometanorm.pipeline.check_preconditions",
            side_effect=OSError("precondition failed"),
        ):
            result = self._scan()[0]

        self.assertEqual(result.status, DatasetStatus.ERROR)
        self.assertEqual((result.changed_field_count, result.issue_count), (0, 2))
        self.assertFalse((meta / "info_norm.json").exists())
        self.assertFalse((meta / "info_norm_review.json").exists())

    def test_partial_writer_failure_preserves_known_normalization_changes(self) -> None:
        import os

        meta = self.fixture.candidate.info_path.parent
        real_replace = os.replace
        replace_calls = 0

        def fail_second_replace(*args: object, **kwargs: object) -> None:
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise OSError("review replace failed")
            real_replace(*args, **kwargs)

        with patch("robometanorm.writer.os.replace", side_effect=fail_second_replace):
            result = self._normalize(self._success_vlm())[0]

        self.assertEqual(replace_calls, 2)
        self.assertEqual(result.status, DatasetStatus.ERROR)
        self.assertEqual((result.changed_field_count, result.issue_count), (4, 1))
        normalized = self._read_output(self.fixture, "info_norm.json")
        self.assertEqual(normalized["robot_type"], "acme_robotics_testbot_one")
        features = normalized["features"]
        self.assertIn("observation.images.cam_front_wrist_rgb", features)
        expected_names = [f"left_arm_joint_{index}_rad" for index in range(6)]
        self.assertEqual(features["action"]["names"], expected_names)
        self.assertEqual(features["observation.state"]["names"], expected_names)
        self.assertFalse((meta / "info_norm_review.json").exists())

    def test_issue_order_is_evidence_precondition_vlm_range_then_standard(self) -> None:
        (self.fixture.candidate.info_path.parent / "common_record.json").write_text(
            "{", encoding="utf-8"
        )
        precondition = Issue("PRECONDITION_REVIEW", "review", "preconditions")
        range_issue = Issue("RANGE_REVIEW", "range", "range")
        ambiguous_mapping = replace(
            self.mapping,
            machines=(replace(self.mapping.machines[0], ambiguous=True), *self.mapping.machines[1:]),
        )
        vlm = FakeVlm(
            research_result=(self.profile, None),
            mapping_result=(ambiguous_mapping, None),
        )

        def range_result(candidate: object, evidence: object, profile: object, mapping: object):
            return evidence, (range_issue,)

        with (
            patch("robometanorm.pipeline.check_preconditions", return_value=(precondition,)),
            patch(
                "robometanorm.pipeline.collect_mapped_gripper_ranges",
                side_effect=range_result,
            ),
        ):
            self._normalize(vlm)

        codes = [
            item["code"]
            for item in self._read_output(self.fixture, "info_norm_review.json")[
                "issues"
            ]
        ]
        self.assertLess(codes.index("COMMON_RECORD_INVALID"), codes.index("PRECONDITION_REVIEW"))
        self.assertLess(codes.index("PRECONDITION_REVIEW"), codes.index("RANGE_REVIEW"))
        self.assertLess(codes.index("RANGE_REVIEW"), codes.index("MACHINE_MAPPING_INVALID"))

    def test_generator_is_fixed_and_only_package_not_found_is_suppressed(self) -> None:
        with patch("robometanorm.pipeline.importlib.metadata.version", return_value="9.9"):
            self._normalize(self._success_vlm())
        review = self._read_output(self.fixture, "info_norm_review.json")
        self.assertEqual(
            review["generator"],
            {"name": "robometanorm", "version": "9.9"},
        )
        self.assertNotIn("endpoint", json.dumps(review["generator"]))

        with patch(
            "robometanorm.pipeline.importlib.metadata.version",
            side_effect=RuntimeError("metadata secret"),
        ):
            with self.assertRaises(RuntimeError):
                self._normalize(self._success_vlm())

    def test_memory_and_process_control_exceptions_propagate(self) -> None:
        for error in (MemoryError("memory"), SystemExit(7), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    self._normalize(_RaisingVlm(error))


if __name__ == "__main__":
    unittest.main()
