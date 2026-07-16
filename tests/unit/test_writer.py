"""Strict mini review payload and two-file writer contract tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.models import (
    DatasetCandidate,
    DatasetEvidence,
    DatasetStatus,
    HardwareProfile,
    IdentityEvidence,
    Issue,
    MappingRecord,
    NormalizationResult,
)
from robometanorm.writer import build_review_payload, write_outputs
from tests.mini_fixtures import DatasetFixture, PipelineFixture


_REVIEW_KEYS = (
    "schema_version",
    "generator",
    "dataset",
    "status",
    "robot_identity",
    "camera_mappings",
    "machine_mappings",
    "issues",
)
_RECORD_KEYS = (
    "source_address",
    "source",
    "output",
    "candidate",
    "changed",
    "vlm_semantics",
    "citations",
    "decision",
    "reason",
)
_ISSUE_KEYS = ("code", "message", "scope", "evidence", "severity")


@dataclass(frozen=True)
class _NotJsonDataclass:
    value: str


class _NotJsonEnum(str, Enum):
    VALUE = "value"


class MiniWriterTest(unittest.TestCase):
    """Verify exact schemas, deterministic bytes, and fail-closed replacement."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.fixture = DatasetFixture.create(
            self.root,
            dataset_name="unicode_dataset",
            info={
                "robot_type": "原始机器人",
                "features": {
                    "observation.images.raw": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "fps": 20,
                        "codec": "h264",
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [2],
                        "names": ["raw_0", "raw_1"],
                    },
                },
            },
        )
        self.candidate = self.fixture.candidate
        self.meta = self.candidate.info_path.parent
        self.identity = IdentityEvidence(
            info_robot_type_state="present",
            info_robot_type="原始机器人",
            common_record_state="present",
            common_record={"vendor": "虚构厂商", "serial": 7},
            tasks_state="present",
            tasks=({"task": "抓取", "rank": 1}, ["放置", 2]),
        )
        self.evidence = DatasetEvidence(
            candidate=self.candidate,
            source_info=deepcopy(self.fixture.info),
            identity=self.identity,
            cameras=(),
            machines=(),
        )
        citations = (
            {
                "source_id": "official-manual",
                "title": "虚构设备手册",
                "url": "https://fixtures.invalid/manual",
                "kind": "official_manual",
            },
        )
        self.identity_record = MappingRecord(
            source_address="robot_type",
            source="原始机器人",
            output="acme_testbot_one",
            candidate="acme_testbot_one",
            changed=True,
            vlm_semantics={
                "manufacturer": "Acme Robotics",
                "model": "TestBot One",
                "confidence": 0.95,
            },
            citations=citations,
            decision="apply",
            reason="本地证据与官方资料一致",
        )
        self.camera_record = MappingRecord(
            source_address="features.observation.images.raw",
            source={"codec": "h264", "fps": 20},
            output={"codec": "h264", "fps": 20},
            candidate="observation.images.cam_front_head_rgb",
            changed=False,
            vlm_semantics={"camera_id": "camera-head", "confidence": 0.4},
            citations=citations,
            decision="review",
            reason="相机置信度低于门槛，保持原字段",
        )
        self.machine_record = MappingRecord(
            source_address="features.action.names",
            source=["raw_0", "raw_1"],
            output=["left_arm_joint_0_rad", "left_arm_joint_1_rad"],
            candidate=["left_arm_joint_0_rad", "left_arm_joint_1_rad"],
            changed=True,
            vlm_semantics={
                "source_feature": "action",
                "slices": [
                    {
                        "start": 0,
                        "end": 2,
                        "component_id": "left-arm",
                        "element_order": ["joint_1", "joint_2"],
                    }
                ],
            },
            citations=citations,
            decision="apply",
            reason="完整顺序已确认",
        )
        self.issues = (
            Issue(
                "CAMERA_MAPPING_UNRESOLVED",
                "相机保持原字段",
                "features.observation.images.raw",
                {"attempt": 1},
                "review",
            ),
            Issue(
                "SECOND_ISSUE",
                "第二条问题",
                "dataset",
                {"ordered": [1, 2]},
                "block",
            ),
        )
        self.result = NormalizationResult(
            normalized_info={
                "robot_type": "acme_testbot_one",
                "features": deepcopy(self.fixture.info["features"]),
            },
            robot_identity=self.identity_record,
            camera_mappings=(self.camera_record,),
            machine_mappings=(self.machine_record,),
            issues=self.issues,
        )
        self.generator = {"name": "robometanorm", "version": "测试版"}
        self.profile = PipelineFixture().hardware_profile()
        self.review = self._build_review()

    def _build_review(
        self,
        *,
        evidence: DatasetEvidence | None = None,
        profile: HardwareProfile | None = None,
        result: NormalizationResult | None = None,
        generator: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return build_review_payload(
            self.candidate,
            DatasetStatus.REVIEW,
            evidence if evidence is not None else self.evidence,
            self.profile if profile is None else profile,
            result if result is not None else self.result,
            generator=self.generator if generator is None else generator,
        )

    def _output_paths(self) -> tuple[Path, Path]:
        return self.meta / "info_norm.json", self.meta / "info_norm_review.json"

    def _seed_existing_outputs(self) -> tuple[bytes, bytes]:
        info_bytes = b'{"old":"info"}\n'
        review_bytes = b'{"old":"review"}\n'
        info_path, review_path = self._output_paths()
        info_path.write_bytes(info_bytes)
        review_path.write_bytes(review_bytes)
        return info_bytes, review_bytes

    def _assert_existing_outputs(self, expected: tuple[bytes, bytes]) -> None:
        info_path, review_path = self._output_paths()
        self.assertEqual(info_path.read_bytes(), expected[0])
        self.assertEqual(review_path.read_bytes(), expected[1])

    def _assert_no_temps(self) -> None:
        self.assertFalse(
            [path.name for path in self.meta.iterdir() if path.name.endswith(".tmp")]
        )

    def test_builds_exact_review_schema_and_preserves_record_and_issue_order(self) -> None:
        payload = self.review

        self.assertEqual(tuple(payload), _REVIEW_KEYS)
        self.assertEqual(payload["schema_version"], "mini-1")
        self.assertEqual(payload["generator"], self.generator)
        self.assertEqual(
            payload["dataset"], {"name": "unicode_dataset", "layout_type": "flat"}
        )
        self.assertEqual(payload["status"], "REVIEW")

        identity = payload["robot_identity"]
        self.assertEqual(tuple(identity), ("local_evidence", *_RECORD_KEYS))
        self.assertEqual(
            identity["local_evidence"],
            {
                "info_robot_type": {"state": "present", "value": "原始机器人"},
                "common_record": {
                    "state": "present",
                    "value": {"vendor": "虚构厂商", "serial": 7},
                },
                "tasks": {
                    "state": "present",
                    "records": [{"task": "抓取", "rank": 1}, ["放置", 2]],
                },
            },
        )
        self.assertEqual(identity["output"], "acme_testbot_one")
        self.assertEqual(identity["citations"][0]["url"], "https://fixtures.invalid/manual")

        self.assertEqual(len(payload["camera_mappings"]), 1)
        self.assertEqual(tuple(payload["camera_mappings"][0]), _RECORD_KEYS)
        self.assertEqual(payload["camera_mappings"][0]["output"], {"codec": "h264", "fps": 20})
        self.assertEqual(
            payload["camera_mappings"][0]["candidate"],
            "observation.images.cam_front_head_rgb",
        )
        self.assertFalse(payload["camera_mappings"][0]["changed"])
        self.assertEqual(tuple(payload["machine_mappings"][0]), _RECORD_KEYS)
        self.assertEqual(
            [item["code"] for item in payload["issues"]],
            ["CAMERA_MAPPING_UNRESOLVED", "SECOND_ISSUE"],
        )
        self.assertTrue(all(tuple(item) == _ISSUE_KEYS for item in payload["issues"]))

    def test_build_payload_does_not_serialize_the_hardware_profile(self) -> None:
        sentinel_profile = replace(
            self.profile,
            sources=(replace(self.profile.sources[0], title="PROFILE_ONLY_SENTINEL"),),
        )

        payload = self._build_review(profile=sentinel_profile)

        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("PROFILE_ONLY_SENTINEL", serialized)
        self.assertNotIn("hardware_profile", payload)

    def test_build_payload_copies_inputs_instead_of_mutating_or_aliasing_them(self) -> None:
        generator_before = deepcopy(self.generator)
        result_before = deepcopy(self.result)
        evidence_before = deepcopy(self.evidence)

        payload = self._build_review()
        payload["generator"]["version"] = "changed"
        payload["robot_identity"]["local_evidence"]["common_record"]["value"]["serial"] = 999
        payload["camera_mappings"][0]["source"]["codec"] = "changed"

        self.assertEqual(self.generator, generator_before)
        self.assertEqual(self.result, result_before)
        self.assertEqual(self.evidence, evidence_before)

    def test_build_payload_rejects_non_json_native_local_evidence_safely(self) -> None:
        bad_identity = replace(self.identity, common_record=Path("secret-path"))
        bad_evidence = replace(self.evidence, identity=bad_identity)

        with self.assertRaisesRegex(ValueError, "^输出不是合法 JSON$"):
            self._build_review(evidence=bad_evidence)

    def test_writes_exactly_two_outputs_and_hashes_the_exact_info_bytes(self) -> None:
        before = {path.name for path in self.meta.iterdir()}

        paths = write_outputs(self.candidate, self.result.normalized_info, self.review)

        info_path, review_path = self._output_paths()
        self.assertEqual(paths, (info_path, review_path))
        after = {path.name for path in self.meta.iterdir()}
        self.assertEqual(after - before, {"info_norm.json", "info_norm_review.json"})
        review = json.loads(review_path.read_text(encoding="utf-8"))
        self.assertEqual(
            review["info_norm_sha256"],
            "sha256:" + hashlib.sha256(info_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            set(review), {*_REVIEW_KEYS, "info_norm_sha256"}
        )
        self._assert_no_temps()

    def test_writes_annotation_only_when_explicitly_supplied(self) -> None:
        annotation = {
            "version": "dataset_annotation_config_v1",
            "robot_type": "acme_robotics_testbot_one",
            "adapter": {"base_type": "LeRobot", "base": {}},
            "robot_channel_schema": {"version": "channel_schema_v1", "channels": {}},
        }

        paths = write_outputs(
            self.candidate,
            self.result.normalized_info,
            self.review,
            annotation=annotation,
        )

        annotation_path = self.meta / "robo_annotation.yaml"
        self.assertEqual(paths, (*self._output_paths(), annotation_path))
        self.assertEqual(yaml.safe_load(annotation_path.read_text(encoding="utf-8")), annotation)

    def test_none_annotation_keeps_exact_two_outputs(self) -> None:
        paths = write_outputs(
            self.candidate,
            self.result.normalized_info,
            self.review,
            annotation=None,
        )

        self.assertEqual(paths, self._output_paths())
        self.assertFalse((self.meta / "robo_annotation.yaml").exists())

    def test_emits_sorted_unicode_utf8_with_two_space_indent_and_one_newline(self) -> None:
        info = {"z_key": "中文", "a_key": {"z": 2, "a": 1}}

        info_path, review_path = write_outputs(self.candidate, info, self.review)

        expected_info = (
            json.dumps(
                info,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(info_path.read_bytes(), expected_info)
        self.assertIn("中文".encode("utf-8"), info_path.read_bytes())
        self.assertTrue(review_path.read_bytes().endswith(b"\n"))
        self.assertFalse(review_path.read_bytes().endswith(b"\n\n"))

    def test_serializes_info_once_and_review_once(self) -> None:
        real_dumps = json.dumps
        with patch("robometanorm.writer.json.dumps", side_effect=real_dumps) as dumps:
            write_outputs(self.candidate, self.result.normalized_info, self.review)

        self.assertEqual(dumps.call_count, 2)

    def test_write_outputs_does_not_mutate_info_or_review_inputs(self) -> None:
        info = deepcopy(self.result.normalized_info)
        review = deepcopy(self.review)
        info_before = deepcopy(info)
        review_before = deepcopy(review)

        write_outputs(self.candidate, info, review)

        self.assertEqual(info, info_before)
        self.assertEqual(review, review_before)
        self.assertNotIn("info_norm_sha256", review)

    def test_rejects_every_non_json_native_value_before_replacement(self) -> None:
        cycle: list[object] = []
        cycle.append(cycle)
        invalid_values: tuple[object, ...] = (
            object(),
            _NotJsonDataclass("value"),
            _NotJsonEnum.VALUE,
            Path("relative/path"),
            ("tuple",),
            {"set"},
            b"bytes",
            math.nan,
            math.inf,
            -math.inf,
            cycle,
            "\ud800",
        )
        expected = self._seed_existing_outputs()
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__):
                with patch("robometanorm.writer.os.replace") as replace_call:
                    with self.assertRaisesRegex(ValueError, "^输出不是合法 JSON$"):
                        write_outputs(self.candidate, {"bad": value}, self.review)
                replace_call.assert_not_called()
                self._assert_existing_outputs(expected)
                self._assert_no_temps()

    def test_rejects_non_string_json_object_keys_before_replacement(self) -> None:
        expected = self._seed_existing_outputs()

        with patch("robometanorm.writer.os.replace") as replace_call:
            with self.assertRaisesRegex(ValueError, "^输出不是合法 JSON$"):
                write_outputs(self.candidate, {1: "bad-key"}, self.review)

        replace_call.assert_not_called()
        self._assert_existing_outputs(expected)

    def test_rejects_non_object_info_before_replacement(self) -> None:
        expected = self._seed_existing_outputs()
        for info in ([], "object", None):
            with self.subTest(info=info):
                with patch("robometanorm.writer.os.replace") as replace_call:
                    with self.assertRaisesRegex(ValueError, "^输出不是合法 JSON$"):
                        write_outputs(self.candidate, info, self.review)
                replace_call.assert_not_called()
                self._assert_existing_outputs(expected)

    def test_rejects_invalid_top_level_review_schema_before_replacement(self) -> None:
        invalid_reviews = (
            {key: value for key, value in self.review.items() if key != "issues"},
            {**self.review, "extra": True},
            {**self.review, "info_norm_sha256": "sha256:premature"},
            {**self.review, "schema_version": "mini-2"},
            {**self.review, "status": "UNKNOWN"},
            {**self.review, "generator": []},
            {**self.review, "dataset": {"name": "unicode_dataset"}},
            {**self.review, "camera_mappings": ()},
            {**self.review, "machine_mappings": {}},
            {**self.review, "issues": ()},
        )
        expected = self._seed_existing_outputs()
        for payload in invalid_reviews:
            with self.subTest(keys=tuple(payload)):
                with patch("robometanorm.writer.os.replace") as replace_call:
                    with self.assertRaisesRegex(ValueError, "^输出不是合法 JSON$"):
                        write_outputs(self.candidate, self.result.normalized_info, payload)
                replace_call.assert_not_called()
                self._assert_existing_outputs(expected)

    def test_rejects_invalid_nested_review_schema_before_replacement(self) -> None:
        missing_record_key = deepcopy(self.review)
        missing_record_key["camera_mappings"][0].pop("reason")
        extra_record_key = deepcopy(self.review)
        extra_record_key["machine_mappings"][0]["extra"] = True
        wrong_changed_type = deepcopy(self.review)
        wrong_changed_type["robot_identity"]["changed"] = 0
        extra_identity_key = deepcopy(self.review)
        extra_identity_key["robot_identity"]["extra"] = True
        missing_local_source = deepcopy(self.review)
        missing_local_source["robot_identity"]["local_evidence"].pop("tasks")
        missing_issue_key = deepcopy(self.review)
        missing_issue_key["issues"][0].pop("severity")
        extra_issue_key = deepcopy(self.review)
        extra_issue_key["issues"][0]["extra"] = True
        bad_citation = deepcopy(self.review)
        bad_citation["camera_mappings"][0]["citations"][0]["extra"] = True
        invalid_reviews = (
            missing_record_key,
            extra_record_key,
            wrong_changed_type,
            extra_identity_key,
            missing_local_source,
            missing_issue_key,
            extra_issue_key,
            bad_citation,
        )
        expected = self._seed_existing_outputs()
        for payload in invalid_reviews:
            with self.subTest(payload=payload):
                with patch("robometanorm.writer.os.replace") as replace_call:
                    with self.assertRaisesRegex(ValueError, "^输出不是合法 JSON$"):
                        write_outputs(self.candidate, self.result.normalized_info, payload)
                replace_call.assert_not_called()
                self._assert_existing_outputs(expected)

    def test_missing_meta_or_info_and_mismatched_info_path_are_rejected(self) -> None:
        mismatched = replace(self.candidate, info_path=self.meta / "other.json")
        missing_source = self.root / "missing-dataset"
        missing = replace(
            self.candidate,
            source_path=missing_source,
            info_path=missing_source / "meta" / "info.json",
        )
        for candidate in (mismatched, missing):
            with self.subTest(candidate=candidate):
                with patch("robometanorm.writer.os.replace") as replace_call:
                    with self.assertRaises(ValueError):
                        write_outputs(candidate, self.result.normalized_info, self.review)
                replace_call.assert_not_called()

    def test_parent_alias_is_rejected_even_when_it_resolves_to_the_dataset(self) -> None:
        alias_directory = self.candidate.source_path / "alias"
        alias_directory.mkdir()
        aliased_source = self.candidate.source_path / "alias" / ".."
        candidate = replace(
            self.candidate,
            source_path=aliased_source,
            info_path=aliased_source / "meta" / "info.json",
        )

        with patch("robometanorm.writer.os.replace") as replace_call:
            with self.assertRaises(ValueError):
                write_outputs(candidate, self.result.normalized_info, self.review)

        replace_call.assert_not_called()

    def test_symlinked_meta_directory_is_rejected(self) -> None:
        dataset = self.root / "symlink-meta-dataset"
        dataset.mkdir()
        real_meta = self.root / "real-meta"
        real_meta.mkdir()
        (real_meta / "info.json").write_text("{}", encoding="utf-8")
        (dataset / "meta").symlink_to(real_meta, target_is_directory=True)
        candidate = replace(
            self.candidate,
            source_path=dataset,
            info_path=dataset / "meta" / "info.json",
        )

        with patch("robometanorm.writer.os.replace") as replace_call:
            with self.assertRaises(ValueError):
                write_outputs(candidate, {}, self.review)

        replace_call.assert_not_called()

    def test_symlinked_or_nonregular_info_file_is_rejected(self) -> None:
        candidates: list[DatasetCandidate] = []
        for suffix, make_info in (
            ("symlink", "symlink"),
            ("directory", "directory"),
        ):
            dataset = self.root / f"bad-info-{suffix}"
            meta = dataset / "meta"
            meta.mkdir(parents=True)
            info_path = meta / "info.json"
            if make_info == "symlink":
                target = self.root / f"target-{suffix}.json"
                target.write_text("{}", encoding="utf-8")
                info_path.symlink_to(target)
            else:
                info_path.mkdir()
            candidates.append(
                replace(
                    self.candidate,
                    source_path=dataset,
                    info_path=info_path,
                )
            )

        for candidate in candidates:
            with self.subTest(candidate=candidate):
                with patch("robometanorm.writer.os.replace") as replace_call:
                    with self.assertRaises(ValueError):
                        write_outputs(candidate, {}, self.review)
                replace_call.assert_not_called()

    def test_symlinked_or_nonregular_existing_target_is_rejected(self) -> None:
        info_target, review_target = self._output_paths()
        outside = self.root / "outside.json"
        outside.write_text("outside", encoding="utf-8")
        info_target.symlink_to(outside)

        with patch("robometanorm.writer.os.replace") as replace_call:
            with self.assertRaises(ValueError):
                write_outputs(self.candidate, {}, self.review)
        replace_call.assert_not_called()
        info_target.unlink()

        review_target.mkdir()
        with patch("robometanorm.writer.os.replace") as replace_call:
            with self.assertRaises(ValueError):
                write_outputs(self.candidate, {}, self.review)
        replace_call.assert_not_called()

    def test_meta_swap_after_path_validation_cannot_write_original_or_outside(self) -> None:
        from robometanorm import writer

        original_meta = self.meta
        moved_meta = self.candidate.source_path / "meta-before-swap"
        outside = self.root / "outside-meta"
        outside.mkdir()
        original_validate = writer._validated_output_paths

        def validate_then_swap(candidate: DatasetCandidate) -> tuple[Path, Path, Path]:
            validated = original_validate(candidate)
            original_meta.rename(moved_meta)
            original_meta.symlink_to(outside, target_is_directory=True)
            return validated

        with patch(
            "robometanorm.writer._validated_output_paths",
            side_effect=validate_then_swap,
        ):
            with self.assertRaises((OSError, ValueError)):
                write_outputs(
                    self.candidate,
                    self.result.normalized_info,
                    self.review,
                )

        self.assertEqual(
            {path.name for path in moved_meta.iterdir()},
            {"info.json"},
        )
        self.assertEqual(list(outside.iterdir()), [])

    def test_meta_swap_after_info_replace_cannot_write_review_to_moved_directory(self) -> None:
        original_meta = self.meta
        moved_meta = self.candidate.source_path / "meta-after-info-replace"
        real_replace = os.replace
        replacements = 0

        def replace_then_swap(source: object, target: object, **kwargs: object) -> None:
            nonlocal replacements
            replacements += 1
            real_replace(source, target, **kwargs)
            if replacements == 1:
                original_meta.rename(moved_meta)
                original_meta.mkdir()

        with patch("robometanorm.writer.os.replace", side_effect=replace_then_swap):
            with self.assertRaises(ValueError):
                write_outputs(self.candidate, self.result.normalized_info, self.review)

        self.assertIn("info_norm.json", {path.name for path in moved_meta.iterdir()})
        self.assertNotIn(
            "info_norm_review.json", {path.name for path in moved_meta.iterdir()}
        )
        self.assertEqual(list(original_meta.iterdir()), [])
        self._assert_no_temps()

    def test_first_temp_creation_failure_replaces_nothing(self) -> None:
        expected = self._seed_existing_outputs()
        real_open = os.open

        def fail_temp_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
            if flags & os.O_CREAT:
                raise OSError("fixture create failure")
            return real_open(path, flags, *args, **kwargs)

        with (
            patch(
                "robometanorm.writer.os.open",
                side_effect=fail_temp_open,
            ),
            patch("robometanorm.writer.os.replace") as replace_call,
        ):
            with self.assertRaises(OSError):
                write_outputs(self.candidate, self.result.normalized_info, self.review)

        replace_call.assert_not_called()
        self._assert_existing_outputs(expected)
        self._assert_no_temps()

    def test_first_temp_write_failure_is_cleaned_and_replaces_nothing(self) -> None:
        expected = self._seed_existing_outputs()

        with (
            patch(
                "robometanorm.writer.os.write",
                side_effect=OSError("fixture write failure"),
            ),
            patch("robometanorm.writer.os.replace") as replace_call,
        ):
            with self.assertRaises(OSError):
                write_outputs(self.candidate, self.result.normalized_info, self.review)

        replace_call.assert_not_called()
        self._assert_existing_outputs(expected)
        self._assert_no_temps()

    def test_first_temp_fsync_failure_is_cleaned_and_replaces_nothing(self) -> None:
        expected = self._seed_existing_outputs()

        with (
            patch("robometanorm.writer.os.fsync", side_effect=OSError("fixture fsync failure")),
            patch("robometanorm.writer.os.replace") as replace_call,
        ):
            with self.assertRaises(OSError):
                write_outputs(self.candidate, self.result.normalized_info, self.review)

        replace_call.assert_not_called()
        self._assert_existing_outputs(expected)
        self._assert_no_temps()

    def test_second_temp_creation_failure_cleans_the_first_and_replaces_nothing(self) -> None:
        expected = self._seed_existing_outputs()
        real_open = os.open
        temp_count = 0

        def fail_second_temp(
            path: object,
            flags: int,
            *args: object,
            **kwargs: object,
        ) -> int:
            nonlocal temp_count
            if flags & os.O_CREAT:
                temp_count += 1
                if temp_count == 2:
                    raise OSError("fixture second create failure")
            return real_open(path, flags, *args, **kwargs)

        with (
            patch(
                "robometanorm.writer.os.open",
                side_effect=fail_second_temp,
            ),
            patch("robometanorm.writer.os.replace") as replace_call,
        ):
            with self.assertRaises(OSError):
                write_outputs(self.candidate, self.result.normalized_info, self.review)

        replace_call.assert_not_called()
        self._assert_existing_outputs(expected)
        self._assert_no_temps()

    def test_both_temps_are_fsynced_before_replacements_in_info_then_review_order(self) -> None:
        events: list[tuple[str, str]] = []
        real_fsync = os.fsync
        real_replace = os.replace

        def record_fsync(file_descriptor: int) -> None:
            events.append(("fsync", str(file_descriptor)))
            real_fsync(file_descriptor)

        def record_replace(source: object, target: object, **kwargs: object) -> None:
            events.append(("replace", Path(target).name))
            real_replace(source, target, **kwargs)

        with (
            patch("robometanorm.writer.os.fsync", side_effect=record_fsync),
            patch("robometanorm.writer.os.replace", side_effect=record_replace),
        ):
            write_outputs(self.candidate, self.result.normalized_info, self.review)

        self.assertEqual([event[0] for event in events], ["fsync", "fsync", "replace", "replace"])
        self.assertEqual(
            [event[1] for event in events if event[0] == "replace"],
            ["info_norm.json", "info_norm_review.json"],
        )
        self._assert_no_temps()

    def test_first_replace_failure_preserves_both_old_outputs_and_cleans_temps(self) -> None:
        expected = self._seed_existing_outputs()

        with patch("robometanorm.writer.os.replace", side_effect=OSError("fixture replace failure")):
            with self.assertRaises(OSError):
                write_outputs(self.candidate, self.result.normalized_info, self.review)

        self._assert_existing_outputs(expected)
        self._assert_no_temps()

    def test_second_replace_failure_leaves_new_info_old_review_and_cleans_temps(self) -> None:
        old_info, old_review = self._seed_existing_outputs()
        real_replace = os.replace
        replace_count = 0

        def fail_second(source: object, target: object, **kwargs: object) -> None:
            nonlocal replace_count
            replace_count += 1
            if replace_count == 2:
                raise OSError("fixture second replace failure")
            real_replace(source, target, **kwargs)

        with patch("robometanorm.writer.os.replace", side_effect=fail_second):
            with self.assertRaises(OSError):
                write_outputs(self.candidate, self.result.normalized_info, self.review)

        info_path, review_path = self._output_paths()
        self.assertNotEqual(info_path.read_bytes(), old_info)
        self.assertEqual(review_path.read_bytes(), old_review)
        self._assert_no_temps()

    def test_meta_swap_inside_second_replace_cannot_report_false_success(self) -> None:
        original_meta = self.meta
        moved_meta = self.candidate.source_path / "meta-during-second-replace"
        outside = self.root / "outside-during-second-replace"
        outside.mkdir()
        real_replace = os.replace
        replace_count = 0
        open_fds_before = len(os.listdir("/proc/self/fd"))

        def swap_before_second(
            source: object,
            target: object,
            **kwargs: object,
        ) -> None:
            nonlocal replace_count
            replace_count += 1
            if replace_count == 2:
                original_meta.rename(moved_meta)
                original_meta.symlink_to(outside, target_is_directory=True)
            real_replace(source, target, **kwargs)

        with patch("robometanorm.writer.os.replace", side_effect=swap_before_second):
            with self.assertRaises(ValueError):
                write_outputs(
                    self.candidate,
                    self.result.normalized_info,
                    self.review,
                )

        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse(
            [path.name for path in moved_meta.iterdir() if path.name.endswith(".tmp")]
        )
        self.assertEqual(len(os.listdir("/proc/self/fd")), open_fds_before)

    def test_memory_and_process_control_exceptions_are_not_wrapped_or_swallowed(self) -> None:
        for exception_type in (MemoryError, KeyboardInterrupt, SystemExit):
            with self.subTest(exception_type=exception_type.__name__):
                with (
                    patch(
                        "robometanorm.writer.json.dumps",
                        side_effect=exception_type("fixture"),
                    ),
                    patch("robometanorm.writer.os.replace") as replace_call,
                ):
                    with self.assertRaises(exception_type):
                        write_outputs(
                            self.candidate,
                            self.result.normalized_info,
                            self.review,
                        )
                replace_call.assert_not_called()
                self._assert_no_temps()


if __name__ == "__main__":
    unittest.main()
