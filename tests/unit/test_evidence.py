"""Raw robot identity evidence collection tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.evidence import (
    collect_camera_evidence,
    collect_dataset_evidence,
    collect_identity_evidence,
    collect_mapped_gripper_ranges,
    collect_machine_evidence,
    extract_midpoint_frame,
    probe_media,
    read_info,
)
from robometanorm.models import (
    CameraEvidence,
    DatasetCandidate,
    DatasetEvidence,
    DatasetMapping,
    FeatureSchema,
    HardwareProfile,
    IdentityEvidence,
    Issue,
    LayoutType,
    MachineAssignment,
    MachineComponent,
    MachineEvidence,
    MachineSlice,
    MediaSample,
    RobotIdentityFact,
)


class IdentityEvidenceTest(unittest.TestCase):
    """Verify identity inputs stay raw while parse failures remain reviewable."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.dataset_path = Path(self.temporary_directory.name) / "acme_dataset"
        self.meta_path = self.dataset_path / "meta"
        self.meta_path.mkdir(parents=True)
        self.candidate = DatasetCandidate(
            dataset_name="acme_dataset",
            task_name=None,
            source_path=self.dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=self.meta_path / "info.json",
            data_path=None,
            video_path=None,
            depth_path=None,
        )

    def test_read_info_returns_json_object_without_normalizing_it(self) -> None:
        payload = {
            "robot_type": "  Acme TestBot V2  ",
            "nested": {"labels": ["Mixed Case", None, 3]},
        }
        self._write_json("info.json", payload)

        source_info = read_info(self.candidate)

        self.assertEqual(source_info, payload)
        self.assertEqual(source_info["robot_type"], "  Acme TestBot V2  ")

    def test_read_info_rejects_non_object_json_values(self) -> None:
        for payload in (["Acme"], "Acme", 4, None):
            with self.subTest(payload=payload):
                self._write_json("info.json", payload)

                with self.assertRaises(ValueError):
                    read_info(self.candidate)

    def test_read_info_rejects_malformed_json(self) -> None:
        self.candidate.info_path.write_text(
            '{"robot_type": "Acme TestBot"', encoding="utf-8"
        )

        with self.assertRaises(ValueError):
            read_info(self.candidate)

    def test_read_info_converts_deep_json_recursion_to_safe_value_error(self) -> None:
        deep_content = "[" * 2_000 + "0" + "]" * 2_000
        self.candidate.info_path.write_text(deep_content, encoding="utf-8")

        with self.assertRaises(ValueError) as caught:
            read_info(self.candidate)

        self.assertIs(type(caught.exception), ValueError)
        self.assertNotIn("[" * 32, str(caught.exception))
        self.assertNotIn(str(self.candidate.info_path), str(caught.exception))

    def test_collects_all_three_present_sources_verbatim(self) -> None:
        source_info = {"robot_type": "  Acme_TestBot V2  ", "untouched": True}
        common_record = ["Acme raw record", {"serial": 7}, None]
        task_records: tuple[object, ...] = (
            {"task": "Acme pick", "model_hint": "TestBot V2"},
            ["raw", 2, None],
            "Acme scalar task record",
        )
        self._write_json("common_record.json", common_record)
        self._write_task_records(*task_records)

        evidence = collect_identity_evidence(self.meta_path, source_info)

        self.assertEqual(evidence.info_robot_type_state, "present")
        self.assertEqual(evidence.info_robot_type, "  Acme_TestBot V2  ")
        self.assertEqual(evidence.common_record_state, "present")
        self.assertEqual(evidence.common_record, common_record)
        self.assertEqual(evidence.tasks_state, "present")
        self.assertEqual(evidence.tasks, task_records)
        self.assertEqual(evidence.issues, ())

    def test_reports_all_three_sources_missing_without_issues(self) -> None:
        evidence = collect_identity_evidence(self.meta_path, {"other": "Acme"})

        self.assertEqual(evidence.info_robot_type_state, "missing")
        self.assertIsNone(evidence.info_robot_type)
        self.assertEqual(evidence.common_record_state, "missing")
        self.assertIsNone(evidence.common_record)
        self.assertEqual(evidence.tasks_state, "missing")
        self.assertEqual(evidence.tasks, ())
        self.assertEqual(evidence.issues, ())

    def test_keeps_present_invalid_robot_type_and_reports_only_its_type(self) -> None:
        for value in (None, 3, "", " \t\n "):
            with self.subTest(value=value):
                evidence = collect_identity_evidence(
                    self.meta_path, {"robot_type": value}
                )

                self.assertEqual(evidence.info_robot_type_state, "present")
                self.assertEqual(evidence.info_robot_type, value)
                self.assertEqual(len(evidence.issues), 1)
                issue = evidence.issues[0]
                self.assertEqual(issue.code, "INFO_ROBOT_TYPE_INVALID")
                self.assertEqual(issue.scope, "identity.info_robot_type")
                self.assertEqual(issue.severity, "review")
                self.assertEqual(
                    issue.evidence, {"value_type": type(value).__name__}
                )

    def test_marks_invalid_common_record_without_exposing_bad_content(self) -> None:
        secret = "sensitive-invalid-common-content"
        (self.meta_path / "common_record.json").write_text(
            secret + " {", encoding="utf-8"
        )

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.common_record_state, "invalid")
        self.assertIsNone(evidence.common_record)
        self.assertEqual(len(evidence.issues), 1)
        issue = evidence.issues[0]
        self.assertEqual(issue.code, "COMMON_RECORD_INVALID")
        self.assertEqual(issue.scope, "identity.common_record")
        self.assertEqual(issue.evidence["error_type"], "JSONDecodeError")
        self._assert_safe_evidence(
            issue,
            allowed_keys={"file_name", "error_type"},
            forbidden_text=secret,
        )

    def test_marks_oversized_common_integer_invalid_without_escaping(self) -> None:
        oversized_integer = "9" * 5_000
        (self.meta_path / "common_record.json").write_text(
            oversized_integer, encoding="utf-8"
        )

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.common_record_state, "invalid")
        self.assertIsNone(evidence.common_record)
        self.assertEqual(len(evidence.issues), 1)
        issue = evidence.issues[0]
        self.assertEqual(issue.code, "COMMON_RECORD_INVALID")
        self.assertEqual(issue.evidence["error_type"], "ValueError")
        self._assert_safe_evidence(
            issue,
            allowed_keys={"file_name", "error_type"},
            forbidden_text="9" * 64,
        )

    def test_marks_non_utf8_common_record_invalid(self) -> None:
        (self.meta_path / "common_record.json").write_bytes(b"\xff")

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.common_record_state, "invalid")
        self.assertIsNone(evidence.common_record)
        self.assertEqual(evidence.issues[0].code, "COMMON_RECORD_INVALID")
        self.assertEqual(
            evidence.issues[0].evidence["error_type"], "UnicodeDecodeError"
        )

    def test_marks_common_record_unreadable_without_exposing_os_error(self) -> None:
        (self.meta_path / "common_record.json").mkdir()

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.common_record_state, "unreadable")
        self.assertIsNone(evidence.common_record)
        self.assertEqual(len(evidence.issues), 1)
        issue = evidence.issues[0]
        self.assertEqual(issue.code, "COMMON_RECORD_UNREADABLE")
        self.assertEqual(issue.scope, "identity.common_record")
        self.assertEqual(issue.evidence["error_type"], "IsADirectoryError")
        self._assert_safe_evidence(
            issue, allowed_keys={"file_name", "error_type"}
        )

    def test_keeps_valid_task_records_around_one_invalid_line(self) -> None:
        first_record = {"task": "Acme pick", "rank": 1}
        last_record = ["Acme place", 2]
        bad_line = "sensitive-invalid-task-line {"
        (self.meta_path / "tasks.jsonl").write_text(
            "\n".join(
                (
                    json.dumps(first_record),
                    bad_line,
                    json.dumps(last_record),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.tasks_state, "invalid")
        self.assertEqual(evidence.tasks, (first_record, last_record))
        self.assertEqual(len(evidence.issues), 1)
        issue = evidence.issues[0]
        self.assertEqual(issue.code, "TASKS_INVALID")
        self.assertEqual(issue.scope, "identity.tasks")
        self.assertEqual(issue.evidence["line_numbers"], [2])
        self.assertEqual(issue.evidence["error_types"], ["JSONDecodeError"])
        self._assert_safe_evidence(
            issue,
            allowed_keys={"file_name", "line_numbers", "error_types"},
            forbidden_text=bad_line,
        )

    def test_aggregates_multiple_bad_task_lines_in_one_issue(self) -> None:
        first_record = {"task": "Acme first"}
        last_record = {"task": "Acme last"}
        (self.meta_path / "tasks.jsonl").write_text(
            "\n".join(
                (
                    json.dumps(first_record),
                    "",
                    "sensitive-not-json",
                    json.dumps(last_record),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.tasks_state, "invalid")
        self.assertEqual(evidence.tasks, (first_record, last_record))
        self.assertEqual(len(evidence.issues), 1)
        issue = evidence.issues[0]
        self.assertEqual(issue.code, "TASKS_INVALID")
        self.assertEqual(issue.evidence["line_numbers"], [2, 3])
        self.assertEqual(issue.evidence["error_types"], ["JSONDecodeError"])
        self.assertNotIn("sensitive-not-json", repr(issue.evidence))

    def test_continues_after_value_and_recursion_errors_in_tasks(self) -> None:
        first_record = {"task": "Acme first"}
        last_record = {"task": "Acme last"}
        oversized_integer = "9" * 5_000
        deeply_nested = "[" * 2_000 + "0" + "]" * 2_000
        (self.meta_path / "tasks.jsonl").write_text(
            "\n".join(
                (
                    json.dumps(first_record),
                    oversized_integer,
                    deeply_nested,
                    json.dumps(last_record),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.tasks_state, "invalid")
        self.assertEqual(evidence.tasks, (first_record, last_record))
        self.assertEqual(len(evidence.issues), 1)
        issue = evidence.issues[0]
        self.assertEqual(issue.code, "TASKS_INVALID")
        self.assertEqual(issue.evidence["line_numbers"], [2, 3])
        self.assertEqual(
            set(issue.evidence["error_types"]), {"ValueError", "RecursionError"}
        )
        self._assert_safe_evidence(
            issue,
            allowed_keys={"file_name", "line_numbers", "error_types"},
            forbidden_text="9" * 64,
        )
        self.assertNotIn("[" * 64, repr(issue.evidence))

    def test_keeps_valid_tasks_around_a_non_utf8_line(self) -> None:
        first_record = {"task": "Acme first"}
        last_record = {"task": "Acme last"}
        (self.meta_path / "tasks.jsonl").write_bytes(
            json.dumps(first_record).encode("utf-8")
            + b"\n\xff\n"
            + json.dumps(last_record).encode("utf-8")
            + b"\n"
        )

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.tasks_state, "invalid")
        self.assertEqual(evidence.tasks, (first_record, last_record))
        self.assertEqual(evidence.issues[0].evidence["line_numbers"], [2])
        self.assertEqual(
            evidence.issues[0].evidence["error_types"], ["UnicodeDecodeError"]
        )

    def test_marks_tasks_unreadable_and_discards_records(self) -> None:
        (self.meta_path / "tasks.jsonl").mkdir()

        evidence = collect_identity_evidence(self.meta_path, {})

        self.assertEqual(evidence.tasks_state, "unreadable")
        self.assertEqual(evidence.tasks, ())
        self.assertEqual(len(evidence.issues), 1)
        issue = evidence.issues[0]
        self.assertEqual(issue.code, "TASKS_UNREADABLE")
        self.assertEqual(issue.scope, "identity.tasks")
        self.assertEqual(issue.evidence["error_type"], "IsADirectoryError")
        self._assert_safe_evidence(
            issue, allowed_keys={"file_name", "error_type"}
        )

    def test_orders_info_common_and_tasks_issues_by_source(self) -> None:
        (self.meta_path / "common_record.json").write_text("bad-common")
        (self.meta_path / "tasks.jsonl").write_text("bad-task\n")

        evidence = collect_identity_evidence(
            self.meta_path, {"robot_type": None}
        )

        self.assertEqual(
            [issue.code for issue in evidence.issues],
            [
                "INFO_ROBOT_TYPE_INVALID",
                "COMMON_RECORD_INVALID",
                "TASKS_INVALID",
            ],
        )

    def test_repeated_collection_does_not_modify_source_info(self) -> None:
        source_info = {
            "robot_type": " Acme_TestBot ",
            "nested": {"items": [1, {"raw": True}]},
        }
        original = deepcopy(source_info)
        self._write_json("common_record.json", {"raw": [1, 2]})
        self._write_task_records({"task": "Acme task"})

        first = collect_identity_evidence(self.meta_path, source_info)
        second = collect_identity_evidence(self.meta_path, source_info)

        self.assertEqual(first, second)
        self.assertEqual(source_info, original)

    def _write_json(self, file_name: str, payload: object) -> None:
        (self.meta_path / file_name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _write_task_records(self, *records: object) -> None:
        (self.meta_path / "tasks.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            ),
            encoding="utf-8",
        )

    def _assert_safe_evidence(
        self,
        issue: Issue,
        *,
        allowed_keys: set[str],
        forbidden_text: str | None = None,
    ) -> None:
        self.assertLessEqual(set(issue.evidence), allowed_keys)
        serialized = json.dumps(issue.evidence, ensure_ascii=False)
        self.assertNotIn(str(self.meta_path), serialized)
        if forbidden_text is not None:
            self.assertNotIn(forbidden_text, serialized)


class ParquetEvidenceTest(unittest.TestCase):
    """Verify bounded Parquet inspection records structure without values."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.dataset_path = Path(self.temporary_directory.name) / "acme_dataset"
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        self.meta_path.mkdir(parents=True)
        self.data_path.mkdir()
        self.candidate = self._candidate(self.dataset_path)

    def test_selects_only_lexicographic_first_and_last_nested_paths(self) -> None:
        first = self._write_parquet(
            "alpha/deep/episode_000002.parquet", {"action": [[1.0, 2.0]]}
        )
        middle = self.data_path / "middle/deeper/episode_000000.parquet"
        middle.parent.mkdir(parents=True)
        middle.write_bytes(b"middle file must never be opened")
        last = self._write_parquet(
            "zeta/episode_000001.parquet", {"action": [[3.0, 4.0]]}
        )

        machines, issues = collect_machine_evidence(
            self.candidate, self._machine_info(action={"shape": [2]})
        )

        self.assertEqual(issues, ())
        self.assertEqual(
            tuple(episode.relative_path for episode in machines[0].episodes),
            (
                first.relative_to(self.dataset_path).as_posix(),
                last.relative_to(self.dataset_path).as_posix(),
            ),
        )
        self.assertNotIn(
            middle.relative_to(self.dataset_path).as_posix(),
            {episode.relative_path for episode in machines[0].episodes},
        )

    def test_zero_one_and_two_parquet_files_are_not_duplicated(self) -> None:
        for file_count in (0, 1, 2):
            with self.subTest(file_count=file_count):
                dataset_path = self.dataset_path / f"case_{file_count}"
                candidate = self._candidate(dataset_path)
                assert candidate.data_path is not None
                candidate.data_path.mkdir(parents=True)
                for index in range(file_count):
                    self._write_candidate_parquet(
                        candidate,
                        f"nested/episode_{index:06d}.parquet",
                        {"action": [[float(index)]]},
                    )

                machines, issues = collect_machine_evidence(
                    candidate, self._machine_info(action={"shape": [1]})
                )

                paths = tuple(item.relative_path for item in machines[0].episodes)
                self.assertEqual(len(paths), file_count)
                self.assertEqual(len(paths), len(set(paths)))
                self.assertEqual(
                    machines[0].episode_lengths, (1,) * file_count
                )
                self.assertEqual(issues, ())

    def test_preserves_schema_and_machine_feature_insertion_order(self) -> None:
        parquet_path = self._write_parquet(
            "episode.parquet",
            {
                "other": [7],
                "observation.state.child": [[1.0, 2.0]],
                "action": [[3.0, 4.0, 5.0]],
                "observation.images.wrist": [[0]],
                "observation.state": [6.0],
            },
        )
        raw_dtype = {"storage": "float32"}
        source_info = {
            "features": {
                "observation.images.wrist": {"dtype": "video"},
                "action": {
                    "dtype": raw_dtype,
                    "shape": [3],
                    "names": ["a", "b", "c"],
                    "fps": "raw-fps",
                    "codec": "raw-codec",
                },
                "other": {"shape": [1]},
                "observation.state.child": {
                    "dtype": "float64",
                    "shape": "not-a-sequence-shape",
                    "names": ("child_0", "child_1"),
                },
                "observation.state": {
                    "dtype": "float64",
                    "shape": (1,),
                    "names": "not-a-sequence-of-names",
                },
            }
        }

        machines, issues = collect_machine_evidence(self.candidate, source_info)

        self.assertEqual(issues, ())
        self.assertEqual(
            tuple(machine.schema.source_key for machine in machines),
            ("action", "observation.state.child", "observation.state"),
        )
        self.assertEqual(machines[0].schema.dtype, raw_dtype)
        self.assertEqual(machines[0].schema.shape, (3,))
        self.assertEqual(machines[0].schema.names, ("a", "b", "c"))
        self.assertEqual(machines[0].schema.fps, "raw-fps")
        self.assertEqual(machines[0].schema.codec, "raw-codec")
        self.assertEqual(machines[1].schema.shape, ())
        self.assertEqual(machines[1].schema.names, ("child_0", "child_1"))
        self.assertEqual(machines[2].schema.names, ())
        expected_columns = (
            "other",
            "observation.state.child",
            "action",
            "observation.images.wrist",
            "observation.state",
        )
        self.assertEqual(machines[0].episodes[0].schema_columns, expected_columns)
        self.assertEqual(
            machines[0].episodes[0].relative_path,
            parquet_path.relative_to(self.dataset_path).as_posix(),
        )

    def test_records_actual_lengths_and_cross_episode_mismatch(self) -> None:
        self._write_parquet(
            "chunk-000/episode_000000.parquet",
            {"action": [[1.0, 2.0]], "observation.state": [3.0]},
        )
        self._write_parquet(
            "chunk-001/episode_000001.parquet",
            {"action": [[1.0, 2.0, 3.0]], "observation.state": [4.0]},
        )
        source_info = self._machine_info(
            action={"shape": [2]},
            **{"observation.state": {"shape": [1]}},
        )

        machines, issues = collect_machine_evidence(self.candidate, source_info)

        by_key = {machine.schema.source_key: machine for machine in machines}
        self.assertEqual(by_key["action"].episode_lengths, (2, 3))
        self.assertEqual(by_key["observation.state"].episode_lengths, (1, 1))
        self.assertEqual(
            tuple(episode.vector_lengths for episode in machines[0].episodes),
            (
                {"action": 2, "observation.state": 1},
                {"action": 3, "observation.state": 1},
            ),
        )
        self.assertEqual(
            [issue.code for issue in issues],
            ["PARQUET_VECTOR_LENGTH_INCONSISTENT"],
        )

    def test_scans_every_batch_and_rejects_mixed_widths_in_one_episode(self) -> None:
        rows = [[1.0, 2.0] for _ in range(512)] + [[1.0, 2.0, 3.0]]
        self._write_parquet("episode.parquet", {"action": rows})

        machines, issues = collect_machine_evidence(
            self.candidate, self._machine_info(action={"shape": [2]})
        )

        self.assertEqual(machines[0].episodes[0].vector_lengths, {"action": None})
        self.assertEqual(machines[0].episode_lengths, ())
        self.assertEqual(
            [issue.code for issue in issues],
            ["PARQUET_VECTOR_LENGTH_INCONSISTENT"],
        )

    def test_empty_and_null_columns_have_unknown_lengths(self) -> None:
        cases = (
            ("empty", pa.array([], type=pa.list_(pa.float64()))),
            ("null", pa.array([None], type=pa.list_(pa.float64()))),
        )
        for name, column in cases:
            with self.subTest(name=name):
                dataset_path = self.dataset_path / name
                candidate = self._candidate(dataset_path)
                assert candidate.data_path is not None
                candidate.data_path.mkdir(parents=True)
                self._write_candidate_parquet(
                    candidate, "episode.parquet", {"action": column}
                )

                machines, issues = collect_machine_evidence(
                    candidate, self._machine_info(action={"shape": [2]})
                )

                self.assertEqual(
                    machines[0].episodes[0].vector_lengths, {"action": None}
                )
                self.assertEqual(machines[0].episode_lengths, ())
                self.assertEqual(
                    [issue.code for issue in issues],
                    ["PARQUET_VECTOR_LENGTH_INCONSISTENT"],
                )

    def test_unsupported_arrow_extension_values_have_unknown_length(self) -> None:
        extension_name = "robometanorm.test.range"

        class RangeScalar(pa.ExtensionScalar):
            def as_py(self, *, maps_as_pydicts: object = None) -> object:
                return range(self.value.as_py())

        class RangeType(pa.ExtensionType):
            def __init__(self) -> None:
                super().__init__(pa.int32(), extension_name)

            def __arrow_ext_serialize__(self) -> bytes:
                return b""

            @classmethod
            def __arrow_ext_deserialize__(
                cls, storage_type: pa.DataType, serialized: bytes
            ) -> RangeType:
                return cls()

            def __arrow_ext_scalar_class__(self) -> type[RangeScalar]:
                return RangeScalar

        extension_type = RangeType()
        pa.register_extension_type(extension_type)
        self.addCleanup(pa.unregister_extension_type, extension_name)
        values = pa.ExtensionArray.from_storage(
            extension_type, pa.array([2, 3], type=pa.int32())
        )
        self._write_parquet("episode.parquet", {"action": values})

        machines, issues = collect_machine_evidence(
            self.candidate, self._machine_info(action={"shape": [2]})
        )

        self.assertEqual(machines[0].episodes[0].vector_lengths, {"action": None})
        self.assertEqual(machines[0].episode_lengths, ())
        self.assertEqual(
            [issue.code for issue in issues],
            ["PARQUET_VECTOR_LENGTH_INCONSISTENT"],
        )

    def test_gripper_names_do_not_trigger_range_collection(self) -> None:
        self._write_parquet(
            "episode.parquet", {"action": [[-1000.0, 1000.0], [0.0, 1.0]]}
        )
        source_info = self._machine_info(
            action={
                "dtype": "float64",
                "shape": [2],
                "names": ["left_gripper", "right_gripper_open"],
            }
        )

        machines, issues = collect_machine_evidence(self.candidate, source_info)

        self.assertEqual(issues, ())
        self.assertEqual(machines[0].episode_lengths, (2,))
        self.assertEqual(machines[0].gripper_ranges, ())

    def test_reports_declared_column_missing_from_exact_top_level_schema(self) -> None:
        self._write_parquet("episode.parquet", {"action": [[1.0, 2.0]]})
        source_info = self._machine_info(
            action={"shape": [2]},
            **{"observation.state": {"shape": [2]}},
        )

        machines, issues = collect_machine_evidence(self.candidate, source_info)

        by_key = {machine.schema.source_key: machine for machine in machines}
        self.assertEqual(
            by_key["observation.state"].episodes[0].vector_lengths[
                "observation.state"
            ],
            None,
        )
        self.assertEqual(by_key["observation.state"].episode_lengths, ())
        missing = [issue for issue in issues if issue.code == "PARQUET_COLUMN_MISSING"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].evidence["feature"], "observation.state")
        self._assert_safe_parquet_issue(missing[0])

    def test_corrupt_parquet_is_safe_and_does_not_stop_other_episode(self) -> None:
        corrupt = self.data_path / "episode_000000.parquet"
        corrupt.write_bytes(b"secret corrupt parquet payload")
        valid = self._write_parquet(
            "episode_000001.parquet", {"action": [[1.0, 2.0]]}
        )

        machines, issues = collect_machine_evidence(
            self.candidate, self._machine_info(action={"shape": [2]})
        )

        self.assertEqual(len(machines[0].episodes), 2)
        self.assertEqual(machines[0].episodes[0].schema_columns, ())
        self.assertEqual(machines[0].episodes[0].vector_lengths, {"action": None})
        self.assertEqual(machines[0].episodes[1].vector_lengths, {"action": 2})
        self.assertEqual(
            machines[0].episodes[1].relative_path,
            valid.relative_to(self.dataset_path).as_posix(),
        )
        self.assertEqual(machines[0].episode_lengths, ())
        failed = [issue for issue in issues if issue.code == "PARQUET_READ_FAILED"]
        self.assertEqual(len(failed), 1)
        self.assertNotIn("secret", repr(failed[0].evidence))
        self._assert_safe_parquet_issue(failed[0])

    def test_arrow_memory_error_propagates(self) -> None:
        self._write_parquet("episode.parquet", {"action": [[1.0, 2.0]]})

        with patch(
            "robometanorm.evidence.pq.ParquetFile",
            side_effect=pa.ArrowMemoryError("out of memory"),
        ):
            with self.assertRaises(MemoryError):
                collect_machine_evidence(
                    self.candidate,
                    self._machine_info(action={"shape": [2]}),
                )

    def test_lazy_read_failure_is_safe_and_does_not_stop_other_feature(self) -> None:
        parquet_path = self._write_parquet(
            "episode.parquet",
            {
                "action": [[1.0, 2.0] for _ in range(513)],
                "observation.state": [3.0 for _ in range(513)],
            },
        )
        real_parquet_file = pq.ParquetFile(parquet_path)
        calls: list[tuple[list[str], int]] = []

        class LazyFailureParquetFile:
            schema_arrow = real_parquet_file.schema_arrow

            def iter_batches(
                self, *, columns: list[str], batch_size: int
            ) -> object:
                calls.append((columns, batch_size))
                batches = real_parquet_file.iter_batches(
                    columns=columns, batch_size=batch_size
                )
                if columns == ["action"]:
                    yield next(batches)
                    raise ValueError("sensitive lazy reader detail")
                yield from batches

        source_info = self._machine_info(
            action={"shape": [2]},
            **{"observation.state": {"shape": [1]}},
        )
        with patch(
            "robometanorm.evidence.pq.ParquetFile",
            return_value=LazyFailureParquetFile(),
        ):
            machines, issues = collect_machine_evidence(
                self.candidate, source_info
            )

        episode = machines[0].episodes[0]
        self.assertEqual(
            episode.vector_lengths,
            {"action": None, "observation.state": 1},
        )
        self.assertEqual(calls, [(["action"], 512), (["observation.state"], 512)])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "PARQUET_READ_FAILED")
        self.assertEqual(issues[0].evidence["error_type"], "ValueError")
        self.assertNotIn("sensitive", repr(issues[0].evidence))
        self._assert_safe_parquet_issue(issues[0])

    def test_rejects_ambiguous_dotted_projection(self) -> None:
        nested = pa.array(
            [{"state": [1.0, 2.0]} for _ in range(513)],
            type=pa.struct([("state", pa.list_(pa.float64()))]),
        )
        parquet_path = self._write_parquet(
            "episode.parquet",
            {
                "observation": nested,
                "observation.state": [[3.0, 4.0] for _ in range(513)],
            },
        )
        real_parquet_file = pq.ParquetFile(parquet_path)
        calls: list[tuple[list[str], int]] = []
        batches_seen = 0

        class TrackingParquetFile:
            schema_arrow = real_parquet_file.schema_arrow

            def iter_batches(
                self, *, columns: list[str], batch_size: int
            ) -> object:
                nonlocal batches_seen
                calls.append((columns, batch_size))
                for batch in real_parquet_file.iter_batches(
                    columns=columns, batch_size=batch_size
                ):
                    batches_seen += 1
                    yield batch

        with patch(
            "robometanorm.evidence.pq.ParquetFile",
            return_value=TrackingParquetFile(),
        ):
            machines, issues = collect_machine_evidence(
                self.candidate,
                self._machine_info(**{"observation.state": {"shape": [2]}}),
            )

        self.assertEqual(
            machines[0].episodes[0].vector_lengths,
            {"observation.state": None},
        )
        self.assertEqual(machines[0].episode_lengths, ())
        self.assertEqual(calls, [(["observation.state"], 512)])
        self.assertEqual(batches_seen, 2)
        self.assertEqual(
            [issue.code for issue in issues], ["PARQUET_COLUMN_AMBIGUOUS"]
        )
        self._assert_safe_parquet_issue(issues[0])

    def test_collection_preserves_info_parquet_bytes_and_creates_no_cache(self) -> None:
        paths = (
            self._write_parquet(
                "a/episode_000000.parquet", {"action": [[1.0, 2.0]]}
            ),
            self._write_parquet(
                "b/episode_000001.parquet", {"action": [[9.0, 9.0]]}
            ),
            self._write_parquet(
                "c/episode_000002.parquet", {"action": [[3.0, 4.0]]}
            ),
        )
        source_info = self._machine_info(
            action={
                "dtype": "float64",
                "shape": [2],
                "names": ["raw_0", "raw_1"],
                "nested": {"untouched": [True, None]},
            }
        )
        original_info = deepcopy(source_info)
        hashes_before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        }

        first = collect_machine_evidence(self.candidate, source_info)
        second = collect_machine_evidence(self.candidate, source_info)

        self.assertEqual(first, second)
        self.assertEqual(source_info, original_info)
        self.assertEqual(
            {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            hashes_before,
        )
        self.assertEqual(list(self.dataset_path.rglob(".robometanorm_cache")), [])

    def _candidate(self, dataset_path: Path) -> DatasetCandidate:
        return DatasetCandidate(
            dataset_name=dataset_path.name,
            task_name=None,
            source_path=dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=dataset_path / "meta" / "info.json",
            data_path=dataset_path / "data",
            video_path=None,
            depth_path=None,
        )

    def _write_parquet(
        self, relative_path: str, columns: dict[str, object]
    ) -> Path:
        return self._write_candidate_parquet(self.candidate, relative_path, columns)

    def _write_candidate_parquet(
        self,
        candidate: DatasetCandidate,
        relative_path: str,
        columns: dict[str, object],
    ) -> Path:
        assert candidate.data_path is not None
        parquet_path = candidate.data_path / relative_path
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), parquet_path)
        return parquet_path

    def _machine_info(self, **features: object) -> dict[str, object]:
        return {"features": features}

    def _assert_safe_parquet_issue(self, issue: Issue) -> None:
        self.assertLessEqual(
            set(issue.evidence),
            {"relative_path", "feature", "length", "error_type"},
        )
        serialized = json.dumps(issue.evidence, ensure_ascii=False)
        self.assertNotIn(str(self.dataset_path), serialized)


class MappedGripperRangeTest(unittest.TestCase):
    """Verify only mapped single-dimension grippers expose bounded ranges."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.dataset_path = Path(self.temporary_directory.name) / "acme_dataset"
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        self.meta_path.mkdir(parents=True)
        self.data_path.mkdir()
        self.candidate = self._candidate(self.dataset_path)

    def test_collects_opaque_mapped_range_from_only_first_and_last(self) -> None:
        first = self._write_parquet(
            "a/episode_000000.parquet",
            {"action": [[100.0, 0.2], [101.0, 0.4]]},
        )
        middle = self._write_parquet(
            "m/episode_000001.parquet",
            {"action": [[102.0, -999.0], [103.0, 999.0]]},
        )
        last = self._write_parquet(
            "z/episode_000002.parquet",
            {"action": [[104.0, 0.8], [105.0, 1.2]]},
        )
        source_info = self._source_info(
            "action", shape=[2], names=["opaque_alpha", "opaque_beta"]
        )
        evidence = self._structural_evidence(source_info)
        original = deepcopy(evidence)
        self.assertEqual(evidence.machines[0].gripper_ranges, ())
        profile = self._profile(
            self._component("arm", "arm_joint"),
            self._component("finger", "gripper_open"),
        )
        mapping = self._mapping(
            "action",
            self._slice(0, "arm"),
            self._slice(1, "finger"),
        )

        collected, issues = collect_mapped_gripper_ranges(
            self.candidate, evidence, profile, mapping
        )

        self.assertEqual(issues, ())
        self.assertEqual(evidence, original)
        self.assertEqual(evidence.machines[0].gripper_ranges, ())
        self.assertEqual(
            collected.machines[0].gripper_ranges,
            (
                self._range(
                    index=1,
                    minimum=0.2,
                    maximum=1.2,
                    finite_count=4,
                    nonfinite_count=0,
                ),
            ),
        )
        selected_paths = {
            item.relative_path for item in collected.machines[0].episodes
        }
        self.assertIn(first.relative_to(self.dataset_path).as_posix(), selected_paths)
        self.assertIn(last.relative_to(self.dataset_path).as_posix(), selected_paths)
        self.assertNotIn(
            middle.relative_to(self.dataset_path).as_posix(), selected_paths
        )

    def test_supports_scale_deduplicates_sorts_and_reads_feature_once(self) -> None:
        parquet_path = self._write_parquet(
            "episode.parquet",
            {"action": [[4.0, 100.0, 0.1], [2.0, 101.0, 0.9]]},
        )
        source_info = self._source_info("action", shape=[3], names=["x", "y", "z"])
        evidence = self._structural_evidence(source_info)
        profile = self._profile(
            self._component("scale", "gripper_open_scale"),
            self._component("open", "gripper_open"),
        )
        mapping = DatasetMapping(
            cameras=(),
            machines=(
                self._assignment(
                    "action", self._slice(2, "scale"), self._slice(0, "open")
                ),
                self._assignment("action", self._slice(2, "scale")),
            ),
        )
        real_parquet_file = pq.ParquetFile(parquet_path)
        constructor_calls: list[Path] = []
        batch_calls: list[tuple[list[str], int]] = []

        class TrackingParquetFile:
            schema_arrow = real_parquet_file.schema_arrow

            def iter_batches(
                self, *, columns: list[str], batch_size: int
            ) -> object:
                batch_calls.append((columns, batch_size))
                return real_parquet_file.iter_batches(
                    columns=columns, batch_size=batch_size
                )

        def open_tracking(path: Path) -> TrackingParquetFile:
            constructor_calls.append(path)
            return TrackingParquetFile()

        with patch(
            "robometanorm.evidence.pq.ParquetFile", side_effect=open_tracking
        ):
            collected, issues = collect_mapped_gripper_ranges(
                self.candidate, evidence, profile, mapping
            )

        self.assertEqual(issues, ())
        self.assertEqual(constructor_calls, [parquet_path])
        self.assertEqual(batch_calls, [(["action"], 512)])
        self.assertEqual(
            collected.machines[0].gripper_ranges,
            (
                self._range(0, 2.0, 4.0, 2, 0),
                self._range(2, 0.1, 0.9, 2, 0),
            ),
        )

    def test_non_gripper_unknown_and_absent_mappings_do_not_open_parquet(self) -> None:
        self._write_parquet("episode.parquet", {"action": [[1.0]]})
        source_info = self._source_info(
            "action", shape=[1], names=["definitely_gripper_open"]
        )
        evidence = self._structural_evidence(source_info)
        cases = (
            (
                self._profile(self._component("arm", "arm_joint")),
                self._mapping("action", self._slice(0, "arm")),
            ),
            (
                self._profile(self._component("grip", "gripper_open")),
                self._mapping("action", self._slice(0, "unknown-component")),
            ),
            (
                self._profile(self._component("grip", "gripper_open")),
                DatasetMapping(cameras=(), machines=()),
            ),
        )

        for profile, mapping in cases:
            with self.subTest(profile=profile, mapping=mapping):
                with patch("robometanorm.evidence.pq.ParquetFile") as parquet_file:
                    collected, issues = collect_mapped_gripper_ranges(
                        self.candidate, evidence, profile, mapping
                    )

                parquet_file.assert_not_called()
                self.assertIs(collected, evidence)
                self.assertEqual(issues, ())

    def test_rejects_all_non_builtin_finite_values(self) -> None:
        parquet_path = self.data_path / "episode.parquet"
        parquet_path.write_bytes(b"reader is replaced by a deterministic fake")
        source_info = self._source_info("action", shape=[1], names=["opaque"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(self._component("grip", "gripper_open"))
        mapping = self._mapping("action", self._slice(0, "grip"))
        values = (
            [float("nan")],
            [float("inf")],
            [float("-inf")],
            [True],
            [None],
            ["0.5"],
            [Decimal("0.5")],
        )

        with patch(
            "robometanorm.evidence.pq.ParquetFile",
            return_value=self._fake_parquet_file("action", values),
        ):
            collected, issues = collect_mapped_gripper_ranges(
                self.candidate, evidence, profile, mapping
            )

        self.assertEqual(issues, ())
        self.assertEqual(
            collected.machines[0].gripper_ranges,
            (self._range(0, None, None, 0, len(values)),),
        )

    def test_out_of_bounds_and_null_rows_are_unusable_and_reviewable(self) -> None:
        self._write_parquet(
            "episode.parquet", {"action": [[1.0, 2.0], [3.0], None, [4.0]]}
        )
        source_info = self._source_info("action", shape=[2], names=["x", "y"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(self._component("grip", "gripper_open"))
        mapping = self._mapping("action", self._slice(1, "grip"))

        collected, issues = collect_mapped_gripper_ranges(
            self.candidate, evidence, profile, mapping
        )

        self.assertEqual(
            collected.machines[0].gripper_ranges,
            (self._range(1, 2.0, 2.0, 1, 3),),
        )
        self.assertEqual(
            [issue.code for issue in issues],
            ["PARQUET_VECTOR_LENGTH_INCONSISTENT"],
        )
        self._assert_safe_parquet_issue(issues[0])

    def test_scalar_rows_are_supported_only_for_index_zero(self) -> None:
        self._write_parquet("episode.parquet", {"action": [1.0, None, 2.5]})
        source_info = self._source_info("action", shape=[1], names=["opaque"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(self._component("grip", "gripper_open"))
        mapping = self._mapping("action", self._slice(0, "grip"))

        collected, issues = collect_mapped_gripper_ranges(
            self.candidate, evidence, profile, mapping
        )

        self.assertEqual(issues, ())
        self.assertEqual(
            collected.machines[0].gripper_ranges,
            (self._range(0, 1.0, 2.5, 2, 1),),
        )

    def test_invalid_gripper_slices_are_safe_and_never_read(self) -> None:
        self._write_parquet("episode.parquet", {"action": [[1.0, 2.0]]})
        source_info = self._source_info("action", shape=[2], names=["x", "y"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(
            self._component("bool-start", "gripper_open"),
            self._component("bool-end", "gripper_open"),
            self._component("negative", "gripper_open"),
            self._component("wide", "gripper_open"),
            self._component("count-two", "gripper_open", count=2),
        )
        invalid_slices = (
            MachineSlice(True, 1, "bool-start", ()),
            MachineSlice(0, True, "bool-end", ()),
            MachineSlice(-1, 0, "negative", ()),
            MachineSlice(0, 2, "wide", ()),
            MachineSlice(1, 2, "count-two", ()),
        )
        mapping = self._mapping("action", *invalid_slices)

        with patch("robometanorm.evidence.pq.ParquetFile") as parquet_file:
            collected, issues = collect_mapped_gripper_ranges(
                self.candidate, evidence, profile, mapping
            )

        parquet_file.assert_not_called()
        self.assertIs(collected, evidence)
        self.assertEqual(
            [issue.code for issue in issues],
            ["MAPPED_GRIPPER_SLICE_INVALID"] * len(invalid_slices),
        )
        for issue, source_slice in zip(issues, invalid_slices):
            self.assertEqual(issue.severity, "review")
            self.assertEqual(
                set(issue.evidence),
                {"source_feature", "component_id", "start", "end"},
            )
            self.assertEqual(issue.evidence["source_feature"], "action")
            self.assertEqual(issue.evidence["component_id"], source_slice.component_id)
            self.assertEqual(issue.evidence["start"], source_slice.start)
            self.assertEqual(issue.evidence["end"], source_slice.end)
            json.dumps(issue.evidence)

    def test_invalid_huge_integer_slice_bounds_remain_json_safe(self) -> None:
        source_info = self._source_info("action", shape=[1], names=["opaque"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(self._component("grip", "gripper_open"))
        huge_integer = 10**5_000
        mapping = self._mapping(
            "action", MachineSlice(huge_integer, huge_integer, "grip", ())
        )

        with patch("robometanorm.evidence.pq.ParquetFile") as parquet_file:
            collected, issues = collect_mapped_gripper_ranges(
                self.candidate, evidence, profile, mapping
            )

        parquet_file.assert_not_called()
        self.assertIs(collected, evidence)
        self.assertEqual(
            [issue.code for issue in issues], ["MAPPED_GRIPPER_SLICE_INVALID"]
        )
        json.dumps(issues[0].evidence)
        self.assertEqual(issues[0].evidence["start"], {"value_type": "int"})
        self.assertEqual(issues[0].evidence["end"], {"value_type": "int"})

    def test_memory_error_while_checking_integer_json_safety_propagates(self) -> None:
        source_info = self._source_info("action", shape=[1], names=["opaque"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(self._component("grip", "gripper_open"))
        mapping = self._mapping("action", MachineSlice(0, 2, "grip", ()))

        with patch(
            "robometanorm.evidence.json.dumps",
            side_effect=MemoryError("out of memory"),
        ):
            with self.assertRaises(MemoryError):
                collect_mapped_gripper_ranges(
                    self.candidate, evidence, profile, mapping
                )

    def test_missing_mapped_source_is_reviewed_without_reading(self) -> None:
        self._write_parquet("episode.parquet", {"action": [[1.0]]})
        source_info = self._source_info("action", shape=[1], names=["opaque"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(self._component("grip", "gripper_open"))
        mapping = self._mapping("ghost.feature", self._slice(0, "grip"))

        with patch("robometanorm.evidence.pq.ParquetFile") as parquet_file:
            collected, issues = collect_mapped_gripper_ranges(
                self.candidate, evidence, profile, mapping
            )

        parquet_file.assert_not_called()
        self.assertIs(collected, evidence)
        self.assertEqual(
            [issue.code for issue in issues], ["MAPPED_GRIPPER_SOURCE_MISSING"]
        )
        self.assertEqual(
            issues[0].evidence,
            {"source_feature": "ghost.feature", "component_id": "grip"},
        )

    def test_missing_corrupt_and_ambiguous_columns_mark_ranges_unsafe(self) -> None:
        cases: list[tuple[str, DatasetCandidate, dict[str, object], str, str]] = []

        missing_candidate = self._case_candidate("missing")
        self._write_candidate_parquet(
            missing_candidate, "episode.parquet", {"other": [[1.0]]}
        )
        cases.append(
            (
                "missing",
                missing_candidate,
                self._source_info("action", shape=[1], names=["opaque"]),
                "action",
                "PARQUET_COLUMN_MISSING",
            )
        )

        corrupt_candidate = self._case_candidate("corrupt")
        assert corrupt_candidate.data_path is not None
        (corrupt_candidate.data_path / "episode.parquet").write_bytes(
            b"sensitive corrupt parquet payload"
        )
        cases.append(
            (
                "corrupt",
                corrupt_candidate,
                self._source_info("action", shape=[1], names=["opaque"]),
                "action",
                "PARQUET_READ_FAILED",
            )
        )

        ambiguous_candidate = self._case_candidate("ambiguous")
        nested = pa.array(
            [{"state": [1.0]}],
            type=pa.struct([("state", pa.list_(pa.float64()))]),
        )
        self._write_candidate_parquet(
            ambiguous_candidate,
            "episode.parquet",
            {"observation": nested, "observation.state": [[2.0]]},
        )
        cases.append(
            (
                "ambiguous",
                ambiguous_candidate,
                self._source_info(
                    "observation.state", shape=[1], names=["opaque"]
                ),
                "observation.state",
                "PARQUET_COLUMN_AMBIGUOUS",
            )
        )

        profile = self._profile(self._component("grip", "gripper_open"))
        for name, candidate, source_info, feature, issue_code in cases:
            with self.subTest(name=name):
                evidence = self._direct_evidence(source_info, feature, candidate)
                mapping = self._mapping(feature, self._slice(0, "grip"))

                collected, issues = collect_mapped_gripper_ranges(
                    candidate, evidence, profile, mapping
                )

                gripper_range = collected.machines[0].gripper_ranges[0]
                self.assertEqual(gripper_range.finite_count, 0)
                self.assertGreater(gripper_range.nonfinite_count, 0)
                self.assertIsNone(gripper_range.minimum)
                self.assertIsNone(gripper_range.maximum)
                self.assertIn(issue_code, [issue.code for issue in issues])
                issue = next(item for item in issues if item.code == issue_code)
                self._assert_safe_parquet_issue(issue, candidate=candidate)
                self.assertNotIn("sensitive", repr(issue.evidence))

    def test_memory_error_propagates(self) -> None:
        (self.data_path / "episode.parquet").write_bytes(b"placeholder")
        source_info = self._source_info("action", shape=[1], names=["opaque"])
        evidence = self._direct_evidence(source_info, "action")
        profile = self._profile(self._component("grip", "gripper_open"))
        mapping = self._mapping("action", self._slice(0, "grip"))

        with patch(
            "robometanorm.evidence.pq.ParquetFile",
            side_effect=MemoryError("out of memory"),
        ):
            with self.assertRaises(MemoryError):
                collect_mapped_gripper_ranges(
                    self.candidate, evidence, profile, mapping
                )

    def test_collection_preserves_bytes_inputs_and_creates_no_cache(self) -> None:
        paths = (
            self._write_parquet("a/episode.parquet", {"action": [[0.1]]}),
            self._write_parquet("m/episode.parquet", {"action": [[-99.0]]}),
            self._write_parquet("z/episode.parquet", {"action": [[0.9]]}),
        )
        source_info = self._source_info("action", shape=[1], names=["opaque"])
        evidence = self._direct_evidence(source_info, "action")
        original_info = deepcopy(source_info)
        original_evidence = deepcopy(evidence)
        hashes_before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        }
        profile = self._profile(self._component("grip", "gripper_open"))
        mapping = self._mapping("action", self._slice(0, "grip"))

        collect_mapped_gripper_ranges(self.candidate, evidence, profile, mapping)

        self.assertEqual(source_info, original_info)
        self.assertEqual(evidence, original_evidence)
        self.assertEqual(
            {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            hashes_before,
        )
        self.assertEqual(list(self.dataset_path.rglob(".robometanorm_cache")), [])

    def _candidate(self, dataset_path: Path) -> DatasetCandidate:
        return DatasetCandidate(
            dataset_name=dataset_path.name,
            task_name=None,
            source_path=dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=dataset_path / "meta" / "info.json",
            data_path=dataset_path / "data",
            video_path=None,
            depth_path=None,
        )

    def _case_candidate(self, name: str) -> DatasetCandidate:
        dataset_path = self.dataset_path / name
        (dataset_path / "meta").mkdir(parents=True)
        (dataset_path / "data").mkdir()
        return self._candidate(dataset_path)

    def _write_parquet(
        self, relative_path: str, columns: dict[str, object]
    ) -> Path:
        return self._write_candidate_parquet(self.candidate, relative_path, columns)

    def _write_candidate_parquet(
        self,
        candidate: DatasetCandidate,
        relative_path: str,
        columns: dict[str, object],
    ) -> Path:
        assert candidate.data_path is not None
        parquet_path = candidate.data_path / relative_path
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), parquet_path)
        return parquet_path

    def _source_info(
        self, feature: str, *, shape: list[int], names: list[str]
    ) -> dict[str, object]:
        return {
            "features": {
                feature: {"dtype": "float64", "shape": shape, "names": names}
            }
        }

    def _structural_evidence(
        self, source_info: dict[str, object]
    ) -> DatasetEvidence:
        machines, issues = collect_machine_evidence(self.candidate, source_info)
        self.assertEqual(issues, ())
        return DatasetEvidence(
            candidate=self.candidate,
            source_info=source_info,
            identity=self._identity_evidence(),
            cameras=(),
            machines=machines,
        )

    def _direct_evidence(
        self,
        source_info: dict[str, object],
        feature: str,
        candidate: DatasetCandidate | None = None,
    ) -> DatasetEvidence:
        selected_candidate = candidate or self.candidate
        feature_data = source_info["features"]
        assert isinstance(feature_data, dict)
        raw_schema = feature_data[feature]
        assert isinstance(raw_schema, dict)
        machine = MachineEvidence(
            schema=FeatureSchema(
                source_key=feature,
                dtype=raw_schema.get("dtype"),
                shape=tuple(raw_schema.get("shape", ())),
                names=tuple(raw_schema.get("names", ())),
                fps=None,
                codec=None,
            ),
            episodes=(),
            episode_lengths=(),
            gripper_ranges=(),
        )
        return DatasetEvidence(
            candidate=selected_candidate,
            source_info=source_info,
            identity=self._identity_evidence(),
            cameras=(),
            machines=(machine,),
        )

    @staticmethod
    def _identity_evidence() -> IdentityEvidence:
        return IdentityEvidence("missing", None, "missing", None, "missing", ())

    def _profile(self, *components: MachineComponent) -> HardwareProfile:
        identity = RobotIdentityFact(
            manufacturer="Acme Robotics",
            model="TestBot",
            confidence=1.0,
            ambiguous=False,
            reason="fictional fixture",
            local_evidence_status="consistent",
            source_ids=(),
            assessments=(),
        )
        return HardwareProfile(identity, (), (), components)

    @staticmethod
    def _component(
        component_id: str, kind: str, *, count: int = 1
    ) -> MachineComponent:
        return MachineComponent(
            component_id=component_id,
            kind=kind,
            side=None,
            count=count,
            element_order=(),
            representation="scalar",
            unit="normalized",
            open_range=None,
            open_direction=None,
            confidence=1.0,
            ambiguous=False,
            reason="fictional fixture",
            source_ids=(),
        )

    @staticmethod
    def _slice(index: int, component_id: str) -> MachineSlice:
        return MachineSlice(index, index + 1, component_id, ())

    @staticmethod
    def _assignment(
        feature: str, *slices: MachineSlice
    ) -> MachineAssignment:
        return MachineAssignment(feature, slices, 0.01, True, "mapping fixture")

    def _mapping(self, feature: str, *slices: MachineSlice) -> DatasetMapping:
        return DatasetMapping(cameras=(), machines=(self._assignment(feature, *slices),))

    @staticmethod
    def _range(
        index: int,
        minimum: float | None,
        maximum: float | None,
        finite_count: int,
        nonfinite_count: int,
    ) -> object:
        from robometanorm.models import GripperRange

        return GripperRange(index, minimum, maximum, finite_count, nonfinite_count)

    @staticmethod
    def _fake_parquet_file(feature: str, values: tuple[object, ...]) -> object:
        class Schema:
            names = (feature,)

        class Column:
            def to_pylist(self) -> list[object]:
                return list(values)

        class Batch:
            num_columns = 1
            schema = Schema()

            def column(self, index: int) -> Column:
                if index != 0:
                    raise IndexError(index)
                return Column()

        class ParquetFile:
            schema_arrow = Schema()

            def iter_batches(
                self, *, columns: list[str], batch_size: int
            ) -> object:
                self.columns = columns
                self.batch_size = batch_size
                yield Batch()

        return ParquetFile()

    def _assert_safe_parquet_issue(
        self, issue: Issue, *, candidate: DatasetCandidate | None = None
    ) -> None:
        self.assertLessEqual(
            set(issue.evidence),
            {"relative_path", "feature", "length", "error_type"},
        )
        serialized = json.dumps(issue.evidence, ensure_ascii=False)
        selected_candidate = candidate or self.candidate
        self.assertNotIn(str(selected_candidate.source_path), serialized)


class CameraEvidenceTest(unittest.TestCase):
    """Verify bounded, exact, read-only camera evidence collection."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.dataset_path = self.root / "acme_camera_dataset"
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        self.video_path = self.dataset_path / "videos"
        self.depth_path = self.dataset_path / "depth"
        for directory in (
            self.meta_path,
            self.data_path,
            self.video_path,
            self.depth_path,
        ):
            directory.mkdir(parents=True)
        self.info_path = self.meta_path / "info.json"
        self.info_path.write_text("{}", encoding="utf-8")
        self.source_key = "observation.images.source_camera"
        self.candidate = DatasetCandidate(
            dataset_name=self.dataset_path.name,
            task_name=None,
            source_path=self.dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=self.info_path,
            data_path=self.data_path,
            video_path=self.video_path,
            depth_path=self.depth_path,
        )
        self.temp_frames = self.root / "temporary-frames"

    def test_camera_schema_accepts_only_exact_image_feature_prefix(self) -> None:
        raw_dtype = {"raw": "dtype"}
        raw_fps = [30, "raw"]
        raw_codec = {"name": "raw-codec"}
        source_info: dict[str, object] = {
            "features": {
                self.source_key: {
                    "dtype": raw_dtype,
                    "shape": [480, "raw", None],
                    "names": ("height", "width", "channel"),
                    "fps": raw_fps,
                    "codec": raw_codec,
                },
                "observation.tactile.palm": {"dtype": "image"},
                "observation.audio.microphone": {"dtype": "audio"},
                "images.source_camera": {"dtype": "video"},
                7: {"dtype": "video"},
                "observation.images.not_a_mapping": "video",
            }
        }

        cameras, issues = collect_camera_evidence(
            self.candidate, source_info, self.temp_frames
        )

        self.assertEqual(len(cameras), 1)
        self.assertIsInstance(cameras[0], CameraEvidence)
        self.assertEqual(cameras[0].schema.source_key, self.source_key)
        self.assertIs(cameras[0].schema.dtype, raw_dtype)
        self.assertEqual(cameras[0].schema.shape, (480, "raw", None))
        self.assertEqual(
            cameras[0].schema.names, ("height", "width", "channel")
        )
        self.assertIs(cameras[0].schema.fps, raw_fps)
        self.assertIs(cameras[0].schema.codec, raw_codec)
        self.assertEqual([issue.code for issue in issues], ["CAMERA_MEDIA_MISSING"])

    def test_matches_only_full_source_key_relative_parent_part(self) -> None:
        exact = self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.MP4",
        )
        self._write_media(
            self.video_path,
            "chunk-000/source_camera/episode_alias.mp4",
        )
        self._write_media(
            self.video_path,
            f"chunk-000/prefix-{self.source_key}-suffix/episode_substring.mp4",
        )
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}.mp4",
        )

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ) as probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ) as extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual(issues, ())
        self.assertEqual([call.args[0] for call in probe.call_args_list], [exact])
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(
            cameras[0].samples[0].relative_path,
            exact.relative_to(self.dataset_path).as_posix(),
        )

    def test_does_not_match_source_key_in_absolute_ancestor(self) -> None:
        ancestor = self.root / self.source_key
        dataset_path = ancestor / "dataset"
        video_path = dataset_path / "videos"
        meta_path = dataset_path / "meta"
        video_path.mkdir(parents=True)
        meta_path.mkdir()
        info_path = meta_path / "info.json"
        info_path.write_text("{}", encoding="utf-8")
        self._write_media(video_path, "chunk-000/episode_000000.mp4")
        candidate = replace(
            self.candidate,
            source_path=dataset_path,
            info_path=info_path,
            data_path=None,
            video_path=video_path,
            depth_path=None,
        )

        with patch("robometanorm.evidence.probe_media") as probe:
            cameras, issues = collect_camera_evidence(
                candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual(cameras[0].samples, ())
        self.assertEqual([issue.code for issue in issues], ["CAMERA_MEDIA_MISSING"])
        probe.assert_not_called()

    def test_three_media_files_sample_only_stable_first_and_last(self) -> None:
        media_paths = [
            self._write_media(
                self.video_path,
                f"chunk-000/{self.source_key}/episode_{index:06d}.mp4",
            )
            for index in range(3)
        ]

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ) as probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ) as extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        selected = [media_paths[0], media_paths[-1]]
        self.assertEqual([call.args[0] for call in probe.call_args_list], selected)
        self.assertEqual([call.args[0] for call in extract.call_args_list], selected)
        self.assertEqual(len(cameras[0].samples), 2)
        self.assertEqual(issues, ())

    def test_one_media_file_is_not_duplicated(self) -> None:
        media_path = self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ) as probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ) as extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual([call.args[0] for call in probe.call_args_list], [media_path])
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(cameras[0].samples), 1)
        self.assertEqual(issues, ())

    def test_two_media_files_are_each_sampled_once(self) -> None:
        media_paths = [
            self._write_media(
                self.video_path,
                f"chunk-000/{self.source_key}/episode_{index:06d}.webm",
            )
            for index in range(2)
        ]

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ) as probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ) as extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual([call.args[0] for call in probe.call_args_list], media_paths)
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(len(cameras[0].samples), 2)
        self.assertEqual(issues, ())

    def test_overlapping_media_roots_deduplicate_same_path(self) -> None:
        nested_root = self.video_path / "nested"
        media_path = self._write_media(
            nested_root,
            f"chunk-000/{self.source_key}/episode_000000.mov",
        )
        candidate = replace(self.candidate, depth_path=nested_root)

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ) as probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ) as extract,
        ):
            cameras, issues = collect_camera_evidence(
                candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual([call.args[0] for call in probe.call_args_list], [media_path])
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(cameras[0].samples), 1)
        self.assertEqual(issues, ())

    def test_external_media_root_uses_safe_logical_relative_path(self) -> None:
        external_root = self.root / "external-media"
        media_path = self._write_media(
            external_root,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )
        candidate = replace(
            self.candidate, video_path=external_root, depth_path=None
        )

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=replace(
                    self._probed_sample(),
                    relative_path=f"/secret/{self.dataset_path.name}.mp4",
                ),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ),
        ):
            cameras, issues = collect_camera_evidence(
                candidate, self._camera_info(), self.temp_frames
            )

        expected = "videos/" + media_path.relative_to(external_root).as_posix()
        self.assertEqual(cameras[0].samples[0].relative_path, expected)
        self.assertFalse(cameras[0].samples[0].relative_path.startswith("/"))
        self.assertNotIn(str(external_root), cameras[0].samples[0].relative_path)
        self.assertEqual(issues, ())

    def test_static_image_is_used_directly_without_media_tools(self) -> None:
        image_path = self._write_media(
            self.depth_path,
            f"chunk-000/{self.source_key}/episode_000000.JPG",
            payload=b"immutable-image",
        )
        before_hash = self._sha256(image_path)

        with (
            patch("robometanorm.evidence.probe_media") as probe,
            patch("robometanorm.evidence.extract_midpoint_frame") as extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        sample = cameras[0].samples[0]
        self.assertEqual(sample.media_type, "image")
        self.assertEqual(sample.frame_path, image_path)
        self.assertEqual(sample.relative_path, "depth/chunk-000/" + self.source_key + "/episode_000000.JPG")
        self.assertIsNone(sample.codec)
        self.assertIsNone(sample.duration_seconds)
        self.assertEqual(self._sha256(image_path), before_hash)
        self.assertEqual(issues, ())
        probe.assert_not_called()
        extract.assert_not_called()

    def test_missing_exact_media_is_review_without_alias_guess(self) -> None:
        self._write_media(
            self.video_path,
            "chunk-000/source_camera/episode_000000.mp4",
        )

        with (
            patch("robometanorm.evidence.probe_media") as probe,
            patch("robometanorm.evidence.extract_midpoint_frame") as extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual(cameras[0].samples, ())
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "CAMERA_MEDIA_MISSING")
        self.assertEqual(issues[0].evidence, {"source_key": self.source_key})
        probe.assert_not_called()
        extract.assert_not_called()

    def test_probe_failure_is_isolated_and_issue_contains_only_safe_evidence(self) -> None:
        media_paths = [
            self._write_media(
                self.video_path,
                f"chunk-000/{self.source_key}/episode_{index:06d}.mp4",
            )
            for index in range(2)
        ]

        def probe(path: Path) -> MediaSample:
            if path == media_paths[0]:
                raise ValueError(f"secret stderr and path {path}")
            return self._probed_sample()

        with (
            patch("robometanorm.evidence.probe_media", side_effect=probe) as mocked_probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ) as extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual(mocked_probe.call_count, 2)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(cameras[0].samples), 1)
        self.assertEqual(issues[0].code, "MEDIA_PROBE_FAILED")
        self.assertEqual(
            issues[0].evidence,
            {
                "source_key": self.source_key,
                "relative_path": media_paths[0]
                .relative_to(self.dataset_path)
                .as_posix(),
                "error_type": "ValueError",
            },
        )
        serialized = json.dumps(issues[0].evidence, ensure_ascii=False)
        self.assertNotIn(str(self.dataset_path), serialized)
        self.assertNotIn("secret", serialized)

    def test_frame_failure_keeps_probe_sample_and_continues(self) -> None:
        media_paths = [
            self._write_media(
                self.video_path,
                f"chunk-000/{self.source_key}/episode_{index:06d}.mkv",
            )
            for index in range(2)
        ]

        def extract(
            path: Path,
            output: Path,
            *,
            duration_seconds: float | None = None,
        ) -> Path:
            self.assertEqual(duration_seconds, 10.0)
            if path == media_paths[0]:
                raise ValueError(f"secret command {path}")
            return self._create_frame(path, output)

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ) as probe,
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=extract,
            ) as mocked_extract,
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual(probe.call_count, 2)
        self.assertEqual(mocked_extract.call_count, 2)
        self.assertEqual(len(cameras[0].samples), 2)
        self.assertIsNone(cameras[0].samples[0].frame_path)
        self.assertTrue(cameras[0].samples[1].frame_path.is_file())
        self.assertEqual(issues[0].code, "FRAME_EXTRACTION_FAILED")
        self.assertEqual(
            issues[0].evidence,
            {
                "source_key": self.source_key,
                "relative_path": media_paths[0]
                .relative_to(self.dataset_path)
                .as_posix(),
                "error_type": "ValueError",
            },
        )

    def test_frame_failure_discards_partial_temporary_output(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )

        def fail_after_write(
            media_path: Path,
            output_path: Path,
            *,
            duration_seconds: float | None = None,
        ) -> Path:
            del media_path
            self.assertEqual(duration_seconds, 10.0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"partial-frame")
            raise ValueError("failed after partial output")

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=fail_after_write,
            ),
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertIsNone(cameras[0].samples[0].frame_path)
        self.assertEqual([issue.code for issue in issues], ["FRAME_EXTRACTION_FAILED"])
        self.assertEqual(list(self.temp_frames.glob("*")), [])

    def test_collection_propagates_probe_memory_error(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )

        with patch(
            "robometanorm.evidence.probe_media", side_effect=MemoryError("oom")
        ):
            with self.assertRaises(MemoryError):
                collect_camera_evidence(
                    self.candidate, self._camera_info(), self.temp_frames
                )

    def test_collection_propagates_frame_memory_error(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=MemoryError("oom"),
            ),
        ):
            with self.assertRaises(MemoryError):
                collect_camera_evidence(
                    self.candidate, self._camera_info(), self.temp_frames
                )

    def test_collection_propagates_unexpected_runtime_errors(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )
        with patch(
            "robometanorm.evidence.probe_media",
            side_effect=RuntimeError("unexpected programmer error"),
        ):
            with self.assertRaises(RuntimeError):
                collect_camera_evidence(
                    self.candidate, self._camera_info(), self.temp_frames
                )

    def test_collection_real_path_probes_and_extracts_each_video_once(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )
        tool_calls: list[str] = []

        def run(command: list[str], **kwargs: object) -> object:
            self.assertEqual(
                kwargs,
                {"capture_output": True, "text": True, "check": False},
            )
            tool_calls.append(command[0])
            if command[0] == "ffprobe":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(self._valid_probe_payload()),
                    stderr="",
                )
            self.assertEqual(command[0], "ffmpeg")
            Path(command[-1]).write_bytes(b"jpeg-frame")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("robometanorm.evidence.subprocess.run", side_effect=run):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        self.assertEqual(tool_calls, ["ffprobe", "ffmpeg"])
        self.assertTrue(cameras[0].samples[0].frame_path.is_file())
        self.assertEqual(issues, ())

    def test_collection_real_path_classifies_probe_and_ffmpeg_failures(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )

        with patch(
            "robometanorm.evidence.subprocess.run",
            return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="unsafe probe error"
            ),
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )
        self.assertEqual(cameras[0].samples, ())
        self.assertEqual([issue.code for issue in issues], ["MEDIA_PROBE_FAILED"])

        def fail_ffmpeg(command: list[str], **kwargs: object) -> object:
            del kwargs
            if command[0] == "ffprobe":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(self._valid_probe_payload()),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=1, stdout="", stderr="unsafe ffmpeg error"
            )

        with patch(
            "robometanorm.evidence.subprocess.run", side_effect=fail_ffmpeg
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )
        self.assertEqual(len(cameras[0].samples), 1)
        self.assertIsNone(cameras[0].samples[0].frame_path)
        self.assertEqual(
            [issue.code for issue in issues], ["FRAME_EXTRACTION_FAILED"]
        )

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=RuntimeError("unexpected programmer error"),
            ),
        ):
            with self.assertRaises(RuntimeError):
                collect_camera_evidence(
                    self.candidate, self._camera_info(), self.temp_frames
                )

    def test_probe_media_parses_first_video_stream_and_uses_safe_command(self) -> None:
        payload = {
            "streams": [
                {"codec_type": "audio", "codec_name": "aac"},
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "r_frame_rate": "30000/1001",
                    "width": 1920,
                    "height": 1080,
                    "duration": "9.5",
                    "pix_fmt": "yuv420p",
                },
                {
                    "codec_type": "video",
                    "codec_name": "ignored",
                    "r_frame_rate": "1/1",
                    "width": 1,
                    "height": 1,
                    "duration": "1",
                },
            ],
            "format": {"duration": "4.25"},
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr=f"secret {self.dataset_path}",
        )

        with patch("robometanorm.evidence.subprocess.run", return_value=completed) as run:
            sample = probe_media(self.video_path / "fictional.mp4")

        self.assertEqual(sample.media_type, "video")
        self.assertEqual(sample.codec, "h264")
        self.assertAlmostEqual(sample.fps, 30000 / 1001)
        self.assertEqual((sample.width, sample.height), (1920, 1080))
        self.assertEqual(sample.duration_seconds, 4.25)
        self.assertEqual(sample.pixel_format, "yuv420p")
        self.assertEqual(sample.relative_path, "")
        self.assertIsNone(sample.frame_path)
        command = run.call_args.args[0]
        self.assertIs(type(command), list)
        self.assertEqual(command[0], "ffprobe")
        self.assertEqual(command[-1], str(self.video_path / "fictional.mp4"))
        self.assertEqual(
            run.call_args.kwargs,
            {"capture_output": True, "text": True, "check": False},
        )

    def test_probe_media_falls_back_to_positive_stream_duration(self) -> None:
        payload = self._valid_probe_payload()
        payload["format"] = {"duration": "N/A"}
        streams = payload["streams"]
        assert isinstance(streams, list)
        streams[0]["duration"] = "2.75"

        with patch(
            "robometanorm.evidence.subprocess.run",
            return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(payload), stderr=""
            ),
        ):
            sample = probe_media(self.video_path / "fictional.mp4")

        self.assertEqual(sample.duration_seconds, 2.75)

    def test_probe_media_rejects_invalid_payload_shapes(self) -> None:
        invalid_payloads: tuple[object, ...] = (
            [],
            {"streams": "not-a-sequence"},
            {"streams": [7]},
            {"streams": [{"codec_type": "audio"}]},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with patch(
                    "robometanorm.evidence.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout=json.dumps(payload), stderr=""
                    ),
                ):
                    with self.assertRaises(ValueError):
                        probe_media(self.video_path / "fictional.mp4")

    def test_probe_media_rejects_nonpositive_nonfinite_and_overflow_values(self) -> None:
        mutations = (
            ("fps", "0/0"),
            ("fps", "0/1"),
            ("fps", "-1/2"),
            ("fps", "N/A"),
            ("fps", "1" + "0" * 3_999),
            ("width", 0),
            ("width", True),
            ("height", -1),
            ("duration", "0"),
            ("duration", "-1"),
            ("duration", "NaN"),
            ("duration", "Infinity"),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=str(value)[:24]):
                payload = self._valid_probe_payload()
                stream = payload["streams"][0]
                if field == "fps":
                    stream["r_frame_rate"] = value
                elif field in {"width", "height"}:
                    stream[field] = value
                else:
                    payload["format"] = {"duration": value}
                    stream.pop("duration", None)
                with patch(
                    "robometanorm.evidence.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout=json.dumps(payload), stderr=""
                    ),
                ):
                    with self.assertRaises(ValueError):
                        probe_media(self.video_path / "fictional.mp4")

    def test_probe_media_converts_tool_and_json_failures_to_safe_value_error(self) -> None:
        failing_results = (
            SimpleNamespace(
                returncode=2,
                stdout="{}",
                stderr=f"secret stderr {self.dataset_path}",
            ),
            SimpleNamespace(returncode=0, stdout="{bad-json", stderr="secret"),
        )
        for result in failing_results:
            with self.subTest(returncode=result.returncode, stdout=result.stdout):
                with patch(
                    "robometanorm.evidence.subprocess.run", return_value=result
                ):
                    with self.assertRaises(ValueError) as caught:
                        probe_media(self.video_path / "fictional.mp4")
                self.assertNotIn(str(self.dataset_path), str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))

        with patch(
            "robometanorm.evidence.subprocess.run",
            side_effect=OSError(f"secret {self.dataset_path}"),
        ):
            with self.assertRaises(ValueError) as caught:
                probe_media(self.video_path / "fictional.mp4")
        self.assertNotIn(str(self.dataset_path), str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_probe_media_converts_timeout_to_value_error(self) -> None:
        with patch(
            "robometanorm.evidence.subprocess.run",
            side_effect=subprocess.TimeoutExpired(
                ["ffprobe", str(self.video_path / "secret.mp4")], 1
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                probe_media(self.video_path / "fictional.mp4")

        self.assertNotIn(str(self.video_path), str(caught.exception))

    def test_probe_media_rejects_noninteger_float_dimensions(self) -> None:
        for field in ("width", "height"):
            payload = self._valid_probe_payload()
            payload["streams"][0][field] = 640.0
            with self.subTest(field=field):
                with patch(
                    "robometanorm.evidence.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout=json.dumps(payload), stderr=""
                    ),
                ):
                    with self.assertRaises(ValueError):
                        probe_media(self.video_path / "fictional.mp4")

        with (
            patch(
                "robometanorm.evidence.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="{}", stderr="secret"
                ),
            ),
            patch(
                "robometanorm.evidence.json.loads",
                side_effect=RecursionError("secret recursion"),
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                probe_media(self.video_path / "fictional.mp4")
        self.assertNotIn("secret", str(caught.exception))

    def test_probe_media_propagates_memory_error(self) -> None:
        with patch(
            "robometanorm.evidence.subprocess.run", side_effect=MemoryError("oom")
        ):
            with self.assertRaises(MemoryError):
                probe_media(self.video_path / "fictional.mp4")

    def test_extract_midpoint_frame_uses_duration_scale_and_safe_command(self) -> None:
        output_path = self.root / "new-parent" / "frame.jpg"

        def run(command: list[str], **kwargs: object) -> object:
            self.assertTrue(output_path.parent.is_dir())
            Path(command[-1]).write_bytes(b"jpeg-frame")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(duration=10.0),
            ) as probe,
            patch("robometanorm.evidence.subprocess.run", side_effect=run) as mocked_run,
        ):
            result = extract_midpoint_frame(
                self.video_path / "fictional.mp4", output_path
            )

        self.assertEqual(result, output_path)
        self.assertEqual(output_path.read_bytes(), b"jpeg-frame")
        probe.assert_called_once_with(self.video_path / "fictional.mp4")
        command = mocked_run.call_args.args[0]
        self.assertIs(type(command), list)
        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(float(command[command.index("-ss") + 1]), 5.0)
        self.assertEqual(
            command[command.index("-vf") + 1],
            "scale=1280:1280:force_original_aspect_ratio=decrease",
        )
        self.assertEqual(command[-1], str(output_path))
        self.assertEqual(
            mocked_run.call_args.kwargs,
            {"capture_output": True, "text": True, "check": False},
        )

    def test_extract_midpoint_frame_propagates_unexpected_resolve_error(self) -> None:
        with (
            patch(
                "pathlib.Path.resolve",
                side_effect=RuntimeError("unexpected resolve failure"),
            ),
            patch("robometanorm.evidence.probe_media") as probe,
            patch("robometanorm.evidence.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected resolve failure"):
                extract_midpoint_frame(
                    self.video_path / "fictional.mp4", self.root / "frame.jpg"
                )

        probe.assert_not_called()
        run.assert_not_called()

    def test_extract_midpoint_frame_rejects_unusable_duration(self) -> None:
        for duration in (None, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                with (
                    patch(
                        "robometanorm.evidence.probe_media",
                        return_value=self._probed_sample(duration=duration),
                    ),
                    patch("robometanorm.evidence.subprocess.run") as run,
                ):
                    with self.assertRaises(ValueError):
                        extract_midpoint_frame(
                            self.video_path / "fictional.mp4",
                            self.root / "frame.jpg",
                        )
                run.assert_not_called()

    def test_extract_midpoint_frame_rejects_tool_failure_or_missing_output(self) -> None:
        results = (
            SimpleNamespace(returncode=1, stdout="", stderr="secret"),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        for result in results:
            with self.subTest(returncode=result.returncode):
                output_path = self.root / f"frame-{result.returncode}.jpg"
                with (
                    patch(
                        "robometanorm.evidence.probe_media",
                        return_value=self._probed_sample(),
                    ),
                    patch(
                        "robometanorm.evidence.subprocess.run", return_value=result
                    ),
                ):
                    with self.assertRaises(ValueError) as caught:
                        extract_midpoint_frame(
                            self.video_path / "fictional.mp4", output_path
                        )
                self.assertNotIn("secret", str(caught.exception))

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.subprocess.run",
                side_effect=OSError(f"secret {self.dataset_path}"),
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                extract_midpoint_frame(
                    self.video_path / "fictional.mp4", self.root / "frame-os.jpg"
                )
        self.assertNotIn(str(self.dataset_path), str(caught.exception))

    def test_extract_midpoint_frame_removes_stale_output_before_running(self) -> None:
        output_path = self.root / "stale-frame.jpg"
        output_path.write_bytes(b"stale-frame-must-not-count")
        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ),
        ):
            with self.assertRaises(ValueError):
                extract_midpoint_frame(
                    self.video_path / "fictional.mp4", output_path
                )

        self.assertFalse(output_path.exists())

    def test_extract_midpoint_frame_discards_partial_output_on_safe_errors(self) -> None:
        tool_error_output = self.root / "partial-tool-error.jpg"

        def write_then_fail(command: list[str], **kwargs: object) -> object:
            del kwargs
            Path(command[-1]).write_bytes(b"partial-tool-output")
            raise OSError("tool failed after writing")

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.subprocess.run",
                side_effect=write_then_fail,
            ),
        ):
            with self.assertRaises(ValueError):
                extract_midpoint_frame(
                    self.video_path / "fictional.mp4", tool_error_output
                )
        self.assertFalse(tool_error_output.exists())

        stat_error_output = self.root / "partial-stat-error.jpg"

        def write_frame(command: list[str], **kwargs: object) -> object:
            del kwargs
            Path(command[-1]).write_bytes(b"partial-stat-output")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.subprocess.run",
                side_effect=write_frame,
            ),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat", side_effect=OSError("stat failed")),
        ):
            with self.assertRaises(ValueError):
                extract_midpoint_frame(
                    self.video_path / "fictional.mp4", stat_error_output
                )
        self.assertFalse(stat_error_output.exists())

    def test_extract_midpoint_frame_never_uses_source_media_as_output(self) -> None:
        media_path = self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/immutable.mp4",
            payload=b"immutable-video",
        )
        before_hash = self._sha256(media_path)
        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch("robometanorm.evidence.subprocess.run") as run,
        ):
            with self.assertRaises(ValueError):
                extract_midpoint_frame(media_path, media_path)

        run.assert_not_called()
        self.assertEqual(self._sha256(media_path), before_hash)

    def test_extract_midpoint_frame_propagates_memory_error(self) -> None:
        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.subprocess.run",
                side_effect=MemoryError("oom"),
            ),
        ):
            with self.assertRaises(MemoryError):
                extract_midpoint_frame(
                    self.video_path / "fictional.mp4", self.root / "frame.jpg"
                )

    def test_dataset_evidence_frames_exist_inside_context_and_are_removed_after(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ),
        ):
            with collect_dataset_evidence(
                self.candidate, self._camera_info()
            ) as evidence:
                frame_paths = tuple(
                    sample.frame_path
                    for camera in evidence.cameras
                    for sample in camera.samples
                )
                self.assertTrue(frame_paths)
                self.assertTrue(
                    all(path is not None and path.is_file() for path in frame_paths)
                )
                self.assertTrue(
                    all(
                        path is not None
                        and path.name.startswith("frame-")
                        and path.parent.name.startswith("robometanorm-mini-")
                        for path in frame_paths
                    )
                )

        self.assertTrue(
            all(path is not None and not path.exists() for path in frame_paths)
        )

    def test_dataset_evidence_removes_frames_after_context_exception(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )
        frame_path: Path | None = None

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                with collect_dataset_evidence(
                    self.candidate, self._camera_info()
                ) as evidence:
                    frame_path = evidence.cameras[0].samples[0].frame_path
                    self.assertIsNotNone(frame_path)
                    assert frame_path is not None
                    self.assertTrue(frame_path.is_file())
                    raise RuntimeError("consumer failed")

        self.assertIsNotNone(frame_path)
        assert frame_path is not None
        self.assertFalse(frame_path.exists())

    def test_static_image_survives_dataset_context_unchanged(self) -> None:
        image_path = self._write_media(
            self.depth_path,
            f"chunk-000/{self.source_key}/episode_000000.png",
            payload=b"immutable-static-image",
        )
        before_hash = self._sha256(image_path)

        with collect_dataset_evidence(
            self.candidate, self._camera_info()
        ) as evidence:
            self.assertEqual(evidence.cameras[0].samples[0].frame_path, image_path)
            self.assertTrue(image_path.is_file())

        self.assertTrue(image_path.is_file())
        self.assertEqual(self._sha256(image_path), before_hash)

    def test_two_cameras_with_same_basename_get_distinct_safe_frame_paths(self) -> None:
        source_keys = (
            "observation.images.left_camera",
            "observation.images.right_camera",
        )
        source_info = {
            "features": {
                source_key: {"dtype": "video", "shape": [1, 1, 3]}
                for source_key in source_keys
            }
        }
        for source_key in source_keys:
            self._write_media(
                self.video_path,
                f"chunk-000/{source_key}/episode_000000.mp4",
            )

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ),
        ):
            cameras, issues = collect_camera_evidence(
                self.candidate, source_info, self.temp_frames
            )

        frame_paths = [camera.samples[0].frame_path for camera in cameras]
        self.assertEqual(len(set(frame_paths)), 2)
        self.assertTrue(
            all(
                path is not None and path.parent == self.temp_frames
                for path in frame_paths
            )
        )
        self.assertEqual(issues, ())

    def test_media_issue_evidence_is_strict_json_safe(self) -> None:
        self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.mp4",
        )
        with patch(
            "robometanorm.evidence.probe_media", side_effect=ValueError("unsafe")
        ):
            _, issues = collect_camera_evidence(
                self.candidate, self._camera_info(), self.temp_frames
            )

        json.dumps(issues[0].evidence, ensure_ascii=False, allow_nan=False)

    def test_dataset_evidence_aggregates_issues_in_declared_order(self) -> None:
        identity_issue = Issue("IDENTITY_REVIEW", "identity", "identity")
        machine_issue = Issue("MACHINE_REVIEW", "machine", "machine")
        camera_issue = Issue("CAMERA_REVIEW", "camera", "camera")
        identity = IdentityEvidence(
            "missing", None, "missing", None, "missing", (), (identity_issue,)
        )
        expected_source_info = self._camera_info()
        observed_temp_paths: list[Path] = []

        def collect_cameras(
            candidate: DatasetCandidate,
            source_info: object,
            temp_frames: Path,
        ) -> tuple[tuple[CameraEvidence, ...], tuple[Issue, ...]]:
            observed_temp_paths.append(temp_frames)
            self.assertEqual(candidate, self.candidate)
            self.assertEqual(source_info, expected_source_info)
            self.assertTrue(temp_frames.name.startswith("robometanorm-mini-"))
            return (), (camera_issue,)

        with (
            patch(
                "robometanorm.evidence.collect_identity_evidence",
                return_value=identity,
            ),
            patch(
                "robometanorm.evidence.collect_machine_evidence",
                return_value=((), (machine_issue,)),
            ),
            patch(
                "robometanorm.evidence.collect_camera_evidence",
                side_effect=collect_cameras,
            ),
        ):
            with collect_dataset_evidence(
                self.candidate, expected_source_info
            ) as evidence:
                self.assertEqual(
                    evidence.issues,
                    (identity_issue, machine_issue, camera_issue),
                )
                self.assertEqual(evidence.source_info, expected_source_info)
                self.assertIsNot(evidence.source_info, expected_source_info)
                self.assertTrue(observed_temp_paths[0].is_dir())

        self.assertFalse(observed_temp_paths[0].exists())

    def test_collection_does_not_mutate_inputs_media_or_create_persistent_artifacts(self) -> None:
        source_info = self._camera_info()
        source_snapshot = deepcopy(source_info)
        media_path = self._write_media(
            self.video_path,
            f"chunk-000/{self.source_key}/episode_000000.avi",
            payload=b"immutable-video",
        )
        info_hash = self._sha256(self.info_path)
        media_hash = self._sha256(media_path)
        before_files = {
            path.relative_to(self.dataset_path).as_posix()
            for path in self.dataset_path.rglob("*")
            if path.is_file()
        }

        with (
            patch(
                "robometanorm.evidence.probe_media",
                return_value=self._probed_sample(),
            ),
            patch(
                "robometanorm.evidence.extract_midpoint_frame",
                side_effect=self._create_frame,
            ),
        ):
            with collect_dataset_evidence(self.candidate, source_info):
                pass

        after_files = {
            path.relative_to(self.dataset_path).as_posix()
            for path in self.dataset_path.rglob("*")
            if path.is_file()
        }
        self.assertEqual(source_info, source_snapshot)
        self.assertEqual(self._sha256(self.info_path), info_hash)
        self.assertEqual(self._sha256(media_path), media_hash)
        self.assertEqual(after_files, before_files)
        self.assertEqual(list(self.dataset_path.rglob(".robometanorm_cache")), [])
        self.assertEqual(list(self.dataset_path.rglob("preview*")), [])

    def _camera_info(self) -> dict[str, object]:
        return {
            "robot_type": "acme_testbot",
            "features": {
                self.source_key: {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "fps": 30,
                    "codec": "h264",
                }
            },
        }

    @staticmethod
    def _valid_probe_payload() -> dict[str, object]:
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "r_frame_rate": "20/1",
                    "width": 640,
                    "height": 480,
                    "duration": "10.0",
                    "pix_fmt": "yuv420p",
                }
            ],
            "format": {"duration": "10.0"},
        }

    @staticmethod
    def _probed_sample(duration: float | None = 10.0) -> MediaSample:
        return MediaSample(
            relative_path="ignored",
            media_type="video",
            codec="h264",
            fps=20.0,
            width=640,
            height=480,
            duration_seconds=duration,
            pixel_format="yuv420p",
            frame_path=None,
        )

    @staticmethod
    def _create_frame(
        media_path: Path,
        output_path: Path,
        *,
        duration_seconds: float | None = None,
    ) -> Path:
        del media_path
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("invalid fixture duration")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"jpeg-fixture")
        return output_path

    @staticmethod
    def _write_media(root: Path, relative_path: str, payload: bytes = b"media") -> Path:
        output_path = root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        return output_path

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
