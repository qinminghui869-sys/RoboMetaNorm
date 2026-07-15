"""Raw robot identity evidence collection tests."""

from __future__ import annotations

from copy import deepcopy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.evidence import collect_identity_evidence, read_info
from robometanorm.models import DatasetCandidate, Issue, LayoutType


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


if __name__ == "__main__":
    unittest.main()
