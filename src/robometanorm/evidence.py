"""Collect raw robot identity evidence from dataset metadata."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from robometanorm.models import DatasetCandidate, IdentityEvidence, Issue


def read_info(candidate: DatasetCandidate) -> dict[str, object]:
    """Read ``info.json`` and require a JSON object at its top level."""
    with candidate.info_path.open("r", encoding="utf-8") as file_handle:
        try:
            source_info = json.load(file_handle)
        except RecursionError as error:
            raise ValueError("info.json could not be parsed") from error
    if not isinstance(source_info, dict):
        raise ValueError("info.json must contain a JSON object")
    return source_info


def collect_identity_evidence(
    meta_path: Path, source_info: Mapping[str, object]
) -> IdentityEvidence:
    """Collect independent raw values and safe parse diagnostics."""
    issues: list[Issue] = []

    if "robot_type" in source_info:
        info_robot_type_state = "present"
        info_robot_type = source_info["robot_type"]
        if not isinstance(info_robot_type, str) or not info_robot_type.strip():
            issues.append(
                Issue(
                    code="INFO_ROBOT_TYPE_INVALID",
                    message="info.json robot_type must be a non-empty string",
                    scope="identity.info_robot_type",
                    evidence={"value_type": type(info_robot_type).__name__},
                )
            )
    else:
        info_robot_type_state = "missing"
        info_robot_type = None

    common_record_state, common_record, common_issue = _read_common_record(
        meta_path / "common_record.json"
    )
    if common_issue is not None:
        issues.append(common_issue)

    tasks_state, tasks, tasks_issue = _read_tasks(meta_path / "tasks.jsonl")
    if tasks_issue is not None:
        issues.append(tasks_issue)

    return IdentityEvidence(
        info_robot_type_state=info_robot_type_state,
        info_robot_type=info_robot_type,
        common_record_state=common_record_state,
        common_record=common_record,
        tasks_state=tasks_state,
        tasks=tasks,
        issues=tuple(issues),
    )


def _read_common_record(path: Path) -> tuple[str, object | None, Issue | None]:
    try:
        raw_content = path.read_bytes()
    except FileNotFoundError:
        return "missing", None, None
    except OSError as error:
        return (
            "unreadable",
            None,
            Issue(
                code="COMMON_RECORD_UNREADABLE",
                message="common_record.json could not be read",
                scope="identity.common_record",
                evidence={
                    "file_name": path.name,
                    "error_type": type(error).__name__,
                },
            ),
        )

    try:
        common_record = json.loads(raw_content.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        return (
            "invalid",
            None,
            Issue(
                code="COMMON_RECORD_INVALID",
                message="common_record.json is not valid UTF-8 JSON",
                scope="identity.common_record",
                evidence={
                    "file_name": path.name,
                    "error_type": type(error).__name__,
                },
            ),
        )
    return "present", common_record, None


def _read_tasks(path: Path) -> tuple[str, tuple[object, ...], Issue | None]:
    try:
        raw_content = path.read_bytes()
    except FileNotFoundError:
        return "missing", (), None
    except OSError as error:
        return (
            "unreadable",
            (),
            Issue(
                code="TASKS_UNREADABLE",
                message="tasks.jsonl could not be read",
                scope="identity.tasks",
                evidence={
                    "file_name": path.name,
                    "error_type": type(error).__name__,
                },
            ),
        )

    records: list[object] = []
    invalid_line_numbers: list[int] = []
    error_types: set[str] = set()
    for line_number, raw_line in enumerate(raw_content.splitlines(), start=1):
        try:
            line = raw_line.decode("utf-8")
            records.append(json.loads(line))
        except (ValueError, RecursionError) as error:
            invalid_line_numbers.append(line_number)
            error_types.add(type(error).__name__)

    if invalid_line_numbers:
        return (
            "invalid",
            tuple(records),
            Issue(
                code="TASKS_INVALID",
                message="tasks.jsonl contains invalid UTF-8 JSON lines",
                scope="identity.tasks",
                evidence={
                    "file_name": path.name,
                    "line_numbers": sorted(set(invalid_line_numbers)),
                    "error_types": sorted(error_types),
                },
            ),
        )
    return "present", tuple(records), None
