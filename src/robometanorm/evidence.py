"""Collect raw robot identity evidence from dataset metadata."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, time, timedelta
from decimal import Decimal
import json
from numbers import Number
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robometanorm.models import (
    DatasetCandidate,
    FeatureSchema,
    IdentityEvidence,
    Issue,
    MachineEvidence,
    ParquetEpisodeEvidence,
)


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


def collect_machine_evidence(
    candidate: DatasetCandidate, source_info: Mapping[str, object]
) -> tuple[tuple[MachineEvidence, ...], tuple[Issue, ...]]:
    """Collect structure-only evidence from representative Parquet files."""
    schemas = _machine_feature_schemas(source_info)
    feature_keys = tuple(schema.source_key for schema in schemas)
    issues: list[Issue] = []
    seen_issues: set[tuple[str, str, str]] = set()
    episodes = tuple(
        _inspect_parquet_episode(
            candidate,
            parquet_path,
            feature_keys,
            issues,
            seen_issues,
        )
        for parquet_path in _representative_parquet_paths(candidate)
    )

    machines: list[MachineEvidence] = []
    for schema in schemas:
        observed_lengths = tuple(
            episode.vector_lengths.get(schema.source_key) for episode in episodes
        )
        all_lengths_known = len(episodes) > 0 and all(
            isinstance(length, int) for length in observed_lengths
        )
        episode_lengths = (
            tuple(
                length for length in observed_lengths if isinstance(length, int)
            )
            if all_lengths_known
            else ()
        )
        if len(set(episode_lengths)) > 1:
            first_length = episode_lengths[0]
            mismatched_episode = next(
                episode
                for episode in episodes
                if (length := episode.vector_lengths.get(schema.source_key))
                is not None
                and length != first_length
            )
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_VECTOR_LENGTH_INCONSISTENT",
                message="Machine feature vector length differs across Parquet episodes",
                relative_path=mismatched_episode.relative_path,
                feature=schema.source_key,
                length=mismatched_episode.vector_lengths[schema.source_key],
            )
        machines.append(
            MachineEvidence(
                schema=schema,
                episodes=episodes,
                episode_lengths=episode_lengths,
                gripper_ranges=(),
            )
        )
    return tuple(machines), tuple(issues)


def _machine_feature_schemas(
    source_info: Mapping[str, object],
) -> tuple[FeatureSchema, ...]:
    features = source_info.get("features")
    if not isinstance(features, Mapping):
        return ()
    schemas: list[FeatureSchema] = []
    for source_key, feature in features.items():
        if not isinstance(source_key, str) or not isinstance(feature, Mapping):
            continue
        if source_key not in {"action", "observation.state"} and not source_key.startswith(
            "observation.state."
        ):
            continue
        shape = feature.get("shape")
        names = feature.get("names")
        schemas.append(
            FeatureSchema(
                source_key=source_key,
                dtype=feature.get("dtype"),
                shape=tuple(shape) if isinstance(shape, (list, tuple)) else (),
                names=tuple(names) if isinstance(names, (list, tuple)) else (),
                fps=feature.get("fps"),
                codec=feature.get("codec"),
            )
        )
    return tuple(schemas)


def _representative_parquet_paths(
    candidate: DatasetCandidate,
) -> tuple[Path, ...]:
    data_path = candidate.data_path
    if data_path is None or not data_path.is_dir():
        return ()
    parquet_paths = sorted(
        (
            path
            for path in data_path.rglob("*")
            if path.is_file() and path.suffix == ".parquet"
        ),
        key=lambda path: path.relative_to(data_path).as_posix(),
    )
    if len(parquet_paths) <= 2:
        return tuple(parquet_paths)
    return parquet_paths[0], parquet_paths[-1]


def _inspect_parquet_episode(
    candidate: DatasetCandidate,
    parquet_path: Path,
    feature_keys: tuple[str, ...],
    issues: list[Issue],
    seen_issues: set[tuple[str, str, str]],
) -> ParquetEpisodeEvidence:
    relative_path = _relative_parquet_path(candidate, parquet_path)
    vector_lengths: dict[str, int | None] = dict.fromkeys(feature_keys)
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        schema_columns = tuple(parquet_file.schema_arrow.names)
    except (
        ValueError,
        OSError,
        pa.ArrowCapacityError,
        pa.ArrowNotImplementedError,
    ) as error:
        for feature in feature_keys:
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_READ_FAILED",
                message="Parquet machine feature could not be read",
                relative_path=relative_path,
                feature=feature,
                error_type=type(error).__name__,
            )
        return ParquetEpisodeEvidence(relative_path, (), vector_lengths)

    for feature in feature_keys:
        if feature not in schema_columns:
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_COLUMN_MISSING",
                message="Declared machine feature is missing from Parquet schema",
                relative_path=relative_path,
                feature=feature,
            )
            continue

        widths: set[int] = set()
        row_count = 0
        unknown_width = False
        ambiguous_projection = False
        try:
            for batch in parquet_file.iter_batches(
                columns=[feature], batch_size=512
            ):
                if batch.num_columns != 1 or tuple(batch.schema.names) != (feature,):
                    ambiguous_projection = True
                    continue
                for value in batch.column(0).to_pylist():
                    row_count += 1
                    width = _parquet_value_width(value)
                    if width is None:
                        unknown_width = True
                    else:
                        widths.add(width)
        except (
            ValueError,
            OSError,
            pa.ArrowCapacityError,
            pa.ArrowNotImplementedError,
        ) as error:
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_READ_FAILED",
                message="Parquet machine feature could not be read",
                relative_path=relative_path,
                feature=feature,
                error_type=type(error).__name__,
            )
            continue

        if ambiguous_projection:
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_COLUMN_AMBIGUOUS",
                message="Parquet projection did not resolve to one exact top-level column",
                relative_path=relative_path,
                feature=feature,
            )
        elif row_count == 0 or unknown_width or len(widths) != 1:
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_VECTOR_LENGTH_INCONSISTENT",
                message="Parquet machine feature has no single vector length",
                relative_path=relative_path,
                feature=feature,
            )
        else:
            vector_lengths[feature] = next(iter(widths))

    return ParquetEpisodeEvidence(relative_path, schema_columns, vector_lengths)


def _parquet_value_width(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, (str, bytes, Number, Decimal, date, time, timedelta)):
        return 1
    return None


def _relative_parquet_path(candidate: DatasetCandidate, parquet_path: Path) -> str:
    try:
        return parquet_path.relative_to(candidate.source_path).as_posix()
    except ValueError:
        if candidate.data_path is None:
            return parquet_path.name
        return parquet_path.relative_to(candidate.data_path).as_posix()


def _append_parquet_issue(
    issues: list[Issue],
    seen_issues: set[tuple[str, str, str]],
    *,
    code: str,
    message: str,
    relative_path: str,
    feature: str,
    length: int | None = None,
    error_type: str | None = None,
) -> None:
    issue_key = (relative_path, feature, code)
    if issue_key in seen_issues:
        return
    seen_issues.add(issue_key)
    evidence: dict[str, object] = {
        "relative_path": relative_path,
        "feature": feature,
    }
    if length is not None:
        evidence["length"] = length
    if error_type is not None:
        evidence["error_type"] = error_type
    issues.append(
        Issue(
            code=code,
            message=message,
            scope=f"machine.{feature}",
            evidence=evidence,
        )
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
