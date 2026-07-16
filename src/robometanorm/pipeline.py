"""Small, fail-closed orchestration for the mini normalization pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib.metadata
import math
from pathlib import Path
import time

from robometanorm.adapters.filesystem import discover_datasets
from robometanorm.annotation import (
    compile_annotation,
    preflight_annotation,
)
from robometanorm.evidence import (
    collect_dataset_evidence,
    collect_mapped_gripper_ranges,
    read_info,
)
from robometanorm.models import (
    DatasetCandidate,
    DatasetAnalysis,
    DatasetEvidence,
    DatasetMapping,
    DatasetResult,
    DatasetStatus,
    HardwareProfile,
    Issue,
    LayoutType,
    NormalizationResult,
)
from robometanorm.standard import apply_standard, check_preconditions
from robometanorm.vlm import DatasetVlm
from robometanorm.writer import build_review_payload, write_outputs


_ANALYSIS_INVALID = Issue(
    code="DATASET_ANALYSIS_INVALID",
    message="The dataset analysis result did not satisfy the pipeline contract.",
    scope="vlm",
)
_LOCAL_ROBOT_TYPE_UNAVAILABLE = Issue(
    code="LOCAL_ROBOT_TYPE_UNAVAILABLE",
    message="The local robot_type was not a safe non-empty string.",
    scope="robot_type",
)


ProgressCallback = Callable[[int, int, DatasetResult], None]
StageCallback = Callable[[int, int, DatasetCandidate, str], None]


def status_from_issues(issues: Sequence[Issue]) -> DatasetStatus:
    """Return the most severe status; unknown input fails closed to ERROR."""

    priority = {
        "review": DatasetStatus.REVIEW,
        "block": DatasetStatus.BLOCKED,
        "error": DatasetStatus.ERROR,
    }
    status = DatasetStatus.PASS
    rank = {
        DatasetStatus.PASS: 0,
        DatasetStatus.REVIEW: 1,
        DatasetStatus.BLOCKED: 2,
        DatasetStatus.ERROR: 3,
    }
    try:
        iterator = iter(issues)
    except TypeError:
        return DatasetStatus.ERROR
    for issue in iterator:
        if not isinstance(issue, Issue) or type(issue.severity) is not str:
            return DatasetStatus.ERROR
        candidate = priority.get(issue.severity)
        if candidate is None:
            return DatasetStatus.ERROR
        if rank[candidate] > rank[status]:
            status = candidate
    return status


def scan_datasets(
    root: Path,
    layout: LayoutType = LayoutType.AUTO,
    *,
    progress: ProgressCallback | None = None,
) -> list[DatasetResult]:
    """Collect evidence and preconditions without VLM calls or persistent writes."""

    candidates = list(discover_datasets(root, layout))
    results: list[DatasetResult] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        source_info: dict[str, object] | None = None
        camera_count = 0
        machine_count = 0
        known_issue_count = 0
        try:
            source_info = read_info(candidate)
            with collect_dataset_evidence(candidate, source_info) as evidence:
                camera_count = len(evidence.cameras)
                machine_count = len(evidence.machines)
                known_issue_count = len(evidence.issues)
                precondition_issues = (
                    *check_preconditions(evidence),
                    *preflight_annotation(evidence),
                )
                issues = (*evidence.issues, *precondition_issues)
                known_issue_count = len(issues)
                status = status_from_issues(issues)
        except MemoryError:
            raise
        except Exception:
            _append_result(
                results,
                _error_result(
                    candidate,
                    source_info,
                    camera_count,
                    machine_count,
                    0,
                    known_issue_count,
                ),
                index=index,
                total=total,
                progress=progress,
            )
            continue
        _append_result(
            results,
            DatasetResult(
                candidate=candidate,
                status=status,
                camera_count=camera_count,
                machine_field_count=machine_count,
                changed_field_count=0,
                issue_count=known_issue_count,
                source_info=source_info,
            ),
            index=index,
            total=total,
            progress=progress,
        )
    return results


def normalize_datasets(
    root: Path,
    layout: LayoutType = LayoutType.AUTO,
    *,
    vlm: DatasetVlm,
    confidence_threshold: float,
    progress: ProgressCallback | None = None,
    stage: StageCallback | None = None,
    dataset_timeout_seconds: float | None = None,
) -> list[DatasetResult]:
    """Normalize every discovered dataset independently and conservatively."""

    _validate_confidence_threshold(confidence_threshold)
    _validate_dataset_timeout_seconds(dataset_timeout_seconds)
    generator = _generator_identity()
    candidates = list(discover_datasets(root, layout))
    results: list[DatasetResult] = []
    total = len(candidates)

    for index, candidate in enumerate(candidates, start=1):
        deadline_monotonic = (
            time.monotonic() + dataset_timeout_seconds
            if dataset_timeout_seconds is not None
            else None
        )
        source_info: dict[str, object] | None = None
        camera_count = 0
        machine_count = 0
        known_changed_count = 0
        known_issue_count = 0
        try:
            _report_stage(stage, index, total, candidate, "读取本地证据")
            source_info = read_info(candidate)
            with collect_dataset_evidence(candidate, source_info) as evidence:
                camera_count = len(evidence.cameras)
                machine_count = len(evidence.machines)
                known_issue_count = len(evidence.issues)
                precondition_issues = (
                    *check_preconditions(evidence),
                    *preflight_annotation(evidence),
                )
                profile: HardwareProfile | None = None
                mapping: DatasetMapping | None = None
                extra_issues: list[Issue] = list(precondition_issues)
                known_issue_count = len(evidence.issues) + len(extra_issues)
                robot_type = source_info.get("robot_type")
                if not _safe_text(robot_type):
                    extra_issues.append(_LOCAL_ROBOT_TYPE_UNAVAILABLE)
                elif not any(
                    issue.severity == "block" for issue in precondition_issues
                ):
                    _report_stage(stage, index, total, candidate, "分析相机与关节")
                    analysis, analysis_issue = _analyze_dataset(
                        vlm,
                        evidence,
                        robot_type,
                        deadline_monotonic=deadline_monotonic,
                    )
                    if analysis_issue is not None:
                        extra_issues.append(analysis_issue)
                    elif analysis is not None:
                        profile = analysis.profile
                        mapping = analysis.mapping
                        evidence, range_issues = collect_mapped_gripper_ranges(
                            candidate,
                            evidence,
                            profile,
                            mapping,
                        )
                        extra_issues.extend(range_issues)
                known_issue_count = len(evidence.issues) + len(extra_issues)

                normalization = apply_standard(
                    evidence,
                    profile,
                    mapping,
                    confidence_threshold=confidence_threshold,
                    extra_issues=tuple(extra_issues),
                )
                known_issue_count = len(normalization.issues)
                known_changed_count = _changed_field_count(normalization)
                annotation_result = compile_annotation(
                    evidence,
                    profile,
                    mapping,
                    normalized_info=normalization.normalized_info,
                    confidence_threshold=confidence_threshold,
                    existing_issues=normalization.issues,
                )
                if annotation_result.issues:
                    normalization = NormalizationResult(
                        normalized_info=normalization.normalized_info,
                        robot_identity=normalization.robot_identity,
                        camera_mappings=normalization.camera_mappings,
                        machine_mappings=normalization.machine_mappings,
                        issues=(*normalization.issues, *annotation_result.issues),
                    )
                known_issue_count = len(normalization.issues)
                status = status_from_issues(normalization.issues)
                review = build_review_payload(
                    candidate,
                    status,
                    evidence,
                    profile,
                    normalization,
                    generator=generator,
                )
                write_outputs(
                    candidate,
                    normalization.normalized_info,
                    review,
                    annotation=annotation_result.document,
                )
        except MemoryError:
            raise
        except Exception:
            _append_result(
                results,
                _error_result(
                    candidate,
                    source_info,
                    camera_count,
                    machine_count,
                    known_changed_count,
                    known_issue_count,
                ),
                index=index,
                total=total,
                progress=progress,
            )
            continue

        _append_result(
            results,
            DatasetResult(
                candidate=candidate,
                status=status,
                camera_count=camera_count,
                machine_field_count=machine_count,
                changed_field_count=known_changed_count,
                issue_count=known_issue_count,
                source_info=source_info,
            ),
            index=index,
            total=total,
            progress=progress,
        )
    return results


def _analyze_dataset(
    vlm: DatasetVlm,
    evidence: DatasetEvidence,
    robot_type: str,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[DatasetAnalysis | None, Issue | None]:
    if deadline_monotonic is None:
        raw_result = vlm.analyze_dataset(evidence, robot_type)
    else:
        raw_result = vlm.analyze_dataset(
            evidence,
            robot_type,
            deadline_monotonic=deadline_monotonic,
        )
    if type(raw_result) is not tuple or len(raw_result) != 2:
        return None, _ANALYSIS_INVALID
    value, issue = raw_result
    if issue is not None:
        return None, issue if isinstance(issue, Issue) else _ANALYSIS_INVALID
    if (
        isinstance(value, DatasetAnalysis)
        and isinstance(value.profile, HardwareProfile)
        and isinstance(value.mapping, DatasetMapping)
    ):
        return value, None
    return None, _ANALYSIS_INVALID


def _safe_text(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and not any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    )


def _validate_confidence_threshold(confidence_threshold: object) -> None:
    if (
        type(confidence_threshold) not in {int, float}
        or not math.isfinite(confidence_threshold)
        or not 0 <= confidence_threshold <= 1
    ):
        raise ValueError("confidence_threshold must be a finite number from 0 to 1")


def _validate_dataset_timeout_seconds(dataset_timeout_seconds: object) -> None:
    if dataset_timeout_seconds is None:
        return
    if (
        type(dataset_timeout_seconds) not in {int, float}
        or not math.isfinite(dataset_timeout_seconds)
        or dataset_timeout_seconds <= 0
    ):
        raise ValueError("dataset_timeout_seconds must be a finite positive number")


def _generator_identity() -> dict[str, object]:
    try:
        version = importlib.metadata.version("robometanorm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"name": "robometanorm", "version": version}


def _changed_field_count(result: NormalizationResult) -> int:
    records = (
        *result.camera_mappings,
        *result.machine_mappings,
    )
    return sum(1 for record in records if record.changed)


def _append_result(
    results: list[DatasetResult],
    result: DatasetResult,
    *,
    index: int,
    total: int,
    progress: ProgressCallback | None,
) -> None:
    results.append(result)
    if progress is not None:
        try:
            progress(index, total, result)
        except MemoryError:
            raise
        except Exception:
            pass


def _report_stage(
    stage: StageCallback | None,
    index: int,
    total: int,
    candidate: DatasetCandidate,
    label: str,
) -> None:
    if stage is not None:
        try:
            stage(index, total, candidate, label)
        except MemoryError:
            raise
        except Exception:
            pass


def _error_result(
    candidate: DatasetCandidate,
    source_info: dict[str, object] | None,
    camera_count: int,
    machine_count: int,
    known_changed_count: int,
    known_issue_count: int,
) -> DatasetResult:
    return DatasetResult(
        candidate=candidate,
        status=DatasetStatus.ERROR,
        camera_count=camera_count,
        machine_field_count=machine_count,
        changed_field_count=known_changed_count,
        issue_count=known_issue_count + 1,
        source_info=source_info,
    )
