"""Small, fail-closed orchestration for the mini normalization pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import importlib.metadata
import math
from pathlib import Path
import time

from robometanorm.adapters.filesystem import discover_datasets
from robometanorm.annotation import (
    compile_annotation,
    has_main_follower_candidate,
    preflight_annotation,
)
from robometanorm.evidence import (
    collect_dataset_evidence,
    collect_mapped_gripper_ranges,
    read_info,
)
from robometanorm.models import (
    DatasetCandidate,
    DatasetEvidence,
    DatasetMapping,
    DatasetResult,
    DatasetStatus,
    HardwareProfile,
    IdentityEvidence,
    Issue,
    LayoutType,
    NormalizationResult,
    RobotIdentityFact,
)
from robometanorm.standard import apply_standard, check_preconditions
from robometanorm.vlm import DatasetVlm
from robometanorm.writer import build_review_payload, write_outputs


_RESEARCH_INVALID = Issue(
    code="HARDWARE_RESEARCH_INVALID",
    message="The hardware research result did not satisfy the pipeline contract.",
    scope="vlm",
)
_MAPPING_INVALID = Issue(
    code="DATASET_MAPPING_INVALID",
    message="The dataset mapping result did not satisfy the pipeline contract.",
    scope="vlm",
)
_IDENTITY_UNRESOLVED = Issue(
    code="HARDWARE_IDENTITY_UNRESOLVED",
    message="The researched manufacturer and model were not safe non-empty text.",
    scope="vlm",
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
                vlm_attempted = False

                if not any(issue.severity == "block" for issue in precondition_issues):
                    vlm_attempted = True
                    _report_stage(stage, index, total, candidate, "查询硬件身份")
                    profile, research_issue = _research_hardware(
                        vlm,
                        evidence.identity,
                        deadline_monotonic=deadline_monotonic,
                    )
                    if research_issue is not None:
                        extra_issues.append(research_issue)
                        known_issue_count = len(evidence.issues) + len(extra_issues)
                    elif profile is not None and not _profile_has_safe_identity(profile):
                        extra_issues.append(_IDENTITY_UNRESOLVED)
                        known_issue_count = len(evidence.issues) + len(extra_issues)
                    elif profile is not None:
                        _report_stage(stage, index, total, candidate, "映射相机与关节")
                        mapping, mapping_issue = _map_dataset(
                            vlm,
                            evidence,
                            profile,
                            deadline_monotonic=deadline_monotonic,
                        )
                        if mapping_issue is not None:
                            extra_issues.append(mapping_issue)
                            known_issue_count = len(evidence.issues) + len(extra_issues)
                        elif mapping is not None:
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
                status = status_from_issues(normalization.issues)
                annotation: dict[str, object] | None = None
                if status is DatasetStatus.PASS or (
                    vlm_attempted and has_main_follower_candidate(evidence)
                ):
                    annotation_result = compile_annotation(
                        evidence,
                        profile,
                        mapping,
                        normalized_info=normalization.normalized_info,
                        confidence_threshold=confidence_threshold,
                    )
                    if annotation_result.issues:
                        normalization = replace(
                            normalization,
                            issues=(*normalization.issues, *annotation_result.issues),
                        )
                        known_issue_count = len(normalization.issues)
                        status = status_from_issues(normalization.issues)
                    elif status is DatasetStatus.PASS:
                        annotation = annotation_result.document
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
                    annotation=annotation,
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


def _research_hardware(
    vlm: DatasetVlm,
    identity: IdentityEvidence,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[HardwareProfile | None, Issue | None]:
    if deadline_monotonic is None:
        raw_result = vlm.research_hardware(identity)
    else:
        raw_result = vlm.research_hardware(
            identity,
            deadline_monotonic=deadline_monotonic,
        )
    if type(raw_result) is not tuple or len(raw_result) != 2:
        return None, _RESEARCH_INVALID
    value, issue = raw_result
    if issue is not None:
        return None, issue if isinstance(issue, Issue) else _RESEARCH_INVALID
    if isinstance(value, HardwareProfile):
        return value, None
    return None, _RESEARCH_INVALID


def _map_dataset(
    vlm: DatasetVlm,
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[DatasetMapping | None, Issue | None]:
    if deadline_monotonic is None:
        raw_result = vlm.map_dataset(evidence, profile)
    else:
        raw_result = vlm.map_dataset(
            evidence,
            profile,
            deadline_monotonic=deadline_monotonic,
        )
    if type(raw_result) is not tuple or len(raw_result) != 2:
        return None, _MAPPING_INVALID
    value, issue = raw_result
    if issue is not None:
        return None, issue if isinstance(issue, Issue) else _MAPPING_INVALID
    if isinstance(value, DatasetMapping):
        return value, None
    return None, _MAPPING_INVALID


def _profile_has_safe_identity(profile: HardwareProfile) -> bool:
    fact = profile.identity
    return (
        isinstance(fact, RobotIdentityFact)
        and _safe_text(fact.manufacturer)
        and _safe_text(fact.model)
    )


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
        result.robot_identity,
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
