"""规范建议与复核 JSON 的原子写入。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

from robometanorm import __version__
from robometanorm.camera.models import CameraReviewItem
from robometanorm.domain.models import DatasetCandidate, DatasetStatus, PreconditionReport
from robometanorm.machine.models import GripperTransformProposal, MachineReviewItem
from robometanorm.robot_identity import RobotIdentity, robot_identity_payload


def write_normalization_files(
    candidate: DatasetCandidate,
    normalized_info: Mapping[str, object],
    report: PreconditionReport,
    *,
    camera_review_items: Sequence[CameraReviewItem] = (),
    machine_review_items: Sequence[MachineReviewItem] = (),
    gripper_transform_proposals: Sequence[GripperTransformProposal] = (),
    robot_identity: RobotIdentity | None = None,
    phase: str = "P0",
) -> None:
    """写入 P0/P1/P2 规范建议和人工复核文件。"""
    output_info = deepcopy(dict(normalized_info))
    review = _build_review(
        candidate,
        report,
        camera_review_items,
        machine_review_items,
        gripper_transform_proposals,
        robot_identity,
        phase,
    )
    _validate_payloads(output_info, review)
    _write_pair_atomically(
        candidate.info_path.parent,
        {
            "info_norm.json": output_info,
            "info_norm_review.json": review,
        },
    )


def _build_review(
    candidate: DatasetCandidate,
    report: PreconditionReport,
    camera_review_items: Sequence[CameraReviewItem],
    machine_review_items: Sequence[MachineReviewItem],
    gripper_transform_proposals: Sequence[GripperTransformProposal],
    robot_identity: RobotIdentity | None,
    phase: str,
) -> dict[str, object]:
    """将领域复核项转换为公开 JSON 结构。"""
    return {
        "generator": {
            "version": __version__,
            "phase": phase,
            "source_info_mtime": candidate.info_path.stat().st_mtime_ns,
            "rule_version": "standard-2026-07",
        },
        "dataset": {
            "name": candidate.dataset_name,
            "layout_type": candidate.layout_type.value,
        },
        "robot_identity": (
            robot_identity_payload(robot_identity)
            if robot_identity is not None
            else None
        ),
        "status": _status_with_reviews(
            report.status, camera_review_items, machine_review_items
        ).value,
        "review_required": bool(
            report.review_items or camera_review_items or machine_review_items
        ),
        "review_items": [
            {
                "review_id": item.review_id,
                "category": item.category,
                "severity": item.severity,
                "reason": item.reason,
                "evidence": item.evidence,
                "required_action": item.required_action,
            }
            for item in report.review_items
        ],
        "camera_review_items": [
            {
                "source_key": item.source_key,
                "reason_code": item.reason_code,
                "candidates": [
                    {"target_key": candidate.target_key, "confidence": candidate.confidence}
                    for candidate in item.candidates
                ],
                "evidence": item.evidence,
                "human_decision": {
                    "status": "pending",
                    "selected_target_key": None,
                },
            }
            for item in camera_review_items
        ],
        "machine_review_items": [
            {
                "source_feature": item.source_feature,
                "source_slice": list(item.source_slice) if item.source_slice else None,
                "category": item.category,
                "severity": item.severity,
                "declared_names": list(item.declared_names),
                "vlm_result": item.vlm_result,
                "vlm_error": item.vlm_error,
                "candidates": list(item.candidates),
                "required_action": item.required_action,
                "human_decision": {
                    "status": "pending",
                    "selected_semantic": None,
                    "comment": None,
                },
            }
            for item in machine_review_items
        ],
        "gripper_transform_proposals": [
            {
                "source_feature": item.source_feature,
                "source_index": item.source_index,
                "source_name": item.source_name,
                "target_name": item.target_name,
                "source_closed": item.source_closed,
                "source_open": item.source_open,
                "target_range": list(item.target_range),
                "formula": item.formula,
                "clipping_policy": item.clipping_policy,
                "direction_evidence": item.direction_evidence,
                "range_evidence": item.range_evidence,
                "confidence": item.confidence,
                "transform_required": item.transform_required,
                "observed_profile": item.observed_profile,
            }
            for item in gripper_transform_proposals
        ],
    }


def _status_with_reviews(
    status: DatasetStatus,
    camera_review_items: Sequence[CameraReviewItem],
    machine_review_items: Sequence[MachineReviewItem],
) -> DatasetStatus:
    """P1/P2 复核项不会降低既有 BLOCKED 或 ERROR 状态。"""
    if (camera_review_items or machine_review_items) and status is DatasetStatus.PASS:
        return DatasetStatus.REVIEW
    return status


def _validate_payloads(normalized_info: object, review: object) -> None:
    """在替换目标文件前完成最小 JSON 结构校验。"""
    if not isinstance(normalized_info, Mapping):
        raise ValueError("info_norm.json 必须是 JSON 对象")
    if not isinstance(review, Mapping):
        raise ValueError("info_norm_review.json 必须是 JSON 对象")
    required_keys = {"generator", "status", "review_required", "review_items"}
    if not required_keys.issubset(review):
        raise ValueError("info_norm_review.json 缺少必需字段")
    try:
        json.dumps(normalized_info, ensure_ascii=False)
        json.dumps(review, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("输出内容无法序列化为 JSON") from error


def _write_pair_atomically(directory: Path, payloads: Mapping[str, object]) -> None:
    """先落盘全部临时文件，再逐个原子替换目标文件。"""
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for filename, payload in payloads.items():
            target_path = directory / filename
            temporary_paths.append((target_path, _write_temp_json(target_path, payload)))
        for target_path, temporary_path in temporary_paths:
            os.replace(temporary_path, target_path)
    finally:
        # 写入异常时清理尚未替换的临时文件。
        for _, temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _write_temp_json(target_path: Path, payload: object) -> Path:
    """在目标文件所在目录创建可原子替换的临时文件。"""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent
    )
    temporary_path = Path(temporary_name)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write("\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())
    return temporary_path
