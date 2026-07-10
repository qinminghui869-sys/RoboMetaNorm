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


def write_normalization_files(
    candidate: DatasetCandidate,
    normalized_info: Mapping[str, object],
    report: PreconditionReport,
    *,
    camera_review_items: Sequence[CameraReviewItem] = (),
    phase: str = "P0",
) -> None:
    """写入 P0/P1 规范建议和人工复核文件。"""
    output_info = deepcopy(dict(normalized_info))
    review = _build_review(candidate, report, camera_review_items, phase)
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
        "status": _status_with_camera_reviews(report.status, camera_review_items).value,
        "review_required": bool(report.review_items or camera_review_items),
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
    }


def _status_with_camera_reviews(
    status: DatasetStatus, camera_review_items: Sequence[CameraReviewItem]
) -> DatasetStatus:
    """相机复核项不会降低既有 BLOCKED 或 ERROR 状态。"""
    if camera_review_items and status is DatasetStatus.PASS:
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
