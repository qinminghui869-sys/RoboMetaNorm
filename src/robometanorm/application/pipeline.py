"""P0 扫描、检查和基础输出编排。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from robometanorm.adapters.filesystem import discover_datasets
from robometanorm.application.preconditions import check_preconditions
from robometanorm.camera.normalizer import normalize_cameras
from robometanorm.camera.vlm_classifier import VlmClassifier
from robometanorm.domain.models import (
    DatasetCandidate,
    DatasetResult,
    DatasetStatus,
    LayoutType,
    PreconditionReport,
    ReviewItem,
)
from robometanorm.machine.normalizer import normalize_machine_fields
from robometanorm.machine.parquet_profiler import profile_parquets
from robometanorm.machine.vlm_semantic_resolver import MachineVlmResolver
from robometanorm.writers.json_writer import write_normalization_files


def scan_datasets(root: Path, layout: LayoutType = LayoutType.AUTO) -> list[DatasetResult]:
    """发现数据集、读取 info.json 并执行 P0 前置检查。"""
    results: list[DatasetResult] = []
    for candidate in discover_datasets(root, layout):
        try:
            source_info = _read_info(candidate)
            report = check_preconditions(candidate, source_info)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            results.append(_error_result(candidate, error))
            continue
        results.append(
            DatasetResult(
                candidate=candidate,
                status=report.status,
                review_items=report.review_items,
                camera_count=report.camera_count,
                machine_field_count=report.machine_field_count,
                source_info=source_info,
            )
        )
    return results


def normalize_datasets(
    root: Path,
    layout: LayoutType = LayoutType.AUTO,
    *,
    vlm_classifier: VlmClassifier | None = None,
    machine_vlm_resolver: MachineVlmResolver | None = None,
    confidence_threshold: float = 0.85,
) -> list[DatasetResult]:
    """执行 P1/P2 规范建议并生成两个输出文件。"""
    results = scan_datasets(root, layout)
    for index, result in enumerate(results):
        if result.source_info is None:
            continue
        try:
            camera_result = normalize_cameras(
                result.candidate,
                result.source_info,
                vlm_classifier=vlm_classifier,
                confidence_threshold=confidence_threshold,
            )
            parquet_profile = _profile_first_parquet(result.candidate)
            machine_result = normalize_machine_fields(
                camera_result.normalized_info,
                parquet_profile,
                vlm_resolver=machine_vlm_resolver,
                dataset_name=result.candidate.dataset_name,
            )
            status = _status_after_reviews(
                result.status,
                camera_result.camera_review_items,
                machine_result.machine_review_items,
            )
            report = PreconditionReport(
                status=status,
                review_items=result.review_items,
                camera_count=result.camera_count,
                machine_field_count=result.machine_field_count,
            )
            write_normalization_files(
                result.candidate,
                machine_result.normalized_info,
                report,
                camera_review_items=camera_result.camera_review_items,
                machine_review_items=machine_result.machine_review_items,
                phase="P2",
            )
            results[index] = replace(
                result,
                status=status,
                camera_review_count=len(camera_result.camera_review_items),
                machine_review_count=len(machine_result.machine_review_items),
            )
        except (OSError, RuntimeError, ValueError) as error:
            results[index] = _error_result(result.candidate, error)
    return results


def _profile_first_parquet(candidate: DatasetCandidate):
    """读取受限样本并比较各 Episode 布局，不改写任何源数据。"""
    if candidate.data_path is None:
        return None
    parquet_paths = sorted(candidate.data_path.rglob("*.parquet"))
    if not parquet_paths:
        return None
    return profile_parquets(parquet_paths)


def _status_after_reviews(
    status: DatasetStatus,
    camera_review_items: tuple[object, ...],
    machine_review_items: tuple[object, ...],
) -> DatasetStatus:
    """相机或机器复核只会将 PASS 提升为 REVIEW。"""
    if (camera_review_items or machine_review_items) and status is DatasetStatus.PASS:
        return DatasetStatus.REVIEW
    return status


def _read_info(candidate: DatasetCandidate) -> dict[str, object]:
    """读取且限制 info.json 的顶层为 JSON 对象。"""
    with candidate.info_path.open("r", encoding="utf-8") as file_handle:
        source_info = json.load(file_handle)
    if not isinstance(source_info, dict):
        raise ValueError("info.json 顶层必须是 JSON 对象")
    return source_info


def _error_result(candidate: DatasetCandidate, error: Exception) -> DatasetResult:
    """将单数据集异常转换为可继续汇总的 ERROR 结果。"""
    review_item = ReviewItem(
        review_id="info_read_error",
        category="info_read_error",
        severity="block",
        reason=f"读取或写入元数据失败: {error}",
        evidence={"error_type": type(error).__name__},
        required_action="修复该数据集的元数据或文件权限后重新运行。",
    )
    return DatasetResult(
        candidate=candidate,
        status=DatasetStatus.ERROR,
        review_items=(review_item,),
        camera_count=0,
        machine_field_count=0,
        source_info=None,
    )
