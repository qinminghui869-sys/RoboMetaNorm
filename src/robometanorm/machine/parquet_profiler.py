"""以受限样本读取 Parquet schema 和机器向量画像。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from robometanorm.machine.models import ParquetProfile, VectorProfile


def profile_parquet(parquet_path: Path, sample_rows: int = 512) -> ParquetProfile:
    """读取 schema 和首个有限 batch，不加载整份 Parquet。"""
    if sample_rows <= 0:
        raise ValueError("sample_rows 必须为正数")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("P2 需要安装 pyarrow 才能读取 Parquet") from error

    parquet_file = pq.ParquetFile(parquet_path)
    batch_iterator = parquet_file.iter_batches(batch_size=sample_rows)
    try:
        batch = next(batch_iterator)
    except StopIteration:
        batch = None
    schema_columns = tuple(parquet_file.schema_arrow.names)
    if batch is None:
        return ParquetProfile(
            row_count=parquet_file.metadata.num_rows,
            row_group_count=parquet_file.metadata.num_row_groups,
            schema_columns=schema_columns,
            columns={},
            samples={},
        )

    table = batch.to_pydict()
    samples: dict[str, np.ndarray] = {}
    columns: dict[str, VectorProfile] = {}
    for column_name, values in table.items():
        vector_values = _as_vector_array(values)
        if vector_values is None:
            continue
        samples[column_name] = vector_values
        columns[column_name] = _build_vector_profile(column_name, vector_values)
    return ParquetProfile(
        row_count=parquet_file.metadata.num_rows,
        row_group_count=parquet_file.metadata.num_row_groups,
        schema_columns=schema_columns,
        columns=columns,
        samples=samples,
    )


def profile_parquets(
    parquet_paths: Sequence[Path], sample_rows: int = 512
) -> ParquetProfile:
    """比较每个 Episode 的有限样本布局，并返回首个 Episode 的画像。"""
    if not parquet_paths:
        raise ValueError("至少需要一个 Parquet 文件")
    profiles = [profile_parquet(path, sample_rows) for path in parquet_paths]
    reference = profiles[0]
    inconsistent = _find_inconsistent_columns(reference, profiles[1:])
    return replace(
        reference,
        episode_count=len(profiles),
        inconsistent_columns=tuple(sorted(inconsistent)),
    )


def _find_inconsistent_columns(
    reference: ParquetProfile, others: Sequence[ParquetProfile]
) -> set[str]:
    """比较 schema 与实测向量长度，找出跨 Episode 不一致的列。"""
    inconsistent: set[str] = set()
    reference_names = set(reference.schema_columns)
    for profile in others:
        current_names = set(profile.schema_columns)
        inconsistent.update(reference_names ^ current_names)
        for name in reference_names & current_names:
            reference_length = reference.columns.get(name)
            current_length = profile.columns.get(name)
            if reference_length is None or current_length is None:
                if reference_length is not current_length:
                    inconsistent.add(name)
            elif reference_length.vector_length != current_length.vector_length:
                inconsistent.add(name)
    return inconsistent


def _as_vector_array(values: object) -> np.ndarray | None:
    """仅接受长度固定且数值化的 list 向量列。"""
    if not isinstance(values, list) or not values or not all(isinstance(item, list) for item in values):
        return None
    lengths = {len(item) for item in values}
    if len(lengths) != 1:
        return None
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _build_vector_profile(column_name: str, samples: np.ndarray) -> VectorProfile:
    """从有限向量样本计算不含 NaN/Inf 的统计量。"""
    flattened = samples.reshape(-1)
    finite = flattened[np.isfinite(flattened)]
    total = flattened.size
    if finite.size == 0:
        statistics = (None, None, None, None, None, None, None)
    else:
        statistics = (
            float(np.min(finite)),
            float(np.max(finite)),
            float(np.percentile(finite, 1)),
            float(np.percentile(finite, 50)),
            float(np.percentile(finite, 99)),
            float(np.mean(finite)),
            float(np.std(finite)),
        )
    mean_abs_diff, max_abs_diff = _frame_difference_statistics(samples)
    return VectorProfile(
        column_name=column_name,
        vector_length=int(samples.shape[1]),
        min_value=statistics[0],
        max_value=statistics[1],
        p01=statistics[2],
        p50=statistics[3],
        p99=statistics[4],
        mean_value=statistics[5],
        std_value=statistics[6],
        nan_ratio=float(np.isnan(flattened).sum() / total),
        inf_ratio=float(np.isinf(flattened).sum() / total),
        mean_abs_diff=mean_abs_diff,
        max_abs_diff=max_abs_diff,
        adjacent_correlation=_adjacent_correlation(samples),
        mean_vector_norm=_mean_vector_norm(samples),
        triplet_grouping_possible=bool(samples.shape[1] % 3 == 0),
        quaternion_norm_valid=_quaternion_norm_valid(samples),
    )


def _frame_difference_statistics(samples: np.ndarray) -> tuple[float | None, float | None]:
    """计算有限样本的跨帧绝对变化，NaN/Inf 不参与统计。"""
    if samples.shape[0] < 2:
        return None, None
    differences = np.abs(np.diff(samples, axis=0)).reshape(-1)
    finite = differences[np.isfinite(differences)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.max(finite))


def _adjacent_correlation(samples: np.ndarray) -> float | None:
    """计算相邻维度的平均相关性，常量维度不参与计算。"""
    if samples.shape[0] < 2 or samples.shape[1] < 2:
        return None
    correlations: list[float] = []
    for index in range(samples.shape[1] - 1):
        pair = samples[:, index : index + 2]
        valid_pair = pair[np.isfinite(pair).all(axis=1)]
        if valid_pair.shape[0] < 2 or np.std(valid_pair[:, 0]) == 0 or np.std(valid_pair[:, 1]) == 0:
            continue
        correlations.append(float(np.corrcoef(valid_pair[:, 0], valid_pair[:, 1])[0, 1]))
    return float(np.mean(correlations)) if correlations else None


def _mean_vector_norm(samples: np.ndarray) -> float | None:
    """计算完整有限向量的平均 L2 模长。"""
    finite_rows = samples[np.isfinite(samples).all(axis=1)]
    if finite_rows.size == 0:
        return None
    return float(np.mean(np.linalg.norm(finite_rows, axis=1)))


def _quaternion_norm_valid(samples: np.ndarray) -> bool:
    """仅为四维向量提供四元数模长接近 1 的弱证据。"""
    if samples.shape[1] != 4:
        return False
    finite_rows = samples[np.isfinite(samples).all(axis=1)]
    if finite_rows.size == 0:
        return False
    norms = np.linalg.norm(finite_rows, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= 0.1))
