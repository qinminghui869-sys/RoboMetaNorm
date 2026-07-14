"""Parquet 有限样本画像与安全持久化缓存。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from robometanorm.machine.models import (
    GripperFrameSample,
    ParquetProfile,
    ProfileProgress,
    ScalarProfile,
    VectorProfile,
)


CACHE_VERSION = 3
METADATA_FILENAME = "parquet_profile_v3.json"
SAMPLES_FILENAME = "parquet_samples_v3.npz"


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
    parquet_paths: Sequence[Path],
    sample_rows: int = 512,
    progress: Callable[[ProfileProgress], None] | None = None,
    *,
    gripper_indices: Mapping[str, Sequence[int]] | None = None,
) -> ParquetProfile:
    """比较每个 Episode 的有限样本布局，并返回首个 Episode 的画像。"""
    if not parquet_paths:
        raise ValueError("至少需要一个 Parquet 文件")
    profiles: list[ParquetProfile] = []
    total = len(parquet_paths)
    for current, path in enumerate(parquet_paths, start=1):
        if progress is not None:
            progress(ProfileProgress("episode", current, total, path))
        profiles.append(profile_parquet(path, sample_rows))
    reference = profiles[0]
    inconsistent = _find_inconsistent_columns(reference, profiles[1:])
    return replace(
        reference,
        episode_count=total,
        inconsistent_columns=tuple(sorted(inconsistent)),
        gripper_profiles=_profile_gripper_dimensions(
            parquet_paths, gripper_indices or {}
        ),
    )


def _profile_gripper_dimensions(
    parquet_paths: Sequence[Path],
    gripper_indices: Mapping[str, Sequence[int]],
) -> dict[str, ScalarProfile]:
    """只投影夹爪所在向量列，并聚合首末 Episode 的目标维度。"""
    requested = {
        column_name: tuple(sorted(set(indices)))
        for column_name, indices in gripper_indices.items()
        if indices
    }
    if not requested:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("P2 需要安装 pyarrow 才能读取 Parquet") from error

    values_by_key: dict[str, list[np.ndarray]] = {
        _gripper_profile_key(column_name, index): []
        for column_name, indices in requested.items()
        for index in indices
    }
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        columns = [name for name in requested if name in available]
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
            payload = batch.to_pydict()
            for column_name, indices in requested.items():
                vector_values = _as_vector_array(payload.get(column_name))
                if vector_values is None:
                    continue
                for index in indices:
                    if 0 <= index < vector_values.shape[1]:
                        values_by_key[_gripper_profile_key(column_name, index)].append(
                            vector_values[:, index]
                        )
    return {
        key: _build_scalar_profile(np.concatenate(parts))
        for key, parts in values_by_key.items()
        if parts
    }


def sample_gripper_extremes(
    parquet_paths: Sequence[Path],
    feature_name: str,
    index: int,
    *,
    fps: float | None = None,
) -> tuple[GripperFrameSample, GripperFrameSample] | None:
    """只投影目标维度和时间列，返回接近 5%/95% 分位数的同步样本。"""
    if not parquet_paths or index < 0:
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("P2 需要安装 pyarrow 才能读取 Parquet") from error

    samples: list[GripperFrameSample] = []
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        if feature_name not in available:
            continue
        time_column = next(
            (
                name
                for name in ("timestamp", "time", "frame_index", "index")
                if name in available
            ),
            None,
        )
        if time_column is None and (fps is None or fps <= 0):
            continue
        columns = [feature_name]
        if time_column is not None:
            columns.append(time_column)
        row_offset = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
            payload = batch.to_pydict()
            vectors = _as_vector_array(payload.get(feature_name))
            if vectors is None or index >= vectors.shape[1]:
                row_offset += batch.num_rows
                continue
            timestamps = _batch_timestamps(
                payload.get(time_column) if time_column is not None else None,
                time_column,
                vectors.shape[0],
                row_offset,
                fps,
            )
            if timestamps is None:
                row_offset += batch.num_rows
                continue
            for local_index, (value, timestamp) in enumerate(
                zip(vectors[:, index], timestamps)
            ):
                if np.isfinite(value) and np.isfinite(timestamp) and timestamp >= 0:
                    samples.append(
                        GripperFrameSample(
                            parquet_path=parquet_path,
                            row_index=row_offset + local_index,
                            timestamp_seconds=float(timestamp),
                            value=float(value),
                        )
                    )
            row_offset += batch.num_rows

    values = np.asarray([sample.value for sample in samples], dtype=np.float64)
    if values.size < 2 or np.unique(values).size < 2:
        return None
    low_target, high_target = np.percentile(values, [5, 95])
    low = min(samples, key=lambda sample: abs(sample.value - low_target))
    high = min(samples, key=lambda sample: abs(sample.value - high_target))
    if low.value >= high.value:
        return None
    return low, high


def _batch_timestamps(
    values: object,
    column_name: str | None,
    row_count: int,
    row_offset: int,
    fps: float | None,
) -> np.ndarray | None:
    """优先使用 Episode 内时间戳，否则用帧序号和声明帧率换算。"""
    if isinstance(values, list) and len(values) == row_count:
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if column_name in {"frame_index", "index"}:
            return numeric / fps if fps is not None and fps > 0 else None
        return numeric
    if fps is None or fps <= 0:
        return None
    return np.arange(row_offset, row_offset + row_count, dtype=np.float64) / fps


def _gripper_profile_key(column_name: str, index: int) -> str:
    return f"{column_name}:{index}"


def _build_scalar_profile(values: np.ndarray) -> ScalarProfile:
    """计算夹爪标量的稳健分位数、异常值比例和离散度。"""
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flattened[np.isfinite(flattened)]
    total = flattened.size
    if finite.size:
        quantiles = np.percentile(finite, [1, 5, 50, 95, 99])
        return ScalarProfile(
            sample_count=int(finite.size),
            min_value=float(np.min(finite)),
            max_value=float(np.max(finite)),
            p01=float(quantiles[0]),
            p05=float(quantiles[1]),
            p50=float(quantiles[2]),
            p95=float(quantiles[3]),
            p99=float(quantiles[4]),
            mean_value=float(np.mean(finite)),
            std_value=float(np.std(finite)),
            nan_ratio=float(np.isnan(flattened).sum() / total),
            inf_ratio=float(np.isinf(flattened).sum() / total),
            unique_count=int(np.unique(finite).size),
        )
    return ScalarProfile(
        sample_count=0,
        min_value=None,
        max_value=None,
        p01=None,
        p05=None,
        p50=None,
        p95=None,
        p99=None,
        mean_value=None,
        std_value=None,
        nan_ratio=float(np.isnan(flattened).sum() / total) if total else 0.0,
        inf_ratio=float(np.isinf(flattened).sum() / total) if total else 0.0,
        unique_count=0,
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


def load_or_profile_parquets(
    parquet_paths: Sequence[Path],
    cache_directory: Path,
    sample_rows: int = 512,
    progress: Callable[[ProfileProgress], None] | None = None,
    *,
    gripper_indices: Mapping[str, Sequence[int]] | None = None,
) -> ParquetProfile:
    """优先读取有效缓存，否则画像并以 JSON/NPZ 原子落盘。"""
    paths = tuple(parquet_paths)
    if not paths:
        raise ValueError("至少需要一个 Parquet 文件")
    requested_grippers = gripper_indices or {}
    fingerprint = _fingerprint(
        paths, sample_rows, cache_directory, requested_grippers
    )
    cached = _load(cache_directory, fingerprint)
    if cached is not None:
        if progress is not None:
            progress(ProfileProgress("cache_hit", len(paths), len(paths)))
        return cached

    if progress is not None:
        progress(ProfileProgress("cache_miss", 0, len(paths)))
    profile = profile_parquets(
        paths,
        sample_rows,
        progress,
        gripper_indices=requested_grippers,
    )
    try:
        _write(cache_directory, fingerprint, sample_rows, profile)
    except Exception as error:
        if progress is not None:
            progress(
                ProfileProgress(
                    "cache_write_warning",
                    len(paths),
                    len(paths),
                    message=str(error),
                )
            )
    return profile


def _fingerprint(
    parquet_paths: Sequence[Path],
    sample_rows: int,
    cache_directory: Path,
    gripper_indices: Mapping[str, Sequence[int]],
) -> str:
    """以文件清单、大小、mtime、采样配置和版本计算缓存指纹。"""
    dataset_root = cache_directory.parent.parent
    files: list[dict[str, object]] = []
    for path in sorted(parquet_paths):
        stat = path.stat()
        try:
            display_path = path.resolve().relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            display_path = path.resolve().as_posix()
        files.append(
            {
                "path": display_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "cache_version": CACHE_VERSION,
        "sample_rows": sample_rows,
        "gripper_indices": {
            name: sorted(set(indices))
            for name, indices in sorted(gripper_indices.items())
        },
        "files": files,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load(cache_directory: Path, fingerprint: str) -> ParquetProfile | None:
    """加载并校验缓存；任何损坏或不匹配均按未命中处理。"""
    metadata_path = cache_directory / METADATA_FILENAME
    samples_path = cache_directory / SAMPLES_FILENAME
    if not metadata_path.is_file() or not samples_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
        if not isinstance(metadata, Mapping):
            return None
        if metadata.get("cache_version") != CACHE_VERSION:
            return None
        if metadata.get("fingerprint") != fingerprint:
            return None

        sample_keys = _string_mapping(metadata.get("samples"), "samples")
        samples: dict[str, np.ndarray] = {}
        with np.load(samples_path, allow_pickle=False) as archive:
            stored_fingerprint = str(np.asarray(archive["__fingerprint__"]).item())
            if stored_fingerprint != fingerprint:
                return None
            for column_name, array_key in sample_keys.items():
                array = np.asarray(archive[array_key])
                if array.ndim != 2 or array.dtype.kind not in "fiu":
                    return None
                samples[column_name] = np.asarray(array, dtype=np.float64).copy()

        profile_payload = metadata.get("profile")
        if not isinstance(profile_payload, Mapping):
            return None
        columns_payload = profile_payload.get("columns")
        if not isinstance(columns_payload, Mapping):
            return None
        columns: dict[str, VectorProfile] = {}
        for column_name, value in columns_payload.items():
            if not isinstance(column_name, str) or not isinstance(value, Mapping):
                return None
            columns[column_name] = VectorProfile(**dict(value))

        gripper_profiles_payload = profile_payload.get("gripper_profiles")
        if not isinstance(gripper_profiles_payload, Mapping):
            return None
        gripper_profiles: dict[str, ScalarProfile] = {}
        for profile_key, value in gripper_profiles_payload.items():
            if not isinstance(profile_key, str) or not isinstance(value, Mapping):
                return None
            gripper_profiles[profile_key] = ScalarProfile(**dict(value))

        schema_columns = _string_sequence(
            profile_payload.get("schema_columns"), "schema_columns"
        )
        inconsistent_columns = _string_sequence(
            profile_payload.get("inconsistent_columns"), "inconsistent_columns"
        )
        row_count = _integer(profile_payload.get("row_count"), "row_count")
        row_group_count = _integer(
            profile_payload.get("row_group_count"), "row_group_count"
        )
        episode_count = _integer(
            profile_payload.get("episode_count"), "episode_count"
        )
        return ParquetProfile(
            row_count=row_count,
            row_group_count=row_group_count,
            schema_columns=schema_columns,
            columns=columns,
            samples=samples,
            episode_count=episode_count,
            inconsistent_columns=inconsistent_columns,
            gripper_profiles=gripper_profiles,
        )
    except (
        EOFError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        zipfile.BadZipFile,
    ):
        return None


def _write(
    cache_directory: Path,
    fingerprint: str,
    sample_rows: int,
    profile: ParquetProfile,
) -> None:
    """先写临时 NPZ/JSON，再替换固定缓存文件。"""
    cache_directory.mkdir(parents=True, exist_ok=True)
    sample_keys: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {
        "__fingerprint__": np.asarray(fingerprint),
    }
    for index, (column_name, values) in enumerate(sorted(profile.samples.items())):
        array_key = f"sample_{index:04d}"
        sample_keys[column_name] = array_key
        arrays[array_key] = np.asarray(values, dtype=np.float64)

    metadata = {
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "sample_rows": sample_rows,
        "samples": sample_keys,
        "profile": {
            "row_count": profile.row_count,
            "row_group_count": profile.row_group_count,
            "schema_columns": list(profile.schema_columns),
            "columns": {
                name: asdict(value) for name, value in profile.columns.items()
            },
            "gripper_profiles": {
                name: asdict(value)
                for name, value in profile.gripper_profiles.items()
            },
            "episode_count": profile.episode_count,
            "inconsistent_columns": list(profile.inconsistent_columns),
        },
    }

    temporary_samples = _write_temporary_npz(cache_directory, arrays)
    temporary_metadata: Path | None = None
    try:
        temporary_metadata = _write_temporary_json(cache_directory, metadata)
        os.replace(temporary_samples, cache_directory / SAMPLES_FILENAME)
        os.replace(temporary_metadata, cache_directory / METADATA_FILENAME)
    finally:
        temporary_samples.unlink(missing_ok=True)
        if temporary_metadata is not None:
            temporary_metadata.unlink(missing_ok=True)


def _write_temporary_npz(
    directory: Path, arrays: Mapping[str, np.ndarray]
) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".parquet_samples.", suffix=".tmp", dir=directory
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file_handle:
            np.savez_compressed(file_handle, **arrays)
            file_handle.flush()
            os.fsync(file_handle.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_temporary_json(directory: Path, payload: object) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".parquet_profile.", suffix=".tmp", dir=directory
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(f"缓存 {label} 必须是字符串映射")
    return dict(value)


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"缓存 {label} 必须是字符串数组")
    return tuple(value)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"缓存 {label} 必须是非负整数")
    return value
