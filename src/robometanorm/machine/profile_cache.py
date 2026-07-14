"""Parquet 画像的安全持久化缓存。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from robometanorm.machine.models import (
    ParquetProfile,
    ProfileProgress,
    VectorProfile,
)
from robometanorm.machine.parquet_profiler import profile_parquets


CACHE_VERSION = 2
METADATA_FILENAME = "parquet_profile_v2.json"
SAMPLES_FILENAME = "parquet_samples_v2.npz"


def load_or_profile_parquets(
    parquet_paths: Sequence[Path],
    cache_directory: Path,
    sample_rows: int = 512,
    progress: Callable[[ProfileProgress], None] | None = None,
) -> ParquetProfile:
    """优先读取有效缓存，否则画像并以 JSON/NPZ 原子落盘。"""
    paths = tuple(parquet_paths)
    if not paths:
        raise ValueError("至少需要一个 Parquet 文件")
    fingerprint = _fingerprint(paths, sample_rows, cache_directory)
    cached = _load(cache_directory, fingerprint)
    if cached is not None:
        if progress is not None:
            progress(ProfileProgress("cache_hit", len(paths), len(paths)))
        return cached

    if progress is not None:
        progress(ProfileProgress("cache_miss", 0, len(paths)))
    profile = profile_parquets(paths, sample_rows, progress)
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
    parquet_paths: Sequence[Path], sample_rows: int, cache_directory: Path
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
