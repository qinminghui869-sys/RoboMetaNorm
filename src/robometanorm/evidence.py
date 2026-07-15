"""Collect bounded, read-only evidence from local dataset inputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, time, timedelta
from decimal import Decimal
from fractions import Fraction
import json
import math
from numbers import Number
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

from robometanorm.models import (
    CameraEvidence,
    DatasetCandidate,
    DatasetEvidence,
    DatasetMapping,
    FeatureSchema,
    GripperRange,
    HardwareProfile,
    IdentityEvidence,
    Issue,
    MachineEvidence,
    MediaSample,
    ParquetEpisodeEvidence,
)


_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_MEDIA_SUFFIXES = _VIDEO_SUFFIXES | _IMAGE_SUFFIXES
_MEDIA_TOOL_TIMEOUT_SECONDS = 120.0
_STATIC_IMAGE_BYTE_LIMIT = 20 * 1024 * 1024
# Large enough for feature-rich metadata and task manifests, while still
# bounding every local JSON input before parsing or prompt construction.
_LOCAL_JSON_BYTE_LIMIT = 8 * 1024 * 1024


class _EvidenceFileTooLarge(ValueError):
    """A local evidence file exceeded its explicit byte budget."""


def _directory_open_flags() -> int:
    try:
        return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    except AttributeError as error:
        raise OSError("secure evidence reads are unavailable") from error


def _file_open_flags() -> int:
    try:
        return (
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0)
        )
    except AttributeError as error:
        raise OSError("secure evidence reads are unavailable") from error


def _path_start_and_components(path: Path) -> tuple[str, tuple[str, ...]]:
    path_parts = path.parts
    if path.is_absolute():
        start = os.sep
        components = tuple(path_parts[1:])
    else:
        start = "."
        components = tuple(path_parts)
    if not components or any(part in {"", ".", ".."} for part in components):
        raise OSError("unsafe evidence path")
    return start, components


@contextmanager
def _open_path_fd(path: Path, *, directory: bool) -> Iterator[int]:
    """Open every component with ``openat`` and validate the live object."""
    start, components = _path_start_and_components(path)
    descriptor = os.open(start, _directory_open_flags())
    descriptors = [descriptor]
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("evidence path parent is not a directory")
        for index, component in enumerate(components):
            component_is_directory = directory or index < len(components) - 1
            descriptor = os.open(
                component,
                (
                    _directory_open_flags()
                    if component_is_directory
                    else _file_open_flags()
                ),
                dir_fd=descriptor,
            )
            descriptors.append(descriptor)
            component_status = os.fstat(descriptor)
            if component_is_directory and not stat.S_ISDIR(
                component_status.st_mode
            ):
                raise OSError("evidence path parent is not a directory")
            if not component_is_directory:
                if stat.S_ISDIR(component_status.st_mode):
                    raise IsADirectoryError("evidence path is a directory")
                if not stat.S_ISREG(component_status.st_mode):
                    raise OSError("evidence path is not a regular file")
        yield descriptor
    finally:
        for open_descriptor in reversed(descriptors):
            try:
                os.close(open_descriptor)
            except OSError:
                pass


@contextmanager
def _open_directory_fd(path: Path) -> Iterator[int]:
    with _open_path_fd(path, directory=True) as descriptor:
        yield descriptor


@contextmanager
def _open_regular_fd(path: Path) -> Iterator[int]:
    with _open_path_fd(path, directory=False) as descriptor:
        yield descriptor


@contextmanager
def _open_regular_binary(path: Path) -> Iterator[BinaryIO]:
    with _open_regular_fd(path) as descriptor:
        file_handle = os.fdopen(descriptor, "rb", closefd=False)
        try:
            yield file_handle
        finally:
            file_handle.close()


def _read_bounded_regular(path: Path, byte_limit: int) -> bytes:
    with _open_regular_fd(path) as descriptor:
        if os.fstat(descriptor).st_size > byte_limit:
            raise _EvidenceFileTooLarge("local evidence exceeds byte limit")
        chunks: list[bytes] = []
        remaining = byte_limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            remaining -= len(chunk)
        raise _EvidenceFileTooLarge("local evidence exceeds byte limit")


def _reject_json_constant(value: str) -> object:
    del value
    raise ValueError("non-finite JSON number")


def _strict_json_loads(raw_content: bytes) -> object:
    value = json.loads(
        raw_content.decode("utf-8"),
        parse_constant=_reject_json_constant,
    )
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("JSON contains unsafe text") from error
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("JSON contains a non-finite number")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


def read_info(candidate: DatasetCandidate) -> dict[str, object]:
    """Read ``info.json`` and require a JSON object at its top level."""
    try:
        raw_content = _read_bounded_regular(
            candidate.info_path, _LOCAL_JSON_BYTE_LIMIT
        )
    except (OSError, _EvidenceFileTooLarge) as error:
        raise ValueError("info.json could not be read safely") from error
    try:
        source_info = _strict_json_loads(raw_content)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("info.json could not be parsed") from error
    if not isinstance(source_info, dict):
        raise ValueError("info.json must contain a JSON object")
    return source_info


def collect_camera_evidence(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
    temp_frames: Path,
) -> tuple[tuple[CameraEvidence, ...], tuple[Issue, ...]]:
    """Collect at most two exact-path media samples for each image feature."""
    schemas = _camera_feature_schemas(source_info)
    discovered_media, unsafe_roots = _discover_media(candidate)
    issues: list[Issue] = [
        Issue(
            code="MEDIA_PATH_UNSAFE",
            message="A media path could not be traversed without following links",
            scope="camera",
            evidence={"logical_root": logical_root},
        )
        for logical_root in unsafe_roots
        if schemas
    ]
    cameras: list[CameraEvidence] = []
    frame_sequence = 0

    for schema in schemas:
        matching_media = [
            (path, relative_path, media_type)
            for path, relative_path, media_type, parent_parts in discovered_media
            if schema.source_key in parent_parts
        ]
        if len(matching_media) > 2:
            matching_media = [matching_media[0], matching_media[-1]]
        if not matching_media:
            issues.append(
                Issue(
                    code="CAMERA_MEDIA_MISSING",
                    message="No media matched the exact camera source path",
                    scope=f"camera.{schema.source_key}",
                    evidence={"source_key": schema.source_key},
                )
            )

        samples: list[MediaSample] = []
        for media_path, relative_path, media_type in matching_media:
            if media_type == "image":
                output_path = temp_frames / (
                    f"image-{frame_sequence:06d}{media_path.suffix.lower()}"
                )
                frame_sequence += 1
                try:
                    _copy_static_image(media_path, output_path)
                    samples.append(
                        MediaSample(
                            relative_path=relative_path,
                            media_type="image",
                            codec=None,
                            fps=None,
                            width=None,
                            height=None,
                            duration_seconds=None,
                            pixel_format=None,
                            frame_path=output_path,
                        )
                    )
                except (OSError, ValueError) as error:
                    _discard_frame_output(output_path)
                    issues.append(
                        _media_issue(
                            code="MEDIA_READ_FAILED",
                            message="A static image could not be read safely",
                            source_key=schema.source_key,
                            relative_path=relative_path,
                            error=error,
                        )
                    )
                continue

            try:
                probed_sample = probe_media(media_path)
                sample = replace(
                    probed_sample,
                    relative_path=relative_path,
                    media_type="video",
                    frame_path=None,
                )
            except ValueError as error:
                issues.append(
                    _media_issue(
                        code="MEDIA_PROBE_FAILED",
                        message="Video metadata could not be probed",
                        source_key=schema.source_key,
                        relative_path=relative_path,
                        error=error,
                    )
                )
                continue

            output_path = temp_frames / f"frame-{frame_sequence:06d}.jpg"
            frame_sequence += 1
            try:
                extracted_path = extract_midpoint_frame(
                    media_path,
                    output_path,
                    duration_seconds=sample.duration_seconds,
                )
                if extracted_path != output_path or not output_path.is_file():
                    raise ValueError("frame extraction returned no safe output")
                sample = replace(sample, frame_path=output_path)
            except ValueError as error:
                _discard_frame_output(output_path)
                issues.append(
                    _media_issue(
                        code="FRAME_EXTRACTION_FAILED",
                        message="A representative video frame could not be extracted",
                        source_key=schema.source_key,
                        relative_path=relative_path,
                        error=error,
                    )
                )
            samples.append(sample)

        cameras.append(CameraEvidence(schema=schema, samples=tuple(samples)))

    return tuple(cameras), tuple(issues)


def probe_media(media_path: Path) -> MediaSample:
    """Probe one video with ffprobe without exposing tool diagnostics."""
    try:
        with _open_regular_fd(media_path) as media_descriptor:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                (
                    "stream=codec_type,codec_name,r_frame_rate,width,height,duration,pix_fmt:"
                    "format=duration"
                ),
                "-of",
                "json",
                f"/proc/self/fd/{media_descriptor}",
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_MEDIA_TOOL_TIMEOUT_SECONDS,
                pass_fds=(media_descriptor,),
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("ffprobe could not be executed") from error
    if completed.returncode != 0:
        raise ValueError("ffprobe did not return usable metadata")

    try:
        payload = json.loads(completed.stdout)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("ffprobe returned invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("ffprobe JSON must be an object")

    streams = payload.get("streams")
    if not isinstance(streams, Sequence) or isinstance(
        streams, (str, bytes, bytearray)
    ):
        raise ValueError("ffprobe streams must be a sequence")

    video_stream: Mapping[str, object] | None = None
    for stream in streams:
        if not isinstance(stream, Mapping):
            raise ValueError("ffprobe stream must be an object")
        if video_stream is None and stream.get("codec_type") == "video":
            video_stream = stream
    if video_stream is None:
        raise ValueError("ffprobe returned no video stream")

    fps = _positive_fraction(video_stream.get("r_frame_rate"))
    width = _positive_dimension(video_stream.get("width"))
    height = _positive_dimension(video_stream.get("height"))
    if fps is None or width is None or height is None:
        raise ValueError("ffprobe returned invalid video dimensions or frame rate")

    raw_format = payload.get("format")
    if raw_format is not None and not isinstance(raw_format, Mapping):
        raise ValueError("ffprobe format must be an object")
    format_duration = (
        raw_format.get("duration") if isinstance(raw_format, Mapping) else None
    )
    duration = _positive_float(format_duration)
    if duration is None:
        duration = _positive_float(video_stream.get("duration"))
    if duration is None:
        raise ValueError("ffprobe returned no usable duration")

    codec = video_stream.get("codec_name")
    pixel_format = video_stream.get("pix_fmt")
    return MediaSample(
        relative_path="",
        media_type="video",
        codec=codec if isinstance(codec, str) else None,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=duration,
        pixel_format=pixel_format if isinstance(pixel_format, str) else None,
        frame_path=None,
    )


def extract_midpoint_frame(
    media_path: Path,
    output_path: Path,
    *,
    duration_seconds: float | None = None,
) -> Path:
    """Extract one bounded JPEG at the temporal midpoint of a video."""
    try:
        if os.path.abspath(media_path) == os.path.abspath(output_path):
            raise ValueError("frame output must differ from source media")
    except OSError as error:
        raise ValueError("frame paths could not be validated") from error

    if duration_seconds is None:
        duration_seconds = probe_media(media_path).duration_seconds
    duration = _positive_float(duration_seconds)
    if duration is None:
        raise ValueError("video duration is unusable")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
    except OSError as error:
        raise ValueError("frame output could not be prepared") from error

    try:
        with _open_regular_fd(media_path) as media_descriptor:
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(duration * 0.5),
                "-i",
                f"/proc/self/fd/{media_descriptor}",
                "-frames:v",
                "1",
                "-vf",
                "scale=1280:1280:force_original_aspect_ratio=decrease",
                str(output_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_MEDIA_TOOL_TIMEOUT_SECONDS,
                pass_fds=(media_descriptor,),
            )
    except (OSError, subprocess.SubprocessError) as error:
        _discard_frame_output(output_path)
        raise ValueError("ffmpeg could not be executed") from error
    if completed.returncode != 0:
        _discard_frame_output(output_path)
        raise ValueError("ffmpeg did not extract a frame")
    try:
        output_is_usable = output_path.is_file() and output_path.stat().st_size > 0
    except OSError as error:
        _discard_frame_output(output_path)
        raise ValueError("extracted frame could not be inspected") from error
    if not output_is_usable:
        _discard_frame_output(output_path)
        raise ValueError("ffmpeg returned no frame")
    return output_path


@contextmanager
def collect_dataset_evidence(
    candidate: DatasetCandidate,
    source_info: Mapping[str, object],
) -> Iterator[DatasetEvidence]:
    """Collect all local evidence while representative frames are available."""
    with tempfile.TemporaryDirectory(prefix="robometanorm-mini-") as temp_name:
        identity = collect_identity_evidence(candidate.info_path.parent, source_info)
        machines, machine_issues = collect_machine_evidence(candidate, source_info)
        cameras, camera_issues = collect_camera_evidence(
            candidate, source_info, Path(temp_name)
        )
        yield DatasetEvidence(
            candidate=candidate,
            source_info=dict(source_info),
            identity=identity,
            cameras=cameras,
            machines=machines,
            issues=(*identity.issues, *machine_issues, *camera_issues),
        )


def _camera_feature_schemas(
    source_info: Mapping[str, object],
) -> tuple[FeatureSchema, ...]:
    features = source_info.get("features")
    if not isinstance(features, Mapping):
        return ()
    schemas: list[FeatureSchema] = []
    for source_key, feature in features.items():
        if (
            not isinstance(source_key, str)
            or not source_key.startswith("observation.images.")
            or not isinstance(feature, Mapping)
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


@dataclass(frozen=True)
class _SecureWalkResult:
    files: tuple[tuple[Path, Path], ...]
    unsafe: bool


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _secure_walk_regular_files(
    root: Path,
    suffixes: frozenset[str],
) -> _SecureWalkResult:
    """Walk a directory by live descriptors and never enter a link."""
    discovered: list[tuple[Path, Path]] = []
    unsafe = False

    def visit(descriptor: int, relative_directory: Path) -> None:
        nonlocal unsafe
        try:
            names = sorted(os.listdir(descriptor))
        except OSError:
            unsafe = True
            return
        for name in names:
            relative_path = relative_directory / name
            try:
                entry_status = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                unsafe = True
                continue

            if stat.S_ISLNK(entry_status.st_mode):
                unsafe = True
                continue
            if stat.S_ISDIR(entry_status.st_mode):
                child_descriptor: int | None = None
                try:
                    child_descriptor = os.open(
                        name,
                        _directory_open_flags(),
                        dir_fd=descriptor,
                    )
                    opened_status = os.fstat(child_descriptor)
                    if not stat.S_ISDIR(opened_status.st_mode) or not _same_file(
                        entry_status, opened_status
                    ):
                        raise OSError("directory changed during evidence walk")
                except OSError:
                    unsafe = True
                    if child_descriptor is not None:
                        try:
                            os.close(child_descriptor)
                        except OSError:
                            pass
                    continue
                assert child_descriptor is not None
                try:
                    visit(child_descriptor, relative_path)
                finally:
                    os.close(child_descriptor)
                continue
            if (
                stat.S_ISREG(entry_status.st_mode)
                and relative_path.suffix.lower() in suffixes
            ):
                discovered.append((root / relative_path, relative_path))

    try:
        with _open_directory_fd(root) as root_descriptor:
            visit(root_descriptor, Path())
    except (OSError, RecursionError):
        unsafe = True
        discovered.clear()

    return _SecureWalkResult(
        tuple(sorted(discovered, key=lambda item: item[1].as_posix())),
        unsafe,
    )


def _discover_media(
    candidate: DatasetCandidate,
) -> tuple[
    tuple[tuple[Path, str, str, frozenset[str]], ...],
    tuple[str, ...],
]:
    roots = (
        (candidate.video_path, "videos"),
        (candidate.depth_path, "depth"),
    )
    discovered: dict[Path, tuple[str, str, set[str]]] = {}
    unsafe_roots: list[str] = []
    for root, logical_name in roots:
        if root is None:
            continue
        walk = _secure_walk_regular_files(root, _MEDIA_SUFFIXES)
        if walk.unsafe:
            unsafe_roots.append(logical_name)
            continue
        for path, relative_to_root in walk.files:
            relative_path = _safe_media_relative_path(
                candidate, path, relative_to_root, logical_name
            )
            media_type = (
                "video" if path.suffix.lower() in _VIDEO_SUFFIXES else "image"
            )
            previous = discovered.get(path)
            if previous is None:
                discovered[path] = (
                    relative_path,
                    media_type,
                    set(relative_to_root.parent.parts),
                )
            else:
                previous[2].update(relative_to_root.parent.parts)

    return (
        tuple(
            (path, relative_path, media_type, frozenset(parent_parts))
            for path, (relative_path, media_type, parent_parts) in sorted(
                discovered.items(), key=lambda item: item[1][0]
            )
        ),
        tuple(unsafe_roots),
    )


def _copy_static_image(source_path: Path, output_path: Path) -> None:
    output_descriptor: int | None = None
    with _open_regular_fd(source_path) as source_descriptor:
        if os.fstat(source_descriptor).st_size > _STATIC_IMAGE_BYTE_LIMIT:
            raise _EvidenceFileTooLarge("static image exceeds byte limit")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_left = _STATIC_IMAGE_BYTE_LIMIT + 1
        try:
            with _open_directory_fd(output_path.parent) as output_directory:
                output_descriptor = os.open(
                    output_path.name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=output_directory,
                )
                while bytes_left > 0:
                    read_size = min(64 * 1024, bytes_left)
                    chunk = os.read(source_descriptor, read_size)
                    if not chunk:
                        return
                    if len(chunk) > read_size:
                        raise _EvidenceFileTooLarge(
                            "static image exceeds byte limit"
                        )
                    bytes_left -= len(chunk)
                    unwritten = memoryview(chunk)
                    while unwritten:
                        written = os.write(output_descriptor, unwritten)
                        if written <= 0:
                            raise OSError("static image copy made no progress")
                        unwritten = unwritten[written:]
                raise _EvidenceFileTooLarge("static image exceeds byte limit")
        finally:
            if output_descriptor is not None:
                os.close(output_descriptor)


def _safe_media_relative_path(
    candidate: DatasetCandidate,
    media_path: Path,
    relative_to_root: Path,
    logical_root: str,
) -> str:
    try:
        return media_path.relative_to(candidate.source_path).as_posix()
    except ValueError:
        return (Path(logical_root) / relative_to_root).as_posix()


def _media_issue(
    *,
    code: str,
    message: str,
    source_key: str,
    relative_path: str,
    error: BaseException,
) -> Issue:
    return Issue(
        code=code,
        message=message,
        scope=f"camera.{source_key}",
        evidence={
            "source_key": source_key,
            "relative_path": relative_path,
            "error_type": type(error).__name__,
        },
    )


def _positive_fraction(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = float(Fraction(value))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _positive_dimension(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _positive_float(value: object) -> float | None:
    if type(value) not in {str, int, float}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _discard_frame_output(output_path: Path) -> None:
    try:
        output_path.unlink(missing_ok=True)
    except OSError:
        pass


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
    parquet_selection = _representative_parquet_selection(candidate)
    if parquet_selection.unsafe:
        for feature in feature_keys:
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_PATH_UNSAFE",
                message="A Parquet path could not be traversed without following links",
                relative_path="data",
                feature=feature,
            )
        return (), tuple(issues)
    episodes = tuple(
        _inspect_parquet_episode(
            candidate,
            parquet_path,
            feature_keys,
            issues,
            seen_issues,
        )
        for parquet_path in parquet_selection.paths
    )
    if any(issue.code == "PARQUET_PATH_UNSAFE" for issue in issues):
        return (), tuple(issues)

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


@dataclass
class _MappedGripperStats:
    minimum: float | None = None
    maximum: float | None = None
    finite_count: int = 0
    nonfinite_count: int = 0

    def observe(self, value: object) -> None:
        finite_value = _finite_number(value)
        if finite_value is None:
            self.nonfinite_count += 1
            return
        self.finite_count += 1
        if self.minimum is None or finite_value < self.minimum:
            self.minimum = finite_value
        if self.maximum is None or finite_value > self.maximum:
            self.maximum = finite_value

    def mark_unusable(self) -> None:
        self.nonfinite_count += 1


def collect_mapped_gripper_ranges(
    candidate: DatasetCandidate,
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    mapping: DatasetMapping,
) -> tuple[DatasetEvidence, tuple[Issue, ...]]:
    """Collect ranges only for single-value grippers confirmed by ``mapping``."""
    issues: list[Issue] = []
    gripper_components = {
        component.component_id: component
        for component in profile.components
        if component.kind in {"gripper_open", "gripper_open_scale"}
    }
    mapped_targets: dict[str, dict[int, str]] = {}

    for assignment in mapping.machines:
        for source_slice in assignment.slices:
            component = gripper_components.get(source_slice.component_id)
            if component is None:
                continue
            if not (
                type(source_slice.start) is int
                and type(source_slice.end) is int
                and source_slice.start >= 0
                and source_slice.end - source_slice.start == 1
                and type(component.count) is int
                and component.count == 1
            ):
                issues.append(
                    Issue(
                        code="MAPPED_GRIPPER_SLICE_INVALID",
                        message="Mapped gripper must use one safe non-negative index",
                        scope=f"machine.{assignment.source_feature}",
                        evidence={
                            "source_feature": _safe_issue_value(
                                assignment.source_feature
                            ),
                            "component_id": _safe_issue_value(
                                source_slice.component_id
                            ),
                            "start": _safe_issue_value(source_slice.start),
                            "end": _safe_issue_value(source_slice.end),
                        },
                    )
                )
                continue
            targets = mapped_targets.setdefault(assignment.source_feature, {})
            targets.setdefault(source_slice.start, source_slice.component_id)

    machine_features = {machine.schema.source_key for machine in evidence.machines}
    readable_targets: dict[str, dict[int, str]] = {}
    for feature, targets in mapped_targets.items():
        if feature in machine_features:
            readable_targets[feature] = targets
            continue
        for component_id in targets.values():
            issues.append(
                Issue(
                    code="MAPPED_GRIPPER_SOURCE_MISSING",
                    message="Mapped gripper source feature is absent from evidence",
                    scope=f"machine.{feature}",
                    evidence={
                        "source_feature": _safe_issue_value(feature),
                        "component_id": _safe_issue_value(component_id),
                    },
                )
            )

    if not readable_targets:
        return evidence, tuple(issues)

    stats = {
        feature: {index: _MappedGripperStats() for index in targets}
        for feature, targets in readable_targets.items()
    }
    parquet_issues_seen: set[tuple[str, str, str]] = set()
    expected_widths: dict[str, int] = {}
    parquet_selection = _representative_parquet_selection(candidate)
    if parquet_selection.unsafe:
        for feature in stats:
            _append_parquet_issue(
                issues,
                parquet_issues_seen,
                code="PARQUET_PATH_UNSAFE",
                message="Mapped gripper source path could not be opened safely",
                relative_path="data",
                feature=feature,
            )
        return replace(evidence, machines=()), tuple(issues)

    for feature, feature_stats in stats.items():
        for parquet_path in parquet_selection.paths:
            _collect_mapped_feature_ranges(
                candidate,
                parquet_path,
                feature,
                feature_stats,
                expected_widths,
                issues,
                parquet_issues_seen,
            )

    if any(issue.code == "PARQUET_PATH_UNSAFE" for issue in issues):
        return replace(evidence, machines=()), tuple(issues)

    machines = tuple(
        replace(
            machine,
            gripper_ranges=tuple(
                GripperRange(
                    index=index,
                    minimum=feature_stats[index].minimum,
                    maximum=feature_stats[index].maximum,
                    finite_count=feature_stats[index].finite_count,
                    nonfinite_count=feature_stats[index].nonfinite_count,
                )
                for index in sorted(feature_stats)
            ),
        )
        if (feature_stats := stats.get(machine.schema.source_key)) is not None
        else machine
        for machine in evidence.machines
    )
    return replace(evidence, machines=machines), tuple(issues)


def _collect_mapped_feature_ranges(
    candidate: DatasetCandidate,
    parquet_path: Path,
    feature: str,
    stats: dict[int, _MappedGripperStats],
    expected_widths: dict[str, int],
    issues: list[Issue],
    seen_issues: set[tuple[str, str, str]],
) -> None:
    relative_path = _relative_parquet_path(candidate, parquet_path)
    try:
        with _open_parquet_file(parquet_path) as parquet_file:
            _collect_mapped_open_parquet(
                parquet_file,
                relative_path,
                feature,
                stats,
                expected_widths,
                issues,
                seen_issues,
            )
    except OSError as error:
        _mark_mapped_targets_unusable(stats)
        _append_parquet_issue(
            issues,
            seen_issues,
            code="PARQUET_PATH_UNSAFE",
            message="Mapped gripper source path could not be opened safely",
            relative_path=relative_path,
            feature=feature,
            error_type=type(error).__name__,
        )
    except (
        ValueError,
        pa.ArrowCapacityError,
        pa.ArrowNotImplementedError,
    ) as error:
        _mark_mapped_targets_unusable(stats)
        _append_parquet_issue(
            issues,
            seen_issues,
            code="PARQUET_READ_FAILED",
            message="Mapped gripper feature could not be read",
            relative_path=relative_path,
            feature=feature,
            error_type=type(error).__name__,
        )


@contextmanager
def _open_parquet_file(path: Path) -> Iterator[object]:
    """Keep the validated source descriptor alive for all PyArrow reads."""
    with _open_regular_binary(path) as file_handle:
        yield pq.ParquetFile(file_handle)


def _collect_mapped_open_parquet(
    parquet_file: object,
    relative_path: str,
    feature: str,
    stats: dict[int, _MappedGripperStats],
    expected_widths: dict[str, int],
    issues: list[Issue],
    seen_issues: set[tuple[str, str, str]],
) -> None:
    schema_columns = tuple(parquet_file.schema_arrow.names)

    column_count = schema_columns.count(feature)
    if column_count == 0:
        _mark_mapped_targets_unusable(stats)
        _append_parquet_issue(
            issues,
            seen_issues,
            code="PARQUET_COLUMN_MISSING",
            message="Mapped gripper feature is missing from Parquet schema",
            relative_path=relative_path,
            feature=feature,
        )
        return
    if column_count != 1:
        _mark_mapped_targets_unusable(stats)
        _append_parquet_issue(
            issues,
            seen_issues,
            code="PARQUET_COLUMN_AMBIGUOUS",
            message="Mapped gripper feature is ambiguous in Parquet schema",
            relative_path=relative_path,
            feature=feature,
        )
        return

    ambiguous_projection = False
    read_error: Exception | None = None
    try:
        for batch in parquet_file.iter_batches(columns=[feature], batch_size=512):
            if batch.num_columns != 1 or tuple(batch.schema.names) != (feature,):
                ambiguous_projection = True
                continue
            for row in batch.column(0).to_pylist():
                width = _parquet_value_width(row)
                if width is not None:
                    expected_width = expected_widths.setdefault(feature, width)
                    if width != expected_width:
                        _append_parquet_issue(
                            issues,
                            seen_issues,
                            code="PARQUET_VECTOR_LENGTH_INCONSISTENT",
                            message="Mapped gripper source vector length is inconsistent",
                            relative_path=relative_path,
                            feature=feature,
                            length=width,
                        )
                _observe_mapped_row(
                    row,
                    stats,
                    relative_path,
                    feature,
                    issues,
                    seen_issues,
                )
    except (
        ValueError,
        OSError,
        pa.ArrowCapacityError,
        pa.ArrowNotImplementedError,
    ) as error:
        read_error = error

    if ambiguous_projection:
        _mark_mapped_targets_unusable(stats)
        _append_parquet_issue(
            issues,
            seen_issues,
            code="PARQUET_COLUMN_AMBIGUOUS",
            message="Parquet projection did not resolve to one exact top-level column",
            relative_path=relative_path,
            feature=feature,
        )
    if read_error is not None:
        _mark_mapped_targets_unusable(stats)
        _append_parquet_issue(
            issues,
            seen_issues,
            code="PARQUET_READ_FAILED",
            message="Mapped gripper feature could not be read",
            relative_path=relative_path,
            feature=feature,
            error_type=type(read_error).__name__,
        )


def _observe_mapped_row(
    row: object,
    stats: dict[int, _MappedGripperStats],
    relative_path: str,
    feature: str,
    issues: list[Issue],
    seen_issues: set[tuple[str, str, str]],
) -> None:
    if row is None:
        _mark_mapped_targets_unusable(stats)
        return
    if isinstance(row, (list, tuple)):
        for index, target_stats in stats.items():
            if index >= len(row):
                target_stats.mark_unusable()
                _append_parquet_issue(
                    issues,
                    seen_issues,
                    code="PARQUET_VECTOR_LENGTH_INCONSISTENT",
                    message="Mapped gripper index is outside the source vector",
                    relative_path=relative_path,
                    feature=feature,
                    length=len(row),
                )
            else:
                target_stats.observe(row[index])
        return

    for index, target_stats in stats.items():
        if index == 0:
            target_stats.observe(row)
        else:
            target_stats.mark_unusable()
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_VECTOR_LENGTH_INCONSISTENT",
                message="Mapped gripper index is outside a scalar source value",
                relative_path=relative_path,
                feature=feature,
                length=1,
            )


def _mark_mapped_targets_unusable(
    stats: dict[int, _MappedGripperStats],
) -> None:
    for target_stats in stats.values():
        target_stats.mark_unusable()


def _finite_number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        finite_value = float(value)
    except (OverflowError, ValueError):
        return None
    return finite_value if math.isfinite(finite_value) else None


def _safe_issue_value(value: object) -> object:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        try:
            json.dumps(value)
        except ValueError:
            return {"value_type": "int"}
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is str:
        return value
    return {"value_type": type(value).__name__}


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


@dataclass(frozen=True)
class _ParquetSelection:
    paths: tuple[Path, ...]
    unsafe: bool


def _representative_parquet_selection(
    candidate: DatasetCandidate,
) -> _ParquetSelection:
    data_path = candidate.data_path
    if data_path is None:
        return _ParquetSelection((), False)
    walk = _secure_walk_regular_files(data_path, frozenset({".parquet"}))
    parquet_paths = [path for path, _ in walk.files]
    if len(parquet_paths) <= 2:
        selected = tuple(parquet_paths)
    else:
        selected = (parquet_paths[0], parquet_paths[-1])
    return _ParquetSelection(selected, walk.unsafe)


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
        with _open_parquet_file(parquet_path) as parquet_file:
            return _inspect_open_parquet_episode(
                parquet_file,
                relative_path,
                feature_keys,
                vector_lengths,
                issues,
                seen_issues,
            )
    except OSError as error:
        for feature in feature_keys:
            _append_parquet_issue(
                issues,
                seen_issues,
                code="PARQUET_PATH_UNSAFE",
                message="Parquet source path could not be opened safely",
                relative_path=relative_path,
                feature=feature,
                error_type=type(error).__name__,
            )
        return ParquetEpisodeEvidence(relative_path, (), vector_lengths)
    except (
        ValueError,
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


def _inspect_open_parquet_episode(
    parquet_file: object,
    relative_path: str,
    feature_keys: tuple[str, ...],
    vector_lengths: dict[str, int | None],
    issues: list[Issue],
    seen_issues: set[tuple[str, str, str]],
) -> ParquetEpisodeEvidence:
    schema_columns = tuple(parquet_file.schema_arrow.names)
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
        raw_content = _read_bounded_regular(path, _LOCAL_JSON_BYTE_LIMIT)
    except FileNotFoundError:
        return "missing", None, None
    except _EvidenceFileTooLarge as error:
        return (
            "invalid",
            None,
            Issue(
                code="COMMON_RECORD_INVALID",
                message="common_record.json is not valid bounded UTF-8 JSON",
                scope="identity.common_record",
                evidence={
                    "file_name": path.name,
                    "error_type": type(error).__name__,
                },
            ),
        )
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
        common_record = _strict_json_loads(raw_content)
    except (UnicodeError, ValueError, RecursionError) as error:
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
        raw_content = _read_bounded_regular(path, _LOCAL_JSON_BYTE_LIMIT)
    except FileNotFoundError:
        return "missing", (), None
    except _EvidenceFileTooLarge as error:
        return (
            "invalid",
            (),
            Issue(
                code="TASKS_INVALID",
                message="tasks.jsonl exceeds the bounded UTF-8 JSON input limit",
                scope="identity.tasks",
                evidence={
                    "file_name": path.name,
                    "line_numbers": [],
                    "error_types": [type(error).__name__],
                },
            ),
        )
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
            records.append(_strict_json_loads(raw_line))
        except (UnicodeError, ValueError, RecursionError) as error:
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
