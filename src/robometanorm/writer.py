"""Deterministic, fail-closed writer for the two mini output files."""

from __future__ import annotations

from collections.abc import Mapping
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat

from robometanorm.models import (
    DatasetCandidate,
    DatasetEvidence,
    DatasetStatus,
    HardwareProfile,
    IdentityEvidence,
    Issue,
    LayoutType,
    MappingRecord,
    NormalizationResult,
)


_INVALID_JSON_MESSAGE = "输出不是合法 JSON"
_REVIEW_KEYS = frozenset(
    {
        "schema_version",
        "generator",
        "dataset",
        "status",
        "robot_identity",
        "camera_mappings",
        "machine_mappings",
        "issues",
    }
)
_RECORD_KEYS = frozenset(
    {
        "source_address",
        "source",
        "output",
        "candidate",
        "changed",
        "vlm_semantics",
        "citations",
        "decision",
        "reason",
    }
)
_IDENTITY_KEYS = _RECORD_KEYS | {"local_evidence"}
_ISSUE_KEYS = frozenset({"code", "message", "scope", "evidence", "severity"})
_CITATION_KEYS = frozenset({"source_id", "title", "url", "kind"})
_STATUS_VALUES = frozenset(status.value for status in DatasetStatus)
_LAYOUT_VALUES = frozenset(layout.value for layout in LayoutType)
_INFO_OUTPUT_NAME = "info_norm.json"
_REVIEW_OUTPUT_NAME = "info_norm_review.json"


def _invalid_json() -> ValueError:
    return ValueError(_INVALID_JSON_MESSAGE)


def _clone_json_native(
    value: object,
    active_containers: set[int] | None = None,
) -> object:
    """Copy exact JSON-native values while rejecting coercions and cycles."""

    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise _invalid_json()
        return value
    if value_type not in {list, dict}:
        raise _invalid_json()

    active = active_containers if active_containers is not None else set()
    identity = id(value)
    if identity in active:
        raise _invalid_json()
    active.add(identity)
    try:
        if value_type is list:
            return [_clone_json_native(item, active) for item in value]
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise _invalid_json()
            output[key] = _clone_json_native(item, active)
        return output
    finally:
        active.remove(identity)


def _safe_clone_json_native(value: object) -> object:
    try:
        return _clone_json_native(value)
    except (TypeError, ValueError, RecursionError, OverflowError):
        raise _invalid_json() from None


def _json_bytes(payload: object) -> bytes:
    """Serialize one already structured payload to canonical UTF-8 bytes."""

    try:
        _clone_json_native(payload)
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        return (text + "\n").encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError):
        raise _invalid_json() from None


def _record_payload(record: MappingRecord) -> dict[str, object]:
    if not isinstance(record, MappingRecord):
        raise _invalid_json()
    if (
        type(record.source_address) is not str
        or type(record.changed) is not bool
        or type(record.vlm_semantics) is not dict
        or type(record.citations) is not tuple
        or type(record.decision) is not str
        or type(record.reason) is not str
    ):
        raise _invalid_json()

    citations: list[dict[str, object]] = []
    for citation in record.citations:
        cloned = _safe_clone_json_native(citation)
        if type(cloned) is not dict or set(cloned) != _CITATION_KEYS:
            raise _invalid_json()
        if any(type(cloned[key]) is not str for key in _CITATION_KEYS):
            raise _invalid_json()
        citations.append(cloned)

    semantics = _safe_clone_json_native(record.vlm_semantics)
    if type(semantics) is not dict:
        raise _invalid_json()
    return {
        "source_address": record.source_address,
        "source": _safe_clone_json_native(record.source),
        "output": _safe_clone_json_native(record.output),
        "candidate": _safe_clone_json_native(record.candidate),
        "changed": record.changed,
        "vlm_semantics": semantics,
        "citations": citations,
        "decision": record.decision,
        "reason": record.reason,
    }


def _local_evidence_payload(identity: IdentityEvidence) -> dict[str, object]:
    if not isinstance(identity, IdentityEvidence) or type(identity.tasks) is not tuple:
        raise _invalid_json()
    states = (
        identity.info_robot_type_state,
        identity.common_record_state,
        identity.tasks_state,
    )
    if any(type(state) is not str for state in states):
        raise _invalid_json()
    return {
        "info_robot_type": {
            "state": identity.info_robot_type_state,
            "value": _safe_clone_json_native(identity.info_robot_type),
        },
        "common_record": {
            "state": identity.common_record_state,
            "value": _safe_clone_json_native(identity.common_record),
        },
        "tasks": {
            "state": identity.tasks_state,
            "records": [
                _safe_clone_json_native(record) for record in identity.tasks
            ],
        },
    }


def _issue_payload(issue: Issue) -> dict[str, object]:
    if not isinstance(issue, Issue) or any(
        type(value) is not str
        for value in (issue.code, issue.message, issue.scope, issue.severity)
    ):
        raise _invalid_json()
    evidence = _safe_clone_json_native(issue.evidence)
    if type(evidence) is not dict:
        raise _invalid_json()
    return {
        "code": issue.code,
        "message": issue.message,
        "scope": issue.scope,
        "evidence": evidence,
        "severity": issue.severity,
    }


def build_review_payload(
    candidate: DatasetCandidate,
    status: DatasetStatus,
    evidence: DatasetEvidence,
    profile: HardwareProfile | None,
    result: NormalizationResult,
    *,
    generator: Mapping[str, object],
) -> dict[str, object]:
    """Build the fixed eight-key review object before its info hash is added."""

    if (
        not isinstance(candidate, DatasetCandidate)
        or type(status) is not DatasetStatus
        or not isinstance(evidence, DatasetEvidence)
        or evidence.candidate != candidate
        or (profile is not None and not isinstance(profile, HardwareProfile))
        or not isinstance(result, NormalizationResult)
        or not isinstance(generator, Mapping)
        or type(result.camera_mappings) is not tuple
        or type(result.machine_mappings) is not tuple
        or type(result.issues) is not tuple
    ):
        raise _invalid_json()
    try:
        generator_copy = _safe_clone_json_native(dict(generator))
        identity_payload = {
            "local_evidence": _local_evidence_payload(evidence.identity),
            **_record_payload(result.robot_identity),
        }
        payload = {
            "schema_version": "mini-1",
            "generator": generator_copy,
            "dataset": {
                "name": candidate.dataset_name,
                "layout_type": candidate.layout_type.value,
            },
            "status": status.value,
            "robot_identity": identity_payload,
            "camera_mappings": [
                _record_payload(record) for record in result.camera_mappings
            ],
            "machine_mappings": [
                _record_payload(record) for record in result.machine_mappings
            ],
            "issues": [_issue_payload(issue) for issue in result.issues],
        }
        _validate_review_without_hash(payload)
        return payload
    except (TypeError, ValueError, AttributeError, KeyError, RecursionError, OverflowError):
        raise _invalid_json() from None


def _validate_record_payload(record: object) -> None:
    if type(record) is not dict or set(record) != _RECORD_KEYS:
        raise _invalid_json()
    if (
        type(record["source_address"]) is not str
        or type(record["changed"]) is not bool
        or type(record["vlm_semantics"]) is not dict
        or type(record["citations"]) is not list
        or type(record["decision"]) is not str
        or type(record["reason"]) is not str
    ):
        raise _invalid_json()
    for citation in record["citations"]:
        if type(citation) is not dict or set(citation) != _CITATION_KEYS:
            raise _invalid_json()
        if any(type(citation[key]) is not str for key in _CITATION_KEYS):
            raise _invalid_json()


def _validate_local_evidence(payload: object) -> None:
    if type(payload) is not dict or set(payload) != {
        "info_robot_type",
        "common_record",
        "tasks",
    }:
        raise _invalid_json()
    for source_name in ("info_robot_type", "common_record"):
        source = payload[source_name]
        if (
            type(source) is not dict
            or set(source) != {"state", "value"}
            or type(source["state"]) is not str
        ):
            raise _invalid_json()
    tasks = payload["tasks"]
    if (
        type(tasks) is not dict
        or set(tasks) != {"state", "records"}
        or type(tasks["state"]) is not str
        or type(tasks["records"]) is not list
    ):
        raise _invalid_json()


def _validate_review_without_hash(review: object) -> None:
    if type(review) is not dict or set(review) != _REVIEW_KEYS:
        raise _invalid_json()
    if review["schema_version"] != "mini-1" or type(review["schema_version"]) is not str:
        raise _invalid_json()
    if type(review["generator"]) is not dict:
        raise _invalid_json()
    dataset = review["dataset"]
    if (
        type(dataset) is not dict
        or set(dataset) != {"name", "layout_type"}
        or type(dataset["name"]) is not str
        or type(dataset["layout_type"]) is not str
        or dataset["layout_type"] not in _LAYOUT_VALUES
    ):
        raise _invalid_json()
    if type(review["status"]) is not str or review["status"] not in _STATUS_VALUES:
        raise _invalid_json()

    identity = review["robot_identity"]
    if type(identity) is not dict or set(identity) != _IDENTITY_KEYS:
        raise _invalid_json()
    _validate_local_evidence(identity["local_evidence"])
    _validate_record_payload(
        {key: identity[key] for key in _RECORD_KEYS}
    )

    for collection_name in ("camera_mappings", "machine_mappings"):
        records = review[collection_name]
        if type(records) is not list:
            raise _invalid_json()
        for record in records:
            _validate_record_payload(record)

    issues = review["issues"]
    if type(issues) is not list:
        raise _invalid_json()
    for issue in issues:
        if type(issue) is not dict or set(issue) != _ISSUE_KEYS:
            raise _invalid_json()
        if any(
            type(issue[key]) is not str
            for key in ("code", "message", "scope", "severity")
        ) or type(issue["evidence"]) is not dict:
            raise _invalid_json()
    _safe_clone_json_native(review)


def _validated_output_paths(
    candidate: DatasetCandidate,
) -> tuple[Path, Path, Path]:
    if not isinstance(candidate, DatasetCandidate):
        raise ValueError("输出路径不安全")
    source_path = candidate.source_path
    info_path = candidate.info_path
    if not isinstance(source_path, Path) or not isinstance(info_path, Path):
        raise ValueError("输出路径不安全")
    expected_info_path = source_path / "meta" / "info.json"
    if info_path != expected_info_path:
        raise ValueError("输出路径不安全")
    meta_dir = info_path.parent
    if (
        not source_path.is_absolute()
        or source_path.is_symlink()
        or not source_path.is_dir()
        or source_path.resolve(strict=True) != source_path
        or meta_dir.is_symlink()
        or not meta_dir.is_dir()
        or meta_dir.resolve(strict=True) != meta_dir
        or info_path.is_symlink()
        or not info_path.is_file()
        or info_path.resolve(strict=True) != info_path
    ):
        raise ValueError("输出路径不安全")

    info_output = meta_dir / _INFO_OUTPUT_NAME
    review_output = meta_dir / _REVIEW_OUTPUT_NAME
    for output in (info_output, review_output):
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise ValueError("输出路径不安全")
        if output.exists() and output.resolve(strict=True) != output:
            raise ValueError("输出路径不安全")
    return meta_dir, info_output, review_output


def _directory_open_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if directory is None or nofollow is None:
        raise OSError(errno.ENOTSUP, "当前平台不支持安全目录写入")
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _assert_meta_attached(source_fd: int, meta_fd: int) -> None:
    entry = os.stat("meta", dir_fd=source_fd, follow_symlinks=False)
    opened = os.fstat(meta_fd)
    if not stat.S_ISDIR(entry.st_mode) or not _same_file(entry, opened):
        raise ValueError("输出路径不安全")


def _validate_relative_file(meta_fd: int, name: str, *, required: bool) -> None:
    try:
        entry = os.stat(name, dir_fd=meta_fd, follow_symlinks=False)
    except FileNotFoundError:
        if required:
            raise ValueError("输出路径不安全") from None
        return
    if not stat.S_ISREG(entry.st_mode):
        raise ValueError("输出路径不安全")


def _open_output_directories(source_path: Path) -> tuple[int, int]:
    flags = _directory_open_flags()
    source_fd = os.open(source_path, flags)
    meta_fd: int | None = None
    completed = False
    try:
        source_entry = os.stat(source_path, follow_symlinks=False)
        if not stat.S_ISDIR(source_entry.st_mode) or not _same_file(
            source_entry, os.fstat(source_fd)
        ):
            raise ValueError("输出路径不安全")
        meta_fd = os.open("meta", flags, dir_fd=source_fd)
        _assert_meta_attached(source_fd, meta_fd)
        _validate_relative_file(meta_fd, "info.json", required=True)
        _validate_relative_file(meta_fd, _INFO_OUTPUT_NAME, required=False)
        _validate_relative_file(meta_fd, _REVIEW_OUTPUT_NAME, required=False)
        completed = True
        return source_fd, meta_fd
    finally:
        if not completed:
            if meta_fd is not None:
                os.close(meta_fd)
            os.close(source_fd)


def _remove_temp(meta_fd: int, name: str | None) -> None:
    if name is None:
        return
    try:
        os.unlink(name, dir_fd=meta_fd)
    except FileNotFoundError:
        pass


def _create_temp(meta_fd: int, prefix: str) -> tuple[str, int]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError(errno.ENOTSUP, "当前平台不支持安全临时文件")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(16):
        name = f".{prefix}.{secrets.token_hex(16)}.tmp"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=meta_fd)
        except FileExistsError:
            continue
    raise FileExistsError(errno.EEXIST, "无法创建唯一临时文件")


def _write_temp(
    source_fd: int,
    meta_fd: int,
    *,
    prefix: str,
    payload: bytes,
) -> str:
    temp_name: str | None = None
    completed = False
    try:
        _assert_meta_attached(source_fd, meta_fd)
        temp_name, temp_fd = _create_temp(meta_fd, prefix)
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(temp_fd, view[written:])
                if count <= 0:
                    raise OSError(errno.EIO, "临时文件写入不完整")
                written += count
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
        completed = True
        return temp_name
    finally:
        if not completed:
            _remove_temp(meta_fd, temp_name)


def write_outputs(
    candidate: DatasetCandidate,
    info_norm: Mapping[str, object],
    review_without_hash: Mapping[str, object],
) -> tuple[Path, Path]:
    """Atomically replace only the normalized info and its matching review."""

    if type(info_norm) is not dict or type(review_without_hash) is not dict:
        raise _invalid_json()
    _safe_clone_json_native(info_norm)
    review_copy = _safe_clone_json_native(review_without_hash)
    _validate_review_without_hash(review_copy)

    info_bytes = _json_bytes(info_norm)
    info_digest = "sha256:" + hashlib.sha256(info_bytes).hexdigest()
    review_copy["info_norm_sha256"] = info_digest
    review_bytes = _json_bytes(review_copy)
    _, info_output, review_output = _validated_output_paths(candidate)

    source_fd: int | None = None
    meta_fd: int | None = None
    info_temp: str | None = None
    review_temp: str | None = None
    try:
        source_fd, meta_fd = _open_output_directories(candidate.source_path)
        info_temp = _write_temp(
            source_fd,
            meta_fd,
            prefix="info_norm",
            payload=info_bytes,
        )
        review_temp = _write_temp(
            source_fd,
            meta_fd,
            prefix="info_norm_review",
            payload=review_bytes,
        )
        _assert_meta_attached(source_fd, meta_fd)
        os.replace(
            info_temp,
            _INFO_OUTPUT_NAME,
            src_dir_fd=meta_fd,
            dst_dir_fd=meta_fd,
        )
        _assert_meta_attached(source_fd, meta_fd)
        os.replace(
            review_temp,
            _REVIEW_OUTPUT_NAME,
            src_dir_fd=meta_fd,
            dst_dir_fd=meta_fd,
        )
        _assert_meta_attached(source_fd, meta_fd)
    finally:
        if meta_fd is not None:
            try:
                _remove_temp(meta_fd, info_temp)
                _remove_temp(meta_fd, review_temp)
            finally:
                try:
                    os.close(meta_fd)
                finally:
                    if source_fd is not None:
                        os.close(source_fd)
    return info_output, review_output
