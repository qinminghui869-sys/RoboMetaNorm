"""从数据集元数据解析可追溯的机器人身份。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class RobotIdentityEvidence:
    """单个元数据来源提供的机器人身份候选。"""

    source: str
    raw_value: str
    canonical_id: str


@dataclass(frozen=True)
class RobotIdentityConflict:
    """低优先级候选与已选机器人系列不一致。"""

    source: str
    raw_value: str
    canonical_id: str


@dataclass(frozen=True)
class RobotIdentity:
    """按来源优先级合并后的机器人身份。"""

    canonical_id: str | None
    selected_source: str | None
    selected_value: str | None
    evidence: tuple[RobotIdentityEvidence, ...] = ()
    conflicts: tuple[RobotIdentityConflict, ...] = ()


def robot_identity_payload(identity: RobotIdentity) -> dict[str, object]:
    """转换为复核文件和 VLM 证据共用的 JSON 结构。"""
    return {
        "canonical_id": identity.canonical_id,
        "selected_source": identity.selected_source,
        "selected_value": identity.selected_value,
        "evidence": [
            {
                "source": item.source,
                "raw_value": item.raw_value,
                "canonical_id": item.canonical_id,
            }
            for item in identity.evidence
        ],
        "conflicts": [
            {
                "source": item.source,
                "raw_value": item.raw_value,
                "canonical_id": item.canonical_id,
            }
            for item in identity.conflicts
        ],
    }


_KNOWN_IDENTITIES = (
    "agilex_cobot_magic",
    "galaxea_r1_lite",
    "dexterous_hand_v1",
    "airbot_mmk2",
    "galbot_g1",
    "dexterous_hand",
    "galaxea",
    "galbot",
    "aloha",
)

_TASK_IDENTITIES = frozenset(
    {
        "agilex_cobot_magic",
        "galaxea_r1_lite",
        "dexterous_hand_v1",
        "airbot_mmk2",
        "galbot_g1",
    }
)

_PLACEHOLDER_VALUES = frozenset(
    {"unknown", "none", "null", "na", "n_a", "not_applicable", "unspecified"}
)

_FAMILIES = {
    "agilex_cobot_magic": "agilex_cobot_magic",
    "airbot_mmk2": "airbot_mmk2",
    "galaxea_r1_lite": "galaxea",
    "galaxea": "galaxea",
    "galbot_g1": "galbot",
    "galbot": "galbot",
    "dexterous_hand_v1": "dexterous_hand",
    "dexterous_hand": "dexterous_hand",
    "aloha": "aloha",
}


def resolve_robot_identity(
    meta_path: Path, source_info: Mapping[str, object]
) -> RobotIdentity:
    """按 info、common_record、tasks 的顺序合并机器人身份。"""
    evidence: list[RobotIdentityEvidence] = []
    for key in ("robot_type", "root_type"):
        value = source_info.get(key)
        if isinstance(value, str) and value.strip() and not _is_placeholder(value):
            canonical_id = _canonicalize_info_value(value)
            if canonical_id:
                evidence.append(
                    RobotIdentityEvidence(f"info.{key}", value, canonical_id)
                )

    machine_id = _read_machine_id(meta_path / "common_record.json")
    if machine_id is not None:
        canonical_id = _canonicalize_machine_id(machine_id)
        if canonical_id is not None:
            evidence.append(
                RobotIdentityEvidence(
                    "common_record.machine_id", machine_id, canonical_id
                )
            )

    for task_hint in _read_task_model_hints(meta_path / "tasks.jsonl"):
        canonical_id = _find_task_identity(task_hint)
        if canonical_id is not None:
            candidate = RobotIdentityEvidence(
                "tasks.model_hint", task_hint, canonical_id
            )
            if candidate not in evidence:
                evidence.append(candidate)

    if not evidence:
        return RobotIdentity(None, None, None)

    selected = evidence[0]
    canonical_id = selected.canonical_id
    conflicts: list[RobotIdentityConflict] = []
    for candidate in evidence[1:]:
        if _family(candidate.canonical_id) != _family(canonical_id):
            conflicts.append(
                RobotIdentityConflict(
                    candidate.source,
                    candidate.raw_value,
                    candidate.canonical_id,
                )
            )
            continue
        canonical_id = _more_specific(canonical_id, candidate.canonical_id)

    return RobotIdentity(
        canonical_id=canonical_id,
        selected_source=selected.source,
        selected_value=selected.raw_value,
        evidence=tuple(evidence),
        conflicts=tuple(conflicts),
    )


def _canonicalize_info_value(value: str) -> str | None:
    known = _find_known_identity(value)
    if known is not None:
        return known
    normalized = _normalize(value)
    return normalized or None


def _canonicalize_machine_id(value: str) -> str | None:
    known = _find_known_identity(value)
    if known is not None:
        return known
    _, separator, suffix = value.partition("_")
    if not separator or not suffix.strip():
        return None
    normalized = _normalize(suffix)
    if not normalized or normalized in _PLACEHOLDER_VALUES:
        return None
    return normalized


def _find_known_identity(value: str) -> str | None:
    normalized = f"_{_normalize(value)}_"
    for canonical_id in _KNOWN_IDENTITIES:
        if f"_{canonical_id}_" in normalized:
            return canonical_id
    return None


def _find_task_identity(value: str) -> str | None:
    canonical_id = _find_known_identity(value)
    return canonical_id if canonical_id in _TASK_IDENTITIES else None


def _normalize(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip(
        "_"
    )


def _is_placeholder(value: str) -> bool:
    return _normalize(value) in _PLACEHOLDER_VALUES


def _family(canonical_id: str) -> str:
    return _FAMILIES.get(canonical_id, canonical_id)


def _more_specific(first: str, second: str) -> str:
    if first == second:
        return first
    if _family(first) != _family(second):
        return first
    return max((first, second), key=lambda value: value.count("_") + len(value))


def _read_machine_id(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return _find_string_value(payload, "machine_id")


def _find_string_value(value: object, key: str) -> str | None:
    if isinstance(value, Mapping):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        for child in value.values():
            found = _find_string_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_string_value(child, key)
            if found is not None:
                return found
    return None


def _read_task_model_hints(path: Path) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    hints: list[str] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        for text in _string_values(payload):
            if _find_task_identity(text) is not None:
                hints.append(text)
    return tuple(dict.fromkeys(hints))


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text for child in value.values() for text in _string_values(child)
        )
    if isinstance(value, list):
        return tuple(text for child in value for text in _string_values(child))
    return ()
