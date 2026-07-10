"""精确映射和字段名正则解析。"""

from __future__ import annotations

import re

from robometanorm.camera.mapping_registry import CAMERA_NAME_MAP
from robometanorm.camera.models import CameraNameProposal
from robometanorm.camera.name_builder import (
    BODY_PART_TOKENS,
    EXTERNAL_DIRECTION_TOKENS,
    build_camera_key,
)


def propose_camera_name(source_key: str) -> CameraNameProposal | None:
    """为相机源字段生成确定性目标名，无法判断时返回空。"""
    target_key = CAMERA_NAME_MAP.get(source_key)
    if target_key:
        return CameraNameProposal(
            source_key=source_key,
            target_key=target_key,
            modality=_modality_from_target(target_key),
            method="exact",
        )

    field_name = source_key.rsplit(".", maxsplit=1)[-1].lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", field_name) if token]
    direction_tokens = [token for token in tokens if token in EXTERNAL_DIRECTION_TOKENS]
    body_parts = [token for token in tokens if token in BODY_PART_TOKENS]
    if len(set(body_parts)) > 1:
        return None
    modality = "depth" if "depth" in tokens else "rgb"
    target_key = build_camera_key(direction_tokens, body_parts[0] if body_parts else None, modality)
    if target_key is None:
        return None
    return CameraNameProposal(
        source_key=source_key,
        target_key=target_key,
        modality=modality,
        method="regex",
    )


def _modality_from_target(target_key: str) -> str:
    """从标准目标名恢复已验证的模态后缀。"""
    return "depth" if target_key.endswith("_depth") else "rgb"
