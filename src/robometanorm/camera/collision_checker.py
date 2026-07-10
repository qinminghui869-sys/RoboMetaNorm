"""相机目标名称重名检测。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from robometanorm.camera.models import CameraNameProposal


def find_colliding_sources(proposals: Iterable[CameraNameProposal]) -> set[str]:
    """返回解析到相同目标字段的全部源字段。"""
    by_target: dict[str, list[str]] = defaultdict(list)
    for proposal in proposals:
        by_target[proposal.target_key].append(proposal.source_key)
    return {
        source_key
        for source_keys in by_target.values()
        if len(source_keys) > 1
        for source_key in source_keys
    }
