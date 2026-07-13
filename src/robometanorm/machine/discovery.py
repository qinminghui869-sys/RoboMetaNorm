"""发现 action、observation.state 及其子字段。"""

from __future__ import annotations

from collections.abc import Mapping


PARENT_MACHINE_FEATURES = ("action", "observation.state")


def discover_machine_features(info: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """返回机器父字段和 observation.state 子字段定义。"""
    features = info.get("features")
    if not isinstance(features, Mapping):
        return {}
    return {
        str(key): value
        for key, value in features.items()
        if isinstance(value, Mapping)
        and (str(key) in PARENT_MACHINE_FEATURES or str(key).startswith("observation.state."))
    }
