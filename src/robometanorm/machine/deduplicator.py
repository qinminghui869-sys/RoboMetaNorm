"""action/state 与父子字段的事实去重工具。"""

from __future__ import annotations

import numpy as np

from robometanorm.machine.models import ParquetProfile


def action_equals_state(profile: ParquetProfile | None) -> bool:
    """仅在样本形状和值均一致时认定 action 与 state 可复用。"""
    if profile is None:
        return False
    action = profile.samples.get("action")
    state = profile.samples.get("observation.state")
    return bool(
        action is not None
        and state is not None
        and action.shape == state.shape
        and np.array_equal(action, state, equal_nan=True)
    )
