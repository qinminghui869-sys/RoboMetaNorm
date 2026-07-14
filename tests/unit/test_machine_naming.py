"""机器字段命名范围测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.machine.rules import (
    build_confirmed_machine_name,
    build_names_from_semantics,
    risk_categories,
    unknown_unit_indices,
)


class MachineNamingTest(unittest.TestCase):
    """验证灵巧手名称不在夹爪末端规范范围内。"""

    def test_rejects_hand_joint_names_but_keeps_gripper_names(self) -> None:
        hand_semantics = SimpleNamespace(
            semantic_type="hand_joint", side="left", unit="rad"
        )

        self.assertIsNone(build_confirmed_machine_name("left_hand_joint_0_rad"))
        self.assertIsNone(build_names_from_semantics(hand_semantics, 2))
        self.assertEqual(
            build_confirmed_machine_name("left_gripper_open"),
            "left_gripper_open",
        )

    def test_recognizes_unit_tokens_before_structural_suffixes(self) -> None:
        names = [
            "follower_left_arm_joint_1_rad.pos",
            "right_joint_1",
            "left_eef_position_x_m.value",
        ]

        self.assertEqual(unknown_unit_indices(names), (1,))
        self.assertNotIn(
            "UNKNOWN_UNIT",
            risk_categories(["follower_left_arm_joint_1_rad.pos"]),
        )
        self.assertIn("UNKNOWN_UNIT", risk_categories(["right_joint_1"]))


if __name__ == "__main__":
    unittest.main()
