"""机器字段命名范围测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.machine.rules import (
    gripper_direction_from_name,
    infer_gripper_range,
    build_confirmed_machine_name,
    build_names_from_semantics,
    risk_categories,
    unknown_unit_indices,
)
from robometanorm.machine.models import ScalarProfile


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

    def test_infers_supported_gripper_scale_with_small_boundary_overshoot(self) -> None:
        profile = ScalarProfile(
            sample_count=1000,
            min_value=-3.0,
            max_value=103.0,
            p01=0.0,
            p05=0.0,
            p50=50.0,
            p95=100.0,
            p99=100.0,
            mean_value=50.0,
            std_value=40.0,
            nan_ratio=0.0,
            inf_ratio=0.0,
            unique_count=101,
        )

        inferred = infer_gripper_range(profile)

        self.assertEqual(inferred.closed_value, 0.0)
        self.assertEqual(inferred.open_value, 100.0)
        self.assertTrue(inferred.clipping_required)
        self.assertGreaterEqual(inferred.confidence, 0.9)

    def test_uses_only_explicit_opening_semantics_for_rule_direction(self) -> None:
        self.assertEqual(
            gripper_direction_from_name("left_gripper_open"),
            "increasing_is_open",
        )
        self.assertEqual(
            gripper_direction_from_name("leader_left_gripper_degree_mm.pos"),
            "increasing_is_open",
        )
        self.assertIsNone(gripper_direction_from_name("right_gripper"))

    def test_rejects_range_with_extreme_outlier(self) -> None:
        profile = ScalarProfile(
            sample_count=1000,
            min_value=0.0,
            max_value=10000.0,
            p01=0.0,
            p05=0.0,
            p50=50.0,
            p95=100.0,
            p99=100.0,
            mean_value=60.0,
            std_value=300.0,
            nan_ratio=0.0,
            inf_ratio=0.0,
            unique_count=101,
        )

        self.assertIsNone(infer_gripper_range(profile))


if __name__ == "__main__":
    unittest.main()
