"""P2 机器语义 VLM 协议与裁决测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.machine.name_builder import build_names_from_semantics
from robometanorm.machine.prompt_builder import build_machine_prompt
from robometanorm.machine.vlm_semantic_resolver import (
    can_apply_semantics,
    parse_machine_semantics,
)


class MachineVlmTest(unittest.TestCase):
    """验证机器 VLM 不越过规则命名边界。"""

    def test_parses_semantics_without_accepting_final_field_name(self) -> None:
        evidence = {
            "dataset_name": "dataset_001",
            "robot_type": "test_robot",
            "parent_feature": "observation.state",
            "source_feature": "observation.state.head",
            "source_slice": [0, 4],
            "shape": [4],
            "declared_names": ["head_rotation_quat"],
            "numeric_profile": {"quaternion_norm_valid": True},
            "relations": {"is_parent_slice": True},
            "rule_candidates": ["head_orientation_quaternion"],
        }
        system_prompt, user_prompt = build_machine_prompt(evidence)
        semantics = parse_machine_semantics(
            {
                "semantic_type": "head_orientation_quaternion",
                "side": "none",
                "body_part": "head",
                "representation": "quaternion_xyzw",
                "unit": "unknown",
                "declared_name_status": "partially_correct",
                "standardizable": "direct",
                "required_transform": "none",
                "confidence": 0.94,
                "alternatives": [],
                "need_human_review": False,
                "reason": "四元数模长稳定。",
            }
        )

        self.assertNotIn("target_key", system_prompt + user_prompt)
        self.assertTrue(can_apply_semantics(semantics, vector_length=4))
        self.assertEqual(
            build_names_from_semantics(semantics, vector_length=4),
            ["head_orient_quat_x", "head_orient_quat_y", "head_orient_quat_z", "head_orient_quat_w"],
        )
        with self.assertRaises(ValueError):
            parse_machine_semantics({"target_key": "head_orient_quat_x"})

    def test_refuses_low_confidence_or_transform_required_semantics(self) -> None:
        semantics = parse_machine_semantics(
            {
                "semantic_type": "eef_rotation_euler",
                "side": "left",
                "body_part": "arm",
                "representation": "quaternion_xyzw",
                "unit": "rad",
                "declared_name_status": "misleading",
                "standardizable": "needs_transform",
                "required_transform": "quaternion_to_euler",
                "confidence": 0.91,
                "alternatives": [],
                "need_human_review": False,
                "reason": "需要旋转表示转换。",
            }
        )

        self.assertFalse(can_apply_semantics(semantics, vector_length=4))

    def test_refuses_semantics_with_wrong_vector_representation(self) -> None:
        semantics = parse_machine_semantics(
            {
                "semantic_type": "head_orientation_quaternion",
                "side": "none",
                "body_part": "head",
                "representation": "scalar",
                "unit": "unknown",
                "declared_name_status": "misleading",
                "standardizable": "direct",
                "required_transform": "none",
                "confidence": 0.95,
                "alternatives": [],
                "need_human_review": False,
                "reason": "表示形式错误。",
            }
        )

        self.assertFalse(can_apply_semantics(semantics, vector_length=4))


if __name__ == "__main__":
    unittest.main()
