"""P2 机器语义 VLM 分段协议与裁决测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.machine.rules import build_names_from_semantics
from robometanorm.machine.vlm import (
    OpenAICompatibleMachineVlmResolver,
    build_machine_prompt,
    can_apply_semantics,
    parse_machine_semantics,
)


class _SequencedClient:
    """按顺序返回 JSON，并记录每次请求。"""

    def __init__(self, payloads: list[dict[str, object] | None]) -> None:
        self.payloads = list(payloads)
        self.requests: list[tuple[str, str]] = []
        self.last_error = "transport failed"

    def request_json(
        self, system_prompt: str, user_prompt: str, image_paths: tuple[object, ...]
    ) -> dict[str, object] | None:
        self.requests.append((system_prompt, user_prompt))
        return self.payloads.pop(0)


class MachineVlmTest(unittest.TestCase):
    """验证复合字段可表达且不能越过规则命名边界。"""

    def test_parses_contiguous_composite_segments(self) -> None:
        semantics = self._parse(self._two_arm_payload(), vector_length=6)

        self.assertEqual(
            [segment.local_slice for segment in semantics.segments],
            [(0, 3), (3, 6)],
        )
        self.assertEqual(
            [segment.side for segment in semantics.segments], ["left", "right"]
        )
        self.assertTrue(can_apply_semantics(semantics, vector_length=6))
        self.assertEqual(
            [
                name
                for segment in semantics.segments
                for name in (
                    build_names_from_semantics(
                        segment, segment.local_slice[1] - segment.local_slice[0]
                    )
                    or []
                )
            ],
            [
                "left_arm_joint_0_rad",
                "left_arm_joint_1_rad",
                "left_arm_joint_2_rad",
                "right_arm_joint_0_rad",
                "right_arm_joint_1_rad",
                "right_arm_joint_2_rad",
            ],
        )

    def test_wraps_legacy_payload_as_one_full_segment(self) -> None:
        semantics = self._parse(self._legacy_head_payload(), vector_length=4)

        self.assertEqual(len(semantics.segments), 1)
        self.assertEqual(semantics.segments[0].local_slice, (0, 4))
        self.assertTrue(can_apply_semantics(semantics, vector_length=4))

    def test_rejects_gap_overlap_reverse_and_out_of_bounds_segments(self) -> None:
        invalid_slices = (
            ([0, 2], [3, 6]),
            ([0, 4], [3, 6]),
            ([0, 4], [4, 7]),
            ([0, 4], [4, 3]),
        )
        for slices in invalid_slices:
            payload = self._two_arm_payload()
            segments = payload["segments"]
            assert isinstance(segments, list)
            segments[0]["local_slice"] = slices[0]
            segments[1]["local_slice"] = slices[1]
            with self.subTest(slices=slices), self.assertRaises(ValueError):
                self._parse(payload, vector_length=6)

    def test_rejects_empty_segments_invalid_enum_and_final_field_name(self) -> None:
        empty = self._two_arm_payload()
        empty["segments"] = []
        with self.assertRaises(ValueError):
            self._parse(empty, vector_length=6)

        invalid_enum = self._two_arm_payload()
        segments = invalid_enum["segments"]
        assert isinstance(segments, list)
        segments[0]["representation"] = "pose_7d"
        with self.assertRaises(ValueError):
            self._parse(invalid_enum, vector_length=6)

        forbidden = self._two_arm_payload()
        segments = forbidden["segments"]
        assert isinstance(segments, list)
        segments[0]["target_key"] = "left_arm_joint_0_rad"
        with self.assertRaises(ValueError):
            self._parse(forbidden, vector_length=6)

        missing_segment_boolean = self._two_arm_payload()
        segments = missing_segment_boolean["segments"]
        assert isinstance(segments, list)
        segments[0].pop("need_human_review")
        with self.assertRaises(ValueError):
            self._parse(missing_segment_boolean, vector_length=6)

        missing_root_reason = self._two_arm_payload()
        missing_root_reason.pop("reason")
        with self.assertRaises(ValueError):
            self._parse(missing_root_reason, vector_length=6)

    def test_rejects_dexterous_hand_semantics_but_accepts_gripper(self) -> None:
        for semantic_type in ("hand_joint", "hand_keypoints"):
            payload = self._legacy_head_payload()
            payload["semantic_type"] = semantic_type
            with self.subTest(semantic_type=semantic_type), self.assertRaisesRegex(
                ValueError, "semantic_type 不合法"
            ):
                self._parse(payload, vector_length=4)

        gripper = self._legacy_head_payload()
        gripper.update(
            {
                "semantic_type": "gripper_open",
                "body_part": "gripper",
                "representation": "scalar",
                "unit": "none",
            }
        )
        semantics = self._parse(gripper, vector_length=1)
        self.assertEqual(semantics.segments[0].semantic_type, "gripper_open")

    def test_refuses_low_confidence_or_transform_required_segment(self) -> None:
        payload = self._two_arm_payload()
        segments = payload["segments"]
        assert isinstance(segments, list)
        segments[1]["confidence"] = 0.91
        semantics = self._parse(payload, vector_length=6)
        self.assertFalse(can_apply_semantics(semantics, vector_length=6))

        payload = self._two_arm_payload()
        segments = payload["segments"]
        assert isinstance(segments, list)
        segments[1]["standardizable"] = "needs_transform"
        segments[1]["required_transform"] = "quaternion_to_euler"
        semantics = self._parse(payload, vector_length=6)
        self.assertFalse(can_apply_semantics(semantics, vector_length=6))

    def test_prompt_contains_exact_json_segment_contract(self) -> None:
        system_prompt, user_prompt = build_machine_prompt(self._evidence(6))
        prompt = system_prompt + user_prompt

        self.assertIn("JSON", prompt)
        self.assertIn('"segments"', prompt)
        self.assertIn('"local_slice"', prompt)
        self.assertIn("quaternion_xyzw", prompt)
        self.assertIn("连续覆盖 [0, 6]", prompt)
        self.assertIn("复合 JSON 示例", prompt)
        self.assertIn('"local_slice": [\n        3,\n        7\n      ]', prompt)
        self.assertIn("不得输出最终标准字段名", prompt)

    def test_prompt_limits_machine_scope_to_gripper_end_effectors(self) -> None:
        system_prompt, user_prompt = build_machine_prompt(self._evidence(6))

        self.assertIn("机械臂末端为夹爪", system_prompt)
        self.assertIn("semantic_type=unknown", system_prompt)
        self.assertIn("standardizable=not_covered", system_prompt)
        self.assertIn("need_human_review=true", system_prompt)
        self.assertNotIn("hand_joint", user_prompt)
        self.assertNotIn("hand_keypoints", user_prompt)
        self.assertIn("gripper_open", user_prompt)

    def test_retries_one_invalid_schema_response_then_accepts_valid_json(self) -> None:
        invalid = self._two_arm_payload()
        segments = invalid["segments"]
        assert isinstance(segments, list)
        segments[0]["representation"] = "pose_7d"
        client = _SequencedClient([invalid, self._two_arm_payload()])
        resolver = OpenAICompatibleMachineVlmResolver(client)

        semantics = resolver.resolve(self._evidence(6))

        self.assertIsNotNone(semantics)
        self.assertEqual(len(client.requests), 2)
        self.assertIn("上次 JSON 不符合协议", client.requests[1][1])

    def test_raises_sanitized_error_after_second_invalid_schema_response(self) -> None:
        invalid = self._two_arm_payload()
        segments = invalid["segments"]
        assert isinstance(segments, list)
        segments[0]["representation"] = "pose_7d"
        client = _SequencedClient([invalid, invalid])
        resolver = OpenAICompatibleMachineVlmResolver(client)

        with self.assertRaises(RuntimeError) as raised:
            resolver.resolve(self._evidence(6))

        self.assertEqual(len(client.requests), 2)
        self.assertIn("representation", str(raised.exception))
        self.assertNotIn("Authorization", str(raised.exception))

    def test_raises_transport_error_when_client_returns_none(self) -> None:
        resolver = OpenAICompatibleMachineVlmResolver(_SequencedClient([None]))

        with self.assertRaises(RuntimeError) as raised:
            resolver.resolve(self._evidence(6))

        self.assertIn("transport failed", str(raised.exception))

    def test_sanitizes_transport_error_detail(self) -> None:
        client = _SequencedClient([None])
        client.last_error = (
            "Authorization: Bearer sk-secret; network request failed"
        )
        resolver = OpenAICompatibleMachineVlmResolver(client)

        with self.assertRaises(RuntimeError) as raised:
            resolver.resolve(self._evidence(6))

        message = str(raised.exception)
        self.assertIn("network request failed", message)
        self.assertNotIn("Authorization", message)
        self.assertNotIn("sk-secret", message)

    def _parse(self, payload: dict[str, object], vector_length: int):
        return parse_machine_semantics(payload, vector_length=vector_length)

    @staticmethod
    def _segment(start: int, end: int, side: str) -> dict[str, object]:
        return {
            "local_slice": [start, end],
            "semantic_type": "arm_joint",
            "side": side,
            "body_part": "arm",
            "representation": "joint_vector",
            "unit": "rad",
            "declared_name_status": "partially_correct",
            "standardizable": "direct",
            "required_transform": "none",
            "confidence": 0.96,
            "alternatives": [],
            "need_human_review": False,
            "reason": "分段和单位由字段证据确认。",
        }

    @classmethod
    def _two_arm_payload(cls) -> dict[str, object]:
        return {
            "segments": [cls._segment(0, 3, "left"), cls._segment(3, 6, "right")],
            "need_human_review": False,
            "reason": "左右臂连续分段明确。",
        }

    @staticmethod
    def _legacy_head_payload() -> dict[str, object]:
        return {
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

    @staticmethod
    def _evidence(vector_length: int) -> dict[str, object]:
        return {
            "dataset_name": "dataset_001",
            "robot_type": "test_robot",
            "parent_feature": "observation.state",
            "source_feature": "observation.state.arms",
            "source_slice": [0, vector_length],
            "shape": [vector_length],
            "declared_names": ["left_arm", "right_arm"],
            "numeric_profile": {},
            "relations": {"is_parent_slice": True},
            "rule_candidates": [],
        }


if __name__ == "__main__":
    unittest.main()
