"""P2 机器字段保守规范化测试。"""

from __future__ import annotations

from copy import deepcopy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.machine.models import ParquetProfile, VectorProfile
from robometanorm.machine.normalizer import normalize_machine_fields
from robometanorm.machine.vlm import (
    MachineSemanticSegment,
    MachineSemantics,
    MachineVlmResolutionError,
)


class _FixedMachineResolver:
    """返回固定语义，验证规则层的最终裁决。"""

    def __init__(self, semantics: MachineSemantics):
        self.semantics = semantics

    def resolve(self, evidence: dict[str, object]) -> MachineSemantics:
        return self.semantics


class _CountingMachineResolver(_FixedMachineResolver):
    """记录 VLM 语义请求次数。"""

    def __init__(self, semantics: MachineSemantics):
        super().__init__(semantics)
        self.call_count = 0

    def resolve(self, evidence: dict[str, object]) -> MachineSemantics:
        self.call_count += 1
        return super().resolve(evidence)


class _FailingMachineResolver:
    """模拟已经脱敏的 VLM 协议错误。"""

    def resolve(self, evidence: dict[str, object]) -> MachineSemantics:
        raise MachineVlmResolutionError(
            "Authorization: Bearer sk-secret; segments[0].representation 不合法: pose_7d"
        )


class MachineNormalizerTest(unittest.TestCase):
    """验证 P2 只改写有充分事实支撑的名称。"""

    def test_normalizes_confirmed_head_quaternion_and_reuses_equal_action_state(self) -> None:
        names = [
            "head_pose_x",
            "head_pose_y",
            "head_pose_z",
            "head_rotation_quat_x",
            "head_rotation_quat_y",
            "head_rotation_quat_z",
            "head_rotation_quat_w",
        ]
        source_info = self._info({"action": self._feature(names), "observation.state": self._feature(names)})
        original_info = deepcopy(source_info)

        result = normalize_machine_fields(source_info, self._profile({"action": 7, "observation.state": 7}, equal_action_state=True))

        expected = [
            "head_pose_x",
            "head_pose_y",
            "head_pose_z",
            "head_orient_quat_x",
            "head_orient_quat_y",
            "head_orient_quat_z",
            "head_orient_quat_w",
        ]
        self.assertEqual(result.normalized_info["features"]["action"]["names"], expected)
        self.assertEqual(result.normalized_info["features"]["observation.state"]["names"], expected)
        self.assertEqual(source_info, original_info)
        self.assertNotIn("ACTION_EQUALS_STATE", {item.category for item in result.machine_review_items})
        self.assertIn("UNKNOWN_UNIT", {item.category for item in result.machine_review_items})

    def test_keeps_unsafe_names_and_records_machine_review_categories(self) -> None:
        names = ["left_wrist_pose_x", "left_joint_1", "left_gripper"]
        source_info = self._info({"action": self._feature(names)})

        result = normalize_machine_fields(source_info, self._profile({"action": 3}))

        self.assertEqual(result.normalized_info["features"]["action"]["names"], names)
        self.assertEqual(
            {item.category for item in result.machine_review_items},
            {
                "WRIST_EEF_RELATION_UNKNOWN",
                "GRIPPER_RANGE_UNKNOWN",
                "GRIPPER_DIRECTION_UNKNOWN",
                "UNKNOWN_UNIT",
            },
        )

    def test_keeps_feature_when_declared_names_do_not_match_actual_vector_length(self) -> None:
        source_info = self._info({"action": self._feature(["head_rotation_quat_x"], shape=4)})

        result = normalize_machine_fields(source_info, self._profile({"action": 4}))

        self.assertEqual(result.normalized_info["features"]["action"]["names"], ["head_rotation_quat_x"])
        self.assertEqual(
            [item.category for item in result.machine_review_items], ["NAMES_ORDER_MISMATCH"]
        )

    def test_applies_only_high_confidence_unitless_vlm_semantics(self) -> None:
        source_info = self._info(
            {
                "observation.state.head": self._feature(
                    ["head_raw_0", "head_raw_1", "head_raw_2", "head_raw_3"]
                )
            }
        )
        semantics = self._head_quaternion_semantics()

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.head": 4}),
            vlm_resolver=_FixedMachineResolver(semantics),
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.head"]["names"],
            [
                "head_orient_quat_x",
                "head_orient_quat_y",
                "head_orient_quat_z",
                "head_orient_quat_w",
            ],
        )
        self.assertEqual(result.machine_review_items, ())

    def test_records_actual_parent_child_slice_on_machine_review(self) -> None:
        names = ["left_wrist_pose_x", "left_wrist_pose_y", "left_wrist_pose_z"]
        source_info = self._info(
            {
                "observation.state": self._feature(names),
                "observation.state.wrist": self._feature(names),
            }
        )
        parent_samples = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        profile = self._profile({"observation.state": 3, "observation.state.wrist": 3})
        profile = ParquetProfile(
            profile.row_count,
            profile.row_group_count,
            profile.schema_columns,
            profile.columns,
            {
                "observation.state": parent_samples,
                "observation.state.wrist": parent_samples.copy(),
            },
        )

        result = normalize_machine_fields(source_info, profile)

        wrist_review = next(
            item
            for item in result.machine_review_items
            if item.source_feature == "observation.state.wrist"
            and item.category == "WRIST_EEF_RELATION_UNKNOWN"
        )
        self.assertEqual(wrist_review.source_slice, (0, 3))

    def test_reuses_child_analysis_to_build_parent_names_without_duplicate_vlm_calls(self) -> None:
        raw_names = ["head_raw_0", "head_raw_1", "head_raw_2", "head_raw_3"]
        source_info = self._info(
            {
                "observation.state": self._feature(raw_names),
                "observation.state.head": self._feature(raw_names),
            }
        )
        samples = np.array([[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0.99]])
        profile = self._profile({"observation.state": 4, "observation.state.head": 4})
        profile = ParquetProfile(
            profile.row_count,
            profile.row_group_count,
            profile.schema_columns,
            profile.columns,
            {"observation.state": samples, "observation.state.head": samples.copy()},
        )
        resolver = _CountingMachineResolver(self._head_quaternion_semantics())

        result = normalize_machine_fields(source_info, profile, vlm_resolver=resolver)

        expected = [
            "head_orient_quat_x",
            "head_orient_quat_y",
            "head_orient_quat_z",
            "head_orient_quat_w",
        ]
        self.assertEqual(result.normalized_info["features"]["observation.state"]["names"], expected)
        self.assertEqual(
            result.normalized_info["features"]["observation.state.head"]["names"],
            expected,
        )
        self.assertEqual(resolver.call_count, 1)

    def test_keeps_names_when_episode_layout_is_inconsistent(self) -> None:
        names = [
            "head_rotation_quat_x",
            "head_rotation_quat_y",
            "head_rotation_quat_z",
            "head_rotation_quat_w",
        ]
        source_info = self._info({"action": self._feature(names)})
        profile = self._profile({"action": 4})
        profile = ParquetProfile(
            profile.row_count,
            profile.row_group_count,
            profile.schema_columns,
            profile.columns,
            profile.samples,
            episode_count=2,
            inconsistent_columns=("action",),
        )

        result = normalize_machine_fields(source_info, profile)

        self.assertEqual(result.normalized_info["features"]["action"]["names"], names)
        self.assertIn(
            "CROSS_EPISODE_LAYOUT_INCONSISTENT",
            {item.category for item in result.machine_review_items},
        )

    def test_resolves_grouped_names_only_when_vlm_semantics_fix_vector_order(self) -> None:
        source_info = self._info(
            {
                "observation.state.head": {
                    "dtype": "float32",
                    "shape": [4],
                    "names": ["head_rotation_quat"],
                }
            }
        )

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.head": 4}),
            vlm_resolver=_FixedMachineResolver(self._head_quaternion_semantics()),
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.head"]["names"],
            [
                "head_orient_quat_x",
                "head_orient_quat_y",
                "head_orient_quat_z",
                "head_orient_quat_w",
            ],
        )
        self.assertEqual(result.machine_review_items, ())

    def test_keeps_grouped_names_when_vlm_returns_no_supported_target(self) -> None:
        source_info = self._info(
            {
                "observation.state.grouped": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["grouped_value"],
                }
            }
        )
        unknown_semantics = self._unknown_semantics(3)

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.grouped": 3}),
            vlm_resolver=_FixedMachineResolver(unknown_semantics),
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.grouped"]["names"],
            ["grouped_value"],
        )
        self.assertIn(
            "NAMES_ORDER_MISMATCH",
            {item.category for item in result.machine_review_items},
        )
        review = next(item for item in result.machine_review_items if item.category == "NAMES_ORDER_MISMATCH")
        self.assertEqual(review.candidates, ("unknown",))

    def test_keeps_dexterous_hand_names_for_generic_review_without_vlm(self) -> None:
        names = ["left_hand_joint_0_rad", "left_finger_joint_0_rad"]
        source_info = self._info(
            {
                "observation.state.dexterous_hand": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": names,
                    "unit": "rad",
                }
            }
        )
        resolver = _CountingMachineResolver(self._unknown_semantics(2))

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.dexterous_hand": 2}),
            vlm_resolver=resolver,
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.dexterous_hand"]["names"],
            names,
        )
        self.assertEqual(resolver.call_count, 0)
        self.assertEqual(
            [item.category for item in result.machine_review_items],
            ["OUT_OF_SCOPE_MACHINE_FIELD"],
        )
        review = result.machine_review_items[0]
        self.assertEqual(review.candidates, ())
        self.assertIsNone(review.vlm_result)

    def test_keeps_grouped_hand_field_for_generic_review_without_vlm(self) -> None:
        source_info = self._info(
            {
                "observation.state.hand": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["hand_group"],
                }
            }
        )
        resolver = _CountingMachineResolver(self._unknown_semantics(3))

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.hand": 3}),
            vlm_resolver=resolver,
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.hand"]["names"],
            ["hand_group"],
        )
        self.assertEqual(resolver.call_count, 0)
        self.assertEqual(
            [item.category for item in result.machine_review_items],
            ["OUT_OF_SCOPE_MACHINE_FIELD"],
        )

    def test_skips_all_machine_analysis_when_skeleton_fields_are_out_of_scope(self) -> None:
        source_info = self._info(
            {
                "action": self._feature(["left_arm_joint_0_rad"]),
                "observation.state": self._feature(
                    ["left_arm_joint_0_rad", "left_skeleton_point_0"]
                ),
                "observation.state.full_skeleton": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["full_skeleton"],
                },
            }
        )
        original_info = deepcopy(source_info)
        resolver = _CountingMachineResolver(self._unknown_semantics(3))

        result = normalize_machine_fields(
            source_info,
            self._profile(
                {
                    "action": 1,
                    "observation.state": 2,
                    "observation.state.full_skeleton": 3,
                }
            ),
            vlm_resolver=resolver,
        )

        self.assertEqual(result.normalized_info, original_info)
        self.assertEqual(resolver.call_count, 0)
        self.assertEqual(len(result.machine_review_items), 1)
        review = result.machine_review_items[0]
        self.assertEqual(review.category, "OUT_OF_SCOPE_MACHINE_FIELD")
        self.assertIn("保留全部源字段", review.required_action)

    def test_records_transform_and_declared_name_conflict_from_vlm(self) -> None:
        source_info = self._info(
            {"observation.state.rotation": self._feature(["raw_0", "raw_1", "raw_2", "raw_3"])}
        )
        semantics = MachineSemantics(
            segments=(
                MachineSemanticSegment(
                    local_slice=(0, 4),
                    semantic_type="eef_rotation_euler",
                    side="left",
                    body_part="arm",
                    representation="quaternion_xyzw",
                    unit="rad",
                    declared_name_status="misleading",
                    standardizable="needs_transform",
                    required_transform="quaternion_to_euler",
                    confidence=0.95,
                    alternatives=(),
                    need_human_review=False,
                    reason="四元数需要转换为欧拉角。",
                ),
            ),
            need_human_review=False,
            reason="四元数需要转换为欧拉角。",
        )

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.rotation": 4}),
            vlm_resolver=_FixedMachineResolver(semantics),
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.rotation"]["names"],
            ["raw_0", "raw_1", "raw_2", "raw_3"],
        )
        self.assertTrue(
            {
                "QUATERNION_REQUIRES_EULER_CONVERSION",
                "DECLARED_NAME_CONFLICT",
            }.issubset({item.category for item in result.machine_review_items})
        )

    def test_does_not_reclassify_action_when_state_child_analysis_is_incomplete(self) -> None:
        source_info = self._info(
            {
                "action": self._feature(["raw_0", "raw_1", "raw_2"]),
                "observation.state": self._feature(["raw_0", "raw_1", "raw_2"]),
                "observation.state.auxiliary": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["auxiliary_group"],
                },
            }
        )
        unknown_semantics = self._unknown_semantics(3)
        profile = self._profile(
            {"action": 3, "observation.state": 3, "observation.state.auxiliary": 3},
            equal_action_state=True,
        )
        samples = profile.samples["action"]
        profile = ParquetProfile(
            profile.row_count,
            profile.row_group_count,
            profile.schema_columns,
            profile.columns,
            {
                "action": samples,
                "observation.state": samples.copy(),
                "observation.state.auxiliary": samples.copy(),
            },
        )
        resolver = _CountingMachineResolver(unknown_semantics)

        result = normalize_machine_fields(source_info, profile, vlm_resolver=resolver)

        self.assertEqual(resolver.call_count, 1)
        self.assertNotIn("ACTION_EQUALS_STATE", {item.category for item in result.machine_review_items})

    def test_limits_unknown_unit_review_to_affected_dimensions(self) -> None:
        names = [
            "left_arm_joint_0_rad",
            "left_wrist_pose_x",
            "left_arm_joint_1_rad",
        ]
        source_info = self._info({"action": self._feature(names)})

        result = normalize_machine_fields(
            source_info,
            self._profile({"action": 3}),
        )

        review = next(
            item
            for item in result.machine_review_items
            if item.category == "UNKNOWN_UNIT"
        )
        self.assertEqual(review.source_slice, (1, 2))
        self.assertEqual(review.declared_names, ("left_wrist_pose_x",))

    def test_concatenates_names_only_when_all_segments_are_safe(self) -> None:
        source_info = self._info(
            {
                "observation.state.arms": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": ["left_arm", "right_arm"],
                    "unit": "rad",
                }
            }
        )
        semantics = MachineSemantics(
            segments=(
                self._arm_segment((0, 3), "left"),
                self._arm_segment((3, 6), "right"),
            ),
            need_human_review=False,
            reason="左右臂分段明确。",
        )

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.arms": 6}),
            vlm_resolver=_FixedMachineResolver(semantics),
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.arms"]["names"],
            [
                "left_arm_joint_0_rad",
                "left_arm_joint_1_rad",
                "left_arm_joint_2_rad",
                "right_arm_joint_0_rad",
                "right_arm_joint_1_rad",
                "right_arm_joint_2_rad",
            ],
        )
        self.assertEqual(result.machine_review_items, ())

    def test_keeps_names_when_any_segment_needs_review(self) -> None:
        source_info = self._info(
            {
                "observation.state.arms": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": ["left_arm", "right_arm"],
                    "unit": "rad",
                }
            }
        )
        right = self._arm_segment((3, 6), "right")
        right = MachineSemanticSegment(
            **{
                **right.__dict__,
                "unit": "unknown",
                "standardizable": "review",
                "need_human_review": True,
            }
        )
        semantics = MachineSemantics(
            segments=(self._arm_segment((0, 3), "left"), right),
            need_human_review=True,
            reason="右臂单位未知。",
        )

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.arms": 6}),
            vlm_resolver=_FixedMachineResolver(semantics),
        )

        self.assertEqual(
            result.normalized_info["features"]["observation.state.arms"]["names"],
            ["left_arm", "right_arm"],
        )
        review = next(
            item
            for item in result.machine_review_items
            if item.source_feature == "observation.state.arms"
        )
        self.assertEqual(len(review.vlm_result["segments"]), 2)

    def test_records_sanitized_vlm_error_on_review_item(self) -> None:
        source_info = self._info(
            {"observation.state.raw": self._feature(["raw_0"])}
        )

        result = normalize_machine_fields(
            source_info,
            self._profile({"observation.state.raw": 1}),
            vlm_resolver=_FailingMachineResolver(),
        )

        review = next(
            item
            for item in result.machine_review_items
            if item.category == "VLM_RESOLUTION_FAILED"
        )
        vlm_error = getattr(review, "vlm_error", None)
        self.assertIsInstance(vlm_error, str)
        self.assertIn("segments[0].representation 不合法: pose_7d", vlm_error)
        self.assertNotIn("Authorization", vlm_error)
        self.assertNotIn("sk-secret", vlm_error)

    @staticmethod
    def _feature(names: list[str], shape: int | None = None) -> dict[str, object]:
        return {"dtype": "float32", "shape": [shape or len(names)], "names": names}

    @staticmethod
    def _info(features: dict[str, object]) -> dict[str, object]:
        return {"robot_type": "test_robot", "features": features}

    @staticmethod
    def _head_quaternion_semantics() -> MachineSemantics:
        return MachineSemantics(
            segments=(
                MachineSemanticSegment(
                    local_slice=(0, 4),
                    semantic_type="head_orientation_quaternion",
                    side="none",
                    body_part="head",
                    representation="quaternion_xyzw",
                    unit="unknown",
                    declared_name_status="partially_correct",
                    standardizable="direct",
                    required_transform="none",
                    confidence=0.94,
                    alternatives=(),
                    need_human_review=False,
                    reason="四元数维度与数值特征一致。",
                ),
            ),
            need_human_review=False,
            reason="四元数维度与数值特征一致。",
        )

    @staticmethod
    def _unknown_semantics(vector_length: int) -> MachineSemantics:
        return MachineSemantics(
            segments=(
                MachineSemanticSegment(
                    local_slice=(0, vector_length),
                    semantic_type="unknown",
                    side="unknown",
                    body_part="unknown",
                    representation="unknown",
                    unit="unknown",
                    declared_name_status="unknown",
                    standardizable="review",
                    required_transform="none",
                    confidence=0.5,
                    alternatives=(),
                    need_human_review=True,
                    reason="证据不足。",
                ),
            ),
            need_human_review=True,
            reason="证据不足。",
        )

    @staticmethod
    def _arm_segment(
        local_slice: tuple[int, int], side: str
    ) -> MachineSemanticSegment:
        return MachineSemanticSegment(
            local_slice=local_slice,
            semantic_type="arm_joint",
            side=side,
            body_part="arm",
            representation="joint_vector",
            unit="rad",
            declared_name_status="partially_correct",
            standardizable="direct",
            required_transform="none",
            confidence=0.96,
            alternatives=(),
            need_human_review=False,
            reason="分段和单位由字段证据确认。",
        )

    @staticmethod
    def _profile(lengths: dict[str, int], equal_action_state: bool = False) -> ParquetProfile:
        samples: dict[str, np.ndarray] = {}
        columns: dict[str, VectorProfile] = {}
        for index, (name, length) in enumerate(lengths.items()):
            values = np.arange(length * 2, dtype=np.float64).reshape(2, length) + index
            if equal_action_state and name == "observation.state":
                values = samples["action"].copy()
            samples[name] = values
            columns[name] = VectorProfile(name, length, 0.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.1, 0.0, 0.0)
        return ParquetProfile(2, 1, tuple(lengths), columns, samples)


if __name__ == "__main__":
    unittest.main()
