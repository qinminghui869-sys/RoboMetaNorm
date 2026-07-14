"""P0 JSON 原子输出测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.domain.models import (
    DatasetCandidate,
    DatasetStatus,
    LayoutType,
    PreconditionReport,
)
from robometanorm.camera.models import CameraReviewCandidate, CameraReviewItem
from robometanorm.machine.models import (
    GripperTransformProposal,
    MachineReviewItem,
)
from robometanorm.robot_identity import RobotIdentity, RobotIdentityEvidence
from robometanorm.writers.json_writer import write_normalization_files


class JsonWriterTest(unittest.TestCase):
    """验证规范建议与复核文件的 P0 契约。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset_001"
        (self.dataset_path / "meta").mkdir(parents=True)
        (self.dataset_path / "data").mkdir()
        self.info = {
            "fps": 20,
            "features": {"action": {"dtype": "float32", "shape": [2]}},
            "extension": {"values": [1, 2, 3]},
        }
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.info_path.write_text(json.dumps(self.info), encoding="utf-8")
        self.candidate = DatasetCandidate(
            dataset_name="dataset_001",
            task_name=None,
            source_path=self.dataset_path,
            layout_type=LayoutType.FLAT,
            info_path=self.info_path,
            data_path=self.dataset_path / "data",
            video_path=None,
            depth_path=None,
        )
        self.report = PreconditionReport(
            status=DatasetStatus.PASS,
            review_items=(),
            camera_count=0,
            machine_field_count=2,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_writes_unchanged_info_and_review_without_temp_files(self) -> None:
        write_normalization_files(self.candidate, self.info, self.report)

        normalized = json.loads(
            (self.dataset_path / "meta" / "info_norm.json").read_text(encoding="utf-8")
        )
        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(normalized, self.info)
        self.assertEqual(review["status"], "PASS")
        self.assertFalse(review["review_required"])
        self.assertEqual(review["review_items"], [])
        self.assertEqual(review["generator"]["phase"], "P0")
        self.assertEqual(list((self.dataset_path / "meta").glob(".*.tmp")), [])

    def test_writes_p1_camera_review_items_and_promotes_status_to_review(self) -> None:
        camera_review = CameraReviewItem(
            source_key="observation.images.camera_1",
            reason_code="UNKNOWN_CAMERA_NAME",
            candidates=(
                CameraReviewCandidate("observation.images.cam_front_rgb", 0.81),
            ),
            evidence={"sample_policy": "first_stage"},
        )

        write_normalization_files(
            self.candidate,
            self.info,
            self.report,
            camera_review_items=(camera_review,),
            phase="P1",
        )

        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(review["generator"]["phase"], "P1")
        self.assertEqual(review["status"], "REVIEW")
        self.assertTrue(review["review_required"])
        self.assertEqual(review["camera_review_items"][0]["source_key"], camera_review.source_key)
        self.assertEqual(
            review["camera_review_items"][0]["candidates"][0]["target_key"],
            "observation.images.cam_front_rgb",
        )
        self.assertEqual(
            review["camera_review_items"][0]["human_decision"],
            {"status": "pending", "selected_target_key": None},
        )

    def test_writes_traceable_robot_identity_summary(self) -> None:
        identity = RobotIdentity(
            canonical_id="airbot_mmk2",
            selected_source="info.robot_type",
            selected_value="Airbot_MMK2",
            evidence=(
                RobotIdentityEvidence(
                    "info.robot_type", "Airbot_MMK2", "airbot_mmk2"
                ),
            ),
        )

        write_normalization_files(
            self.candidate,
            self.info,
            self.report,
            robot_identity=identity,
        )

        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(review["robot_identity"]["canonical_id"], "airbot_mmk2")
        self.assertEqual(
            review["robot_identity"]["evidence"][0]["source"],
            "info.robot_type",
        )

    def test_writes_p2_machine_review_items_and_promotes_status_to_review(self) -> None:
        machine_review = MachineReviewItem(
            source_feature="observation.state.arm",
            source_slice=(7, 10),
            category="UNKNOWN_UNIT",
            severity="confirmation",
            declared_names=("left_arm_joint_0",),
            vlm_result={"semantic_type": "arm_joint", "confidence": 0.8},
            candidates=("left_arm_joint_0_rad",),
            required_action="确认物理单位后再添加 _rad。",
            vlm_error="segments[0].unit 不合法",
        )

        write_normalization_files(
            self.candidate,
            self.info,
            self.report,
            machine_review_items=(machine_review,),
            phase="P2",
        )

        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(review["generator"]["phase"], "P2")
        self.assertEqual(review["status"], "REVIEW")
        self.assertTrue(review["review_required"])
        self.assertEqual(
            review["machine_review_items"][0]["source_slice"], [7, 10]
        )
        self.assertEqual(
            review["machine_review_items"][0]["human_decision"],
            {
                "status": "pending",
                "selected_semantic": None,
                "comment": None,
            },
        )
        self.assertEqual(
            review["machine_review_items"][0]["vlm_error"],
            "segments[0].unit 不合法",
        )

    def test_writes_gripper_transform_proposal_without_forcing_review(self) -> None:
        proposal = GripperTransformProposal(
            source_feature="action",
            source_index=7,
            source_name="leader_left_gripper_degree_mm.pos",
            target_name="left_gripper_open",
            source_closed=0.0,
            source_open=100.0,
            target_range=(0.0, 1.0),
            formula="clip(x / 100, 0, 1)",
            clipping_policy="clip_to_unit_interval",
            direction_evidence="declared_name",
            range_evidence="parquet_percentiles",
            confidence=0.96,
            transform_required=True,
            observed_profile={"p01": 0.0, "p99": 100.0},
        )

        write_normalization_files(
            self.candidate,
            self.info,
            self.report,
            gripper_transform_proposals=(proposal,),
            phase="P2",
        )

        review = json.loads(
            (self.dataset_path / "meta" / "info_norm_review.json").read_text(
                encoding="utf-8"
            )
        )
        serialized = review["gripper_transform_proposals"][0]
        self.assertEqual(serialized["target_name"], "left_gripper_open")
        self.assertEqual(serialized["target_range"], [0.0, 1.0])
        self.assertEqual(serialized["formula"], "clip(x / 100, 0, 1)")
        self.assertFalse(review["review_required"])


if __name__ == "__main__":
    unittest.main()
