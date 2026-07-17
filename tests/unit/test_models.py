"""Root domain model contract tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.models import (
    DatasetAnalysis,
    DatasetCandidate,
    DatasetStatus,
    Issue,
    LayoutType,
)
from tests.mini_fixtures import (
    DatasetFixture,
    FakeVlm,
    PipelineFixture,
    StubTransport,
    VlmFixture,
)


class DatasetStatusTest(unittest.TestCase):
    """Verify the stable status values exposed to serialized output."""

    def test_exposes_exactly_the_four_uppercase_status_values(self) -> None:
        self.assertEqual(
            tuple(status.value for status in DatasetStatus),
            ("PASS", "REVIEW", "BLOCKED", "ERROR"),
        )


class IssueTest(unittest.TestCase):
    """Verify issue evidence remains directly JSON serializable."""

    def test_evidence_is_json_safe(self) -> None:
        issue = Issue(
            code="ACME_TEST_WARNING",
            message="Fictional fixture warning",
            scope="dataset",
            evidence={"frame": 3, "valid": True, "labels": ["left", "wrist"]},
        )

        self.assertEqual(
            json.loads(json.dumps(asdict(issue))),
            {
                "code": "ACME_TEST_WARNING",
                "message": "Fictional fixture warning",
                "scope": "dataset",
                "evidence": {
                    "frame": 3,
                    "valid": True,
                    "labels": ["left", "wrist"],
                },
                "severity": "review",
            },
        )

    def test_default_evidence_is_not_shared(self) -> None:
        first = Issue(code="FIRST", message="first", scope="dataset")
        second = Issue(code="SECOND", message="second", scope="dataset")

        self.assertEqual(first.evidence, {})
        self.assertIsNot(first.evidence, second.evidence)


class DatasetCandidateTest(unittest.TestCase):
    """Verify discovery can describe both supported dataset layouts."""

    def test_preserves_task_grouped_layout_contract(self) -> None:
        source_path = Path("/tmp/acme_pick/pick_001")
        candidate = DatasetCandidate(
            dataset_name="pick_001",
            task_name="acme_pick",
            source_path=source_path,
            layout_type=LayoutType.TASK_GROUPED,
            info_path=source_path / "meta" / "info.json",
            data_path=source_path / "data",
            video_path=source_path / "videos",
            depth_path=None,
        )

        self.assertEqual(candidate.dataset_name, "pick_001")
        self.assertEqual(candidate.task_name, "acme_pick")
        self.assertEqual(candidate.layout_type, LayoutType.TASK_GROUPED)
        self.assertEqual(candidate.info_path, source_path / "meta" / "info.json")
        self.assertEqual(candidate.data_path, source_path / "data")
        self.assertEqual(candidate.video_path, source_path / "videos")
        self.assertIsNone(candidate.depth_path)


class MiniFixtureTest(unittest.TestCase):
    """Verify reusable mini fixtures stay deterministic and protocol-complete."""

    def test_dataset_analysis_groups_profile_and_mapping(self) -> None:
        fixture = PipelineFixture()
        profile = fixture.hardware_profile()
        mapping = fixture.dataset_mapping()

        analysis = DatasetAnalysis(profile=profile, mapping=mapping)

        self.assertIs(analysis.profile, profile)
        self.assertIs(analysis.mapping, mapping)
        with self.assertRaises(FrozenInstanceError):
            analysis.mapping = mapping  # type: ignore[misc]

    def test_dataset_fixture_isolates_source_info_from_later_caller_changes(self) -> None:
        source_info = DatasetFixture.default_info()
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = DatasetFixture.create(Path(temp_dir), info=source_info)

            features = source_info["features"]
            self.assertIsInstance(features, dict)
            features["observation.state"]["shape"][0] = 999

            on_disk = json.loads(fixture.candidate.info_path.read_text(encoding="utf-8"))

        self.assertEqual(fixture.info, on_disk)
        self.assertNotEqual(fixture.info, source_info)

    def test_pipeline_fixture_uses_canonical_acme_profile_values(self) -> None:
        profile = PipelineFixture().hardware_profile()

        self.assertEqual(profile.cameras[0].direction_tokens, ("front",))
        arm = profile.components[0]
        self.assertEqual(arm.kind, "arm_joint")
        self.assertEqual(arm.side, "left")
        self.assertEqual(arm.representation, "joint_vector")
        self.assertEqual(arm.unit, "rad")
        self.assertIsNone(arm.open_range)
        self.assertIsNone(arm.open_direction)
        self.assertEqual(
            tuple(item.local_source for item in profile.identity.assessments),
            ("info_robot_type", "common_record", "tasks"),
        )

    def test_vlm_fixture_payloads_match_every_model_field(self) -> None:
        hardware = VlmFixture.valid_hardware_payload()
        mapping = VlmFixture.valid_mapping_payload()

        self.assertEqual(set(hardware), {"identity", "sources", "cameras", "components"})
        self.assertEqual(
            set(hardware["identity"]),
            {
                "manufacturer",
                "model",
                "confidence",
                "ambiguous",
                "reason",
                "local_evidence_status",
                "source_ids",
                "assessments",
            },
        )
        self.assertEqual(len(hardware["identity"]["assessments"]), 3)
        self.assertEqual(
            set(hardware["identity"]["assessments"][0]),
            {"local_source", "relation", "explanation"},
        )
        self.assertEqual(
            set(hardware["sources"][0]), {"source_id", "title", "url", "kind"}
        )
        self.assertEqual(
            set(hardware["cameras"][0]),
            {
                "camera_id",
                "interface_name",
                "mount_type",
                "direction_tokens",
                "body_part",
                "modality",
                "confidence",
                "ambiguous",
                "reason",
                "source_ids",
            },
        )
        self.assertEqual(
            set(hardware["components"][0]),
            {
                "component_id",
                "kind",
                "side",
                "count",
                "element_order",
                "representation",
                "unit",
                "open_range",
                "open_direction",
                "confidence",
                "ambiguous",
                "reason",
                "source_ids",
            },
        )
        self.assertEqual(set(mapping), {"cameras", "machines"})
        self.assertEqual(
            set(mapping["cameras"][0]),
            {"source_key", "camera_id", "confidence", "ambiguous", "reason"},
        )
        self.assertEqual(
            set(mapping["machines"][0]),
            {"source_feature", "slices", "confidence", "ambiguous", "reason"},
        )
        self.assertEqual(
            set(mapping["machines"][0]["slices"][0]),
            {"start", "end", "component_id", "element_order"},
        )
        json.dumps(hardware)
        json.dumps(mapping)

    def test_fake_vlm_uses_frozen_results_and_records_calls(self) -> None:
        evidence = object()
        profile = PipelineFixture().hardware_profile()
        mapping = PipelineFixture().dataset_mapping()
        analysis = DatasetAnalysis(profile, mapping)
        fake = FakeVlm(analysis_result=(analysis, None))

        self.assertEqual(fake.analysis_calls, 0)
        self.assertEqual(fake.analyze_dataset(evidence, "acme_testbot"), (analysis, None))
        self.assertEqual(fake.analysis_calls, 1)
        self.assertEqual(fake.analysis_robot_types, ["acme_testbot"])

        missing = FakeVlm()
        missing_analysis, issue = missing.analyze_dataset(evidence, "acme_testbot")
        self.assertIsNone(missing_analysis)
        self.assertEqual(issue.code, "VLM_CONFIG_MISSING")
        self.assertEqual(issue.severity, "review")

    def test_stub_transport_returns_frozen_results_and_records_attempts(self) -> None:
        web_payload = {"sources": [{"title": "Acme Test Manual"}]}
        chat_payload = {"camera_id": "wrist_rgb"}
        transport = StubTransport(
            web_payload=web_payload,
            web_issue=None,
            chat_payload=chat_payload,
            chat_issue=None,
        )
        image_paths = (Path("/tmp/acme-frame.png"),)

        self.assertEqual(transport.web_attempts, 0)
        self.assertEqual(transport.chat_attempts, 0)
        self.assertEqual(
            transport.request_web_json("web system", "web user"),
            (web_payload, None),
        )
        self.assertEqual(
            transport.request_json("chat system", "chat user", image_paths),
            (chat_payload, None),
        )
        self.assertEqual(transport.web_attempts, 1)
        self.assertEqual(transport.chat_attempts, 1)


if __name__ == "__main__":
    unittest.main()
