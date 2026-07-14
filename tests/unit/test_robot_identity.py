"""机器人身份强证据解析测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.robot_identity import resolve_robot_identity


class RobotIdentityTest(unittest.TestCase):
    """验证身份来源优先级、兼容细化和冲突审计。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.meta_path = Path(self.temporary_directory.name) / "meta"
        self.meta_path.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_prefers_info_robot_type_and_preserves_all_evidence(self) -> None:
        self._write_json(
            "common_record.json", {"machine_id": "AkOrnESsjcE_galaxea"}
        )
        self._write_tasks("Galaxea_R1_Lite organize clothes")

        identity = resolve_robot_identity(
            self.meta_path, {"robot_type": "Airbot_MMK2"}
        )

        self.assertEqual(identity.canonical_id, "airbot_mmk2")
        self.assertEqual(identity.selected_source, "info.robot_type")
        self.assertEqual(identity.selected_value, "Airbot_MMK2")
        self.assertEqual(len(identity.evidence), 3)
        self.assertEqual(len(identity.conflicts), 2)

    def test_reads_root_type_as_compatible_info_alias(self) -> None:
        identity = resolve_robot_identity(
            self.meta_path, {"root_type": "Agilex_Cobot_Magic"}
        )

        self.assertEqual(identity.canonical_id, "agilex_cobot_magic")
        self.assertEqual(identity.selected_source, "info.root_type")
        self.assertEqual(identity.conflicts, ())

    def test_refines_generic_machine_family_with_compatible_task_model(self) -> None:
        self._write_json(
            "common_record.json", {"machine_id": "AkOrnESsjcE_galaxea"}
        )
        self._write_tasks("Galaxea_R1_Lite organize clothes")

        identity = resolve_robot_identity(self.meta_path, {})

        self.assertEqual(identity.canonical_id, "galaxea_r1_lite")
        self.assertEqual(identity.selected_source, "common_record.machine_id")
        self.assertEqual(identity.selected_value, "AkOrnESsjcE_galaxea")
        self.assertEqual(identity.conflicts, ())

    def test_extracts_unlisted_robot_suffix_from_machine_id(self) -> None:
        self._write_json(
            "common_record.json", {"machine_id": "CFvFyRtw8T1_franka"}
        )

        identity = resolve_robot_identity(self.meta_path, {})

        self.assertEqual(identity.canonical_id, "franka")
        self.assertEqual(identity.selected_source, "common_record.machine_id")

    def test_does_not_treat_bare_machine_serial_as_robot_identity(self) -> None:
        self._write_json(
            "common_record.json", {"machine_id": "CFvFyRtw8T1"}
        )

        identity = resolve_robot_identity(self.meta_path, {})

        self.assertIsNone(identity.canonical_id)
        self.assertEqual(identity.evidence, ())

    def test_ignores_info_placeholders_and_uses_valid_machine_id(self) -> None:
        self._write_json(
            "common_record.json", {"machine_id": "sample_galbot_g1"}
        )

        identity = resolve_robot_identity(
            self.meta_path, {"robot_type": "unknown", "root_type": "N/A"}
        )

        self.assertEqual(identity.canonical_id, "galbot_g1")
        self.assertEqual(identity.selected_source, "common_record.machine_id")
        self.assertEqual(identity.conflicts, ())

    def test_ignores_placeholder_root_type_beside_valid_robot_type(self) -> None:
        identity = resolve_robot_identity(
            self.meta_path,
            {"robot_type": "Airbot_MMK2", "root_type": "null"},
        )

        self.assertEqual(identity.canonical_id, "airbot_mmk2")
        self.assertEqual(len(identity.evidence), 1)
        self.assertEqual(identity.conflicts, ())

    def test_does_not_infer_model_from_natural_language_task(self) -> None:
        self._write_tasks(
            "take the items off the building blocks with both hands"
        )

        identity = resolve_robot_identity(self.meta_path, {})

        self.assertIsNone(identity.canonical_id)
        self.assertIsNone(identity.selected_source)
        self.assertEqual(identity.evidence, ())

    def test_does_not_treat_aloha_greeting_as_robot_model(self) -> None:
        self._write_tasks("say aloha to the guests")

        identity = resolve_robot_identity(self.meta_path, {})

        self.assertIsNone(identity.canonical_id)
        self.assertEqual(identity.evidence, ())

    def test_skips_malformed_task_lines_and_uses_valid_model_hint(self) -> None:
        tasks_path = self.meta_path / "tasks.jsonl"
        tasks_path.write_text(
            "not-json\n" + json.dumps({"task": "Galbot_G1 pull curtains"}) + "\n",
            encoding="utf-8",
        )

        identity = resolve_robot_identity(self.meta_path, {})

        self.assertEqual(identity.canonical_id, "galbot_g1")
        self.assertEqual(identity.selected_source, "tasks.model_hint")

    def test_missing_optional_files_produces_unknown_identity(self) -> None:
        identity = resolve_robot_identity(self.meta_path, {})

        self.assertIsNone(identity.canonical_id)
        self.assertEqual(identity.evidence, ())
        self.assertEqual(identity.conflicts, ())

    def _write_json(self, filename: str, payload: object) -> None:
        (self.meta_path / filename).write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _write_tasks(self, task: str) -> None:
        (self.meta_path / "tasks.jsonl").write_text(
            json.dumps({"task": task}) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
