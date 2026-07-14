"""机器人相机拓扑联网查询与严格解析测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.topology import (
    OpenAICompatibleRobotCameraTopologyResolver,
    RobotCameraTopologyValidationError,
    TOPOLOGY_SYSTEM_PROMPT,
    parse_robot_camera_topology,
)


class _TopologyClient:
    """记录联网拓扑请求次数的最小客户端。"""

    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "qwen3.7-plus"
    last_error = None
    last_error_code = None
    last_error_evidence: dict[str, object] = {}

    def __init__(self, payload: dict[str, object] | None) -> None:
        self.payload = payload
        self.call_count = 0

    def request_web_json(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, object] | None:
        self.call_count += 1
        return self.payload


def _airbot_payload() -> dict[str, object]:
    return {
        "robot_id": "airbot_mmk2",
        "camera_mounts": [
            {
                "mount_type": "on_robot",
                "direction_tokens": [],
                "body_part": "head",
            },
            {
                "mount_type": "on_robot",
                "direction_tokens": ["left"],
                "body_part": "wrist",
            },
            {
                "mount_type": "on_robot",
                "direction_tokens": ["right"],
                "body_part": "wrist",
            },
        ],
        "confidence": 0.95,
        "ambiguous": False,
    }


class CameraTopologyTest(unittest.TestCase):
    """验证联网结果只能形成受约束的机器人本体相机槽位。"""

    def test_parses_valid_robot_camera_topology(self) -> None:
        topology = parse_robot_camera_topology(_airbot_payload())

        self.assertEqual(topology.robot_id, "airbot_mmk2")
        self.assertEqual(len(topology.camera_mounts), 3)
        self.assertEqual(topology.camera_mounts[0].body_part, "head")
        self.assertEqual(
            topology.camera_mounts[1].direction_tokens, ("left",)
        )
        self.assertEqual(topology.confidence, 0.95)
        self.assertFalse(topology.ambiguous)

    def test_rejects_final_names_at_the_top_level(self) -> None:
        with self.assertRaises(RobotCameraTopologyValidationError) as caught:
            parse_robot_camera_topology(
                {**_airbot_payload(), "target_key": "observation.images.cam_head_rgb"}
            )
        self.assertEqual(caught.exception.field, "target_key")

    def test_accepts_platform_external_cameras_from_the_robot_configuration(self) -> None:
        external = _airbot_payload()
        external["robot_id"] = "agilex_cobot_magic"
        external["camera_mounts"] = [
            {
                "mount_type": "external",
                "direction_tokens": ["top"],
                "body_part": None,
            }
        ]

        topology = parse_robot_camera_topology(external)

        self.assertEqual(topology.camera_mounts[0].mount_type, "external")
        self.assertEqual(topology.camera_mounts[0].direction_tokens, ("top",))
        self.assertFalse(topology.partial)
        self.assertIn("标准平台配置中的固定外部相机", TOPOLOGY_SYSTEM_PROMPT)

    def test_keeps_valid_mounts_and_records_rejected_mounts(self) -> None:
        invalid = _airbot_payload()
        invalid["camera_mounts"].extend(
            [
            {
                "mount_type": "on_robot",
                "direction_tokens": ["top"],
                "body_part": None,
            },
            {
                "mount_type": "on_robot",
                "direction_tokens": [],
                "body_part": "head",
                "target_key": "observation.images.cam_head_rgb",
            },
            ]
        )

        topology = parse_robot_camera_topology(invalid)

        self.assertEqual(len(topology.camera_mounts), 3)
        self.assertTrue(topology.partial)
        self.assertEqual(len(topology.rejected_mounts), 2)
        self.assertEqual(topology.rejected_mounts[0].field, "camera_mounts[3]")
        self.assertEqual(
            topology.rejected_mounts[0].value,
            {
                "mount_type": "on_robot",
                "direction_tokens": ["top"],
                "body_part": None,
            },
        )
        self.assertEqual(
            topology.rejected_mounts[1].field,
            "camera_mounts[4].target_key",
        )

    def test_resolver_exposes_partial_topology_evidence(self) -> None:
        payload = _airbot_payload()
        payload["camera_mounts"].append(
            {
                "mount_type": "on_robot",
                "direction_tokens": ["top"],
                "body_part": None,
            }
        )
        resolver = OpenAICompatibleRobotCameraTopologyResolver(
            _TopologyClient(payload)
        )

        topology = resolver.resolve("airbot_mmk2")

        self.assertIsNotNone(topology)
        self.assertTrue(topology.partial)
        self.assertEqual(resolver.last_error_code, "ROBOT_TOPOLOGY_PARTIAL")
        self.assertEqual(
            resolver.last_error_evidence["rejected_mounts"][0]["field"],
            "camera_mounts[3]",
        )

    def test_resolver_queries_each_robot_only_once(self) -> None:
        client = _TopologyClient(_airbot_payload())
        resolver = OpenAICompatibleRobotCameraTopologyResolver(client)

        first = resolver.resolve("airbot_mmk2")
        second = resolver.resolve("airbot_mmk2")

        self.assertIs(first, second)
        self.assertEqual(client.call_count, 1)

    def test_resolver_caches_transport_failure(self) -> None:
        client = _TopologyClient(None)
        client.last_error = "network unavailable"
        client.last_error_code = "VLM_UNAVAILABLE"
        resolver = OpenAICompatibleRobotCameraTopologyResolver(client)

        self.assertIsNone(resolver.resolve("airbot_mmk2"))
        self.assertIsNone(resolver.resolve("airbot_mmk2"))

        self.assertEqual(client.call_count, 1)
        self.assertEqual(resolver.last_error_code, "VLM_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
