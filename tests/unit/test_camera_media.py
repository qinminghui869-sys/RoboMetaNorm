"""相机媒体处理和 VLM 语义接口测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.frame_sampler import first_stage_ratios, second_stage_ratios
from robometanorm.camera.media_probe import probe_media
from robometanorm.camera.prompt_builder import build_vlm_prompt
from robometanorm.camera.vlm_classifier import parse_vlm_semantics


class CameraMediaTest(unittest.TestCase):
    """验证 P1 媒体与 VLM 的确定性边界。"""

    def test_uses_specified_two_stage_sampling_ratios(self) -> None:
        self.assertEqual(first_stage_ratios(), (0.1, 0.5, 0.9))
        self.assertEqual(second_stage_ratios(1), (0.1, 0.3, 0.5, 0.7, 0.9))
        self.assertEqual(second_stage_ratios(3), (0.2, 0.5, 0.8))

    def test_reads_media_metadata_from_ffprobe_json(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "30000/1001",
                    "nb_frames": "90",
                    "pix_fmt": "yuv420p",
                }
            ],
            "format": {"duration": "3.003"},
        }
        completed = CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")

        with patch("robometanorm.camera.media_probe.subprocess.run", return_value=completed):
            media = probe_media(Path("episode.mp4"))

        self.assertEqual(media.codec, "h264")
        self.assertEqual(media.width, 640)
        self.assertAlmostEqual(media.fps, 30000 / 1001)
        self.assertEqual(media.frame_count, 90)

    def test_vlm_prompt_and_response_are_semantic_only(self) -> None:
        system_prompt, user_prompt = build_vlm_prompt(
            dataset_name="dataset_001",
            robot_type="aloha",
            source_key="observation.images.camera_1",
            feature={"dtype": "video", "shape": [480, 640, 3]},
            declared_fps=30,
            media=None,
            other_camera_keys=("observation.images.image_left",),
        )
        semantics = parse_vlm_semantics(
            {
                "modality": "rgb",
                "mount_type": "body",
                "direction_tokens": ["left"],
                "body_part": "wrist",
                "is_primary": False,
                "confidence": 0.94,
                "ambiguous": False,
                "alternatives": [],
                "need_human_review": False,
            }
        )

        self.assertNotIn("target_key", system_prompt + user_prompt)
        self.assertEqual(semantics.body_part, "wrist")
        with self.assertRaises(ValueError):
            parse_vlm_semantics({"target_key": "observation.images.cam_left_rgb"})


if __name__ == "__main__":
    unittest.main()
