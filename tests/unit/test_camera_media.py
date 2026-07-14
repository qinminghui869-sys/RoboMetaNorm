"""相机媒体处理和 VLM 语义接口测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.camera.media import (
    extract_rgb_frame_at,
    find_camera_media,
    first_stage_ratios,
    probe_media,
    second_stage_ratios,
)
from robometanorm.camera.normalizer import _select_second_stage_episodes
from robometanorm.camera.vlm import (
    CameraSemanticsValidationError,
    build_vlm_prompt,
    parse_vlm_semantics,
)
from robometanorm.domain.models import DatasetCandidate, LayoutType


class CameraMediaTest(unittest.TestCase):
    """验证 P1 媒体与 VLM 的确定性边界。"""

    def test_uses_specified_two_stage_sampling_ratios(self) -> None:
        self.assertEqual(first_stage_ratios(), (0.1, 0.5, 0.9))
        self.assertEqual(second_stage_ratios(1), (0.1, 0.3, 0.5, 0.7, 0.9))
        self.assertEqual(second_stage_ratios(3), (0.2, 0.5, 0.8))

    def test_discovers_only_first_and_last_camera_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory)
            media_directory = (
                dataset_path / "videos" / "observation.images.camera_1"
            )
            media_directory.mkdir(parents=True)
            episodes = tuple(
                media_directory / f"episode_{index:06d}.mp4"
                for index in range(4)
            )
            for episode in episodes:
                episode.touch()
            candidate = DatasetCandidate(
                dataset_name="dataset",
                task_name=None,
                source_path=dataset_path,
                layout_type=LayoutType.FLAT,
                info_path=dataset_path / "meta" / "info.json",
                data_path=None,
                video_path=dataset_path / "videos",
                depth_path=None,
            )

            self.assertEqual(
                find_camera_media(candidate, "observation.images.camera_1"),
                (episodes[0], episodes[-1]),
            )

    def test_matches_only_safe_left_right_media_key_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory)
            media_directory = (
                dataset_path / "videos" / "observation.images.image_left"
            )
            media_directory.mkdir(parents=True)
            episodes = tuple(
                media_directory / f"episode_{index:06d}.mp4"
                for index in range(3)
            )
            for episode in episodes:
                episode.touch()
            candidate = DatasetCandidate(
                dataset_name="dataset",
                task_name=None,
                source_path=dataset_path,
                layout_type=LayoutType.FLAT,
                info_path=dataset_path / "meta" / "info.json",
                data_path=None,
                video_path=dataset_path / "videos",
                depth_path=None,
            )

            self.assertEqual(
                find_camera_media(candidate, "observation.images.left_image"),
                (episodes[0], episodes[-1]),
            )

    def test_second_stage_selects_only_first_and_last_camera_episode(self) -> None:
        episodes = tuple(
            Path(f"episode_{index:06d}.mp4") for index in range(4)
        )

        self.assertEqual(
            _select_second_stage_episodes(episodes),
            (episodes[0], episodes[-1]),
        )

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

        with patch("robometanorm.camera.media.subprocess.run", return_value=completed):
            media = probe_media(Path("episode.mp4"))

        self.assertEqual(media.codec, "h264")
        self.assertEqual(media.width, 640)
        self.assertAlmostEqual(media.fps, 30000 / 1001)
        self.assertEqual(media.frame_count, 90)

    def test_extracts_one_frame_at_an_exact_episode_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "sample.jpg"

            def create_frame(command: list[str], **_: object) -> CompletedProcess:
                Path(command[-1]).touch()
                return CompletedProcess(command, 0, stdout="", stderr="")

            with patch(
                "robometanorm.camera.media.subprocess.run",
                side_effect=create_frame,
            ) as run:
                result = extract_rgb_frame_at(
                    Path("episode_000001.mp4"), 1.25, target
                )

        self.assertEqual(result, target)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "1.250000")
        self.assertEqual(command[command.index("-frames:v") + 1], "1")

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
                "mount_type": "on_robot",
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
        self.assertIn("direction_tokens 仅允许", system_prompt)
        self.assertIn("未知时返回空数组 []", system_prompt)
        self.assertIn("body_part 仅允许", system_prompt)
        self.assertIn("未知时返回 null", system_prompt)
        self.assertEqual(semantics.body_part, "wrist")
        with self.assertRaises(ValueError):
            parse_vlm_semantics({"target_key": "observation.images.cam_left_rgb"})

    def test_normalizes_safe_unknown_sentinels_but_rejects_unknown_tokens(self) -> None:
        payload = {
            "modality": "rgb",
            "mount_type": "external",
            "direction_tokens": ["unknown"],
            "body_part": "unknown",
            "is_primary": False,
            "confidence": 0.61,
            "ambiguous": True,
            "alternatives": [],
            "need_human_review": True,
        }

        semantics = parse_vlm_semantics(payload)

        self.assertEqual(semantics.direction_tokens, ())
        self.assertIsNone(semantics.body_part)
        invalid_payload = {**payload, "direction_tokens": ["high"]}
        with self.assertRaises(CameraSemanticsValidationError) as caught:
            parse_vlm_semantics(invalid_payload)
        self.assertEqual(caught.exception.field, "direction_tokens")
        self.assertEqual(caught.exception.value, ["high"])
        with self.assertRaises(CameraSemanticsValidationError) as caught:
            parse_vlm_semantics({**payload, "modality": ["rgb"]})
        self.assertEqual(caught.exception.field, "modality")

    def test_vlm_mount_type_uses_the_embedded_camera_standard(self) -> None:
        payload = {
            "modality": "rgb",
            "mount_type": "external",
            "direction_tokens": ["left"],
            "body_part": None,
            "is_primary": False,
            "confidence": 0.94,
            "ambiguous": False,
            "alternatives": [],
            "need_human_review": False,
        }

        semantics = parse_vlm_semantics(payload)

        self.assertEqual(semantics.mount_type, "external")
        with self.assertRaises(CameraSemanticsValidationError) as caught:
            parse_vlm_semantics({**payload, "mount_type": "fixed"})
        self.assertEqual(caught.exception.field, "mount_type")
        with self.assertRaises(CameraSemanticsValidationError) as caught:
            parse_vlm_semantics(
                {
                    **payload,
                    "mount_type": "external",
                    "body_part": "wrist",
                }
            )
        self.assertEqual(caught.exception.field, "body_part")
        with self.assertRaises(CameraSemanticsValidationError) as caught:
            parse_vlm_semantics(
                {
                    **payload,
                    "mount_type": "on_robot",
                    "direction_tokens": ["env"],
                    "body_part": "head",
                }
            )
        self.assertEqual(caught.exception.field, "direction_tokens")


if __name__ == "__main__":
    unittest.main()
