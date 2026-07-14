"""夹爪同步视频方向判定编排测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.application.pipeline import (
    _match_episode_media,
    _resolve_gripper_directions,
    _select_gripper_camera_key,
    _unresolved_grippers,
)
from robometanorm.domain.models import DatasetCandidate, LayoutType
from robometanorm.machine.models import GripperDirectionEvidence
from robometanorm.machine.profiling import profile_parquets


class _DirectionResolver:
    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, object], tuple[Path, ...]]] = []

    def resolve(
        self, evidence: dict[str, object], image_paths: tuple[Path, ...]
    ) -> GripperDirectionEvidence:
        self.requests.append((evidence, image_paths))
        return GripperDirectionEvidence(
            "increasing_is_open", 0.96, "synchronized_video", evidence
        )


class GripperPipelineTest(unittest.TestCase):
    def test_uses_two_representative_episodes_and_same_side_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "dataset"
            data = dataset / "data"
            videos = dataset / "videos" / "observation.images.right_wrist"
            meta = dataset / "meta"
            data.mkdir(parents=True)
            videos.mkdir(parents=True)
            meta.mkdir()
            parquet_paths = (
                data / "episode_000000.parquet",
                data / "episode_000009.parquet",
            )
            pq.write_table(
                pa.table({"timestamp": [0.0, 0.5], "action": [[0.0], [0.2]]}),
                parquet_paths[0],
            )
            pq.write_table(
                pa.table({"timestamp": [0.25, 0.75], "action": [[0.8], [1.0]]}),
                parquet_paths[1],
            )
            for episode in ("episode_000000.mp4", "episode_000009.mp4"):
                (videos / episode).touch()
            candidate = DatasetCandidate(
                dataset_name="dataset",
                task_name=None,
                source_path=dataset,
                layout_type=LayoutType.FLAT,
                info_path=meta / "info.json",
                data_path=data,
                video_path=dataset / "videos",
                depth_path=None,
            )
            info = {
                "fps": 20,
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [1],
                        "names": ["right_gripper"],
                    },
                    "observation.images.right_wrist": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
                },
            }
            profile = profile_parquets(
                parquet_paths, gripper_indices={"action": (0,)}
            )
            resolver = _DirectionResolver()

            def create_frame(
                _: Path, __: float, target: Path
            ) -> Path:
                target.touch()
                return target

            with patch(
                "robometanorm.application.pipeline.extract_rgb_frame_at",
                side_effect=create_frame,
            ):
                directions = _resolve_gripper_directions(
                    candidate,
                    info,
                    parquet_paths,
                    profile,
                    resolver,
                )

        self.assertEqual(directions["action:0"].direction, "increasing_is_open")
        evidence, image_paths = resolver.requests[0]
        self.assertEqual(evidence["camera_key"], "observation.images.right_wrist")
        self.assertEqual((evidence["low_value"], evidence["high_value"]), (0.0, 1.0))
        self.assertEqual(len(image_paths), 2)

    def test_skips_gripper_like_fields_when_dataset_contains_fingers(self) -> None:
        info = {
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["left_gripper_finger"],
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            parquet_path = Path(temporary_directory) / "episode_000000.parquet"
            pq.write_table(pa.table({"action": [[0.0], [1.0]]}), parquet_path)
            profile = profile_parquets(
                (parquet_path,), gripper_indices={"action": (0,)}
            )

        self.assertEqual(_unresolved_grippers(info, profile), {})

    def test_does_not_pair_a_missing_episode_with_the_only_video(self) -> None:
        first_parquet = Path("episode_000000.parquet")
        last_parquet = Path("episode_000009.parquet")
        first_video = Path("episode_000000.mp4")

        matched = _match_episode_media(
            last_parquet,
            (first_parquet, last_parquet),
            (first_video,),
        )

        self.assertIsNone(matched)

    def test_prefers_rgb_over_same_side_depth_camera(self) -> None:
        info = {
            "features": {
                "observation.images.left_wrist_depth": {
                    "dtype": "video",
                    "shape": [480, 640, 1],
                },
                "observation.images.left_wrist_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
            }
        }

        selected = _select_gripper_camera_key(info, "left")

        self.assertEqual(selected, "observation.images.left_wrist_rgb")


if __name__ == "__main__":
    unittest.main()
