"""按 P1 两阶段策略从视频抽取临时帧。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess

from robometanorm.camera.media_probe import MediaInfo, probe_media


def first_stage_ratios() -> tuple[float, float, float]:
    """返回第一阶段的 10%、50%、90% 采样比例。"""
    return (0.1, 0.5, 0.9)


def second_stage_ratios(episode_count: int) -> tuple[float, ...]:
    """按 episode 数量返回第二阶段采样比例。"""
    if episode_count <= 1:
        return (0.1, 0.3, 0.5, 0.7, 0.9)
    return (0.2, 0.5, 0.8)


def extract_rgb_frames(
    media_path: Path,
    ratios: Sequence[float],
    output_directory: Path,
    media: MediaInfo | None = None,
) -> tuple[Path, ...]:
    """用 FFmpeg 按比例抽取最长边不超过 1280 的 JPG。"""
    media = media or probe_media(media_path)
    if media.duration_seconds is None or media.duration_seconds <= 0:
        raise ValueError("媒体缺少可用时长，无法按比例抽帧")
    output_directory.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, ratio in enumerate(ratios):
        if not 0 < ratio < 1:
            raise ValueError("抽样比例必须在 0 与 1 之间")
        target_path = output_directory / f"frame_{index:02d}.jpg"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{media.duration_seconds * ratio:.6f}",
            "-i",
            str(media_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1280:-2:force_original_aspect_ratio=decrease",
            str(target_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not target_path.is_file():
            raise ValueError(f"FFmpeg 抽帧失败: {completed.stderr.strip()}")
        frames.append(target_path)
    return tuple(frames)
