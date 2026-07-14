"""相机媒体发现、探测、抽帧与预览。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from robometanorm.domain.models import DatasetCandidate
from robometanorm.episode_sampling import select_representative_episodes


MEDIA_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
_SAFE_MEDIA_KEY_ALIASES = {
    "observation.images.left_image": "observation.images.image_left",
    "observation.images.right_image": "observation.images.image_right",
}


@dataclass(frozen=True)
class MediaInfo:
    """单个视频流的关键媒体属性。"""

    codec: str | None
    fps: float | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    frame_count: int | None
    pixel_format: str | None


def discover_camera_features(info: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """返回 observation.images 前缀的相机特征定义。"""
    features = info.get("features")
    if not isinstance(features, Mapping):
        return {}
    return {
        str(key): value
        for key, value in features.items()
        if str(key).startswith("observation.images.") and isinstance(value, Mapping)
    }


def find_camera_media(candidate: DatasetCandidate, source_key: str) -> tuple[Path, ...]:
    """按视频目录名匹配源字段，返回首、末 Episode 文件。"""
    roots = [path for path in (candidate.video_path, candidate.depth_path) if path is not None]
    accepted_keys = {source_key}
    alias = _SAFE_MEDIA_KEY_ALIASES.get(source_key)
    if alias is not None:
        accepted_keys.add(alias)
    files = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in MEDIA_SUFFIXES
        and any(key in path.parts for key in accepted_keys)
    ]
    return select_representative_episodes(files)


def discover_camera_media_keys(candidate: DatasetCandidate) -> tuple[str, ...]:
    """列出实际视频路径中的相机字段目录，用于诊断元数据键不一致。"""
    roots = [path for path in (candidate.video_path, candidate.depth_path) if path is not None]
    keys = {
        part
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
        for part in path.parts
        if part.startswith("observation.images.")
    }
    return tuple(sorted(keys))


def probe_media(media_path: Path) -> MediaInfo:
    """调用 FFprobe 并解析首个视频流。"""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(media_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise ValueError(f"无法执行 FFprobe: {error}") from error
    if completed.returncode != 0:
        raise ValueError(f"FFprobe 无法读取媒体: {completed.stderr.strip()}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("FFprobe 输出不是合法 JSON") from error
    streams = payload.get("streams", [])
    stream = next(
        (item for item in streams if item.get("codec_type", "video") == "video"), None
    )
    if not isinstance(stream, dict):
        raise ValueError("媒体中没有视频流")
    format_info = payload.get("format", {})
    return MediaInfo(
        codec=_string_or_none(stream.get("codec_name")),
        fps=_fraction_to_float(stream.get("r_frame_rate")),
        width=_integer_or_none(stream.get("width")),
        height=_integer_or_none(stream.get("height")),
        duration_seconds=_float_or_none(format_info.get("duration")),
        frame_count=_integer_or_none(stream.get("nb_frames")),
        pixel_format=_string_or_none(stream.get("pix_fmt")),
    )


def _fraction_to_float(value: object) -> float | None:
    """将 FFprobe 的有理数字符串安全地转为浮点数。"""
    if not isinstance(value, str) or value in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _integer_or_none(value: object) -> int | None:
    """转换可用的整数字段。"""
    try:
        return int(value) if value not in {None, "N/A"} else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    """转换可用的浮点字段。"""
    try:
        return float(value) if value not in {None, "N/A"} else None
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object) -> str | None:
    """转换可用的字符串字段。"""
    return value if isinstance(value, str) and value else None


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


def extract_rgb_frame_at(
    media_path: Path,
    timestamp_seconds: float,
    output_path: Path,
) -> Path:
    """在 Episode 内指定时刻抽取一张 RGB 帧。"""
    if timestamp_seconds < 0:
        raise ValueError("抽帧时间不能为负数")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{timestamp_seconds:.6f}",
        "-i",
        str(media_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=1280:-2:force_original_aspect_ratio=decrease",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise ValueError(f"无法执行 FFmpeg: {error}") from error
    if completed.returncode != 0 or not output_path.is_file():
        raise ValueError(f"FFmpeg 抽帧失败: {completed.stderr.strip()}")
    return output_path


def write_depth_previews(depth_array: object, output_directory: Path) -> tuple[Path, Path]:
    """以有效值 2%–98% 分位数归一化生成两张预览图。"""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Depth 预览需要安装 numpy 和 opencv-python") from error

    values = np.asarray(depth_array, dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        raise ValueError("Depth 数组没有有效深度值")
    lower, upper = np.percentile(valid, [2, 98])
    if upper <= lower:
        upper = lower + 1.0
    normalized = np.clip((values - lower) / (upper - lower), 0, 1)
    normalized[~np.isfinite(normalized)] = 0
    grayscale = (normalized * 255).astype(np.uint8)
    color = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    output_directory.mkdir(parents=True, exist_ok=True)
    gray_path = output_directory / "depth_gray.png"
    color_path = output_directory / "depth_pseudocolor.png"
    if not cv2.imwrite(str(gray_path), grayscale) or not cv2.imwrite(str(color_path), color):
        raise ValueError("无法写入 Depth 预览图")
    return gray_path, color_path
