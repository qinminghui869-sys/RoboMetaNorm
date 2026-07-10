"""通过 FFprobe 获取媒体一致性校验信息。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import subprocess


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
