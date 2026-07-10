"""构造只要求相机语义的 VLM 提示词。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from robometanorm.camera.media_probe import MediaInfo


SYSTEM_PROMPT = """你是机器人数据集相机语义分类器。
根据字段元数据和视频采样图判断模态、安装类型、方位、本体部位、主摄像头与歧义。
left/right 表示安装位置而非画面物体位置；wrist 表示末端执行器附近且随机械臂运动。
证据不足时使用 unknown 或 ambiguous=true。不得输出最终标准字段名，只输出合法 JSON。"""


def build_vlm_prompt(
    *,
    dataset_name: str,
    robot_type: str | None,
    source_key: str,
    feature: Mapping[str, object],
    declared_fps: object,
    media: MediaInfo | None,
    other_camera_keys: Sequence[str],
) -> tuple[str, str]:
    """返回系统提示词和带媒体证据的用户提示词。"""
    codec = media.codec if media else "unknown"
    resolution = f"{media.width}x{media.height}" if media else "unknown"
    actual_fps = media.fps if media else "unknown"
    user_prompt = "\n".join(
        [
            "请识别以下相机的语义：",
            f"dataset_name: {dataset_name}",
            f"robot_type: {robot_type or 'unknown'}",
            f"source_key: {source_key}",
            f"dtype: {feature.get('dtype')}",
            f"shape: {feature.get('shape')}",
            f"declared_fps: {declared_fps}",
            f"actual_codec: {codec}",
            f"actual_resolution: {resolution}",
            f"actual_fps: {actual_fps}",
            f"other_camera_keys: {list(other_camera_keys)}",
            "采样图按时间顺序排列。",
            "请返回 modality、mount_type、direction_tokens、body_part、is_primary、confidence、ambiguous、alternatives、need_human_review。",
        ]
    )
    return SYSTEM_PROMPT, user_prompt
