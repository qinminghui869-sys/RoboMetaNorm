"""P1 全局相机字段精确映射。"""

from __future__ import annotations


CAMERA_NAME_MAP = {
    "observation.images.image_left": "observation.images.cam_left_rgb",
    "observation.images.image_right": "observation.images.cam_right_rgb",
    "observation.images.image_top": "observation.images.cam_top_rgb",
    "observation.images.image_wrist": "observation.images.cam_wrist_rgb",
    "observation.images.image_wrist_left": "observation.images.cam_left_wrist_rgb",
    "observation.images.image_wrist_right": "observation.images.cam_right_wrist_rgb",
    "observation.images.image_top_depth": "observation.images.cam_top_depth",
    "observation.images.image_left_depth": "observation.images.cam_left_depth",
    "observation.images.image_right_depth": "observation.images.cam_right_depth",
    "observation.images.image_wrist_depth": "observation.images.cam_wrist_depth",
}
