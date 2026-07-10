"""Depth 数组的灰度与伪彩预览生成。"""

from __future__ import annotations

from pathlib import Path


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
