"""Command-line boundary for the mini normalization pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
import os
from pathlib import Path

from robometanorm.models import DatasetResult, LayoutType
from robometanorm.pipeline import normalize_datasets, scan_datasets
from robometanorm.vlm import OpenAICompatibleDatasetVlm, OpenAICompatibleTransport


DEFAULT_VLM_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VLM_MODEL = "qwen3.7-plus"
DEFAULT_VLM_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_CONFIDENCE_THRESHOLD = 0.85


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and print the fixed four-column dataset summary."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    layout = LayoutType(arguments.layout)
    try:
        if arguments.command == "scan":
            results = scan_datasets(arguments.root, layout)
        else:
            vlm = _build_vlm(arguments, parser)
            results = normalize_datasets(
                arguments.root,
                layout,
                vlm=vlm,
                confidence_threshold=arguments.confidence_threshold,
            )
    except ValueError as error:
        parser.error(str(error))
        return 2
    print(_format_summary(results))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="扫描机器人数据集并生成严格、保守的规范输出。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("scan", "normalize"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--root", type=Path, required=True, help="数据集根目录"
        )
        command_parser.add_argument(
            "--layout",
            choices=[layout.value for layout in LayoutType],
            default=LayoutType.AUTO.value,
            help="输入目录布局，默认自动识别",
        )
        if command == "normalize":
            command_parser.add_argument(
                "--vlm-endpoint",
                default=DEFAULT_VLM_ENDPOINT,
                help=f"OpenAI-compatible VLM 服务地址，默认 {DEFAULT_VLM_ENDPOINT}",
            )
            command_parser.add_argument(
                "--vlm-model",
                default=DEFAULT_VLM_MODEL,
                help=f"VLM 模型名称，默认 {DEFAULT_VLM_MODEL}",
            )
            command_parser.add_argument(
                "--vlm-api-key-env",
                default=DEFAULT_VLM_API_KEY_ENV,
                help=f"保存 VLM API Key 的环境变量名，默认 {DEFAULT_VLM_API_KEY_ENV}",
            )
            command_parser.add_argument(
                "--confidence-threshold",
                type=float,
                default=DEFAULT_CONFIDENCE_THRESHOLD,
                help=(
                    "VLM 自动采纳阈值，默认 "
                    f"{DEFAULT_CONFIDENCE_THRESHOLD}"
                ),
            )
            command_parser.add_argument(
                "--vlm-timeout-seconds",
                type=int,
                default=120,
                help="单次 VLM 请求超时秒数，默认 120",
            )
            command_parser.add_argument(
                "--vlm-max-retries",
                type=int,
                default=2,
                help="429、5xx 和网络异常的最大重试次数，默认 2",
            )
            command_parser.add_argument(
                "--vlm-retry-backoff-seconds",
                type=float,
                default=1.0,
                help="VLM 指数退避初始等待秒数，默认 1.0",
            )
            command_parser.add_argument(
                "--vlm-max-tokens",
                type=int,
                default=1024,
                help="VLM 最大输出 token 数，默认 1024",
            )
    return parser


def _build_vlm(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> OpenAICompatibleDatasetVlm:
    """Build one transport and wrap it in one dataset-level VLM service."""

    threshold = arguments.confidence_threshold
    if (
        type(threshold) not in {int, float}
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        parser.error("--confidence-threshold 必须是 0 到 1 的有限数字")
    transport = OpenAICompatibleTransport(
        arguments.vlm_endpoint,
        arguments.vlm_model,
        os.environ.get(arguments.vlm_api_key_env, ""),
        api_key_env=arguments.vlm_api_key_env,
        timeout_seconds=arguments.vlm_timeout_seconds,
        max_retries=arguments.vlm_max_retries,
        retry_backoff_seconds=arguments.vlm_retry_backoff_seconds,
        max_tokens=arguments.vlm_max_tokens,
    )
    return OpenAICompatibleDatasetVlm(transport)


def _format_summary(results: Sequence[DatasetResult]) -> str:
    lines = [
        "Dataset | Status | Changed Fields | Issues",
        "--- | --- | --- | ---",
    ]
    lines.extend(
        " | ".join(
            (
                result.candidate.dataset_name,
                result.status.value,
                str(result.changed_field_count),
                str(result.issue_count),
            )
        )
        for result in results
    )
    return "\n".join(lines)
