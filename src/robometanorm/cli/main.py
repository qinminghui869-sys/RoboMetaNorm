"""RoboMetaNorm P0 命令行。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path

from robometanorm.application.pipeline import normalize_datasets, scan_datasets
from robometanorm.camera.vlm_classifier import OpenAICompatibleVlmClassifier, VlmClassifier
from robometanorm.domain.models import DatasetResult, LayoutType
from robometanorm.machine.vlm_semantic_resolver import OpenAICompatibleMachineVlmResolver


def main(argv: Sequence[str] | None = None) -> int:
    """运行 scan 或 normalize 子命令并打印汇总表。"""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    layout = LayoutType(arguments.layout)
    try:
        if arguments.command == "scan":
            results = scan_datasets(arguments.root, layout)
        else:
            vlm_classifier = _build_vlm_classifier(arguments, parser)
            results = normalize_datasets(
                arguments.root,
                layout,
                vlm_classifier=vlm_classifier,
                machine_vlm_resolver=_build_machine_vlm_resolver(vlm_classifier),
                confidence_threshold=arguments.confidence_threshold,
            )
    except ValueError as error:
        parser.error(str(error))
        return 2
    print(_format_summary(results))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构建 P0/P1 所需的命令行参数接口。"""
    parser = argparse.ArgumentParser(description="扫描机器人数据集并生成规范建议。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("scan", "normalize"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", type=Path, required=True, help="数据集根目录")
        command_parser.add_argument(
            "--layout",
            choices=[layout.value for layout in LayoutType],
            default=LayoutType.AUTO.value,
            help="输入目录布局，默认自动识别",
        )
        if command == "normalize":
            command_parser.add_argument(
                "--vlm-endpoint",
                help="可选的 OpenAI-compatible VLM 服务地址；未提供时不调用 VLM",
            )
            command_parser.add_argument("--vlm-model", help="VLM 模型名称")
            command_parser.add_argument(
                "--vlm-api-key-env",
                default="OPENAI_API_KEY",
                help="保存 VLM API Key 的环境变量名，默认 OPENAI_API_KEY",
            )
            command_parser.add_argument(
                "--confidence-threshold",
                type=float,
                default=0.85,
                help="VLM 自动采纳阈值，默认 0.85",
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


def _build_vlm_classifier(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> VlmClassifier | None:
    """仅在 endpoint 和 model 同时提供时启用外部 VLM。"""
    if not arguments.vlm_endpoint and not arguments.vlm_model:
        return None
    if not arguments.vlm_endpoint or not arguments.vlm_model:
        parser.error("--vlm-endpoint 与 --vlm-model 必须同时提供")
    if not 0 <= arguments.confidence_threshold <= 1:
        parser.error("--confidence-threshold 必须在 0 到 1 之间")
    return OpenAICompatibleVlmClassifier(
        arguments.vlm_endpoint,
        arguments.vlm_model,
        os.environ.get(arguments.vlm_api_key_env, ""),
        api_key_env=arguments.vlm_api_key_env,
        timeout_seconds=arguments.vlm_timeout_seconds,
        max_retries=arguments.vlm_max_retries,
        retry_backoff_seconds=arguments.vlm_retry_backoff_seconds,
        max_tokens=arguments.vlm_max_tokens,
    )


def _build_machine_vlm_resolver(
    vlm_classifier: VlmClassifier | None,
) -> OpenAICompatibleMachineVlmResolver | None:
    """机器字段复用相同 VLM 连接配置，但始终只请求语义 JSON。"""
    if isinstance(vlm_classifier, OpenAICompatibleVlmClassifier):
        return OpenAICompatibleMachineVlmResolver(vlm_classifier)
    return None


def _format_summary(results: Sequence[DatasetResult]) -> str:
    """使用标准库输出框架约定的执行汇总表。"""
    headers = ["Dataset", "Status", "Cameras", "Machine Fields", "Reviews"]
    rows = [
        [
            result.candidate.dataset_name,
            result.status.value,
            str(result.camera_count),
            str(result.machine_field_count),
            str(
                len(result.review_items)
                + result.camera_review_count
                + result.machine_review_count
            ),
        ]
        for result in results
    ]
    if not rows:
        return "未发现有效数据集。"

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "-+-".join("-" * width for width in widths)
    lines = [_format_row(headers, widths), separator]
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines)


def _format_row(values: Sequence[str], widths: Sequence[int]) -> str:
    """按列宽对齐一行表格。"""
    return " | ".join(value.ljust(width) for value, width in zip(values, widths))
