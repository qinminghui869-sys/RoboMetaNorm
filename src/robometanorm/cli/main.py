"""Command-line boundary for the mini normalization pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
import os
from pathlib import Path
import sys
from typing import TextIO
import unicodedata

from robometanorm.models import DatasetCandidate, DatasetResult, LayoutType
from robometanorm.pipeline import normalize_datasets, scan_datasets
from robometanorm.vlm import OpenAICompatibleDatasetVlm, OpenAICompatibleTransport


DEFAULT_VLM_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VLM_MODEL = "qwen3.7-plus"
DEFAULT_VLM_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_DATASET_TIMEOUT_SECONDS = 180.0
DEFAULT_VLM_MAX_TOKENS = 4096


class _ProgressRenderer:
    """Render dataset phases and completion on one interactive terminal line."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._rendered_width = 0
        self._wrote_update = False

    def update(self, index: int, total: int, result: DatasetResult) -> None:
        self._write(f"处理中 [{index}/{total}] {result.candidate.dataset_name}")

    def stage(
        self,
        index: int,
        total: int,
        candidate: DatasetCandidate,
        label: str,
    ) -> None:
        self._write(f"处理中 [{index}/{total}] {candidate.dataset_name}：{label}")

    def _write(self, message: str) -> None:
        width = _display_width(message)
        padding = " " * max(0, self._rendered_width - width)
        self._stream.write(f"\r{message}{padding}")
        self._stream.flush()
        self._rendered_width = max(self._rendered_width, width)
        self._wrote_update = True

    def finish(self) -> None:
        if self._wrote_update:
            self._stream.write("\n")
            self._stream.flush()


def _display_width(text: str) -> int:
    """Return terminal cells for the simple status text emitted by this CLI."""

    return sum(
        0
        if unicodedata.combining(character)
        else 2
        if unicodedata.east_asian_width(character) in {"F", "W"}
        else 1
        for character in text
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and print the fixed four-column dataset summary."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    layout = LayoutType(arguments.layout)
    progress = _ProgressRenderer(sys.stderr) if sys.stderr.isatty() else None
    try:
        if arguments.command == "scan":
            results = scan_datasets(
                arguments.root,
                layout,
                progress=progress.update if progress else None,
            )
        else:
            _validate_dataset_timeout_seconds(
                arguments.dataset_timeout_seconds,
                parser,
            )
            vlm = _build_vlm(arguments, parser)
            results = normalize_datasets(
                arguments.root,
                layout,
                vlm=vlm,
                confidence_threshold=arguments.confidence_threshold,
                progress=progress.update if progress else None,
                stage=progress.stage if progress else None,
                dataset_timeout_seconds=arguments.dataset_timeout_seconds,
                tolerate_vlm_network_errors=arguments.ignore_vlm_network_errors,
            )
    except ValueError as error:
        parser.error(str(error))
        return 2
    finally:
        if progress is not None:
            try:
                progress.finish()
            except MemoryError:
                raise
            except Exception:
                pass
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
                "--dataset-timeout-seconds",
                type=float,
                default=DEFAULT_DATASET_TIMEOUT_SECONDS,
                help="单个数据集的 VLM 总等待秒数，默认 180",
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
                default=DEFAULT_VLM_MAX_TOKENS,
                help=f"VLM 最大输出 token 数，默认 {DEFAULT_VLM_MAX_TOKENS}",
            )
            command_parser.add_argument(
                "--ignore-vlm-network-errors",
                action="store_true",
                help=(
                    "VLM 联网异常直接转为 REVIEW 输出，不把该数据集标记为 ERROR"
                ),
            )
    return parser


def _validate_dataset_timeout_seconds(
    timeout_seconds: object,
    parser: argparse.ArgumentParser,
) -> None:
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        parser.error("--dataset-timeout-seconds 必须是正的有限数字")


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
