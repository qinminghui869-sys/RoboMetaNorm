"""End-to-end tests for the intentionally small mini CLI."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from robometanorm.cli.main import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DATASET_TIMEOUT_SECONDS,
    _ProgressRenderer,
    _build_parser,
    _build_vlm,
    main,
)
from robometanorm.models import (
    DatasetAnalysis,
    DatasetMapping,
    DatasetResult,
    DatasetStatus,
    Issue,
    MediaSample,
)
from tests.mini_fixtures import DatasetFixture, FakeVlm, PipelineFixture


class _InteractiveStderr(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FinishFailingStderr(_InteractiveStderr):
    def write(self, value: str) -> int:
        if value == "\n":
            raise OSError("terminal closed")
        return super().write(value)


class CliIntegrationTest(unittest.TestCase):
    """Run scan and normalize against complete, fictional local datasets."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.fixture = self._create_dataset("dataset-a")
        builder = PipelineFixture()
        self.profile = builder.hardware_profile()
        state_assignment = builder.dataset_mapping().machines[0]
        self.mapping = DatasetMapping(
            cameras=builder.dataset_mapping().cameras,
            machines=(
                replace(state_assignment, source_feature="action"),
                state_assignment,
            ),
        )

    @staticmethod
    def _source_info() -> dict[str, object]:
        raw_names = [f"left_joint_{index}" for index in range(1, 7)]
        return {
            "robot_type": "acme_testbot",
            "fps": 30,
            "features": {
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "fps": 30,
                    "codec": "av1",
                },
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": list(raw_names),
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": list(raw_names),
                },
            },
        }

    def _create_dataset(self, name: str) -> DatasetFixture:
        fixture = DatasetFixture.create(
            self.root,
            dataset_name=name,
            info=self._source_info(),
            with_data=True,
            with_videos=True,
        )
        rows = {
            "action": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]],
            "observation.state": [[0.5, 0.4, 0.3, 0.2, 0.1, 0.0]],
        }
        fixture.write_parquet(rows, relative_path="chunk-000/episode_000000.parquet")
        fixture.write_parquet(rows, relative_path="chunk-001/episode_000001.parquet")
        fixture.write_media_placeholder(
            relative_path="chunk-000/observation.images.wrist/episode_000000.mp4",
            payload=b"immutable-source-video",
        )
        return fixture

    @staticmethod
    def _probe_media(*args: object, **kwargs: object) -> MediaSample:
        return MediaSample(
            relative_path="",
            media_type="video",
            codec="av1",
            fps=30.0,
            width=640,
            height=480,
            duration_seconds=1.0,
            pixel_format="yuv420p",
            frame_path=None,
        )

    @staticmethod
    def _extract_frame(
        media_path: Path,
        output_path: Path,
        *,
        duration_seconds: float | None = None,
    ) -> Path:
        output_path.write_bytes(b"ephemeral-frame")
        return output_path

    def _success_vlm(self) -> FakeVlm:
        return FakeVlm(analysis_result=(DatasetAnalysis(self.profile, self.mapping), None))

    def _run(
        self,
        *arguments: str,
        vlm: object | None = None,
        use_real_builder: bool = False,
        stderr: io.StringIO | None = None,
    ) -> tuple[str, str]:
        stdout = io.StringIO()
        stderr = stderr if stderr is not None else io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(
                patch("robometanorm.evidence.probe_media", side_effect=self._probe_media)
            )
            stack.enter_context(
                patch(
                    "robometanorm.evidence.extract_midpoint_frame",
                    side_effect=self._extract_frame,
                )
            )
            if not use_real_builder and arguments[0] == "normalize":
                stack.enter_context(
                    patch(
                        "robometanorm.cli.main._build_vlm",
                        return_value=vlm if vlm is not None else self._success_vlm(),
                    )
                )
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            exit_code = main(list(arguments))
        self.assertEqual(exit_code, 0)
        return stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_scan_is_read_only_and_uses_only_the_four_summary_columns(self) -> None:
        output, _ = self._run("scan", "--root", str(self.root))

        self.assertEqual(
            output.splitlines()[0],
            "Dataset | Status | Changed Fields | Issues",
        )
        self.assertIn("dataset-a", output)
        self.assertIn("PASS", output)
        for obsolete in ("Cameras", "Machine Fields", "Cam C/I/U", "Topology Errors"):
            self.assertNotIn(obsolete, output)
        meta = self.fixture.candidate.info_path.parent
        self.assertFalse((meta / "info_norm.json").exists())
        self.assertFalse((meta / "info_norm_review.json").exists())

    def test_scan_renders_progress_only_to_interactive_stderr(self) -> None:
        interactive_stderr = _InteractiveStderr()

        output, progress = self._run(
            "scan",
            "--root",
            str(self.root),
            stderr=interactive_stderr,
        )

        self.assertEqual(progress, "\r处理中 [1/1] dataset-a\n")
        self.assertIn("dataset-a | PASS", output)

        _, noninteractive_stderr = self._run("scan", "--root", str(self.root))
        self.assertEqual(noninteractive_stderr, "")

    def test_normalize_renders_progress_only_to_interactive_stderr(self) -> None:
        interactive_stderr = _InteractiveStderr()

        output, progress = self._run(
            "normalize",
            "--root",
            str(self.root),
            vlm=self._success_vlm(),
            stderr=interactive_stderr,
        )

        self.assertIn("处理中 [1/1] dataset-a：分析相机与关节", progress)
        self.assertLess(
            progress.index("处理中 [1/1] dataset-a：分析相机与关节"),
            progress.rindex("\r处理中 [1/1] dataset-a"),
        )
        self.assertTrue(progress.endswith("\n"))
        self.assertEqual(
            output,
            "Dataset | Status | Changed Fields | Issues\n"
            "--- | --- | --- | ---\n"
            "dataset-a | PASS | 3 | 0\n",
        )

        _, noninteractive_stderr = self._run(
            "normalize",
            "--root",
            str(self.root),
            vlm=self._success_vlm(),
        )
        self.assertEqual(noninteractive_stderr, "")

    def test_progress_renderer_clears_wide_phase_suffix_before_completion(self) -> None:
        stderr = _InteractiveStderr()
        renderer = _ProgressRenderer(stderr)

        renderer.stage(1, 1, self.fixture.candidate, "映射相机与关节")
        renderer.update(
            1,
            1,
            DatasetResult(
                candidate=self.fixture.candidate,
                status=DatasetStatus.PASS,
                camera_count=0,
                machine_field_count=0,
                changed_field_count=0,
                issue_count=0,
                source_info=None,
            ),
        )
        renderer.finish()

        self.assertEqual(
            stderr.getvalue(),
            "\r处理中 [1/1] dataset-a：映射相机与关节"
            "\r处理中 [1/1] dataset-a"
            + " " * 16
            + "\n",
        )

    def test_normalize_forwards_dataset_timeout_and_stage_callback(self) -> None:
        cases = (
            ((), 180.0, False),
            (("--dataset-timeout-seconds", "45"), 45.0, False),
            (("--ignore-vlm-network-errors",), 180.0, True),
        )
        for option, expected_timeout, expected_ignore_network_errors in cases:
            with self.subTest(option=option):
                stderr = _InteractiveStderr()
                stdout = io.StringIO()
                with (
                    patch("robometanorm.cli.main._build_vlm", return_value=object()),
                    patch(
                        "robometanorm.cli.main.normalize_datasets",
                        return_value=[],
                    ) as normalize,
                    redirect_stderr(stderr),
                    redirect_stdout(stdout),
                ):
                    exit_code = main(
                        ["normalize", "--root", str(self.root), *option]
                    )

                self.assertEqual(exit_code, 0)
                self.assertEqual(
                    normalize.call_args.kwargs["dataset_timeout_seconds"],
                    expected_timeout,
                )
                self.assertEqual(
                    normalize.call_args.kwargs["tolerate_vlm_network_errors"],
                    expected_ignore_network_errors,
                )
                self.assertIsNotNone(normalize.call_args.kwargs["stage"])
                self.assertEqual(
                    stdout.getvalue(),
                    "Dataset | Status | Changed Fields | Issues\n"
                    "--- | --- | --- | ---\n",
                )

        self.assertEqual(DEFAULT_DATASET_TIMEOUT_SECONDS, 180.0)

    def test_normalize_rejects_invalid_dataset_timeout(self) -> None:
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    main(
                        [
                            "normalize",
                            "--root",
                            str(self.root),
                            f"--dataset-timeout-seconds={value}",
                        ]
                    )
                self.assertIn(
                    "--dataset-timeout-seconds 必须是正的有限数字",
                    stderr.getvalue(),
                )

    def test_interrupted_scan_finishes_rendered_progress_before_propagating(self) -> None:
        interactive_stderr = _InteractiveStderr()

        def interrupt_after_progress(
            root: Path,
            layout: object,
            *,
            progress: object = None,
        ) -> list[DatasetResult]:
            self.assertIsNotNone(progress)
            progress(
                1,
                1,
                DatasetResult(
                    candidate=self.fixture.candidate,
                    status=DatasetStatus.PASS,
                    camera_count=0,
                    machine_field_count=0,
                    changed_field_count=0,
                    issue_count=0,
                    source_info=None,
                ),
            )
            raise KeyboardInterrupt

        with (
            patch(
                "robometanorm.cli.main.scan_datasets",
                side_effect=interrupt_after_progress,
            ),
            redirect_stderr(interactive_stderr),
            self.assertRaises(KeyboardInterrupt),
        ):
            main(["scan", "--root", str(self.root)])

        self.assertEqual(interactive_stderr.getvalue(), "\r处理中 [1/1] dataset-a\n")

    def test_finish_stream_error_does_not_prevent_successful_summary(self) -> None:
        stderr = _FinishFailingStderr()

        output, progress = self._run(
            "scan",
            "--root",
            str(self.root),
            stderr=stderr,
        )

        self.assertEqual(progress, "\r处理中 [1/1] dataset-a")
        self.assertEqual(
            output,
            "Dataset | Status | Changed Fields | Issues\n"
            "--- | --- | --- | ---\n"
            "dataset-a | PASS | 0 | 0\n",
        )

    def test_normalize_writes_annotation_and_preserves_all_sources(self) -> None:
        source_paths = tuple(
            sorted(
                (
                    self.fixture.candidate.info_path,
                    *self.fixture.candidate.data_path.rglob("*.parquet"),
                    *self.fixture.candidate.video_path.rglob("*.mp4"),
                )
            )
        )
        before = {path: self._hash(path) for path in source_paths}
        before_files = set(self.fixture.candidate.source_path.rglob("*"))

        output, _ = self._run(
            "normalize", "--root", str(self.root), vlm=self._success_vlm()
        )

        self.assertIn("dataset-a | PASS | 3 | 0", output)
        self.assertEqual({path: self._hash(path) for path in source_paths}, before)
        after_files = set(self.fixture.candidate.source_path.rglob("*"))
        new_files = {path for path in after_files - before_files if path.is_file()}
        meta = self.fixture.candidate.info_path.parent
        self.assertEqual(
            new_files,
            {
                meta / "info_norm.json",
                meta / "info_norm_review.json",
                meta / "robo_annotation.yaml",
            },
        )
        normalized = json.loads((meta / "info_norm.json").read_text(encoding="utf-8"))
        self.assertEqual(normalized["robot_type"], "acme_testbot")
        features = normalized["features"]
        self.assertIn("observation.images.cam_front_wrist_rgb", features)
        expected = [f"left_arm_joint_{index}_rad" for index in range(6)]
        self.assertEqual(features["action"]["names"], expected)
        self.assertEqual(features["observation.state"]["names"], expected)
        annotation = yaml.safe_load((meta / "robo_annotation.yaml").read_text(encoding="utf-8"))
        self.assertEqual(annotation["adapter"]["cameras"], {
            "observation.images.cam_front_wrist_rgb": "observation.images.wrist"
        })
        self.assertIn("arm.left.joint", annotation["robot_channel_schema"]["channels"])

    def test_missing_api_key_never_opens_network_and_degrades_to_review(self) -> None:
        key_name = "ROBOMETANORM_TEST_MISSING_KEY"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("urllib.request.urlopen", side_effect=AssertionError("network used")),
        ):
            output, _ = self._run(
                "normalize",
                "--root",
                str(self.root),
                "--vlm-api-key-env",
                key_name,
                use_real_builder=True,
            )

        self.assertIn("REVIEW", output)
        meta = self.fixture.candidate.info_path.parent
        normalized = json.loads((meta / "info_norm.json").read_text(encoding="utf-8"))
        review = json.loads(
            (meta / "info_norm_review.json").read_text(encoding="utf-8")
        )
        self.assertEqual(normalized, self._source_info())
        self.assertIn("VLM_CONFIG_MISSING", {item["code"] for item in review["issues"]})
        annotation = yaml.safe_load((meta / "robo_annotation.yaml").read_text(encoding="utf-8"))
        self.assertTrue(annotation["review"]["required"])

    def test_network_failure_from_vlm_keeps_source_and_records_reason(self) -> None:
        failure = Issue("VLM_NETWORK_ERROR", "offline", "vlm")
        vlm = FakeVlm(analysis_result=(None, failure))
        output, _ = self._run("normalize", "--root", str(self.root), vlm=vlm)

        self.assertIn("REVIEW", output)
        meta = self.fixture.candidate.info_path.parent
        self.assertEqual(
            json.loads((meta / "info_norm.json").read_text(encoding="utf-8")),
            self._source_info(),
        )
        review = json.loads(
            (meta / "info_norm_review.json").read_text(encoding="utf-8")
        )
        self.assertIn("VLM_NETWORK_ERROR", {item["code"] for item in review["issues"]})
        annotation = yaml.safe_load((meta / "robo_annotation.yaml").read_text(encoding="utf-8"))
        self.assertTrue(annotation["review"]["required"])

    def test_blocked_dataset_still_writes_review_annotation_and_source_preserving_outputs(self) -> None:
        source = self._source_info()
        del source["features"]["action"]
        self.fixture.candidate.info_path.write_text(json.dumps(source), encoding="utf-8")
        vlm = self._success_vlm()

        output, _ = self._run("normalize", "--root", str(self.root), vlm=vlm)

        self.assertIn("BLOCKED", output)
        self.assertEqual(vlm.analysis_calls, 0)
        meta = self.fixture.candidate.info_path.parent
        self.assertEqual(
            json.loads((meta / "info_norm.json").read_text(encoding="utf-8")), source
        )
        self.assertTrue((meta / "info_norm_review.json").is_file())
        self.assertTrue((meta / "robo_annotation.yaml").is_file())

    def test_writer_failure_is_isolated_and_second_dataset_completes(self) -> None:
        second = self._create_dataset("dataset-b")
        from robometanorm.writer import write_outputs as real_write

        calls = 0

        def fail_first(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("disk failure")
            return real_write(*args, **kwargs)

        with patch("robometanorm.pipeline.write_outputs", side_effect=fail_first):
            output, _ = self._run(
                "normalize", "--root", str(self.root), vlm=self._success_vlm()
            )

        self.assertEqual(calls, 2)
        self.assertIn("dataset-a | ERROR", output)
        self.assertIn("dataset-b | PASS", output)
        self.assertTrue(
            (second.candidate.info_path.parent / "info_norm_review.json").is_file()
        )

    def test_builder_has_one_transport_wrapper_and_forwards_all_arguments(self) -> None:
        arguments = argparse.Namespace(
            vlm_endpoint="https://vlm.invalid/v1",
            vlm_model="fictional-model",
            vlm_api_key_env="FICTIONAL_KEY",
            confidence_threshold=0.9,
            vlm_timeout_seconds=90,
            vlm_max_retries=3,
            vlm_retry_backoff_seconds=0.5,
            vlm_max_tokens=2048,
        )
        parser = _build_parser()
        transport = object()
        service = object()
        with (
            patch.dict(os.environ, {"FICTIONAL_KEY": "test-secret"}),
            patch(
                "robometanorm.cli.main.OpenAICompatibleTransport",
                return_value=transport,
            ) as transport_type,
            patch(
                "robometanorm.cli.main.OpenAICompatibleDatasetVlm",
                return_value=service,
            ) as service_type,
        ):
            built = _build_vlm(arguments, parser)

        self.assertIs(built, service)
        transport_type.assert_called_once_with(
            "https://vlm.invalid/v1",
            "fictional-model",
            "test-secret",
            api_key_env="FICTIONAL_KEY",
            timeout_seconds=90,
            max_retries=3,
            retry_backoff_seconds=0.5,
            max_tokens=2048,
        )
        service_type.assert_called_once_with(transport)

    def test_nonfinite_out_of_range_and_direct_bool_thresholds_are_rejected(self) -> None:
        for value in ("nan", "inf", "-inf", "-0.1", "1.1"):
            with self.subTest(value=value):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(
                        [
                            "normalize",
                            "--root",
                            str(self.root),
                            "--confidence-threshold",
                            value,
                        ]
                    )

        arguments = _build_parser().parse_args(
            ["normalize", "--root", str(self.root)]
        )
        arguments.confidence_threshold = True
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _build_vlm(arguments, _build_parser())

        source = Path(__file__).parents[2] / "src" / "robometanorm" / "cli" / "main.py"
        self.assertEqual(source.read_text(encoding="utf-8").count("0.85"), 1)
        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.85)

    def test_module_help_is_runnable_without_network_or_dataset_access(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-m", "robometanorm", "--help"],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("scan", completed.stdout)
        self.assertIn("normalize", completed.stdout)


if __name__ == "__main__":
    unittest.main()
