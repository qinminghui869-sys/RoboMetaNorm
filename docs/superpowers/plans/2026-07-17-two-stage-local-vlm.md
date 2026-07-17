# Two-Stage Local VLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a validated local profile and dataset mapping in two VLM calls with 4096 output tokens and field-oriented validation diagnostics.

**Architecture:** Keep the pipeline-facing `analyze_dataset()` API, but make the production implementation orchestrate a profile-only request followed by the existing mapping-only request. Parse each boundary independently and return safe stage/path evidence when validation fails.

**Tech Stack:** Python 3.10+, dataclasses, unittest, OpenAI-compatible Chat Completions transport.

---

### Task 1: Lock the two-call service contract

**Files:**
- Modify: `tests/mini_fixtures.py`
- Modify: `tests/unit/test_vlm.py`

- [x] Add queued chat payload support to `StubTransport` while preserving its single-payload behavior.
- [x] Add a profile-only fixture derived from the valid hardware payload without identity, sources, or `source_ids`.
- [x] Change the service test to expect two ordered chat requests, zero web requests, and the same deadline on both.
- [x] Add first-stage and second-stage invalid-payload tests that assert `DATASET_PROFILE_INVALID`/`DATASET_MAPPING_INVALID`, `stage`, `error_type`, and `validation_error`.
- [x] Run the focused VLM tests and confirm they fail because production still performs one combined request.

### Task 2: Implement profile-only prompting and parsing

**Files:**
- Modify: `src/robometanorm/vlm.py`
- Test: `tests/unit/test_vlm.py`

- [x] Add `build_profile_prompt(evidence, robot_type)` with the full local camera/component grammar and bounded serialized evidence.
- [x] Add `parse_dataset_profile(payload, robot_type)` that injects `_local_analysis_identity()` and reuses `parse_hardware_profile()`.
- [x] Add a bounded validation-issue helper that records stage, exception type, and parser message without raw payload values.
- [x] Run profile prompt/parser tests and confirm they pass.

### Task 3: Orchestrate the second mapping call

**Files:**
- Modify: `src/robometanorm/vlm.py`
- Modify: `tests/unit/test_vlm.py`

- [x] Replace the production combined request in `OpenAICompatibleDatasetVlm.analyze_dataset()` with profile request/parse followed by `map_dataset()`.
- [x] Add locatable diagnostics to `map_dataset()` prompt and parse failures.
- [x] Preserve transport issues exactly and stop before mapping when profile fails.
- [x] Run the focused service tests and confirm two-call behavior passes.

### Task 4: Raise output budget and update documentation

**Files:**
- Modify: `src/robometanorm/cli/main.py`
- Modify: `src/robometanorm/vlm.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `README.md`

- [x] Change CLI and transport defaults from 1024 to 4096.
- [x] Add a CLI parser/builder assertion for the 4096 default.
- [x] Document two local VLM calls, 4096 output tokens, and stage-specific REVIEW diagnostics.
- [x] Run the focused CLI and transport tests.

### Task 5: Verify pipeline integration

**Files:**
- Test: `tests/unit/test_vlm.py`
- Test: `tests/unit/test_pipeline.py`
- Test: `tests/integration/test_cli.py`

- [x] Run the complete VLM unit module.
- [x] Run the annotation module and primary pipeline/CLI degradation tests.
- [x] Run `git diff --check` and inspect the staged scope.
- [x] Commit only the design, plan, implementation, documentation, and related tests; preserve unrelated worktree changes.
