# Robo Annotation YAML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a strict, executable `meta/robo_annotation.yaml` for PASS datasets and block unsafe joint layouts with source-file diagnostics.

**Architecture:** `annotation.py` performs a small lexical preflight and compiles the approved evidence/mapping into the example-shaped YAML mapping. `pipeline.py` runs the preflight before VLM and compiles only after a PASS normalization. `writer.py` adds the optional third atomically staged output without changing non-PASS behavior.

**Tech Stack:** Python 3.10, existing dataclasses and unittest suite, PyYAML safe serializer.

---

### Task 1: Annotation layout preflight and compiler

**Files:**
- Create: `src/robometanorm/annotation.py`
- Test: `tests/unit/test_annotation.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing preflight tests**

```python
def test_no_side_joint_names_block_before_vlm() -> None:
    issues = annotation_preconditions(
        {"features": {"action": {"names": ["joint_1"]}}}
    )
    assert issues[0].code == "ANNOTATION_JOINT_AMBIGUOUS"
    assert issues[0].evidence["source_file"] == "meta/info.json"


def test_left_and_right_contiguous_joint_names_are_candidates() -> None:
    issues = annotation_preconditions(two_arm_info())
    assert issues == ()
```

- [ ] **Step 2: Run the new tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.unit.test_annotation -v`

Expected: FAIL because `robometanorm.annotation` does not exist.

- [ ] **Step 3: Add the smallest annotation module**

```python
def annotation_preconditions(source_info: Mapping[str, object]) -> tuple[Issue, ...]:
    state_layout = _parse_layout(source_info, "observation.state")
    action_layout = _parse_layout(source_info, "action")
    if state_layout.issue is not None:
        return (state_layout.issue,)
    if action_layout.issue is not None:
        return (action_layout.issue,)
    if state_layout.roles != action_layout.roles:
        return (_layout_issue("action and qpos layouts differ"),)
    return ()


def compile_annotation(
    evidence: DatasetEvidence,
    profile: HardwareProfile,
    mapping: DatasetMapping,
    normalized_info: Mapping[str, object],
) -> tuple[dict[str, object] | None, Issue | None]:
    layout = _parse_layout(evidence.source_info, "observation.state")
    if layout.issue is not None:
        return None, layout.issue
    if not _mapping_confirms_layout(layout, profile, mapping):
        return None, _layout_issue("hardware mapping is incomplete")
    return _annotation_payload(normalized_info, evidence, layout), None
```

The parser accepts canonical names and sided one-based `left/right_joint_N`
sequences.  It emits only `arm.<side>.joint`, `arm.<side>.eef`, and
`gripper.<side>` keys; validates complete coverage, qpos/action semantic
equivalence, PDF gripper ranges, and a full sourced VLM mapping.

- [ ] **Step 4: Add exact output tests**

```python
def test_compiles_single_arm_example_shape() -> None:
    payload, issue = compile_annotation(
        fixture.evidence, fixture.profile, fixture.mapping, fixture.info_norm
    )
    assert issue is None
    assert tuple(payload) == (
        "version", "robot_type", "adapter", "robot_channel_schema"
    )
    assert "arm.right.eef" in payload["robot_channel_schema"]["channels"]


def test_compiles_dual_arm_without_hardcoded_offsets() -> None:
    payload, issue = compile_annotation(
        fixture.evidence, fixture.profile, fixture.mapping, fixture.info_norm
    )
    assert issue is None
    assert set(payload["robot_channel_schema"]["channels"]) >= {
        "arm.left.joint", "arm.right.joint", "gripper.left", "gripper.right"
    }
```

- [ ] **Step 5: Add YAML dependency and run focused tests**

Add `PyYAML` to `pyproject.toml`; use
`yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)` and reparse
with `yaml.safe_load` inside the compiler.

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.unit.test_annotation -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/robometanorm/annotation.py tests/unit/test_annotation.py
git commit -m "feat: compile strict robo annotations"
```

### Task 2: Pipeline refusal and PASS-only compilation

**Files:**
- Modify: `src/robometanorm/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`
- Test: `tests/unit/test_annotation.py`

- [ ] **Step 1: Write failing pipeline tests**

```python
def test_ambiguous_joint_preflight_skips_vlm_and_annotation() -> None:
    result = normalize_dataset_with_names(["joint_1"])
    assert result.status is DatasetStatus.BLOCKED
    assert fake_vlm.calls == 0
    assert not annotation_path.exists()


def test_sided_joint_candidates_reach_normal_mapping() -> None:
    result = normalize_two_arm_dataset()
    assert result.status is DatasetStatus.PASS
    assert annotation_path.exists()
```

- [ ] **Step 2: Run the focused pipeline tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.unit.test_pipeline -v`

Expected: FAIL because the pipeline does not call annotation preflight or compiler.

- [ ] **Step 3: Integrate at the two explicit boundaries**

```python
precondition_issues = (
    *check_preconditions(evidence),
    *annotation_preconditions(source_info),
)
# Skip VLM when any precondition has severity "block".

if status is DatasetStatus.PASS:
    annotation, annotation_issue = compile_annotation(
        evidence, profile, mapping, normalization.normalized_info
    )
```

When compilation returns an Issue, append it to the normalization issues,
recompute status, and pass `None` to the writer.  This records the exact
`meta/info.json` error evidence in `info_norm_review.json` and emits no YAML.

- [ ] **Step 4: Run focused regression tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.unit.test_annotation tests.unit.test_pipeline -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/robometanorm/pipeline.py tests/unit/test_annotation.py tests/unit/test_pipeline.py
git commit -m "feat: block unsafe annotation joints"
```

### Task 3: Deterministic YAML output bundle

**Files:**
- Modify: `src/robometanorm/writer.py`
- Modify: `tests/unit/test_writer.py`
- Modify: `tests/integration/test_cli.py`

- [ ] **Step 1: Write failing writer tests**

```python
def test_pass_bundle_writes_annotation_with_exact_schema() -> None:
    paths = write_outputs(candidate, info_norm, review, annotation=payload)
    assert paths[2].name == "robo_annotation.yaml"
    assert yaml.safe_load(paths[2].read_text()) == payload


def test_none_annotation_keeps_exact_two_outputs() -> None:
    paths = write_outputs(candidate, info_norm, review, annotation=None)
    assert len(paths) == 2
    assert not (candidate.info_path.parent / "robo_annotation.yaml").exists()
```

- [ ] **Step 2: Run writer tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.unit.test_writer -v`

Expected: FAIL because `write_outputs` has no `annotation` parameter.

- [ ] **Step 3: Add optional third staged file**

```python
def write_outputs(
    candidate: DatasetCandidate,
    info_norm: Mapping[str, object],
    review_without_hash: Mapping[str, object],
    *,
    annotation: Mapping[str, object] | None = None,
) -> tuple[Path, ...]:
    return _write_output_bundle(
        candidate, info_norm, review_without_hash, annotation=annotation
    )
```

Validate and serialize YAML before opening output files.  When annotation is
present, stage, fsync, and replace `robo_annotation.yaml` with the existing
safe descriptor-relative writer.  Keep the old two-file behavior exactly when
annotation is absent.

- [ ] **Step 4: Add CLI integration coverage**

```python
def test_pass_emits_annotation_and_nonpass_does_not() -> None:
    assert (meta / "robo_annotation.yaml").is_file()
    assert yaml.safe_load((meta / "robo_annotation.yaml").read_text())["version"] == (
        "dataset_annotation_config_v1"
    )
```

- [ ] **Step 5: Run complete verification**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 6: Commit and push**

```bash
git add src/robometanorm/writer.py tests/unit/test_writer.py tests/integration/test_cli.py
git commit -m "feat: write pass-only robo annotations"
git push origin HEAD:mini
```
