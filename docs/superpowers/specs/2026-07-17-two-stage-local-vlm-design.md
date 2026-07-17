# Two-Stage Local VLM Design

## Goal

Replace the atomic dataset-analysis response with two local, non-web VLM calls:
first infer a dataset-local hardware profile, then map dataset sources to the
validated profile. Keep `robot_type` fixed to `meta/info.json`, raise the default
output budget to 4096 tokens, and report validation failures with a safe stage and
field-oriented message.

## Architecture

`DatasetVlm.analyze_dataset()` remains the pipeline-facing operation so pipeline
and test fakes do not gain a second public responsibility. The production
implementation performs two internal requests:

1. `build_profile_prompt()` sends robot type, bounded camera evidence, bounded
   machine schemas, and representative frames. It requests exactly `cameras` and
   `components`; no web search, citations, identity rewrite, or final names are
   permitted.
2. `parse_dataset_profile()` injects the fixed local identity and validates the
   same camera/component grammar used by the deterministic standard renderer.
3. `build_mapping_prompt()` receives the validated profile and asks only for
   camera/component assignments. `parse_dataset_mapping()` verifies exact source
   coverage and ID/slice integrity.
4. The two validated values are combined into `DatasetAnalysis`.

The existing combined prompt/parser remain available only for compatibility
tests during migration; the production service no longer calls them.

## Prompt Contract

The profile prompt must enumerate every accepted component kind and its exact
`side`, `representation`, `unit`, `count`, `element_order`, gripper range, and
open-direction constraints. It must also state camera grammar, ID uniqueness,
finite confidence range, non-empty reasons, and the prohibition on final target
names.

The mapping prompt must state non-negative slices, `end > start`, element-order
length equality, known-ID references, unique resolved camera IDs, and exact
coverage of all supplied camera and machine source keys.

## Errors

Profile prompt construction, missing payload, or profile parsing returns
`DATASET_PROFILE_INVALID`. Mapping failures return `DATASET_MAPPING_INVALID`.
Parser errors include only safe diagnostics:

- `stage`: `profile_prompt`, `profile_parse`, `mapping_prompt`, or `mapping_parse`;
- `error_type`: the exception class;
- `validation_error`: the bounded parser message, which contains schema paths but
  no raw response values.

Transport issues remain unchanged. A first-stage failure prevents the second
request. A second-stage failure discards the incomplete analysis and preserves
the existing REVIEW fallback.

## Configuration

The CLI and transport default `max_tokens` becomes 4096. An explicit
`--vlm-max-tokens` value still overrides it. The same dataset deadline is forwarded
to both requests, so the second call consumes only the remaining dataset budget.

## Verification

Tests must prove two ordered chat calls, no web call, shared deadline propagation,
complete profile grammar in the first prompt, mapping-only content in the second,
stage-specific validation evidence, first-stage short-circuiting, 4096 defaults,
and unchanged pipeline fallback behavior for VLM issues.
