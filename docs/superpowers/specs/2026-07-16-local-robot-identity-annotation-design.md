# Local Robot Identity and Guaranteed Annotation Design

## Goal

Temporarily remove the remote hardware-profile research phase from `normalize`.
Use only `meta/info.json`'s `robot_type` as the robot identity, retain one VLM
request for dataset-level semantic mapping, and write `robo_annotation.yaml`
whenever the dataset metadata can be read—even when preflight or VLM analysis
cannot produce a complete mapping.

The longer-term hardware-profile cache is intentionally out of scope. The new
one-call boundary must leave a clear insertion point where a profile keyed by
`robot_type` can later be reused across datasets of the same model.

## Current Constraint

The current pipeline first calls `research_hardware` with web search to obtain a
`HardwareProfile`, then calls `map_dataset` to associate dataset fields with that
profile. Both `standard.py` and `annotation.py` require the profile and mapping,
and the writer emits `robo_annotation.yaml` only for a final `PASS` result.

This coupling means a remote research failure prevents annotation output even
when the local schema contains enough deterministic information for a partial
descriptor.

## Selected Approach

Replace the two-step research-and-map sequence with one dataset-analysis VLM
operation. Its inputs are the local `robot_type`, bounded camera evidence, and
the `action` and `observation.state` schemas and samples already collected by
the pipeline. Its response contains both:

- the camera and machine-component structure observed for this dataset; and
- the assignments from source fields and slices to that structure.

The VLM request is a normal multimodal JSON request, not a web-search request.
The prompt treats the local `robot_type` as fixed input and does not permit the
model to replace it.

The pipeline validates the combined response locally. A complete validated
response follows the full annotation path. A failed, incomplete, or rejected
response follows a deterministic best-effort annotation path.

## Processing Flow

For every discovered dataset, `normalize` performs these steps:

1. Read `meta/info.json` and collect the existing bounded local evidence.
2. Read `robot_type` directly from the source object. Never derive a replacement
   identity from the VLM response.
3. Run existing local preflight checks.
4. If `robot_type` is a safe non-empty string and preflight permits semantic
   analysis, make one VLM request that returns dataset structure and mapping.
5. Validate the response and apply only locally confirmed normalization changes.
6. Compile either a complete or best-effort annotation.
7. Atomically write `info_norm.json`, `info_norm_review.json`, and
   `robo_annotation.yaml`.

Preflight failures still skip the VLM request, but no longer skip annotation
compilation. Missing or invalid `robot_type` also skips the VLM request.

## Component Boundaries

### VLM boundary

`vlm.py` exposes one dataset-analysis operation for the normalization pipeline.
The operation accepts `DatasetEvidence` and the validated local `robot_type` and
returns a combined, typed result containing hardware structure and field
mapping, plus an optional `Issue`.

The OpenAI-compatible implementation uses `request_json`; it never calls
`request_web_json` on this path. Response parsing continues to enforce exact
keys, source coverage, finite confidence values, non-ambiguous identifiers, and
slice bounds before constructing domain objects.

The combined operation is the future cache insertion point: a later profile
provider may resolve the structure by `robot_type` and invoke only the mapping
portion for subsequent datasets. No persistent cache or cross-dataset reuse is
implemented now.

### Pipeline boundary

`pipeline.py` removes the hardware-research stage from `normalize_datasets`.
The stage callback reports local evidence collection and dataset mapping only.
The local source identity is preserved regardless of VLM outcome.

The pipeline invokes annotation compilation for every dataset whose source
metadata was successfully read. Annotation issues contribute to the dataset
status but do not suppress YAML output. A filesystem serialization or atomic
write failure remains an `ERROR` and is not reported as successful output.

### Standardization boundary

`standard.py` no longer requires an official web citation to preserve the local
identity. `info_norm.json` keeps the exact JSON-native source `robot_type`; a
safe string is recorded as a local keep decision, not as a VLM-derived apply
decision. The VLM cannot rename `robot_type`.

Camera and machine-field changes remain gated by the existing confidence,
ambiguity, schema, media, component, and slice checks. Failure of the combined
analysis preserves the original fields.

### Annotation boundary

`annotation.py` supports two deterministic compilation outcomes:

- **Complete:** all required camera and machine assignments pass the existing
  checks.
- **Best effort:** one or more checks, preconditions, or VLM operations fail.
  Only locally provable adapter entries and channels are included.

Neither path contains model-specific hardcoded rules.

### Writer boundary

`writer.py` serializes and atomically writes the annotation whenever source
metadata was read and an annotation document was produced. The writer remains
responsible only for validation, serialization, and atomic replacement; it does
not infer annotation content.

## Annotation Contract

Both complete and best-effort annotations retain the existing top-level fields:

- `version`
- `robot_type`
- `adapter`
- `robot_channel_schema`

They add a fifth top-level field:

```yaml
review:
  required: true
  issues:
    - code: VLM_NETWORK_ERROR
      message: The VLM service could not be reached.
```

For a complete annotation, `review.required` is `false` and `review.issues` is
empty. For a best-effort annotation, `review.required` is `true` and the list
contains a stable, compact projection of relevant issues. Each issue contains
only `code` and `message`; full evidence remains in `info_norm_review.json`.

The annotation's `robot_type` is the exact local value when it is a safe string.
If the key is missing or its value is not a safe string, the annotation uses
`null` and records a review issue. It does not invent an identity.

The best-effort compiler follows these rules:

- Write `adapter.base.qpos` and `adapter.base.action` only when the corresponding
  source features exist.
- Reuse a camera only when its source key is already canonical or its semantics
  are otherwise locally deterministic. Do not invent a camera direction or body
  position.
- Compile continuous, ordered, sided `left`/`right` joint layouts into matching
  arm slices.
- Compile a locally valid `main_follower_joint_*` layout into a `main` slice,
  but mark the annotation for review when single-arm hardware was not confirmed
  by a valid VLM result.
- Omit ambiguous, discontinuous, conflicting, or unsupported channels and add a
  compact review issue for each omitted category.
- Permit empty `cameras`, `channels`, and `group_weights` mappings so the YAML
  remains structurally readable when no semantic mapping is safe.

Thus file presence does not imply that every downstream channel is executable;
consumers can check `review.required` before automatic use.

## Error Handling

- Missing, null, non-string, whitespace-padded, control-character-containing, or
  empty `robot_type`: skip VLM, preserve the source value in JSON outputs, write
  `robot_type: null` in YAML, and require review.
- Local preflight block: skip VLM, preserve source normalization fields, and
  write a best-effort annotation containing only safe local structure.
- VLM configuration, network, timeout, HTTP, parsing, or contract failure:
  preserve source normalization fields and write a best-effort annotation.
- Partially valid VLM response: reject the conflicting response under the
  existing fail-closed contract rather than mixing trusted and untrusted object
  fragments; use the best-effort path.
- Annotation serialization or filesystem failure: retain the pipeline's error
  handling and do not claim that YAML was written.

All original issues remain in `info_norm_review.json`. The YAML review list is
derived from the final ordered issue sequence and deduplicated by code and
message.

## Compatibility

The command-line interface and its VLM configuration flags remain unchanged.
The web endpoint capability is simply unused by `normalize` during this
temporary phase. Existing callers that implement the old two-method VLM
protocol must be updated to implement the one-call analysis contract; this is an
internal package boundary rather than a documented public API.

Existing four-field annotation consumers must tolerate the new `review` field.
The addition is required so downstream users can distinguish confirmed output
from a file written only to satisfy the guaranteed-output requirement.

## Test Strategy

Implementation follows red-green-refactor. Tests cover:

1. A successful normalization performs zero hardware-research/web calls and one
   dataset-analysis/chat call.
2. A successful combined analysis preserves the exact local `robot_type`,
   applies validated field mappings, and emits a complete annotation with
   `review.required: false`.
3. VLM configuration, network, timeout, malformed response, and incomplete
   response cases each emit a best-effort annotation with
   `review.required: true`.
4. Preflight-blocked ambiguous and invalid joint layouts skip VLM but still emit
   YAML without unsafe semantic channels.
5. Missing and invalid `robot_type` values skip VLM, emit `robot_type: null`, and
   retain the source value in the JSON outputs.
6. Locally recognizable sided and `main_follower` layouts generate only their
   deterministic slices in best-effort mode.
7. No model-name hardcoding is introduced, and architecture tree guardrails stay
   satisfied.
8. CLI integration, multiple-dataset isolation, source preservation, YAML
   round-trip validation, and atomic writer tests continue to pass.

## Out of Scope

- Persistent hardware-profile cache.
- Sharing one profile across datasets in the current run.
- Model-specific mappings or aliases.
- URDF lookup or parsing.
- Numerical transforms, media transcoding, or changes to source files.
