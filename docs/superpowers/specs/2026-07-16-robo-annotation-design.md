# Robo Annotation YAML Design

## Goal

For each usable LeRobot dataset, generate the executable
`meta/robo_annotation.yaml`.  The file follows the provided YAML layout and
uses PDF-standard camera names.  A dataset emits this file only when its
normalization result is `PASS`.

## Scope

- Implement only in the `mini-implementation` worktree.
- Keep the existing `info_norm.json` and `info_norm_review.json` outputs.
- Add `robo_annotation.yaml` only for `PASS` datasets.
- Do not add URDF, tactile, or audio processing.
- Do not add robot-model-specific rules.

## YAML Contract

The YAML has exactly these top-level keys:

```yaml
version: dataset_annotation_config_v1
robot_type: canonical_robot_type
adapter:
  base_type: LeRobot
  base:
    qpos: observation.state
    action: action
  cameras:
    observation.images.cam_top_rgb: observation.images.image_top
robot_channel_schema:
  version: channel_schema_v1
  robot_type: canonical_robot_type
  channels: {}
  group_weights: {}
```

`adapter.cameras` keys must use the PDF grammar
`observation.images.cam_<position>_<rgb|depth>`.  They point to the original
dataset feature so the consumer can read the unmodified data.  The mapped
camera must preserve FPS and use AV1 for RGB or FFV1 for depth as required by
the PDF.

Each channel has exactly `source`, `field`, `slice`, `group`, `unit`, `norm`,
`weight`, and `optional`.  `slice` is `[start, end)`.  `source` and `field`
are `qpos`; `adapter.base.qpos` resolves them to `observation.state`.

The supported channel identifiers are deliberately small and match the
example:

- `arm.left.joint`, `arm.right.joint`
- `arm.left.eef`, `arm.right.eef`
- `gripper.left`, `gripper.right`

Single-arm datasets emit only their confirmed side.  Dual-arm datasets emit
both sides.  No joint count or slice offset is hard-coded.

## Joint Inference and Refusal

The compiler inspects `info.json` names for `action` and
`observation.state` before VLM use.  A contiguous, one-based sequence such as
`left_joint_1...left_joint_N` or `right_joint_1...right_joint_N` is an arm
candidate, not an automatic refusal.  It is accepted only when:

1. the side is explicit and indices are continuous without duplicates;
2. action and qpos have the same inferred ordered channel layout;
3. qpos contains no unrepresented machine dimensions;
4. the later VLM hardware profile and mapping confirm the side, complete
   slice, unit, and representation.

Names with no side, such as `joint1`, `joint_1`, or `j1`, are blocked before
network access.  Non-contiguous indices, side conflicts, missing names,
action/qpos disagreement, or incomplete VLM confirmation also block YAML
generation.

The parser also recognizes the existing canonical machine names.  A full
position-plus-Euler pose segment becomes `arm.<side>.eef`; a scalar confirmed
by the PDF gripper range rule becomes `gripper.<side>`.

On refusal, `info_norm_review.json` receives an
`ANNOTATION_JOINT_AMBIGUOUS` or related annotation issue.  Its evidence names
the failing input file (`meta/info.json`), feature, half-open source slice,
observed names, and a remediation hint.  No separate error artifact is
created.

## Output and Validation

The compiler runs after standard normalization and only receives a `PASS`
result.  It validates exact YAML keys, PDF camera names, channel keys, slice
bounds, qpos/action semantic equivalence, and permitted gripper ranges.
Serialization is deterministic and re-parsed before writing.  The writer
stages and fsyncs all output files; any annotation validation or write failure
leaves the dataset without `robo_annotation.yaml` and reports an error result.

## Example Consequence

The sandwich dataset's `left_joint_1...` and `right_joint_1...` names are not
rejected merely for using ordinals: their explicit sides, contiguous indices,
pose segments, grippers, action/qpos agreement, and hardware mapping can
confirm the two-arm layout.  A no-side `joint_1...joint_6` layout remains
unusable.
