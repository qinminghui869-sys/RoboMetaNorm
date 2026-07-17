# Side-Neutral Single-Arm Annotation Design

## Goal

Allow a confirmed single-arm dataset with `main_follower_joint_*` fields to
produce semantic `arm.main.*` and `gripper.main` channels without weakening the
rejection of ordinary side-less joint names.

## Candidate Detection

The local preflight treats a machine vector as a deferred single-arm candidate
only when both `action` and `observation.state` have the same ordered,
non-interleaved `main_follower_joint_<index>` sequence. Indices must be
contiguous and start at zero or one. A malformed, gapped, interleaved, or
mismatched sequence remains a blocking annotation issue with source evidence.

All other side-less numbered joint labels remain blocked before any VLM request.
The candidate itself is not yet trusted; it only permits the existing hardware
research and whole-dataset mapping stages to run.

## Confirmation and Output

The compiler emits side-neutral channels only when all of the following hold:

1. `info.json` has a safe non-empty `robot_type`.
2. The VLM hardware profile has a confident, unambiguous identity and exactly
   one confirmed arm-joint component.
3. The VLM mapping completely and consistently covers `action` and
   `observation.state` with that one arm and any confirmed end-effector and
   gripper components.

The normalizer may internally use the profile component side to validate the
mapping, but the annotation output replaces that implementation detail with:

- `arm.main.joint`
- `arm.main.eef`, when position and rotation are both confirmed
- `gripper.main`, when the gripper is confirmed

Dual-arm datasets retain `arm.left/right.*` and `gripper.left/right`. Cameras
keep the existing PDF-conformant `observation.images.cam_<position>_<modality>`
keys and their original source feature values.

## Failure Behavior

Candidate data that fails single-arm confirmation writes no YAML. The existing
review output records the reason, while generic or malformed side-less joint
layouts remain blocked before VLM use. Only a final `PASS` writes
`meta/robo_annotation.yaml`.

## Verification

Tests cover a confirmed `main_follower` single arm, a dual-arm profile rejected
for that candidate, a malformed candidate blocked before VLM, unchanged generic
joint rejection, and PASS-only YAML writing with `arm.main.*` keys.
