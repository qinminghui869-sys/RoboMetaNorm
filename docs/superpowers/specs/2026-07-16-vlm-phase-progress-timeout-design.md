# VLM phase progress and dataset timeout design

## Goal

Make `normalize` visibly report its VLM phase and bound remote VLM waiting for
each dataset without changing the conservative normalization result contract.

## User-visible behavior

- Interactive terminals show the current dataset and phase on stderr before
  local evidence collection, hardware research, and mapping. VLM phases are
  `查询硬件身份` and `映射相机与关节`.
- `normalize` adds `--dataset-timeout-seconds`, a positive finite number with
  a default of `180` seconds.
- The budget starts when a candidate begins. It limits all VLM HTTP attempts
  and retry backoffs for that candidate. Local filesystem work is not forcibly
  interrupted.
- When the remaining budget is exhausted, the current dataset receives the
  `VLM_DATASET_TIMEOUT` review issue, writes its normal JSON/review outputs,
  does not write YAML, and processing continues with the next dataset.
- `scan`, non-interactive stderr, final stdout summary, and existing per-call
  timeout/retry options retain their current behavior.

## Structure

- `normalize_datasets` receives optional stage-progress and dataset-budget
  parameters. Stage callbacks are display-only: ordinary callback failures are
  ignored, while `MemoryError` and control-flow exceptions propagate.
- The CLI renderer receives both a stage update and a completion update, using
  the same one-line stderr display.
- A monotonic deadline is passed from the pipeline through the dataset VLM
  service into the HTTP transport. Before each attempt and backoff, the
  transport checks remaining time; each HTTP request uses the smaller of the
  configured per-request timeout and the remaining dataset budget.
- Timeout is represented as a structured VLM issue, so normal standard
  application creates source-preserving outputs and a REVIEW result.

## Validation

- Pipeline tests cover ordered stage callbacks, callback-failure isolation, a
  timeout review that continues to the next dataset, and no YAML for timeout.
- Transport tests cover a deadline that prevents a request or retry and emits
  `VLM_DATASET_TIMEOUT`.
- CLI tests cover phase rendering, the 180-second default, custom timeout
  forwarding, non-TTY silence, and unchanged stdout summary.
