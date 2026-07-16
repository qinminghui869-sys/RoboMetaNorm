# CLI terminal progress design

## Goal

Show concise terminal progress for both `scan` and `normalize` without adding a
runtime dependency or changing the machine-readable summary printed to stdout.

## Behavior

- The pipeline discovers all candidates first, so the renderer knows the total.
- In an interactive terminal, each command updates one stderr line in the form
  `处理中 [3/11] <dataset-name>` after each dataset completes.
- The renderer finishes with one newline. Empty inputs produce no progress line.
- Non-interactive stderr emits no progress, preserving existing redirected and
  test output.
- The final four-column summary remains on stdout exactly as it is today.

## Structure

- `scan_datasets` and `normalize_datasets` accept an optional callback invoked
  once for every completed candidate with its one-based index, total, and
  `DatasetResult`.
- The CLI owns terminal rendering and passes the callback only when
  `sys.stderr.isatty()` is true.
- Progress rendering does not affect processing, result ordering, errors, or
  generated dataset files.

## Validation

- Unit tests prove the callback fires once per candidate in input order for
  scan and normalize.
- CLI tests prove interactive progress is written to stderr, the final summary
  remains stdout-only, and non-interactive execution has no progress output.
