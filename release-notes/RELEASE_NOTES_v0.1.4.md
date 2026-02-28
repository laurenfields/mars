# Mars v0.1.4 Release Notes

**Release Date:** TBD

## Overview

This release includes bug fixes and improvements.

## New Features

- **`--mzML` flag alias**: The `--mzml` option now also accepts `--mzML` (matching the file extension casing) across all commands (`calibrate`, `qc`, `apply`).
- **Unquoted wildcards**: Shell-expanded wildcards now work without quotes. For example, `mars calibrate --mzml *.mzML` works the same as `mars calibrate --mzml "*.mzML"`.
- **Positional file arguments**: mzML files can now be passed as positional arguments without the `--mzml` flag, e.g., `mars calibrate *.mzML --prism-csv report.csv`.
- **Repeatable `--mzml`**: The `--mzml` option can be specified multiple times to pass individual files, e.g., `--mzml a.mzML --mzml b.mzML`.

## Bug Fixes

(No bug fixes yet)

## Changes

- The `--mzml` option now uses `multiple=True` internally, accepting one or more values.
- All three subcommands (`calibrate`, `qc`, `apply`) accept a trailing `[INPUT_FILES]...` positional argument for mzML file paths.

## Compatibility

- Fully backward compatible with v0.1.3
- Supported spectral library formats:
  - blib (BiblioSpec)
  - PRISM CSV
  - DIA-NN parquet (`report-lib.parquet` + `report.parquet`)
- Output mzML files are compatible with:
  - DIA-NN
  - SeeMS (ProteoWizard)
  - MSConvert
  - Skyline
  - Other standard mzML readers

## Upgrade Notes

```bash
pip install --upgrade mars-ms
```
