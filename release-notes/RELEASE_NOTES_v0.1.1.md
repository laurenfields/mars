# Mars v0.1.1 Release Notes

## Overview

Patch release v0.1.1 focuses on code quality improvements, fixing reported linting issues, and minor bug fixes in the command-line interface.

## Changes

### CLI Improvements
- Added missing `--max-isolation-window` parameter to the `mars qc` command, allowing consistent filtering between calibration and QC steps.

### Code Quality
- Addressed multiple linting issues identified by `ruff`.
- Updated obsolete string formatting to use modern f-strings in `mzml.py`.
- Fixed potential logic errors by ensuring strict iterables in `zip()` calls.
- Removed unused variables and unnecessary function arguments.
- Improved code readability and standard compliance.
