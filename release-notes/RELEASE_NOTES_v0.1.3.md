# Mars v0.1.3 Release Notes

**Release Date:** January 2026

## Overview

This release fixes a critical issue with mzML output files being unreadable by downstream tools. The mzML writer has been completely rewritten to use the psims library, which produces properly indexed mzML files.

## Bug Fixes

### Fixed: Broken mzML Output Files

The previous mzML writer used lxml to directly modify XML, which caused several issues:

- **Invalid index offsets**: The `<indexList>` section at the end of indexed mzML files contained stale byte offsets after XML rewriting, causing random-access tools to fail
- **XML formatting changes**: Attribute reordering and whitespace changes could break strict parsers

The new implementation uses the [psims](https://github.com/mobiusklein/psims) library, which is the standard tool for writing PSI-MS format files and handles index generation correctly.

### Fixed: Missing Source File ID in mzML Output

The psims library requires an `id` attribute for source file entries in the mzML metadata. Added proper ID generation for source file references.

### Fixed: mzML Schema Compliance Warning

Added missing `instrumentConfigurationList` section to output mzML files. The mzML schema requires this section before `dataProcessingList`, and psims was emitting a `StateTransitionWarning` without it.

## Changes

### New mzML Writer Implementation

- **Uses psims `MzMLWriter`** instead of raw lxml for writing mzML files
- **Produces properly indexed mzML** with correct byte offsets for each spectrum
- **Preserves all spectrum data**:
  - MS1 spectra are written unchanged
  - MS2 spectra have calibrated m/z values; intensity arrays remain unchanged
  - Scan time, TIC, injection time, and precursor/isolation window information preserved
  - Activation parameters (CID/HCD, collision energy) preserved

### Wide-Window MS2 Spectra Now Excluded

- When `--max-isolation-window` is specified, MS2 spectra exceeding that width are now **completely excluded** from the output mzML file
- Previously, these spectra were written unchanged (without calibration)
- This ensures the output file only contains calibrated data matching the training criteria

### New Dependency

- Added `psims>=1.3` to dependencies for mzML writing

## Technical Details

The new writer workflow:

1. Reads all spectra from input mzML using pyteomics
2. Filters out MS2 spectra exceeding `--max-isolation-window` (if specified)
3. For each remaining MS2 spectrum:
   - Extracts metadata (RT, TIC, injection time, temperatures)
   - Applies calibration function to m/z array
   - Writes with calibrated m/z values
4. MS1 spectra are written unchanged
5. psims automatically generates the index with correct byte offsets

### Output mzML Metadata

The output mzML files include:

- Source file reference to the original mzML
- Mars software entry (version 0.1.3)
- Data processing record indicating m/z calibration was applied

## Compatibility

- Fully backward compatible with v0.1.2
- Output mzML files are now compatible with all standard mzML readers
- The calibration model format is unchanged

## Upgrade Notes

No action required. Simply update to v0.1.3 and re-run calibration to generate valid mzML files.

```bash
pip install --upgrade mars-ms
```
