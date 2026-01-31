# Mars v0.1.3 Release Notes

**Release Date:** January 2026

## Overview

This release adds support for DIA-NN parquet library files and fixes critical compatibility issues with mzML output files. The mzML writer has been completely rewritten to use a passthrough approach that preserves all original file metadata, ensuring compatibility with downstream tools like DIA-NN, SeeMS, and MSConvert.

## New Features

### DIA-NN Parquet Library Support

Mars now supports loading spectral libraries directly from DIA-NN parquet output files as an alternative to blib or PRISM CSV formats.

**Usage:**

```bash
mars calibrate --mzml input.mzML --library report-lib.parquet --output calibrated.mzML
```

**How it works:**

- **Library file (`report-lib.parquet`)**: Contains fragment ion information (m/z, ion types, charges) used for matching
- **Report file (`report.parquet`)**: Contains per-file retention time windows (RT.Start, RT.Stop) for each precursor
- The report file is automatically detected in the same directory as the library file
- If `report.parquet` is not found, Mars will exit with an error

**Optional filtering:**

Use the `--diann-report` option to specify a different report file location:

```bash
mars calibrate --mzml input.mzML --library report-lib.parquet --diann-report /path/to/report.parquet
```

**File type detection:**

Mars automatically detects if you accidentally provide the wrong file type (e.g., `report.parquet` instead of `report-lib.parquet`) based on column content, not filename. You'll receive a helpful error message pointing to the correct file.

## Bug Fixes

### Fixed: DIA-NN Compatibility

The previous psims-based writer generated mzML files that DIA-NN could not read. The issue was caused by differences in:

- **CV reference IDs**: psims uses `cvRef="PSI-MS"` while ProteoWizard uses `cvRef="MS"`
- **Missing metadata**: Thermo nativeID format, instrument configuration, and other CV terms were not preserved
- **Altered file structure**: The psims writer generated a different XML structure than the original

The new passthrough writer preserves the original file structure byte-for-byte, only modifying the m/z binary data for MS2 spectra.

### Fixed: SeeMS Metadata Display

SeeMS now correctly displays spectrum metadata in separate columns (Controllertype, Controllernumber, Scan) instead of a combined ID field. This is because the original `Thermo nativeID format` CV term is now preserved.

### Fixed: Broken mzML Output Files

The original lxml-based writer caused several issues:

- **Invalid index offsets**: The `<indexList>` section contained stale byte offsets after XML rewriting
- **XML formatting changes**: Attribute reordering and whitespace changes could break strict parsers

## Changes

### New Passthrough mzML Writer

The new writer uses a fundamentally different approach:

1. **Preserves original file exactly** - All metadata, CV terms, XML structure, and formatting are kept unchanged
2. **Only modifies m/z binary data** - For MS2 spectra, the m/z array is decoded, calibrated, and re-encoded
3. **Regenerates index** - Byte offsets are recalculated after m/z data changes
4. **Regenerates checksum** - The SHA-1 file checksum is updated

This ensures maximum compatibility with all downstream tools.

### Wide-Window MS2 Spectra Handling

- When `--max-isolation-window` is specified, MS2 spectra exceeding that width are left **unchanged** (not calibrated)
- The spectra remain in the output file but with original m/z values
- This differs from the previous behavior where wide-window spectra were excluded entirely

### Dependencies

- Added `lxml` for XML parsing and serialization
- `psims` is no longer used for mzML writing (still available as a dependency)

## Technical Details

The new writer workflow:

1. Reads the original mzML file as raw bytes
2. Parses the mzML content with lxml while preserving structure
3. Reads spectrum metadata with pyteomics for calibration calculations
4. For each MS2 spectrum:
   - Decodes the m/z binary array (base64 + zlib)
   - Applies calibration function
   - Re-encodes with same compression settings
   - Updates the `encodedLength` attribute
5. MS1 spectra and chromatograms remain completely unchanged
6. Regenerates the index with correct byte offsets
7. Recalculates the SHA-1 file checksum

### Preserved Metadata

The following are now correctly preserved from the original file:

- Thermo nativeID format CV term
- Thermo RAW format CV term
- Source file references (RAW file path, SHA-1)
- Instrument configuration (Stellar, serial number)
- Sample information
- All spectrum CV parameters (base peak, TIC, filter string, etc.)
- All chromatograms (TIC, pump pressure, etc.)
- XML namespaces and schema locations

## Compatibility

- Fully backward compatible with v0.1.2
- Supported spectral library formats:
  - blib (BiblioSpec)
  - PRISM CSV
  - DIA-NN parquet (`report-lib.parquet` + `report.parquet`) **NEW**
- Output mzML files are compatible with:
  - DIA-NN
  - SeeMS (ProteoWizard)
  - MSConvert
  - Skyline
  - Other standard mzML readers
- The calibration model format is unchanged

## Upgrade Notes

No action required. Simply update to v0.1.3 and re-run calibration to generate valid mzML files.

```bash
pip install --upgrade mars-ms
```
