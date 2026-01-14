# Mars v0.1.2 Release Notes

**Release Date:** January 2026

## Overview

This release adds support for high-resolution Orbitrap/Astral analyzer data with PPM-based matching and visualization, plus major performance improvements for large PRISM libraries. Mars can now handle both Stellar Ion Trap data (Th-scale errors) and Astral analyzer data (ppm-scale errors) with automatic detection.

## New Features

### PPM Tolerance Support
- **New `--tolerance-ppm` CLI option** for fragment matching in ppm (e.g., `--tolerance-ppm 10` for ±10 ppm)
- When specified, overrides the default `--tolerance` (Th) parameter
- PPM tolerance scales dynamically with m/z, appropriate for high-resolution Orbitrap data

### Delta PPM Metrics
- All match DataFrames now include both `delta_mz` (Th) and `delta_ppm` columns
- After calibration, `delta_ppm_calibrated` is computed alongside `delta_mz_calibrated`
- Logging shows statistics in both units for easier comparison

### Adaptive QC Visualization
- **Auto-detection of ppm vs Th mode** based on MAD (Median Absolute Deviation):
  - If MAD < 0.05 Th → ppm mode (high-resolution data)
  - If MAD ≥ 0.05 Th → Th mode (unit-resolution data)
- All hexbin QC plots updated with `use_ppm` parameter:
  - Histogram
  - Heatmap (RT × fragment m/z)
  - Intensity vs error
  - RT vs error
  - Fragment m/z vs error
  - TIC vs error
  - Injection time vs error
  - TIC×Injection time vs error
  - Fragment ions vs error
  - Temperature vs error
  - Adjacent ion feature plots
- Y-axis limits automatically adjust:
  - ppm mode: ±25 ppm
  - Th mode: ±0.25 Th

## Performance Improvements

### Automatic Replicate Filtering
- PRISM library loading now automatically filters to only the replicates matching the mzML files being processed
- Previously, large multi-replicate PRISM exports (e.g., 67M rows) would load entirely; now only relevant rows are processed
- This dramatically reduces load time and memory usage for large studies

### Optimized PRISM Library Loading
- **Column-selective loading**: Only loads required columns, reducing I/O and memory
- **Vectorized replicate filtering**: Uses pandas string methods instead of row-by-row apply()
- **Vectorized fragment parsing**: Ion type, number, and loss type parsed in bulk
- **Faster iteration**: Uses `itertuples()` instead of `iterrows()` (5-10x faster)
- **Progress logging**: Reports progress every 50,000 peptides for large libraries

### Dependabot Integration
- Added `.github/dependabot.yml` for automated dependency updates
- Monitors both Python (pip) and GitHub Actions dependencies weekly

## Bug Fixes

- Fixed auto-detection logic to use proper MAD (Median Absolute Deviation) instead of median(|delta_mz|) for determining visualization mode
- Fixed `DtypeWarning` when loading large PRISM CSVs with mixed column types
- Hexbin plots now use linear color scale (except injection time vs error which uses log scale)

## Usage Examples

### Stellar Ion Trap (Th-based, default)
```bash
mars calibrate \
  --mzml data.mzML \
  --prism-csv report.csv \
  --tolerance 0.3 \
  --output-dir output/
```

### Astral Analyzer (ppm-based)
```bash
mars calibrate \
  --mzml data.mzML \
  --prism-csv report.csv \
  --tolerance-ppm 10 \
  --output-dir output/
```

### Large Multi-Replicate Studies
```bash
# Mars automatically filters the PRISM library to only the 3 files being processed
mars calibrate \
  --mzml "plasma_samples/*.mzML" \
  --prism-csv full_study_prism_export.csv \
  --tolerance-ppm 10 \
  --output-dir output/
```

## Technical Notes

- The model still trains on `delta_mz` (Th) internally, as the XGBoost model works in absolute units
- PPM conversion is applied at the matching and visualization stages
- Temperature-based features remain relevant for Stellar data but are typically not present in Astral mzML files

## Compatibility

- Fully backward compatible with v0.1.1
- Existing workflows using `--tolerance` (Th) continue to work unchanged
- New Astral/Orbitrap workflows can use `--tolerance-ppm`
