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

### Faster mzML Calibration
- **Vectorized space charge feature computation**: The `_compute_ions_in_range_vectorized` function now uses fully vectorized NumPy operations instead of a Python for-loop
- Uses `np.searchsorted` with array inputs to compute all intensity range sums simultaneously
- Significantly improves calibration speed when writing large mzML files with many spectra

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

---

## Space Charge Modeling Improvements

### Ions Below Features

Added 6 new features to capture space charge effects from ions below the fragment m/z:

- `ions_below_0_1` - Total ions in (X-1.5, X-0.5] Th range below fragment m/z
- `ions_below_1_2` - Total ions in (X-2.5, X-1.5] Th range below fragment m/z
- `ions_below_2_3` - Total ions in (X-3.5, X-2.5] Th range below fragment m/z
- `adjacent_ratio_below_0_1` - ions_below_0_1 / fragment_ions
- `adjacent_ratio_below_1_2` - ions_below_1_2 / fragment_ions
- `adjacent_ratio_below_2_3` - ions_below_2_3 / fragment_ions

These complement the existing "ions above" features and provide a more complete picture of the local ion environment affecting each fragment.

### New QC Plots

Added 6 new QC visualization plots for the ions below features:

- `mars_qc_ions_below_-1.5_-0.5_vs_error.png`
- `mars_qc_ions_below_-2.5_-1.5_vs_error.png`
- `mars_qc_ions_below_-3.5_-2.5_vs_error.png`
- `mars_qc_adjacent_ratio_-1.5_-0.5_vs_error.png`
- `mars_qc_adjacent_ratio_-2.5_-1.5_vs_error.png`
- `mars_qc_adjacent_ratio_-3.5_-2.5_vs_error.png`

### Isotope-Centered m/z Bins

The space charge feature bins have been shifted by 0.5 Th to better center on isotope patterns for 1+ charge fragments:

| Old Range | New Range | Description |
|-----------|-----------|-------------|
| (X, X+1] | (X+0.5, X+1.5] | First isotope region above |
| (X+1, X+2] | (X+1.5, X+2.5] | Second isotope region above |
| (X+2, X+3] | (X+2.5, X+3.5] | Third isotope region above |

The bins now reference the monoisotopic fragment m/z (expected m/z from library) rather than the observed m/z for more consistent feature calculation.

### Space Charge Features Applied During Calibration

Previously, space charge features (`ions_above_*`, `adjacent_ratio_*`) were only computed during model training but set to zero during mzML calibration. Now all 12 space charge features are computed for every peak when writing calibrated mzML files, ensuring the model's learned corrections are fully applied.

### Updated Feature Display Names

Feature importance plots and QC plot filenames now use the new bin naming convention:

- `ions_above_0.5_1.5` (was `ions_above_0_1`)
- `ions_above_1.5_2.5` (was `ions_above_1_2`)
- `ions_above_2.5_3.5` (was `ions_above_2_3`)
- `adjacent_ratio_0.5_1.5` (was `adjacent_ratio_0_1`)
- etc.

### Model Features

The model now supports up to 22 features (when all data is available):

1. `precursor_mz` - DIA isolation window center
2. `fragment_mz` - Fragment ion m/z
3. `log_tic` - Log10 of total ion current
4. `log_intensity` - Log10 of observed peak intensity
5. `absolute_time` - Unix timestamp of acquisition
6. `injection_time` - Ion injection time (seconds)
7. `tic_injection_time` - TIC x injection time product
8. `fragment_ions` - Fragment intensity x injection time
9. `ions_above_0_1`, `ions_above_1_2`, `ions_above_2_3` - Ions in ranges above
10. `ions_below_0_1`, `ions_below_1_2`, `ions_below_2_3` - Ions in ranges below
11. `adjacent_ratio_0_1`, `adjacent_ratio_1_2`, `adjacent_ratio_2_3` - Ratios above
12. `adjacent_ratio_below_0_1`, `adjacent_ratio_below_1_2`, `adjacent_ratio_below_2_3` - Ratios below
13. `rfa2_temp` - RF amplifier temperature
14. `rfc2_temp` - RF electronics temperature
