#!/usr/bin/env python3
"""
Summary of all features and data columns in MARS calibration pipeline.

This document tracks all the features used in the XGBoost model
and all the columns in the matches DataFrame.
"""

print("""
=== MARS Feature and Column Summary ===

MODEL FEATURES (7 total):
1. precursor_mz      - DIA isolation window center m/z
2. fragment_mz       - Theoretical fragment m/z
3. log_tic           - Log10(spectrum TIC)
4. log_intensity     - Log10(observed peak intensity)
5. absolute_time     - Absolute acquisition time (seconds from run start)
6. injection_time    - Ion injection time (seconds)
7. tic_injection_time- TIC × injection_time product

MATCHES DATAFRAME COLUMNS:
Required for features:
- precursor_mz       ✓ from spectrum.precursor_mz_center
- fragment_mz        ✓ from fragment.mz
- log_tic            ✓ calculated from spectrum.tic
- log_intensity      ✓ calculated from observed_intensity
- absolute_time      ✓ calculated from spectrum.absolute_time
- injection_time     ✓ from spectrum.injection_time
- tic_injection_time ✓ calculated from spectrum.tic × injection_time

Additional columns in DataFrame:
- expected_mz        ✓ from fragment.mz
- library_intensity  ✓ from fragment.intensity
- observed_mz        ✓ from peak search
- observed_intensity ✓ from peak search
- delta_mz           ✓ observed_mz - expected_mz (training target)
- fragment_charge    ✓ from fragment.charge
- peptide_sequence   ✓ from entry.modified_sequence
- ion_annotation     ✓ constructed label
- scan_number        ✓ from spectrum.scan_number

OUTPUT COLUMNS (added after calibration):
- delta_mz_calibrated ✓ delta_mz - corrections
- mz_correction      ✓ predicted correction value

VISUALIZATION USAGE:
- Histogram: uses delta_mz (before), delta_mz_calibrated (after)
- Heatmap: uses fragment_mz (y-axis), absolute_time (x-axis), delta_mz
- Intensity vs Error: uses observed_intensity (x-axis), delta_mz (y-axis)
- RT vs Error: uses absolute_time (x-axis), delta_mz (y-axis)
- Fragment m/z vs Error: uses fragment_mz (x-axis), delta_mz (y-axis)
- TIC vs Error: uses log_tic (x-axis), delta_mz (y-axis)

FEATURE AVAILABILITY HANDLING:
- absolute_time: Always available (RT × 60 fallback if no acquisition timestamp)
- injection_time: Optional (excluded from model if missing)
- tic_injection_time: Optional (excluded from model if missing)
- log_tic, log_intensity: Always available (calculated from required fields)

""")
