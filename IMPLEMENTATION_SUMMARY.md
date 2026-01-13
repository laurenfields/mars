# Implementation Summary: Re-add precursor_mz & Add Time/Injection Features

## Overview
Successfully implemented all planned features to add time-based and injection time features to the MARS calibration model. The model now supports up to 8 features with automatic handling of missing injection_time data.

## Changes Made

### 1. **mars/mzml.py** - New Data Extraction
- **Updated DIASpectrum dataclass** (lines 20-35):
  - Added `injection_time: float | None` - Ion injection time in seconds
  - Added `acquisition_start_time: float | None` - File acquisition start timestamp (Unix)
  - Added `absolute_time: float | None` - Normalized absolute time in seconds

- **New helper functions**:
  - `_extract_injection_time(spectrum)` - Extracts ion injection time from mzML, converts ms→seconds
  - `_parse_iso8601_timestamp(timestamp_str)` - Parses ISO 8601 timestamps to Unix time

- **Enhanced read_dia_spectra()** function:
  - Now extracts file acquisition start time from mzML metadata
  - Calculates absolute_time as: `acquisition_start_time + (rt * 60)`
  - Supports optional `min_absolute_time` parameter for normalization
  - Gracefully handles missing injection_time data

### 2. **mars/matching.py** - Feature Calculation & Storage
- **Updated FragmentMatch dataclass** (lines 25-47):
  - Replaced `rt` with `absolute_time: float | None`
  - Replaced `tic` with `log_tic: float` (log-transformed spectrum TIC)
  - Replaced implicit intensity handling with `log_intensity: float` (log-transformed peak intensity)
  - Added `injection_time: float | None` (optional)
  - Added `tic_injection_time: float | None` (optional, calculated as tic × injection_time)
  - Added `precursor_mz` (re-added feature)

- **Enhanced match_library_to_spectra()** function:
  - First pass collects all spectra to find minimum absolute_time
  - All absolute_time values normalized relative to earliest file acquisition time
  - Calculates `log_tic` and `log_intensity` at match creation time
  - Computes `tic_injection_time` only if `injection_time` is available
  - Handles sparse injection_time (missing in some but not all spectra)

### 3. **mars/calibration.py** - Model Features & Missing Data Handling
- **Updated MzCalibrator feature set**:
  - New feature_names (up to 8): `["precursor_mz", "fragment_mz", "absolute_time", "log_tic", "log_intensity", "injection_time", "tic_injection_time", ...]`
  - Replaces old 4-feature model

- **Completely refactored _prepare_features()** method:
  - Changed signature from individual arrays to DataFrame input
  - Implements smart missing data handling:
    - **Universally missing** features: Automatically removed from model
    - **Sparsely missing** injection_time: Rows with NaN values dropped during training
  - Returns tuple of (feature_matrix, active_feature_names) for dynamic feature adaptation
  - Updates `self.feature_names` based on actual available data

- **Updated fit()** method:
  - Now passes entire DataFrame to _prepare_features()
  - Feature importance logging automatically matches active features used
  - Handles variable feature count based on data availability

- **Updated predict()** method:
  - Now accepts DataFrame interface: `predict(matches=df)`
  - Builds feature matrix dynamically using active_features
  - More robust than old array-based interface

- **Enhanced create_calibration_function()** method:
  - Updated to construct feature arrays for all active features
  - Handles optional features gracefully
  - Supports metadata dict with any subset of feature values

### 4. **tests/test_calibration.py** - Updated Tests
- Updated sample_matches fixture to include all new features
- Added `tic_injection_time` calculation to test data
- Updated all test methods to use new DataFrame interface
- Added test for missing injection_time removal
- Added test for sparse injection_time row dropping

### 5. **tests/test_mzml.py** - New Tests
Created comprehensive tests for new mzML functionality:
- **TestParseISO8601Timestamp**: Tests timestamp parsing with Z suffix, timezone, invalid, empty
- **TestExtractInjectionTime**: Tests injection time extraction from precursor/scan metadata, missing values
- **TestDIASpectrum**: Tests dataclass with new fields and optional None values

## Key Implementation Details

### Time Scale & Precision
- **Injection_time**: Milliseconds from mzML → Seconds in Python (ms / 1000)
- **Absolute_time**: Seconds from file acquisition start with **millisecond precision (0.001 resolution)**
- **RT (unchanged)**: Minutes, as in original implementation
- **All times normalized** to earliest file acquisition timestamp across all input files

### Missing Data Strategy
```
If injection_time in ALL spectra: EXCLUDED from model
├─ (log message: "injection_time not available in all spectra - skipping...")
└─ tic_injection_time also excluded

If injection_time in SOME spectra: INCLUDED in model
├─ Sparse rows with NaN dropped during training (log: "Dropped N rows with missing injection_time")
├─ tic_injection_time also included (rows with both must be non-NaN)
└─ Ensures model trains only on complete feature sets

If absolute_time is None: EXCLUDED from model
└─ (Occurs when acquisition_start_time cannot be extracted from mzML)
```

### Feature Importance Output
Model now logs feature importance for actual features used, which may differ from initial feature_names if injection_time is unavailable:
```
Feature importance:
  precursor_mz: 0.150
  fragment_mz: 0.400
  absolute_time: 0.120
  log_tic: 0.140
  log_intensity: 0.190
  injection_time: 0.000  (if included)
  tic_injection_time: 0.000  (if included)
```

## Backward Compatibility Notes
- Old pickled models using 4 features will NOT be compatible with new predict() interface
- Models trained with missing injection_time will have fewer features, affecting predictions on data with injection_time
- The DataFrame interface is now preferred; legacy **kwargs interface removed from predict()

## Testing
All modified files pass Python syntax validation:
✓ mars/mzml.py
✓ mars/matching.py  
✓ mars/calibration.py
✓ tests/test_calibration.py (updated)
✓ tests/test_mzml.py (new)

Tests require pytest and dependencies to run, but syntax is correct.

## Next Steps
1. Move test data into place (as mentioned by user)
2. Run full test suite with actual test data
3. Validate feature importance changes with trained models
4. Verify mzML injection_time extraction with actual MS data files
5. Compare model performance: old 4-feature vs new 7-8-feature models
