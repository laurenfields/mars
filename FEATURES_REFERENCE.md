# Quick Reference: New Features Usage

## Overview of Changes
- **Re-added precursor_mz** to the model
- **Added 3 new features**: ion injection_time, tic_injection_time, absolute_time (replaces RT)
- **Total features**: 7-8 (depends on injection_time availability)

## Data Flow

### 1. Reading mzML Files
```python
from mars.mzml import read_dia_spectra

# Read spectra - extracts injection_time and absolute_time automatically
spectra = list(read_dia_spectra("data.mzML"))

for spectrum in spectra:
    print(f"RT: {spectrum.rt} min")
    print(f"Injection time: {spectrum.injection_time} seconds")  # May be None
    print(f"Absolute time: {spectrum.absolute_time} seconds")  # May be None
    print(f"Acquisition start: {spectrum.acquisition_start_time}")  # Unix timestamp
```

### 2. Matching & Feature Computation
```python
from mars.matching import match_library_to_spectra

# Matches automatically compute:
# - log_tic (log10 of spectrum TIC)
# - log_intensity (log10 of peak intensity)  
# - tic_injection_time (tic × injection_time, if available)
# - absolute_time (normalized to earliest file)
matches_df = match_library_to_spectra(library, spectra)

# DataFrame columns now include:
print(matches_df.columns)
# [..., 'precursor_mz', 'fragment_mz', 'absolute_time', 
#  'log_tic', 'log_intensity', 'injection_time', 'tic_injection_time', ...]
```

### 3. Model Training
```python
from mars.calibration import MzCalibrator

calibrator = MzCalibrator()

# Training automatically:
# - Detects which features are available
# - Drops rows with missing injection_time (if sparse)
# - Removes injection_time features entirely (if universally missing)
# - Logs actual features used
calibrator.fit(matches_df)

# Check which features were actually used:
print(f"Features used: {calibrator.feature_names}")
# Output may be: ['precursor_mz', 'fragment_mz', 'absolute_time', 'log_tic', 'log_intensity']
# if injection_time wasn't available, or up to 8 if it was

# View feature importance for actual features used
print(calibrator.training_stats['feature_importance'])
```

### 4. Prediction & Calibration
```python
# Predict using new interface
corrections = calibrator.predict(matches=matches_df)

# Create calibration function for write_calibrated_mzml
cal_func = calibrator.create_calibration_function()

# Use metadata with new features
metadata = {
    "precursor_mz": 500.0,
    "tic": 1e7,
    "absolute_time": 100.0,  # seconds from earliest file
    "injection_time": 0.05,   # seconds
    # Will automatically compute: tic_injection_time = tic × injection_time
}

calibrated_mz = cal_func(metadata, mz_array, intensity_array)
```

## Handling Missing injection_time

### Scenario 1: Data HAS injection_time
```
✓ injection_time feature included in model
✓ tic_injection_time feature included
✓ Rows with any missing values are automatically dropped
✓ Model trains on complete data only
```

### Scenario 2: Data LACKS injection_time
```
✓ Features automatically excluded (no errors)
✓ Model trains with 6 features: [precursor_mz, fragment_mz, 
                                  absolute_time, log_tic, log_intensity]
✓ Prediction works on any data
```

## Key Differences from Previous Version

| Feature | Old | New | Notes |
|---------|-----|-----|-------|
| precursor_mz | ❌ Removed | ✅ Re-added | Now included in model |
| fragment_mz | ✅ | ✅ | Unchanged |
| RT | ✅ Used directly | ✅ → absolute_time | Normalized to earliest file |
| TIC | ✅ As raw value | ✅ log_tic | Log-transformed |
| Intensity | ✅ Implicit | ✅ log_intensity | Now explicit, log-transformed |
| injection_time | ❌ | ✅ Optional | From mzML ion injection time |
| tic_injection_time | ❌ | ✅ Optional | Ion count in spectrum |

## Time Units

| Field | Unit | Notes |
|-------|------|-------|
| rt | Minutes | Original retention time (unchanged) |
| acquisition_start_time | Unix seconds | Wall-clock time from file metadata |
| absolute_time | Seconds (0.001 precision) | Seconds from earliest file start |
| injection_time | Seconds | Converted from milliseconds in mzML |

## Important Notes

1. **Precision**: Millisecond precision (0.001 s) for absolute_time to handle spectra separated by ms
2. **Normalization**: All absolute_time values are normalized to the earliest file's acquisition time
3. **Missing Values**: Intelligently handled - features excluded if universally missing, rows dropped if sparse
4. **Backward Compatibility**: Old models (4 features) will NOT work with new code - must retrain
5. **DataFrame Interface**: New `predict(matches=df)` is preferred over old array interface

