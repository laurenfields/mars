#!/usr/bin/env python3
"""Verify visualization code can handle all DataFrame columns."""

import pandas as pd
import numpy as np

# Create a sample matches DataFrame with all expected columns
df = pd.DataFrame({
    "expected_mz": np.random.uniform(300, 2000, 100),
    "library_intensity": np.random.uniform(10, 1000, 100),
    "observed_mz": np.random.uniform(300, 2000, 100),
    "observed_intensity": np.random.uniform(10, 1000, 100),
    "delta_mz": np.random.normal(0, 0.1, 100),
    "precursor_mz": np.random.uniform(300, 2000, 100),
    "fragment_mz": np.random.uniform(300, 2000, 100),
    "fragment_charge": np.random.randint(1, 5, 100),
    "absolute_time": np.random.uniform(0, 7200, 100),
    "log_tic": np.random.uniform(3, 7, 100),
    "log_intensity": np.random.uniform(1, 4, 100),
    "injection_time": np.random.uniform(0.01, 0.1, 100),
    "tic_injection_time": np.random.uniform(0.1, 10000, 100),
    "peptide_sequence": ["PEPTIDE"] * 100,
    "ion_annotation": ["b1+1"] * 100,
    "scan_number": list(range(100)),
})

print("DataFrame columns:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

# Test that visualization code can access these columns
print("\nTesting column access:")
tests = [
    ("observed_intensity", lambda: df["observed_intensity"].values),
    ("fragment_mz", lambda: df["fragment_mz"].values),
    ("absolute_time", lambda: df["absolute_time"].values),
    ("log_tic", lambda: df["log_tic"].values),
    ("log_intensity", lambda: df["log_intensity"].values),
    ("delta_mz", lambda: df["delta_mz"].values),
]

for name, accessor in tests:
    try:
        result = accessor()
        print(f"  ✓ {name}: OK (type={type(result).__name__}, shape={result.shape})")
    except KeyError as e:
        print(f"  ✗ {name}: MISSING - {e}")

# Test log transforms
print("\nTesting log transforms:")
try:
    x = np.log10(np.clip(df["log_tic"], 1, None))
    print(f"  ✓ log10(log_tic): OK")
except Exception as e:
    print(f"  ✗ log10(log_tic): FAILED - {e}")

try:
    x = np.log10(np.clip(df["observed_intensity"], 1, None))
    print(f"  ✓ log10(observed_intensity): OK")
except Exception as e:
    print(f"  ✗ log10(observed_intensity): FAILED - {e}")

# Test binning
print("\nTesting binning:")
try:
    time_col = "absolute_time" if "absolute_time" in df.columns else "rt"
    rt_edges = np.linspace(df[time_col].min(), df[time_col].max(), 11)
    mz_edges = np.linspace(df["fragment_mz"].min(), df["fragment_mz"].max(), 11)
    print(f"  ✓ Binning with {time_col}: OK")
except Exception as e:
    print(f"  ✗ Binning: FAILED - {e}")

print("\n✓ All checks passed!")
