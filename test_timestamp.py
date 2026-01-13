#!/usr/bin/env python3
"""Quick test to extract startTimeStamp from mzML."""

from pyteomics import mzml
from datetime import datetime

mzml_file = "example-data/Ste-2024-12-02_HeLa_20msIIT_GPFDIA_500-600_15.mzML"

print(f"\nTesting timestamp extraction from: {mzml_file}\n")

with mzml.MzML(mzml_file) as reader:
    print("Available reader attributes:")
    for attr in ['info', 'metadata', '_run_info']:
        if hasattr(reader, attr):
            val = getattr(reader, attr)
            print(f"  {attr}: {type(val).__name__}")
            if isinstance(val, dict):
                print(f"    Keys: {list(val.keys())}")
                if 'startTimeStamp' in val:
                    print(f"    startTimeStamp: {val['startTimeStamp']}")
    
    # Try to access via iteration
    print("\nChecking reader._run_info directly...")
    if hasattr(reader, '_run_info'):
        print(f"  _run_info type: {type(reader._run_info)}")
        if isinstance(reader._run_info, dict):
            print(f"  _run_info keys: {list(reader._run_info.keys())}")
            if 'startTimeStamp' in reader._run_info:
                timestamp = reader._run_info['startTimeStamp']
                print(f"  Found startTimeStamp: {timestamp}")
                
                # Parse it
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                unix_ts = dt.timestamp()
                print(f"  Parsed as Unix timestamp: {unix_ts}")

print("\nDone.\n")
