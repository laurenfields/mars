#!/usr/bin/env python3
"""Quick script to inspect mzML metadata for timestamp information."""

from pyteomics import mzml
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_mzml.py <mzml_file>")
    sys.exit(1)

mzml_file = sys.argv[1]

print(f"\n=== Inspecting {mzml_file} ===\n")

with mzml.MzML(mzml_file) as reader:
    # Check metadata
    print("Reader metadata:")
    if hasattr(reader, 'metadata') and reader.metadata:
        for key, value in reader.metadata.items():
            print(f"  {key}: {value}")
    else:
        print("  No metadata attribute or empty metadata")
    
    print("\nReader attributes:")
    for attr in dir(reader):
        if not attr.startswith('_'):
            try:
                val = getattr(reader, attr)
                if not callable(val):
                    print(f"  {attr}: {type(val).__name__}")
            except:
                pass
    
    # Check first spectrum
    print("\nFirst MS1 spectrum keys:")
    for i, spectrum in enumerate(reader):
        if spectrum.get('ms level', 1) == 1:
            print(f"  Spectrum {i} keys: {list(spectrum.keys())[:20]}")
            
            # Check scanList
            if 'scanList' in spectrum:
                print(f"\n  scanList keys: {list(spectrum['scanList'].keys())}")
                if 'scan' in spectrum['scanList']:
                    scans = spectrum['scanList']['scan']
                    if scans:
                        print(f"  scan[0] keys: {list(scans[0].keys())}")
                        for key, val in scans[0].items():
                            if 'time' in key.lower() or 'date' in key.lower():
                                print(f"    {key}: {val}")
            break
    
    # Check first MS2 spectrum
    print("\nFirst MS2 spectrum keys:")
    reader.reset()
    for i, spectrum in enumerate(reader):
        if spectrum.get('ms level', 1) == 2:
            print(f"  Spectrum {i} keys: {list(spectrum.keys())[:20]}")
            
            # Check scanList
            if 'scanList' in spectrum:
                print(f"\n  scanList keys: {list(spectrum['scanList'].keys())}")
                if 'scan' in spectrum['scanList']:
                    scans = spectrum['scanList']['scan']
                    if scans:
                        print(f"  scan[0] keys: {list(scans[0].keys())}")
                        for key, val in scans[0].items():
                            if 'time' in key.lower() or 'date' in key.lower():
                                print(f"    {key}: {val}")
            break

print("\n=== Done ===\n")
