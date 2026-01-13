#!/usr/bin/env python3
"""Check all attributes of mzML reader."""

from pyteomics import mzml

mzml_file = "example-data/Ste-2024-12-02_HeLa_20msIIT_GPFDIA_500-600_15.mzML"

print(f"\nChecking mzML reader attributes:\n")

with mzml.MzML(mzml_file) as reader:
    # List all non-private attributes
    attrs = [a for a in dir(reader) if not a.startswith('_')]
    print(f"Public attributes: {attrs}\n")
    
    # Check each one
    for attr in attrs:
        try:
            val = getattr(reader, attr)
            if not callable(val):
                print(f"{attr}: {type(val).__name__} = {str(val)[:100]}")
        except Exception as e:
            print(f"{attr}: Error - {e}")
    
    # Try to get the info via the reader
    print("\n\nTrying to read metadata via reader methods...")
    if hasattr(reader, 'get_info'):
        info = reader.get_info()
        print(f"get_info() returned: {info}")

print("\nDone.\n")
