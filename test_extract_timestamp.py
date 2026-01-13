#!/usr/bin/env python3
"""Extract startTimeStamp from mzML using XML parsing."""

from pyteomics import mzml
import xml.etree.ElementTree as ET

mzml_file = "example-data/Ste-2024-12-02_HeLa_20msIIT_GPFDIA_500-600_15.mzML"

print(f"\nExtracting startTimeStamp from: {mzml_file}\n")

# Method 1: Direct XML parsing (read just the beginning)
print("Method 1: Direct XML parsing...")
with open(mzml_file, 'r') as f:
    # Read first 50 lines to find the run element
    for i, line in enumerate(f):
        if 'startTimeStamp' in line:
            print(f"  Found at line {i+1}: {line.strip()}")
            # Extract the timestamp
            import re
            match = re.search(r'startTimeStamp="([^"]+)"', line)
            if match:
                timestamp = match.group(1)
                print(f"  Extracted: {timestamp}")
                
                # Parse it
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                unix_ts = dt.timestamp()
                print(f"  Unix timestamp: {unix_ts}")
            break

# Method 2: Check pyteomics MzML reader internal state
print("\nMethod 2: Checking reader internals...")
with mzml.MzML(mzml_file) as reader:
    # Access the internal XML reader
    if hasattr(reader, 'source'):
        print(f"  reader.source: {type(reader.source)}")
    
    # Try to get the tree
    try:
        tree = reader.build_tree()
        root = tree.getroot()
        print(f"  Root tag: {root.tag}")
        
        # Find the run element
        ns = {'mzML': 'http://psi.hupo.org/ms/mzml'}
        run = root.find('.//mzML:run', ns)
        if run is not None:
            timestamp = run.get('startTimeStamp')
            print(f"  Found startTimeStamp in run element: {timestamp}")
    except Exception as e:
        print(f"  Error building tree: {e}")

print("\nDone.\n")
