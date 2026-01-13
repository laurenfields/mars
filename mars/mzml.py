"""mzML file reading and writing for DIA data.

Uses pyteomics for parsing mzML files and extracting DIA MS2 spectra
with isolation window information.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyteomics import mzml

logger = logging.getLogger(__name__)


@dataclass
class DIASpectrum:
    """Single MS2 spectrum from DIA acquisition."""

    scan_number: int
    rt: float  # Retention time in minutes
    precursor_mz_low: float  # Isolation window lower bound
    precursor_mz_high: float  # Isolation window upper bound
    precursor_mz_center: float  # Isolation window center
    tic: float  # Total ion current
    mz_array: np.ndarray  # Fragment m/z values
    intensity_array: np.ndarray  # Fragment intensities
    injection_time: float | None = None  # Ion injection time in seconds (optional)
    acquisition_start_time: float | None = None  # File acquisition start time (Unix timestamp)
    absolute_time: float | None = None  # Absolute time in seconds (normalized to earliest file)

    @property
    def n_peaks(self) -> int:
        """Number of peaks in spectrum."""
        return len(self.mz_array)


def _extract_isolation_window(spectrum: dict) -> tuple[float, float, float]:
    """Extract isolation window bounds from spectrum metadata.

    Args:
        spectrum: Pyteomics spectrum dict

    Returns:
        Tuple of (low, high, center) m/z values
    """
    try:
        precursor_list = spectrum.get("precursorList", {})
        precursors = precursor_list.get("precursor", [])

        if precursors:
            precursor = precursors[0]
            isolation = precursor.get("isolationWindow", {})

            target = isolation.get("isolation window target m/z", 0.0)
            lower_offset = isolation.get("isolation window lower offset", 0.0)
            upper_offset = isolation.get("isolation window upper offset", 0.0)

            low = target - lower_offset
            high = target + upper_offset

            return float(low), float(high), float(target)

    except Exception as e:
        logger.debug(f"Failed to extract isolation window: {e}")

    # Fallback: try selected ion m/z
    try:
        precursor_list = spectrum.get("precursorList", {})
        precursors = precursor_list.get("precursor", [])
        if precursors:
            selected_ions = precursors[0].get("selectedIonList", {}).get("selectedIon", [])
            if selected_ions:
                mz = selected_ions[0].get("selected ion m/z", 0.0)
                return float(mz) - 0.5, float(mz) + 0.5, float(mz)
    except Exception:
        pass

    return 0.0, 0.0, 0.0


def _extract_scan_number(spectrum: dict) -> int:
    """Extract scan number from spectrum ID.

    Args:
        spectrum: Pyteomics spectrum dict

    Returns:
        Scan number (0 if not found)
    """
    spec_id = spectrum.get("id", "")

    # Try "scan=NNNN" format
    if "scan=" in spec_id:
        try:
            return int(spec_id.split("scan=")[1].split()[0])
        except (ValueError, IndexError):
            pass

    # Try index from spectrum
    return spectrum.get("index", 0)


def _extract_injection_time(spectrum: dict) -> float | None:
    """Extract ion injection time from spectrum metadata.

    Args:
        spectrum: Pyteomics spectrum dict

    Returns:
        Injection time in seconds, or None if not found
    """
    try:
        precursor_list = spectrum.get("precursorList", {})
        precursors = precursor_list.get("precursor", [])

        if precursors:
            precursor = precursors[0]
            # Look for ion injection time in precursor
            injection_time_ms = precursor.get("ion injection time")
            if injection_time_ms is not None:
                # Convert from milliseconds to seconds
                return float(injection_time_ms) / 1000.0

        # Also check in scan level
        scan_list = spectrum.get("scanList", {})
        scans = scan_list.get("scan", [])
        if scans:
            injection_time_ms = scans[0].get("ion injection time")
            if injection_time_ms is not None:
                return float(injection_time_ms) / 1000.0

    except Exception as e:
        logger.debug(f"Failed to extract injection time: {e}")

    return None


def _parse_iso8601_timestamp(timestamp_str: str) -> float | None:
    """Parse ISO 8601 timestamp to Unix timestamp.

    Args:
        timestamp_str: ISO 8601 formatted timestamp string (e.g., "2023-01-15T10:30:45Z")

    Returns:
        Unix timestamp (float) or None if parsing fails
    """
    try:
        from datetime import datetime

        # Try parsing with timezone info
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            # Fallback: try without timezone
            dt = datetime.fromisoformat(timestamp_str)

        # Convert to Unix timestamp
        return float(dt.timestamp())
    except Exception as e:
        logger.debug(f"Failed to parse ISO 8601 timestamp '{timestamp_str}': {e}")
        return None


def read_dia_spectra(
    mzml_path: Path | str,
    ms_level: int = 2,
    min_absolute_time: float | None = None,
) -> Iterator[DIASpectrum]:
    """Stream DIA MS2 spectra from mzML file.

    Args:
        mzml_path: Path to mzML file
        ms_level: MS level to extract (default: 2 for MS2)
        min_absolute_time: Minimum absolute time to use for normalization (seconds).
                          If provided, all absolute_time values will be normalized to this reference.

    Yields:
        DIASpectrum objects for each matching spectrum
    """
    import re

    mzml_path = Path(mzml_path)
    logger.info(f"Reading DIA spectra from {mzml_path}")

    if not mzml_path.exists():
        raise FileNotFoundError(f"mzML file not found: {mzml_path}")

    n_spectra = 0
    n_yielded = 0
    acquisition_start_time = None

    # Extract startTimeStamp from the <run> element before processing spectra
    try:
        with open(mzml_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if "startTimeStamp" in line:
                    match = re.search(r'startTimeStamp="([^"]+)"', line)
                    if match:
                        timestamp_str = match.group(1)
                        acquisition_start_time = _parse_iso8601_timestamp(timestamp_str)
                        if acquisition_start_time is not None:
                            logger.info(f"Found acquisition start time: {timestamp_str}")
                        break
                # Stop searching after a reasonable number of lines (run element should be near the top)
                if line_num > 200:
                    break
    except Exception as e:
        logger.debug(f"Failed to extract startTimeStamp: {e}")

    if acquisition_start_time is None:
        logger.info("No acquisition start time found in mzML, will use RT as absolute time")

    with mzml.MzML(str(mzml_path)) as reader:
        for spectrum in reader:
            n_spectra += 1

            # Filter by MS level
            spec_ms_level = spectrum.get("ms level", 1)
            if spec_ms_level != ms_level:
                continue

            # Extract arrays
            mz_array = spectrum.get("m/z array", np.array([]))
            intensity_array = spectrum.get("intensity array", np.array([]))

            if len(mz_array) == 0:
                continue

            # Extract metadata
            rt = spectrum.get("scanList", {}).get("scan", [{}])[0].get("scan start time", 0.0)
            # Convert to minutes if in seconds
            rt_unit = (
                spectrum.get("scanList", {})
                .get("scan", [{}])[0]
                .get("scan start time", {"unit_info": "minute"})
            )
            if isinstance(rt_unit, dict) and rt_unit.get("unit_info") == "second":
                rt = rt / 60.0

            # Sometimes RT is stored directly
            if rt == 0.0 and "scan start time" in spectrum:
                rt = spectrum["scan start time"]

            # Calculate TIC
            tic = float(np.sum(intensity_array))

            # Extract isolation window
            low, high, center = _extract_isolation_window(spectrum)

            # Get scan number
            scan_number = _extract_scan_number(spectrum)

            # Extract injection time
            injection_time = _extract_injection_time(spectrum)

            # Calculate absolute_time
            # If acquisition_start_time is available, use it; otherwise use RT in seconds
            if acquisition_start_time is not None:
                # Use acquisition start time + RT offset
                absolute_time = acquisition_start_time + (rt * 60.0)
            else:
                # Fallback: use RT converted to seconds
                absolute_time = rt * 60.0

            # Normalize to min_absolute_time if provided
            if min_absolute_time is not None:
                absolute_time = absolute_time - min_absolute_time

            n_yielded += 1
            yield DIASpectrum(
                scan_number=scan_number,
                rt=float(rt),
                precursor_mz_low=low,
                precursor_mz_high=high,
                precursor_mz_center=center,
                tic=tic,
                mz_array=np.asarray(mz_array, dtype=np.float64),
                intensity_array=np.asarray(intensity_array, dtype=np.float64),
                injection_time=injection_time,
                acquisition_start_time=acquisition_start_time,
                absolute_time=absolute_time,
            )

    logger.info(f"Read {n_yielded} MS2 spectra from {n_spectra} total spectra")


def write_calibrated_mzml(
    input_path: Path | str,
    output_path: Path | str,
    calibration_func,
    max_isolation_window_width: float | None = None,
    temperature_data: dict | None = None,
) -> None:
    """Write calibrated mzML file with corrected m/z values.

    Args:
        input_path: Path to input mzML file
        output_path: Path for output mzML file
        calibration_func: Function that takes (spectrum_metadata, mz_array)
                         and returns calibrated mz_array
        max_isolation_window_width: Maximum isolation window width (m/z) to process.
                                    Spectra with wider windows are skipped
        temperature_data: Dict mapping source names (e.g., 'RFA2', 'RFC2') to TemperatureData objects
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Writing calibrated mzML to {output_path}")

    # For now, we'll use a simple approach: read, modify, write
    # A more sophisticated approach would use streaming
    import base64
    import re
    import struct as struct_module
    import zlib as zlib_module

    from lxml import etree

    # Extract startTimeStamp from the file (for absolute_time calculation)
    acquisition_start_time = None
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "<run" in line and "startTimeStamp" in line:
                match = re.search(r'startTimeStamp="([^"]+)"', line)
                if match:
                    acquisition_start_time = _parse_iso8601_timestamp(match.group(1))
                break
            if "</run>" in line or "<spectrumList" in line:
                break

    # Parse XML
    tree = etree.parse(str(input_path))
    root = tree.getroot()

    # Find namespace
    ns = {"ms": root.nsmap.get(None, "http://psi.hupo.org/ms/mzml")}
    ns_ms = ns["ms"]  # Helper for f-strings

    n_calibrated = 0

    # Find all spectrum elements
    for spectrum in root.iter(f"{{{ns_ms}}}spectrum"):
        # Get MS level
        ms_level = 1
        for cv_param in spectrum.iter(f"{{{ns_ms}}}cvParam"):
            if cv_param.get("name") == "ms level":
                ms_level = int(cv_param.get("value", 1))
                break

        if ms_level != 2:
            continue

        # Extract RT, injection time, and isolation window for calibration
        rt = 0.0
        injection_time = 0.0
        precursor_mz = 0.0
        precursor_mz_low = 0.0
        precursor_mz_high = 0.0
        tic = 0.0

        for scan in spectrum.iter(f"{{{ns_ms}}}scan"):
            for cv_param in scan.iter(f"{{{ns_ms}}}cvParam"):
                name = cv_param.get("name", "")
                if name == "scan start time":
                    rt = float(cv_param.get("value", 0.0))
                elif name == "ion injection time":
                    # Convert ms to seconds
                    injection_time = float(cv_param.get("value", 0.0)) / 1000.0

        for isolation in spectrum.iter(f"{{{ns_ms}}}isolationWindow"):
            for cv_param in isolation.iter(f"{{{ns_ms}}}cvParam"):
                name = cv_param.get("name", "")
                if name == "isolation window target m/z":
                    precursor_mz = float(cv_param.get("value", 0.0))
                elif name == "isolation window lower offset":
                    lower_offset = float(cv_param.get("value", 0.0))
                    precursor_mz_low = precursor_mz - lower_offset
                elif name == "isolation window upper offset":
                    upper_offset = float(cv_param.get("value", 0.0))
                    precursor_mz_high = precursor_mz + upper_offset

        # Skip spectra with wide isolation windows if max_isolation_window_width is set
        if (
            max_isolation_window_width is not None
            and precursor_mz_low > 0
            and precursor_mz_high > 0
        ):
            window_width = precursor_mz_high - precursor_mz_low
            if window_width > max_isolation_window_width:
                continue

        for cv_param in spectrum.iter(f"{{{ns_ms}}}cvParam"):
            if cv_param.get("name") == "total ion current":
                tic = float(cv_param.get("value", 0.0))

        # Calculate absolute time (seconds since acquisition start + RT in minutes)
        absolute_time = 0.0
        if acquisition_start_time is not None:
            absolute_time = acquisition_start_time + rt * 60.0

        # Look up temperatures at this RT
        rfa2_temp = 0.0
        rfc2_temp = 0.0
        if temperature_data is not None:
            if "RFA2" in temperature_data:
                rfa2_temp = temperature_data["RFA2"].get_temperature_at_time(rt)
            if "RFC2" in temperature_data:
                rfc2_temp = temperature_data["RFC2"].get_temperature_at_time(rt)

        # Find m/z and intensity binary data arrays
        mz_array = None
        intensity_array = None
        mz_binary_elem = None
        mz_compressed = False
        mz_precision = 64

        for binary_array in spectrum.iter(f"{{{ns_ms}}}binaryDataArray"):
            is_mz_array = False
            is_intensity_array = False

            for cv_param in binary_array.iter(f"{{{ns_ms}}}cvParam"):
                if cv_param.get("name") == "m/z array":
                    is_mz_array = True
                elif cv_param.get("name") == "intensity array":
                    is_intensity_array = True

            # Get binary element and settings
            binary_elem = binary_array.find(f"{{{ns_ms}}}binary")
            if binary_elem is None or not binary_elem.text:
                continue

            compressed = False
            precision = 64

            for cv_param in binary_array.iter(f"{{{ns_ms}}}cvParam"):
                name = cv_param.get("name", "")
                if "zlib" in name.lower() or "compression" in name.lower():
                    compressed = True
                if "32-bit" in name:
                    precision = 32
                elif "64-bit" in name:
                    precision = 64

            data = base64.b64decode(binary_elem.text)
            if compressed:
                data = zlib_module.decompress(data)

            if precision == 64:
                arr = np.array(struct_module.unpack(f"<{len(data) // 8}d", data))
            else:
                arr = np.array(struct_module.unpack(f"<{len(data) // 4}f", data))

            if is_mz_array:
                mz_array = arr
                mz_binary_elem = binary_elem
                mz_compressed = compressed
                mz_precision = precision
            elif is_intensity_array:
                intensity_array = arr

        # Apply calibration if we have m/z array
        if mz_array is not None and mz_binary_elem is not None:
            metadata = {
                "rt": rt,
                "precursor_mz": precursor_mz,
                "tic": tic,
                "injection_time": injection_time,
                "absolute_time": absolute_time,
                "rfa2_temp": rfa2_temp,
                "rfc2_temp": rfc2_temp,
            }
            calibrated_mz = calibration_func(metadata, mz_array, intensity_array)

            # Re-encode
            if mz_precision == 64:
                new_data = struct_module.pack(f"<{len(calibrated_mz)}d", *calibrated_mz)
            else:
                new_data = struct_module.pack(f"<{len(calibrated_mz)}f", *calibrated_mz)

            if mz_compressed:
                new_data = zlib_module.compress(new_data)

            mz_binary_elem.text = base64.b64encode(new_data).decode("ascii")
            n_calibrated += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), xml_declaration=True, encoding="utf-8")

    logger.info(f"Calibrated {n_calibrated} MS2 spectra")


def get_output_path(input_path: Path | str, output_dir: Path | str | None = None) -> Path:
    """Generate output path for calibrated mzML file.

    Output file is named {input_stem}-mars.mzML

    Args:
        input_path: Input mzML file path
        output_dir: Output directory (uses input dir if None)

    Returns:
        Output file path
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir else input_path.parent

    stem = input_path.stem
    if stem.endswith(".mzML"):
        stem = stem[:-5]

    return output_dir / f"{stem}-mars.mzML"
