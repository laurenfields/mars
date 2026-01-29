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

    Uses psims library to write properly indexed mzML files. Reads the original
    file with pyteomics and writes with psims, modifying only MS2 m/z arrays.

    Args:
        input_path: Path to input mzML file
        output_path: Path for output mzML file
        calibration_func: Function that takes (spectrum_metadata, mz_array, intensity_array)
                         and returns calibrated mz_array
        max_isolation_window_width: Maximum isolation window width (m/z) to include.
                                    MS2 spectra with wider windows are excluded from output.
        temperature_data: Dict mapping source names (e.g., 'RFA2', 'RFC2') to TemperatureData objects
    """
    import re

    from psims.mzml import MzMLWriter

    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Writing calibrated mzML to {output_path}")

    # Extract acquisition start time for absolute_time calculation
    acquisition_start_time = None
    try:
        with open(input_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "startTimeStamp" in line:
                    match = re.search(r'startTimeStamp="([^"]+)"', line)
                    if match:
                        acquisition_start_time = _parse_iso8601_timestamp(match.group(1))
                    break
                if "<spectrumList" in line:
                    break
    except Exception as e:
        logger.debug(f"Failed to extract startTimeStamp: {e}")

    # Read all spectra from input file using pyteomics
    all_spectra = []
    with mzml.MzML(str(input_path)) as reader:
        for spectrum in reader:
            all_spectra.append(spectrum)

    logger.info(f"Read {len(all_spectra)} spectra from input file")

    # Filter out wide-window MS2 spectra if max_isolation_window_width is set
    spectra_data = []
    n_skipped = 0
    window_widths_seen = set()
    for spectrum in all_spectra:
        ms_level = spectrum.get("ms level", 1)
        if ms_level == 2 and max_isolation_window_width is not None:
            low, high, _ = _extract_isolation_window(spectrum)
            window_width = high - low if high > 0 and low > 0 else 0
            window_widths_seen.add(round(window_width, 1))
            if window_width > max_isolation_window_width:
                n_skipped += 1
                continue
        spectra_data.append(spectrum)

    if max_isolation_window_width is not None and window_widths_seen:
        logger.info(f"Isolation window widths found: {sorted(window_widths_seen)}")
    if n_skipped > 0:
        logger.info(f"Skipping {n_skipped} MS2 spectra with isolation window > {max_isolation_window_width} m/z")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_calibrated = 0

    # Write with psims
    with MzMLWriter(str(output_path)) as writer:
        # Write controlled vocabularies
        writer.controlled_vocabularies()

        # Write minimal file description
        writer.file_description(
            file_contents=["MSn spectrum", "centroid spectrum"],
            source_files=[
                {
                    "id": f"source_{input_path.stem}",
                    "name": input_path.name,
                    "location": str(input_path.parent),
                    "file_format": "mzML format",
                }
            ],
        )

        # Write software list (include Mars as processing software)
        writer.software_list(
            [
                {"id": "mars", "version": "0.1.3", "params": ["data processing software"]}
            ]
        )

        # Write instrument configuration (required by mzML schema before data processing)
        writer.instrument_configuration_list(
            [
                {
                    "id": "IC1",
                    "component_list": [],
                    "params": ["instrument model"],
                }
            ]
        )

        # Write data processing
        writer.data_processing_list(
            [
                {
                    "id": "mars_calibration",
                    "processing_methods": [
                        {
                            "order": 1,
                            "software_reference": "mars",
                            "params": ["m/z calibration"],
                        }
                    ],
                }
            ]
        )

        # Start run
        with writer.run(id="run1"):
            # Write spectra
            with writer.spectrum_list(count=len(spectra_data)):
                for spectrum in spectra_data:
                    _write_spectrum_with_calibration(
                        writer,
                        spectrum,
                        calibration_func,
                        max_isolation_window_width,
                        temperature_data,
                        acquisition_start_time,
                    )
                    # Track calibrations (all MS2 in spectra_data are calibrated since we pre-filtered)
                    if spectrum.get("ms level", 1) == 2:
                        n_calibrated += 1

    logger.info(f"Wrote {len(spectra_data)} spectra ({n_calibrated} MS2 calibrated)")


def _write_spectrum_with_calibration(
    writer,
    spectrum: dict,
    calibration_func,
    max_isolation_window_width: float | None,
    temperature_data: dict | None,
    acquisition_start_time: float | None,
) -> None:
    """Write a single spectrum to the mzML writer, applying calibration to MS2.

    Args:
        writer: psims MzMLWriter instance
        spectrum: Pyteomics spectrum dictionary
        calibration_func: Calibration function for m/z correction
        max_isolation_window_width: Max isolation window width to calibrate
        temperature_data: Temperature data dict
        acquisition_start_time: Acquisition start timestamp
    """
    # Extract basic info
    spectrum_id = spectrum.get("id", f"scan={spectrum.get('index', 0)}")
    ms_level = spectrum.get("ms level", 1)
    mz_array = spectrum.get("m/z array", np.array([]))
    intensity_array = spectrum.get("intensity array", np.array([]))

    # Extract scan time
    scan_time = 0.0
    scan_list = spectrum.get("scanList", {})
    scans = scan_list.get("scan", [])
    if scans:
        scan_time = scans[0].get("scan start time", 0.0)

    # Extract TIC
    tic = spectrum.get("total ion current", float(np.sum(intensity_array)))

    # Check if centroided
    centroided = spectrum.get("centroid spectrum", False) or spectrum.get(
        "MS:1000127", False
    )

    # Build params list
    params = [{"ms level": ms_level}]
    if tic > 0:
        params.append({"total ion current": tic})
    if centroided:
        params.append("centroid spectrum")
    else:
        params.append("profile spectrum")

    # Extract injection time if available
    injection_time = _extract_injection_time(spectrum)
    if injection_time is not None:
        # Convert back to milliseconds for mzML
        params.append({"ion injection time": injection_time * 1000.0})

    # For MS2, apply calibration and include precursor info
    precursor_info = None
    if ms_level == 2:
        low, high, center = _extract_isolation_window(spectrum)
        window_width = high - low if high > 0 and low > 0 else 0

        # Apply calibration unless window is too wide
        should_calibrate = max_isolation_window_width is None or (
            window_width <= max_isolation_window_width and window_width > 0
        )

        if should_calibrate and len(mz_array) > 0 and calibration_func is not None:
            # Calculate absolute time
            absolute_time = 0.0
            if acquisition_start_time is not None:
                absolute_time = acquisition_start_time + scan_time * 60.0

            # Look up temperatures
            rfa2_temp = 0.0
            rfc2_temp = 0.0
            if temperature_data is not None:
                if "RFA2" in temperature_data:
                    rfa2_temp = temperature_data["RFA2"].get_temperature_at_time(scan_time)
                if "RFC2" in temperature_data:
                    rfc2_temp = temperature_data["RFC2"].get_temperature_at_time(scan_time)

            # Build metadata for calibration
            metadata = {
                "rt": scan_time,
                "precursor_mz": center,
                "tic": tic,
                "injection_time": injection_time if injection_time else 0.0,
                "absolute_time": absolute_time,
                "rfa2_temp": rfa2_temp,
                "rfc2_temp": rfc2_temp,
            }

            # Apply calibration to m/z array
            mz_array = calibration_func(metadata, mz_array, intensity_array)

        # Build precursor information
        if center > 0:
            # Extract activation info if available
            activation_params = ["collision-induced dissociation"]  # Default
            precursor_list = spectrum.get("precursorList", {})
            precursors = precursor_list.get("precursor", [])
            if precursors:
                activation = precursors[0].get("activation", {})
                # Check for HCD
                if activation.get("beam-type collision-induced dissociation"):
                    activation_params = ["beam-type collision-induced dissociation"]
                # Get collision energy if available
                collision_energy = activation.get("collision energy")
                if collision_energy is not None:
                    activation_params.append({"collision energy": float(collision_energy)})

            precursor_info = {
                "mz": center,
                "isolation_window": (low - center if low > 0 else 0.5, center, high - center if high > 0 else 0.5),
                "activation": activation_params,
            }

    # Write the spectrum
    writer.write_spectrum(
        mz_array,
        intensity_array,
        id=spectrum_id,
        centroided=centroided,
        scan_start_time=scan_time,
        params=params,
        precursor_information=precursor_info,
    )


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
