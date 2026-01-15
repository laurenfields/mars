"""Temperature data loading and interpolation for RF temperature features."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemperatureData:
    """Holds temperature time series data for interpolation."""

    def __init__(self, time_minutes: np.ndarray, temperature: np.ndarray, source: str):
        """Initialize temperature data.

        Args:
            time_minutes: Array of time values in minutes
            temperature: Array of temperature values in °C
            source: Name of temperature source (e.g., 'RFA2', 'RFC2')
        """
        self.time_minutes = np.asarray(time_minutes)
        self.temperature = np.asarray(temperature)
        self.source = source

        # Sort by time for interpolation
        sort_idx = np.argsort(self.time_minutes)
        self.time_minutes = self.time_minutes[sort_idx]
        self.temperature = self.temperature[sort_idx]

    def get_temperature_at_time(self, rt_minutes: float) -> float:
        """Get temperature at a specific retention time using nearest neighbor.

        Args:
            rt_minutes: Retention time in minutes

        Returns:
            Temperature value at the closest time point
        """
        if len(self.time_minutes) == 0:
            return np.nan

        # Find nearest time point
        idx = np.argmin(np.abs(self.time_minutes - rt_minutes))
        return float(self.temperature[idx])

    def get_temperatures_at_times(self, rt_minutes: np.ndarray) -> np.ndarray:
        """Get temperatures at multiple retention times using nearest neighbor.

        Args:
            rt_minutes: Array of retention times in minutes

        Returns:
            Array of temperature values at the closest time points
        """
        if len(self.time_minutes) == 0:
            return np.full(len(rt_minutes), np.nan)

        # Use searchsorted for efficient nearest neighbor lookup
        rt_minutes = np.asarray(rt_minutes)
        indices = np.searchsorted(self.time_minutes, rt_minutes)

        # Handle edge cases
        indices = np.clip(indices, 0, len(self.time_minutes) - 1)

        # Check if previous index is closer
        prev_indices = np.clip(indices - 1, 0, len(self.time_minutes) - 1)
        dist_curr = np.abs(self.time_minutes[indices] - rt_minutes)
        dist_prev = np.abs(self.time_minutes[prev_indices] - rt_minutes)

        # Use previous index where it's closer
        use_prev = dist_prev < dist_curr
        indices[use_prev] = prev_indices[use_prev]

        return self.temperature[indices]


def load_temperature_csv(csv_path: Path | str) -> TemperatureData | None:
    """Load temperature data from a Thermo chromatogram CSV export.

    Expected format:
        CHROMATOGRAM
        <file_path>
        Data points: <n>
        Time(min),<temperature_column_name>
        0,50.123
        0.0025,50.456
        ...

    Args:
        csv_path: Path to temperature CSV file

    Returns:
        TemperatureData object or None if loading fails
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.debug(f"Temperature file not found: {csv_path}")
        return None

    try:
        # Read the file and skip header lines
        with open(csv_path) as f:
            lines = f.readlines()

        # Find the header line (contains "Time")
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("Time"):
                header_idx = i
                break

        if header_idx is None:
            logger.warning(f"Could not find Time header in {csv_path}")
            return None

        # Parse header to extract source name
        header = lines[header_idx].strip()
        cols = header.split(",")
        if len(cols) < 2:
            logger.warning(f"Invalid header format in {csv_path}")
            return None

        # Extract source name from column header (e.g., "32 - RFA2 Temperature {°C}")
        temp_col = cols[1]
        source_match = re.search(r"(RF[AC]\d+)", temp_col)
        source = source_match.group(1) if source_match else csv_path.stem.split("-")[0]

        # Read the data
        df = pd.read_csv(csv_path, skiprows=header_idx, header=0)

        # Get column names
        time_col = df.columns[0]
        temp_col = df.columns[1]

        time_minutes: np.ndarray = df[time_col].to_numpy(dtype=float)
        temperature: np.ndarray = df[temp_col].to_numpy(dtype=float)

        logger.info(
            f"Loaded {len(time_minutes)} temperature points from {csv_path.name} "
            f"(source: {source}, range: {temperature.min():.1f}-{temperature.max():.1f}°C)"
        )

        return TemperatureData(time_minutes, temperature, source)

    except Exception as e:
        logger.warning(f"Failed to load temperature data from {csv_path}: {e}")
        return None


def find_temperature_files(
    mzml_path: Path | str,
    temperature_dir: Path | str | None = None,
    sources: list[str] | None = None,
) -> dict[str, TemperatureData]:
    """Find and load temperature CSV files for an mzML file.

    Looks for files named like: {source}-{mzml_basename}.csv
    e.g., RFA2-Ste-2024-12-02_HeLa_20msIIT_GPFDIA_400-500_14.csv

    Args:
        mzml_path: Path to mzML file
        temperature_dir: Directory containing temperature CSVs (default: same as mzML)
        sources: List of temperature sources to look for (default: ['RFA2', 'RFC2'])

    Returns:
        Dict mapping source name to TemperatureData
    """
    mzml_path = Path(mzml_path)
    mzml_basename = mzml_path.stem  # e.g., "Ste-2024-12-02_HeLa_20msIIT_GPFDIA_400-500_14"

    if temperature_dir is None:
        temperature_dir = mzml_path.parent
    else:
        temperature_dir = Path(temperature_dir)

    if sources is None:
        sources = ["RFA2", "RFC2"]

    temperature_data: dict[str, TemperatureData] = {}

    for source in sources:
        # Look for file like: RFA2-{mzml_basename}.csv
        csv_name = f"{source}-{mzml_basename}.csv"
        csv_path = temperature_dir / csv_name

        temp_data = load_temperature_csv(csv_path)
        if temp_data is not None:
            temperature_data[source] = temp_data

    if temperature_data:
        logger.info(f"Found {len(temperature_data)} temperature file(s) for {mzml_path.name}")
    else:
        logger.debug(f"No temperature files found for {mzml_path.name}")

    return temperature_data
