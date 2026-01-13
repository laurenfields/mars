"""Tests for mars matching module."""

import numpy as np
import pytest

from mars.matching import find_most_intense_peak


class TestFindMostIntensePeak:
    """Tests for find_most_intense_peak function."""

    def test_exact_match(self):
        """Test finding exact m/z match."""
        mz_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensity_array = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

        result = find_most_intense_peak(
            target_mz=300.0,
            mz_array=mz_array,
            intensity_array=intensity_array,
            tolerance=0.5,
            min_intensity=500.0,
        )

        assert result is not None
        observed_mz, observed_intensity = result
        assert observed_mz == 300.0
        assert observed_intensity == 3000.0

    def test_most_intense_within_tolerance(self):
        """Test that most intense peak is selected within tolerance."""
        mz_array = np.array([299.8, 300.0, 300.2])
        intensity_array = np.array([1000.0, 5000.0, 2000.0])

        result = find_most_intense_peak(
            target_mz=300.0,
            mz_array=mz_array,
            intensity_array=intensity_array,
            tolerance=0.5,
            min_intensity=500.0,
        )

        assert result is not None
        observed_mz, observed_intensity = result
        assert observed_mz == 300.0
        assert observed_intensity == 5000.0

    def test_no_peak_within_tolerance(self):
        """Test no match when peaks are outside tolerance."""
        mz_array = np.array([100.0, 200.0, 400.0, 500.0])
        intensity_array = np.array([1000.0, 2000.0, 4000.0, 5000.0])

        result = find_most_intense_peak(
            target_mz=300.0,
            mz_array=mz_array,
            intensity_array=intensity_array,
            tolerance=0.5,
            min_intensity=500.0,
        )

        assert result is None

    def test_below_min_intensity(self):
        """Test no match when peak is below min intensity."""
        mz_array = np.array([300.0])
        intensity_array = np.array([100.0])  # Below min_intensity

        result = find_most_intense_peak(
            target_mz=300.0,
            mz_array=mz_array,
            intensity_array=intensity_array,
            tolerance=0.5,
            min_intensity=500.0,
        )

        assert result is None

    def test_empty_arrays(self):
        """Test with empty arrays."""
        result = find_most_intense_peak(
            target_mz=300.0,
            mz_array=np.array([]),
            intensity_array=np.array([]),
            tolerance=0.5,
            min_intensity=500.0,
        )

        assert result is None

    def test_tolerance_boundary(self):
        """Test peak exactly at tolerance boundary."""
        mz_array = np.array([299.5, 300.5])  # Exactly at +/- 0.5
        intensity_array = np.array([1000.0, 2000.0])

        result = find_most_intense_peak(
            target_mz=300.0,
            mz_array=mz_array,
            intensity_array=intensity_array,
            tolerance=0.5,
            min_intensity=500.0,
        )

        assert result is not None
        # Should find the more intense one
        assert result[1] == 2000.0
