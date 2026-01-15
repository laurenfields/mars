"""Tests for mars mzml module."""

import numpy as np
import pytest

from mars.mzml import (
    _extract_injection_time,
    _extract_isolation_window,
    _parse_iso8601_timestamp,
    DIASpectrum,
)


class TestParseISO8601Timestamp:
    """Tests for ISO 8601 timestamp parsing."""

    def test_parse_iso8601_with_z(self):
        """Test parsing ISO 8601 timestamp with Z suffix."""
        timestamp_str = "2023-01-15T10:30:45Z"
        result = _parse_iso8601_timestamp(timestamp_str)

        assert result is not None
        assert isinstance(result, float)
        # Should be a valid Unix timestamp
        assert result > 0

    def test_parse_iso8601_with_timezone(self):
        """Test parsing ISO 8601 timestamp with timezone offset."""
        timestamp_str = "2023-01-15T10:30:45+00:00"
        result = _parse_iso8601_timestamp(timestamp_str)

        assert result is not None
        assert isinstance(result, float)

    def test_parse_iso8601_invalid(self):
        """Test parsing invalid timestamp returns None."""
        timestamp_str = "not-a-timestamp"
        result = _parse_iso8601_timestamp(timestamp_str)

        assert result is None

    def test_parse_iso8601_empty(self):
        """Test parsing empty string returns None."""
        result = _parse_iso8601_timestamp("")
        assert result is None


class TestExtractInjectionTime:
    """Tests for injection time extraction."""

    def test_extract_injection_time_from_precursor(self):
        """Test extracting injection time from precursor metadata."""
        spectrum = {
            "precursorList": {
                "precursor": [
                    {
                        "ion injection time": 50.0,  # milliseconds
                    }
                ]
            }
        }

        result = _extract_injection_time(spectrum)

        assert result is not None
        assert result == 0.05  # Should be converted to seconds

    def test_extract_injection_time_from_scan(self):
        """Test extracting injection time from scan metadata."""
        spectrum = {
            "precursorList": {"precursor": [{}]},
            "scanList": {
                "scan": [
                    {
                        "ion injection time": 75.5,  # milliseconds
                    }
                ]
            },
        }

        result = _extract_injection_time(spectrum)

        assert result is not None
        assert result == 0.0755  # Should be converted to seconds

    def test_extract_injection_time_missing(self):
        """Test that None is returned when injection time is missing."""
        spectrum = {
            "precursorList": {"precursor": [{}]},
            "scanList": {"scan": [{}]},
        }

        result = _extract_injection_time(spectrum)

        assert result is None

    def test_extract_injection_time_empty_spectrum(self):
        """Test with empty spectrum dict."""
        result = _extract_injection_time({})

        assert result is None


class TestDIASpectrum:
    """Tests for DIASpectrum dataclass."""

    def test_dia_spectrum_creation(self):
        """Test creating a DIASpectrum with new fields."""
        spectrum = DIASpectrum(
            scan_number=1,
            rt=10.5,
            precursor_mz_low=400.0,
            precursor_mz_high=401.0,
            precursor_mz_center=400.5,
            tic=1e7,
            mz_array=np.array([100.0, 200.0]),
            intensity_array=np.array([1000.0, 2000.0]),
            injection_time=0.05,
            acquisition_start_time=1673779845.0,
            absolute_time=630.0,
        )

        assert spectrum.scan_number == 1
        assert spectrum.rt == 10.5
        assert spectrum.injection_time == 0.05
        assert spectrum.acquisition_start_time == 1673779845.0
        assert spectrum.absolute_time == 630.0
        assert spectrum.n_peaks == 2

    def test_dia_spectrum_optional_fields(self):
        """Test DIASpectrum with optional fields as None."""
        spectrum = DIASpectrum(
            scan_number=1,
            rt=10.5,
            precursor_mz_low=400.0,
            precursor_mz_high=401.0,
            precursor_mz_center=400.5,
            tic=1e7,
            mz_array=np.array([100.0, 200.0]),
            intensity_array=np.array([1000.0, 2000.0]),
            injection_time=None,
            acquisition_start_time=None,
            absolute_time=None,
        )

        assert spectrum.injection_time is None
        assert spectrum.acquisition_start_time is None
        assert spectrum.absolute_time is None
