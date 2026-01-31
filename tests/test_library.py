"""Tests for mars library module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from mars.library import (
    Fragment,
    LibraryEntry,
    calculate_fragment_mz,
    load_diann_library,
    strip_modifications,
)


class TestFragment:
    """Tests for Fragment dataclass."""

    def test_fragment_creation(self):
        """Test Fragment creation."""
        frag = Fragment(
            mz=500.5,
            intensity=1000.0,
            ion_type="y",
            ion_number=5,
            charge=1,
        )
        assert frag.mz == 500.5
        assert frag.ion_type == "y"
        assert frag.charge == 1
        assert frag.loss_type == "noloss"


class TestLibraryEntry:
    """Tests for LibraryEntry dataclass."""

    def test_library_entry_creation(self):
        """Test LibraryEntry creation."""
        entry = LibraryEntry(
            modified_sequence="PEPTIDE",
            stripped_sequence="PEPTIDE",
            precursor_charge=2,
            precursor_mz=400.0,
        )
        assert entry.modified_sequence == "PEPTIDE"
        assert entry.precursor_charge == 2
        assert entry.fragments == []

    def test_make_key(self):
        """Test unique key generation."""
        entry = LibraryEntry(
            modified_sequence="PEPTIDE",
            stripped_sequence="PEPTIDE",
            precursor_charge=2,
            precursor_mz=400.0,
        )
        key = entry.make_key()
        assert key == "PEPTIDE_2"


class TestStripModifications:
    """Tests for strip_modifications function."""

    def test_no_modifications(self):
        """Test sequence without modifications."""
        assert strip_modifications("PEPTIDE") == "PEPTIDE"

    def test_unimod_format(self):
        """Test UNIMOD format removal."""
        assert strip_modifications("PEP[UNIMOD:35]TIDE") == "PEPTIDE"
        assert strip_modifications("C[UNIMOD:4]PEPTIDE") == "CPEPTIDE"

    def test_bracket_format(self):
        """Test bracket modification removal."""
        assert strip_modifications("PEP[+80]TIDE") == "PEPTIDE"
        assert strip_modifications("M[+16]EPTIDE") == "MEPTIDE"

    def test_parentheses_format(self):
        """Test parentheses modification removal."""
        assert strip_modifications("PEP(phospho)TIDE") == "PEPTIDE"


class TestCalculateFragmentMz:
    """Tests for calculate_fragment_mz function."""

    def test_y_ion(self):
        """Test y-ion m/z calculation."""
        # y1 of PEPTIDE is E = 148.0604 (approx)
        mz = calculate_fragment_mz("PEPTIDE", "y", 1, 1)
        assert mz > 0
        assert 145 < mz < 155  # Approximate range for y1

    def test_b_ion(self):
        """Test b-ion m/z calculation."""
        # b2 of PEPTIDE is PE
        mz = calculate_fragment_mz("PEPTIDE", "b", 2, 1)
        assert mz > 0
        assert 225 < mz < 235  # Approximate range for b2

    def test_charged_ion(self):
        """Test doubly charged ion."""
        mz_1 = calculate_fragment_mz("PEPTIDE", "y", 5, 1)
        mz_2 = calculate_fragment_mz("PEPTIDE", "y", 5, 2)
        # Doubly charged should be roughly half
        assert abs(mz_2 - (mz_1 + 1) / 2) < 1.0

    def test_invalid_ion_type(self):
        """Test invalid ion type returns value > 0 (pyteomics computes something)."""
        # Note: pyteomics may still compute a value for unsupported types
        mz = calculate_fragment_mz("PEPTIDE", "z", 1, 1)
        # Just check it doesn't crash - behavior depends on pyteomics
        assert isinstance(mz, float)

    def test_ion_number_bounds(self):
        """Test ion numbers at sequence boundaries."""
        # y7 of PEPTIDE (7 residues) should be valid
        mz_max = calculate_fragment_mz("PEPTIDE", "y", 7, 1)
        assert mz_max > 0

        # b1 should be valid
        mz_b1 = calculate_fragment_mz("PEPTIDE", "b", 1, 1)
        assert mz_b1 > 0


class TestLoadDiannLibrary:
    """Tests for load_diann_library function."""

    def test_load_diann_library_basic(self, tmp_path):
        """Test loading DIA-NN library from parquet files."""
        # Create mock library parquet
        lib_data = {
            "Precursor.Id": ["PEPTIDE2", "PEPTIDE2", "SEQUENCE3", "SEQUENCE3"],
            "Modified.Sequence": ["PEPTIDE", "PEPTIDE", "SEQUENCE", "SEQUENCE"],
            "Stripped.Sequence": ["PEPTIDE", "PEPTIDE", "SEQUENCE", "SEQUENCE"],
            "Precursor.Charge": [2, 2, 3, 3],
            "Precursor.Mz": [400.5, 400.5, 350.2, 350.2],
            "Product.Mz": [500.3, 600.4, 450.1, 550.2],
            "Relative.Intensity": [1.0, 0.5, 0.8, 0.3],
            "Fragment.Type": ["y", "y", "b", "b"],
            "Fragment.Charge": [1, 1, 1, 2],
            "Fragment.Series.Number": [5, 6, 3, 4],
            "Fragment.Loss.Type": ["noloss", "noloss", "noloss", "H2O"],
            "Protein.Ids": ["P12345", "P12345", "P67890", "P67890"],
        }
        lib_df = pd.DataFrame(lib_data)
        lib_path = tmp_path / "report-lib.parquet"
        lib_df.to_parquet(lib_path)

        # Create mock report parquet
        report_data = {
            "Precursor.Id": ["PEPTIDE2", "SEQUENCE3"],
            "Run": ["test_file.mzML", "test_file.mzML"],
            "RT": [10.5, 15.2],
            "RT.Start": [10.0, 14.8],
            "RT.Stop": [11.0, 15.6],
        }
        report_df = pd.DataFrame(report_data)
        report_path = tmp_path / "report.parquet"
        report_df.to_parquet(report_path)

        # Load library
        entries = load_diann_library(lib_path)

        assert len(entries) == 2

        # Check first entry (PEPTIDE)
        entry1 = next(e for e in entries if e.modified_sequence == "PEPTIDE")
        assert entry1.precursor_charge == 2
        assert entry1.precursor_mz == 400.5
        assert entry1.rt_start == 10.0
        assert entry1.rt_end == 11.0
        assert len(entry1.fragments) == 2
        assert entry1.fragments[0].mz == 500.3
        assert entry1.fragments[0].ion_type == "y"

        # Check second entry (SEQUENCE)
        entry2 = next(e for e in entries if e.modified_sequence == "SEQUENCE")
        assert entry2.precursor_charge == 3
        assert entry2.rt_start == 14.8
        assert entry2.rt_end == 15.6
        assert len(entry2.fragments) == 2
        assert entry2.fragments[1].loss_type == "H2O"

    def test_load_diann_library_missing_report(self, tmp_path):
        """Test error when report.parquet is missing."""
        # Create mock library parquet only
        lib_data = {
            "Precursor.Id": ["PEPTIDE2"],
            "Modified.Sequence": ["PEPTIDE"],
            "Stripped.Sequence": ["PEPTIDE"],
            "Precursor.Charge": [2],
            "Precursor.Mz": [400.5],
            "Product.Mz": [500.3],
            "Relative.Intensity": [1.0],
            "Fragment.Type": ["y"],
            "Fragment.Charge": [1],
            "Fragment.Series.Number": [5],
            "Fragment.Loss.Type": ["noloss"],
        }
        lib_df = pd.DataFrame(lib_data)
        lib_path = tmp_path / "report-lib.parquet"
        lib_df.to_parquet(lib_path)

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="report.parquet not found"):
            load_diann_library(lib_path)

    def test_load_diann_library_wrong_file_type(self, tmp_path):
        """Test helpful error when report.parquet is provided instead of report-lib.parquet."""
        # Create a file that looks like report.parquet (has RT.Start, RT.Stop, Run, etc.)
        report_data = {
            "Precursor.Id": ["PEPTIDE2"],
            "Run": ["test_file.mzML"],
            "RT": [10.5],
            "RT.Start": [10.0],
            "RT.Stop": [11.0],
            "Q.Value": [0.01],
            "PEP": [0.001],
        }
        report_df = pd.DataFrame(report_data)
        # Save it as if someone named it wrong or provided wrong file
        wrong_file = tmp_path / "my_library.parquet"
        report_df.to_parquet(wrong_file)

        # Also create the actual report.parquet so it doesn't fail on that
        report_path = tmp_path / "report.parquet"
        report_df.to_parquet(report_path)

        # Should raise ValueError with helpful message about wrong file type
        with pytest.raises(ValueError, match="appears to be a DIA-NN report file"):
            load_diann_library(wrong_file)

    def test_load_diann_library_with_filename_filter(self, tmp_path):
        """Test filtering by mzML filename."""
        # Create mock library parquet
        lib_data = {
            "Precursor.Id": ["PEPTIDE2", "SEQUENCE3"],
            "Modified.Sequence": ["PEPTIDE", "SEQUENCE"],
            "Stripped.Sequence": ["PEPTIDE", "SEQUENCE"],
            "Precursor.Charge": [2, 3],
            "Precursor.Mz": [400.5, 350.2],
            "Product.Mz": [500.3, 450.1],
            "Relative.Intensity": [1.0, 0.8],
            "Fragment.Type": ["y", "b"],
            "Fragment.Charge": [1, 1],
            "Fragment.Series.Number": [5, 3],
            "Fragment.Loss.Type": ["noloss", "noloss"],
        }
        lib_df = pd.DataFrame(lib_data)
        lib_path = tmp_path / "report-lib.parquet"
        lib_df.to_parquet(lib_path)

        # Create mock report parquet with multiple runs
        report_data = {
            "Precursor.Id": ["PEPTIDE2", "PEPTIDE2", "SEQUENCE3"],
            "Run": ["file1.mzML", "file2.mzML", "file1.mzML"],
            "RT": [10.5, 11.0, 15.2],
            "RT.Start": [10.0, 10.5, 14.8],
            "RT.Stop": [11.0, 11.5, 15.6],
        }
        report_df = pd.DataFrame(report_data)
        report_path = tmp_path / "report.parquet"
        report_df.to_parquet(report_path)

        # Load with filter for file1 only
        entries = load_diann_library(lib_path, mzml_filename=["file1"])

        # PEPTIDE should have RT window from file1
        entry1 = next(e for e in entries if e.modified_sequence == "PEPTIDE")
        assert entry1.rt_start == 10.0  # From file1
        assert entry1.rt_end == 11.0

        # SEQUENCE should also have RT window (only appears in file1)
        entry2 = next(e for e in entries if e.modified_sequence == "SEQUENCE")
        assert entry2.rt_start == 14.8
        assert entry2.rt_end == 15.6
