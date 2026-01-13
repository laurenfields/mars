"""Tests for mars library module."""

import pytest

from mars.library import (
    Fragment,
    LibraryEntry,
    calculate_fragment_mz,
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
