"""Spectral library loading for blib files.

Supports loading Skyline blib (SQLite-based) spectral libraries with
fragment m/z values calculated from peptide sequences using pyteomics.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyteomics import mass

logger = logging.getLogger(__name__)


# Standard amino acid masses for fragment calculation
AMINO_ACID_MASS = mass.std_aa_mass.copy()

# Common modifications (mass deltas)
MODIFICATIONS = {
    "Carbamidomethyl": 57.021464,  # C alkylation
    "Oxidation": 15.994915,  # M oxidation
    "Phospho": 79.966331,  # S/T/Y phosphorylation
    "Acetyl": 42.010565,  # N-term acetylation
}


@dataclass
class Fragment:
    """Expected fragment ion from spectral library."""

    mz: float  # Calculated or library m/z
    intensity: float  # Relative intensity (0-1, for weighting)
    ion_type: str  # b, y, a, etc.
    ion_number: int  # Position in sequence
    charge: int  # Fragment charge state
    loss_type: str = "noloss"  # Neutral loss: noloss, H2O, NH3, etc.


@dataclass
class LibraryEntry:
    """Peptide entry with expected fragment m/z values."""

    modified_sequence: str  # Modified peptide sequence
    stripped_sequence: str  # Unmodified sequence
    precursor_charge: int
    precursor_mz: float
    rt: float | None = None  # Library retention time
    rt_start: float | None = None  # RT window start (for matching)
    rt_end: float | None = None  # RT window end
    fragments: list[Fragment] = field(default_factory=list)
    protein_ids: list[str] = field(default_factory=list)

    def make_key(self) -> str:
        """Create unique key for this peptide."""
        return f"{self.modified_sequence}_{self.precursor_charge}"


def strip_modifications(sequence: str) -> str:
    """Strip all modifications from a peptide sequence.

    Handles multiple modification formats:
    - Unimod format: M(unimod:35)
    - Mass delta format: M[+15.99491]
    - Skyline format: M[Oxidation (M)]

    Args:
        sequence: Modified peptide sequence

    Returns:
        Unmodified (stripped) sequence
    """
    import re

    # Remove (unimod:N) format
    result = re.sub(r"\(unimod:\d+\)", "", sequence)
    # Remove [+N.NNNNN] format (mass delta)
    result = re.sub(r"\[[+-]?\d+\.?\d*\]", "", result)
    # Remove [Modification Name] format
    result = re.sub(r"\[[^\]]+\]", "", result)
    # Remove any remaining parenthetical modifications
    result = re.sub(r"\([^)]+\)", "", result)

    return result


def calculate_fragment_mz(
    sequence: str,
    ion_type: str,
    ion_number: int,
    charge: int,
    loss_type: str = "noloss",
) -> float:
    """Calculate theoretical fragment m/z using pyteomics.

    Args:
        sequence: Stripped (unmodified) peptide sequence
        ion_type: Fragment type (b, y, a, c, x, z)
        ion_number: Ion number (1-indexed position)
        charge: Fragment charge state
        loss_type: Neutral loss type (noloss, H2O, NH3)

    Returns:
        Calculated m/z value
    """
    # For b-ions: N-terminal fragments
    # For y-ions: C-terminal fragments
    if ion_type in ("b", "a", "c"):
        # N-terminal fragment
        frag_seq = sequence[:ion_number]
    elif ion_type in ("y", "x", "z"):
        # C-terminal fragment
        frag_seq = sequence[-ion_number:]
    else:
        logger.warning(f"Unknown ion type: {ion_type}")
        return 0.0

    try:
        # Calculate base mass
        frag_mass = mass.fast_mass(frag_seq, ion_type=ion_type, charge=charge)

        # Apply neutral losses
        if loss_type == "H2O":
            frag_mass -= 18.010565 / charge
        elif loss_type == "NH3":
            frag_mass -= 17.026549 / charge

        return frag_mass
    except Exception as e:
        logger.warning(f"Failed to calculate m/z for {ion_type}{ion_number}+{charge}: {e}")
        return 0.0


def _decode_mz_blob(blob: bytes) -> np.ndarray:
    """Decode zlib-compressed m/z blob (float64).

    Args:
        blob: Compressed bytes

    Returns:
        Array of m/z values
    """
    try:
        # Check for zlib magic bytes (78 9c)
        if blob[:2] == b"\x78\x9c":
            decompressed = zlib.decompress(blob)
        else:
            try:
                decompressed = zlib.decompress(blob)
            except zlib.error:
                decompressed = blob

        n_values = len(decompressed) // 8
        mz_values = np.array(struct.unpack(f"<{n_values}d", decompressed))
        return mz_values

    except Exception as e:
        logger.warning(f"Failed to decode m/z blob: {e}")
        return np.array([])


def _decode_intensity_blob(blob: bytes) -> np.ndarray:
    """Decode intensity blob (raw float32).

    Args:
        blob: Raw bytes (may be compressed)

    Returns:
        Array of intensity values (normalized to max=1.0)
    """
    try:
        if blob[:2] == b"\x78\x9c":
            try:
                blob = zlib.decompress(blob)
            except zlib.error:
                pass

        n_values = len(blob) // 4
        intensities = np.array(struct.unpack(f"<{n_values}f", blob))

        if len(intensities) > 0 and intensities.max() > 0:
            intensities = intensities / intensities.max()

        return intensities

    except Exception as e:
        logger.warning(f"Failed to decode intensity blob: {e}")
        return np.array([])


def load_blib(
    path: Path | str,
    rt_window: float = 0.083,  # ±5 seconds (in minutes) for non-PRISM peptides
    recalculate_mz: bool = True,
) -> list[LibraryEntry]:
    """Load Skyline blib spectral library.

    Args:
        path: Path to blib file
        rt_window: RT window (±minutes) around library RT for matching.
                   Default is 0.083 min (±5 seconds) for peptides without
                   PRISM CSV data. PRISM CSV provides exact Start/End times.
        recalculate_mz: If True, recalculate fragment m/z from sequence
                       using pyteomics (recommended for ground truth)

    Returns:
        List of LibraryEntry objects with fragments
    """
    path = Path(path)
    logger.info(f"Loading blib library from {path}")

    if not path.exists():
        raise FileNotFoundError(f"Blib file not found: {path}")

    entries: list[LibraryEntry] = []

    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()

        # Check for RefSpectraPeakAnnotations table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='RefSpectraPeakAnnotations'
        """)
        has_annotations = cursor.fetchone() is not None

        # Query RefSpectra for peptide info
        cursor.execute("""
            SELECT id, peptideSeq, precursorMZ, precursorCharge,
                   peptideModSeq, retentionTime, numPeaks
            FROM RefSpectra
        """)

        ref_spectra = cursor.fetchall()
        logger.info(f"Found {len(ref_spectra)} library entries")

        for spec_id, pep_seq, prec_mz, prec_charge, mod_seq, rt, n_peaks in ref_spectra:
            seq = mod_seq if mod_seq else pep_seq
            stripped = strip_modifications(seq) if mod_seq else pep_seq

            entry = LibraryEntry(
                modified_sequence=seq,
                stripped_sequence=stripped,
                precursor_charge=int(prec_charge),
                precursor_mz=float(prec_mz),
                rt=float(rt) if rt else None,
                rt_start=float(rt) - rt_window if rt else None,
                rt_end=float(rt) + rt_window if rt else None,
            )

            # Get peaks
            cursor.execute(
                """
                SELECT peakMZ, peakIntensity
                FROM RefSpectraPeaks
                WHERE RefSpectraID = ?
            """,
                (spec_id,),
            )

            peaks_row = cursor.fetchone()
            if not peaks_row:
                continue

            mz_blob, intensity_blob = peaks_row
            mz_values = _decode_mz_blob(mz_blob)
            intensities = _decode_intensity_blob(intensity_blob)

            if len(mz_values) != len(intensities):
                logger.warning(
                    f"Mismatch in peak count for {entry.make_key()}: "
                    f"{len(mz_values)} m/z vs {len(intensities)} intensities"
                )
                continue

            # Try to get fragment annotations
            annotations = {}
            if has_annotations:
                cursor.execute(
                    """
                    SELECT peakIndex, name, charge
                    FROM RefSpectraPeakAnnotations
                    WHERE RefSpectraID = ?
                """,
                    (spec_id,),
                )
                for peak_idx, name, charge in cursor.fetchall():
                    annotations[peak_idx] = (name, charge)

            # Build fragments
            for i, (mz, intensity) in enumerate(zip(mz_values, intensities)):
                if i in annotations:
                    # Parse annotation like "b3" or "y7"
                    name, frag_charge = annotations[i]
                    ion_type = name[0] if name else "?"
                    try:
                        ion_number = int("".join(c for c in name if c.isdigit()))
                    except ValueError:
                        ion_number = 0

                    # Recalculate m/z if requested
                    if recalculate_mz and ion_type in ("b", "y"):
                        calc_mz = calculate_fragment_mz(
                            stripped,
                            ion_type,
                            ion_number,
                            int(frag_charge) if frag_charge else 1,
                        )
                        if calc_mz > 0:
                            mz = calc_mz

                    entry.fragments.append(
                        Fragment(
                            mz=float(mz),
                            intensity=float(intensity),
                            ion_type=ion_type,
                            ion_number=ion_number,
                            charge=int(frag_charge) if frag_charge else 1,
                        )
                    )
                else:
                    # No annotation - use m/z directly
                    entry.fragments.append(
                        Fragment(
                            mz=float(mz),
                            intensity=float(intensity),
                            ion_type="?",
                            ion_number=0,
                            charge=1,
                        )
                    )

            if entry.fragments:
                entries.append(entry)

    finally:
        conn.close()

    logger.info(f"Loaded {len(entries)} peptides with fragments from blib")
    return entries


def iter_library_fragments(entries: list[LibraryEntry]) -> Iterator[tuple[LibraryEntry, Fragment]]:
    """Iterate over all fragments in the library.

    Yields:
        Tuples of (LibraryEntry, Fragment)
    """
    for entry in entries:
        for fragment in entry.fragments:
            yield entry, fragment


def load_prism_library(
    path: Path | str,
    mzml_filename: str | None = None,
) -> list[LibraryEntry]:
    """Load library from PRISM Skyline report CSV.

    Uses the theoretical m/z values from Product Mz column.
    Filters to MS2 fragments only (excludes precursor ions).

    Args:
        path: Path to PRISM CSV file
        mzml_filename: Optional mzML filename to filter to specific replicate

    Returns:
        List of LibraryEntry objects with fragments
    """
    import re

    import pandas as pd

    path = Path(path)
    logger.info(f"Loading PRISM library from {path}")

    if not path.exists():
        raise FileNotFoundError(f"PRISM CSV file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from PRISM CSV")

    # Required columns
    required_cols = [
        "Peptide Modified Sequence Unimod Ids",
        "Precursor Charge",
        "Precursor Mz",
        "Fragment Ion",
        "Product Charge",
        "Product Mz",
        "Start Time",
        "End Time",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to specific replicate if mzml_filename provided
    if mzml_filename and "Replicate Name" in df.columns:
        # Strip common suffixes from mzML filename
        base_name = mzml_filename
        for suffix in ["_uncalibrated", "-mars", ".mzML", ".raw"]:
            base_name = base_name.replace(suffix, "")

        # Find matching rows
        mask = (
            df["Replicate Name"].astype(str).apply(lambda x: x in base_name or base_name in str(x))
        )
        if mask.sum() > 0:
            df = df[mask]
            logger.info(f"Filtered to {len(df)} rows matching '{base_name}'")

    # Filter to MS2 fragments only (exclude precursor)
    df = df[df["Fragment Ion"] != "precursor"].copy()
    logger.info(f"After filtering precursors: {len(df)} MS2 fragment rows")

    # Parse fragment ion annotations (e.g., "y8", "b4", "y5++")
    def parse_fragment_ion(ion_str: str) -> tuple[str, int, str]:
        """Parse fragment ion string like 'y8', 'b4', 'y5-H2O'.

        Returns: (ion_type, ion_number, loss_type)
        """
        ion_str = str(ion_str)

        # Default values
        ion_type = "?"
        ion_number = 0
        loss_type = "noloss"

        # Parse ion type (first letter)
        if ion_str and ion_str[0] in "ybazcx":
            ion_type = ion_str[0]

        # Parse ion number (digits)
        match = re.search(r"(\d+)", ion_str)
        if match:
            ion_number = int(match.group(1))

        # Parse neutral loss
        if "-H2O" in ion_str or "-18" in ion_str:
            loss_type = "H2O"
        elif "-NH3" in ion_str or "-17" in ion_str:
            loss_type = "NH3"

        return ion_type, ion_number, loss_type

    # Group by peptide+charge to create LibraryEntry objects
    entries: list[LibraryEntry] = []
    peptide_col = "Peptide Modified Sequence Unimod Ids"

    grouped = df.groupby([peptide_col, "Precursor Charge"])

    for (mod_seq, prec_charge), group in grouped:
        # Get precursor info from first row
        first_row = group.iloc[0]

        entry = LibraryEntry(
            modified_sequence=str(mod_seq),
            stripped_sequence=strip_modifications(str(mod_seq)),
            precursor_charge=int(prec_charge),
            precursor_mz=float(first_row["Precursor Mz"]),
            rt=float(first_row["Retention Time"]) if "Retention Time" in group.columns else None,
            rt_start=float(first_row["Start Time"]) if pd.notna(first_row["Start Time"]) else None,
            rt_end=float(first_row["End Time"]) if pd.notna(first_row["End Time"]) else None,
        )

        # Add protein IDs if available
        if "Protein Accession" in group.columns:
            protein = first_row["Protein Accession"]
            if pd.notna(protein):
                entry.protein_ids = [str(protein)]

        # Add fragments
        for _, row in group.iterrows():
            ion_str = str(row["Fragment Ion"])
            ion_type, ion_number, loss_type = parse_fragment_ion(ion_str)

            # Use Product Mz as the theoretical m/z
            product_mz = row["Product Mz"]
            if pd.isna(product_mz) or product_mz <= 0:
                continue

            # Use Area as relative intensity if available
            intensity = 1.0
            if "Area" in group.columns and pd.notna(row["Area"]):
                intensity = float(row["Area"])

            # Get product charge
            product_charge = 1
            if pd.notna(row["Product Charge"]):
                product_charge = int(row["Product Charge"])

            entry.fragments.append(
                Fragment(
                    mz=float(product_mz),
                    intensity=intensity,
                    ion_type=ion_type,
                    ion_number=ion_number,
                    charge=product_charge,
                    loss_type=loss_type,
                )
            )

        if entry.fragments:
            entries.append(entry)

    logger.info(
        f"Loaded {len(entries)} peptides with {sum(len(e.fragments) for e in entries)} fragments from PRISM CSV"
    )
    return entries
