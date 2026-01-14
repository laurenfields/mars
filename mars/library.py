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

        for spec_id, pep_seq, prec_mz, prec_charge, mod_seq, rt, _n_peaks in ref_spectra:
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
            for i, (mz, intensity) in enumerate(zip(mz_values, intensities, strict=True)):
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
        mzml_filename: Optional mzML filename(s) to filter to specific replicate(s).
                      Can be a single filename string or a list of filenames.

    Returns:
        List of LibraryEntry objects with fragments
    """
    import pandas as pd

    path = Path(path)
    logger.info(f"Loading PRISM library from {path}")

    if not path.exists():
        raise FileNotFoundError(f"PRISM CSV file not found: {path}")

    # Define columns we need - load only these for faster I/O
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
    optional_cols = ["Replicate Name", "File Name", "Retention Time", "Protein Accession", "Area"]

    # First pass: check which columns exist (read just header)
    header_df = pd.read_csv(path, nrows=0)
    available_cols = set(header_df.columns)

    missing = [c for c in required_cols if c not in available_cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Load only the columns we need
    cols_to_load = required_cols + [c for c in optional_cols if c in available_cols]

    # Use pyarrow engine for much faster CSV reading (3-10x faster than default)
    import time

    start_time = time.time()
    logger.info(f"Loading {len(cols_to_load)} columns from PRISM CSV (using pyarrow)...")
    df = pd.read_csv(path, usecols=cols_to_load, engine="pyarrow")
    elapsed = time.time() - start_time
    logger.info(f"Loaded {len(df):,} rows from PRISM CSV in {elapsed:.1f}s")

    # Filter to specific replicate(s) if mzml_filename provided
    # Filter to specific replicate(s) if mzml_filename provided
    # Prioritize 'File Name' if available, otherwise use 'Replicate Name'
    filter_col = "File Name" if "File Name" in df.columns else "Replicate Name"

    if mzml_filename and filter_col in df.columns:
        # Normalize to list
        if isinstance(mzml_filename, str):
            filenames = [mzml_filename]
        else:
            filenames = list(mzml_filename)

        # Strip common suffixes from mzML filenames to get base names
        input_base_names = set()
        for fname in filenames:
            base_name = fname
            for suffix in ["_uncalibrated", "-mars", ".mzML", ".raw"]:
                base_name = base_name.replace(suffix, "")
            input_base_names.add(base_name)

        # Get unique values in the CSV (File Name or Replicate Name)
        csv_values = df[filter_col].astype(str).unique()
        n_csv_values = len(csv_values)

        # Find which CSV values correspond to our input files
        matched_values = []
        for csv_val in csv_values:
            csv_val_str = str(csv_val)
            # Normalize CSV value too (remove extension if present)
            csv_base = csv_val_str
            for suffix in [".mzML", ".raw", ".wiff", ".d"]:
                if csv_base.lower().endswith(suffix.lower()):
                    csv_base = csv_base[: -len(suffix)]

            for input_name in input_base_names:
                # Check exact match of base names first (strongest match)
                if csv_base == input_name:
                    matched_values.append(csv_val)
                    break
                # Then check substring match
                elif csv_base in input_name or input_name in csv_base:
                    matched_values.append(csv_val)
                    break

        n_matched = len(matched_values)

        if n_matched > 0:
            if n_matched < n_csv_values:
                # Filter to the matching replicates
                df = df[df[filter_col].isin(matched_values)]
                logger.info(
                    f"Filtered to {n_matched}/{n_csv_values} files ({len(df):,} rows) "
                    f"using '{filter_col}' column"
                )
            else:
                # All replicates matched
                logger.info(f"All {n_csv_values} files in CSV match input ({len(df):,} rows)")
        else:
            # No rows matched
            logger.warning(
                f"Filtering failed! CSV contains {n_csv_values} files ({filter_col}) "
                f"but none matched the {len(input_base_names)} input files."
            )
            # Debug info
            logger.warning(
                f"  Input file bases: {list(input_base_names)[:3]}{'...' if len(input_base_names) > 3 else ''}"
            )
            logger.warning(
                f"  CSV values:       {list(csv_values)[:3]}{'...' if len(csv_values) > 3 else ''}"
            )
            logger.info(f"Using all {len(df):,} rows (no filtering applied)")

    # Filter to MS2 fragments only (exclude precursor) - vectorized
    df = df[df["Fragment Ion"] != "precursor"]
    logger.info(f"After filtering precursors: {len(df):,} MS2 fragment rows")

    if len(df) == 0:
        logger.warning("No MS2 fragments found in library")
        return []

    # Vectorized fragment ion parsing
    logger.info("Parsing fragment annotations...")
    ion_series = df["Fragment Ion"].astype(str)

    # Extract ion type (first character if y, b, a, z, c, x)
    df["_ion_type"] = ion_series.str.extract(r"^([ybazcx])", expand=False).fillna("?")

    # Extract ion number (first digits)
    df["_ion_number"] = ion_series.str.extract(r"(\d+)", expand=False).fillna("0").astype(int)

    # Extract loss type
    df["_loss_type"] = "noloss"
    df.loc[ion_series.str.contains(r"-H2O|-18", regex=True, na=False), "_loss_type"] = "H2O"
    df.loc[ion_series.str.contains(r"-NH3|-17", regex=True, na=False), "_loss_type"] = "NH3"

    # Group by peptide+charge to create LibraryEntry objects
    logger.info("Building library entries...")
    entries: list[LibraryEntry] = []
    peptide_col = "Peptide Modified Sequence Unimod Ids"

    # Pre-compute column indices for faster access
    has_rt = "Retention Time" in df.columns
    has_protein = "Protein Accession" in df.columns
    has_area = "Area" in df.columns

    grouped = df.groupby([peptide_col, "Precursor Charge"], sort=False)
    n_groups = len(grouped)

    for i, ((mod_seq, prec_charge), group) in enumerate(grouped):
        if i % 50000 == 0 and i > 0:
            logger.info(f"  Processed {i:,}/{n_groups:,} peptides...")

        # Get precursor info from first row using iloc (faster than iterrows)
        first_row = group.iloc[0]

        entry = LibraryEntry(
            modified_sequence=str(mod_seq),
            stripped_sequence=strip_modifications(str(mod_seq)),
            precursor_charge=int(prec_charge),
            precursor_mz=float(first_row["Precursor Mz"]),
            rt=float(first_row["Retention Time"])
            if has_rt and pd.notna(first_row["Retention Time"])
            else None,
            rt_start=float(first_row["Start Time"]) if pd.notna(first_row["Start Time"]) else None,
            rt_end=float(first_row["End Time"]) if pd.notna(first_row["End Time"]) else None,
        )

        # Add protein IDs if available
        if has_protein:
            protein = first_row["Protein Accession"]
            if pd.notna(protein):
                entry.protein_ids = [str(protein)]

        # Add fragments using itertuples (10x faster than iterrows)
        for row in group.itertuples(index=False):
            product_mz = row[group.columns.get_loc("Product Mz")]
            if pd.isna(product_mz) or product_mz <= 0:
                continue

            intensity = 1.0
            if has_area:
                area = row[group.columns.get_loc("Area")]
                if pd.notna(area):
                    intensity = float(area)

            product_charge = 1
            charge_val = row[group.columns.get_loc("Product Charge")]
            if pd.notna(charge_val):
                product_charge = int(charge_val)

            entry.fragments.append(
                Fragment(
                    mz=float(product_mz),
                    intensity=intensity,
                    ion_type=row[group.columns.get_loc("_ion_type")],
                    ion_number=row[group.columns.get_loc("_ion_number")],
                    charge=product_charge,
                    loss_type=row[group.columns.get_loc("_loss_type")],
                )
            )

        if entry.fragments:
            entries.append(entry)

    logger.info(
        f"Loaded {len(entries):,} peptides with {sum(len(e.fragments) for e in entries):,} fragments from PRISM CSV"
    )
    return entries
