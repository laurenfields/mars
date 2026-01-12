# Mars: Mass Accuracy Recalibration System

Mass recalibration tool for Thermo Stellar unit resolution DIA mass spectrometry data.

## Overview

Mars learns m/z calibration corrections from spectral library fragment matches. The XGBoost model accounts for:

- **Fragment m/z**: Mass-dependent calibration bias
- **Peak intensity**: Higher intensity peaks provide more reliable calibration
- **Retention time**: Calibration drift over the LC gradient  
- **Spectrum TIC**: Space charge effects from high ion current
- **Precursor window**: DIA isolation window-specific effects

## How It Works

1. **Fragment matching**: For each DIA MS2 spectrum, Mars finds library peptides where:
   - The precursor m/z falls within the DIA isolation window
   - The spectrum RT is within the peptide's elution window

2. **Peak selection**: For each expected fragment, Mars selects the **most intense** peak within the m/z tolerance (not the closest), filtering for minimum intensity

3. **Model training**: Each matched fragment becomes a training point with features: `[precursor_mz, fragment_mz, rt, log_tic, log_intensity]` and target: `delta_mz`

4. **Calibration**: The trained model predicts m/z corrections for all peaks in the mzML

## Installation

```bash
git clone https://github.com/maccoss/mars.git
cd mars
pip install -e .
```

**Requirements**: Python 3.10+, pyteomics, xgboost, numpy, pandas, matplotlib, seaborn, click

## Usage

### With PRISM CSV (Recommended)

Use a PRISM Skyline report CSV for accurate RT windows:

```bash
mars calibrate \
  --mzml data.mzML \
  --library library.blib \
  --prism-csv prism_report.csv \
  --tolerance 0.2 \
  --min-intensity 1500 \
  --output-dir output/
```

### Basic Usage

```bash
mars calibrate --mzml data.mzML --library library.blib --output-dir output/
```

### Batch Processing

```bash
# Multiple files with wildcard
mars calibrate --mzml "*.mzML" --library library.blib --output-dir output/

# All files in directory
mars calibrate --mzml-dir /path/to/data/ --library library.blib --output-dir output/
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mzml` | - | Path to mzML file or glob pattern |
| `--mzml-dir` | - | Directory containing mzML files |
| `--library` | - | Path to blib spectral library (required) |
| `--prism-csv` | - | PRISM Skyline CSV with Start/End Time columns |
| `--tolerance` | 0.7 | m/z tolerance for matching (Th) |
| `--min-intensity` | 500 | Minimum peak intensity for matching |
| `--output-dir` | `.` | Output directory |
| `--model-path` | - | Path to save/load calibration model |
| `--no-recalibrate` | - | Only train model, don't write mzML |

## RT Window Behavior

- **With `--prism-csv`**: Uses exact `Start Time` and `End Time` from Skyline
- **Without `--prism-csv`**: Uses ±5 seconds around the blib library RT

## Output Files

| File | Description |
|------|-------------|
| `{input}-mars.mzML` | Recalibrated mzML file |
| `mars_model.pkl` | Trained XGBoost calibration model |
| `mars_qc_histogram.png` | Delta m/z distribution (before/after) |
| `mars_qc_heatmap.png` | 2D heatmap (RT × m/z, color = delta) |
| `mars_qc_feature_importance.png` | Model feature importance |
| `mars_qc_summary.txt` | Calibration statistics |

## Model Features

The XGBoost model uses 5 features to predict m/z corrections:

1. `precursor_mz` - DIA isolation window center
2. `fragment_mz` - Fragment m/z being calibrated  
3. `rt` - Retention time
4. `log_tic` - Log10 of spectrum total ion current
5. `log_intensity` - Log10 of peak intensity

## Python API

```python
from mars import load_blib, read_dia_spectra, match_library_to_spectra, MzCalibrator

# Load library and match
library = load_blib("library.blib")
spectra = read_dia_spectra("data.mzML")
matches = match_library_to_spectra(library, spectra, mz_tolerance=0.2, min_intensity=1500)

# Train and save model
calibrator = MzCalibrator()
calibrator.fit(matches)
calibrator.save("model.pkl")
```

## Requirements

- **Spectral library**: blib format from Skyline with fragment annotations
- **mzML files**: DIA data from Thermo Stellar (or similar unit resolution instrument)  
- **PRISM CSV** (optional): Skyline report with `Start Time`, `End Time`, `Replicate Name` columns

## License

Apache 2.0
