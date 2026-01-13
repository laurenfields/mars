# Mars v0.1.0 Release Notes

## Overview

Initial release of Mars (Mass Accuracy Recalibration System), a mass calibration tool for Thermo Stellar unit resolution DIA mass spectrometry data. Mars learns m/z calibration corrections from spectral library fragment matches using an XGBoost model.

## Features

### XGBoost Calibration Model
Mars uses a machine learning approach to predict m/z corrections based on:
- **Fragment m/z**: Mass-dependent calibration bias
- **Peak intensity**: Higher intensity peaks provide more reliable calibration
- **Retention time**: Calibration drift over the LC gradient
- **Spectrum TIC**: Space charge effects from high ion current

### Fragment Matching
- Matches library peptides to DIA MS2 spectra using precursor m/z and RT windows
- Selects the most intense peak within m/z tolerance (not closest)
- Configurable minimum intensity threshold

### PRISM Integration
- Optional `--prism-csv` flag for using exact Skyline RT windows (`Start Time`, `End Time`)
- Falls back to ±5 seconds around library RT when PRISM CSV not provided

### Batch Processing
- Process multiple mzML files with glob patterns (`--mzml "*.mzML"`)
- Process entire directories with `--mzml-dir`

### QC Reports
Generated quality control outputs include:
- Delta m/z distribution histogram with MAD and RMS statistics (before/after calibration)
- 2D heatmap visualization (RT × m/z, color = delta)
- Model feature importance plot
- Calibration statistics summary

## Output Files

| File | Description |
|------|-------------|
| `{input}-mars.mzML` | Recalibrated mzML file |
| `mars_model.pkl` | Trained XGBoost calibration model |
| `mars_qc_histogram.png` | Delta m/z distribution with MAD and RMS (before/after) |
| `mars_qc_heatmap.png` | 2D heatmap (RT × m/z, color = delta) |
| `mars_qc_feature_importance.png` | Model feature importance |
| `mars_qc_summary.txt` | Calibration statistics |

## Installation

```bash
git clone https://github.com/maccoss/mars.git
cd mars
pip install -e .
```

Or from PyPI:

```bash
pip install mars-ms==0.1.0
```

## Requirements

- Python 3.10+
- Spectral library in blib format from Skyline
- mzML files from Thermo Stellar (or similar unit resolution instrument)
- PRISM CSV (optional): Skyline report with `Start Time`, `End Time`, `Replicate Name` columns

## License

Apache 2.0
