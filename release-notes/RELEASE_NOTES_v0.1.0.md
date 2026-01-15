# Mars v0.1.0 Release Notes

## Overview

Initial release of Mars (Mass Accuracy Recalibration System), a mass calibration tool for Thermo Stellar unit resolution DIA mass spectrometry data. Mars learns m/z calibration corrections from spectral library fragment matches using an XGBoost model.

## Features

Mars uses a machine learning approach to predict m/z corrections based on:
- **Fragment m/z**: Mass-dependent calibration bias
- **Peak intensity**: Higher intensity peaks provide more reliable calibration
- **Absolute time**: Calibration drift over the acquisition run
- **Spectrum TIC**: Space charge effects from high ion current
- **Ion injection time**: Signal accumulation duration effects
- **Precursor m/z**: DIA isolation window-specific effects
- **Adjacent ion population**: Ion density in neighboring m/z ranges (0.5-1.5, 1.5-2.5, 2.5-3.5 Th above)
- **Adjacent ion ratios**: Relative ion density (adjacent ions / fragment ions)
- **RF temperatures**: Thermal effects from RF amplifier (RFA2) and electronics (RFC2)

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
- Hexbin density plots (intensity, RT, m/z, injection time, TIC, fragment ions vs mass error)
- Model feature importance plot
- Calibration statistics summary

## Output Files

| File | Description |
|------|-------------|
| `{input}-mars.mzML` | Recalibrated mzML file |
| `mars_model.pkl` | Trained XGBoost calibration model |
| `mars_qc_histogram.png` | Delta m/z distribution (before/after) |
| `mars_qc_heatmap.png` | 2D heatmap (RT × m/z, color = delta) |
| `mars_qc_intensity_vs_error.png` | Intensity vs mass error hexbin |
| `mars_qc_rt_vs_error.png` | RT vs mass error hexbin |
| `mars_qc_mz_vs_error.png` | Fragment m/z vs mass error hexbin |
| `mars_qc_tic_vs_error.png` | TIC vs mass error hexbin |
| `mars_qc_injection_time_vs_error.png` | Injection time vs mass error hexbin |
| `mars_qc_tic_injection_time_vs_error.png` | TIC×injection time vs mass error hexbin |
| `mars_qc_fragment_ions_vs_error.png` | Fragment ions vs mass error hexbin |
| `mars_qc_rfa2_temperature_vs_error.png` | RFA2 temperature vs error (if available) |
| `mars_qc_rfc2_temperature_vs_error.png` | RFC2 temperature vs error (if available) |
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

MIT
