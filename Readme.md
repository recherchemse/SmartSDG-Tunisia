# Composite Drought Index Dataset for Tunisia (2000–2025)

This repository provides the source code and scripts used to develop a high-resolution (1 km) **Composite Drought Index (CDI)** dataset for Tunisia for the period 2000–2025, and to forecast drought dynamics using deep learning models.

The CDI integrates multi-source remote sensing and reanalysis datasets (CHIRPS, ERA5-Land, and MODIS) to capture meteorological, agricultural, and thermal drought conditions.


## Paper Reference

**Title:** A Novel Composite Drought Index from Satellite Imagery and Its Forecasting using Deep Learning Models : Case Study for Tunisia 2000-2025

> If you use this dataset or code, please cite the associated paper.

## Project Overview

This project develops a novel CDI using multi-source Earth Observation data (CHIRPS, ERA5-Land, MODIS) at 1 km resolution. It integrates SPI, SPEI, soil moisture anomaly (SMA), NDVI anomaly, and LST anomaly via a logic-based cause-effect framework for drought monitoring and classifies into Normal, Watch, Warning, Alert-1/2, and Urgency stages.

Deep learning forecasts CDI using 12 models (e.g., TimeFormer, SSSLN, LSTM), with TimeFormer performing best (Accuracy: 0.9057).

## Key Features
High-resolution (1 km) national drought dataset for Tunisia 2000-2025.

Captures major events (e.g., 2000-2002, 2016-2018, 2021-2025 droughts).

Validated against ground stations (R²=0.75 for SPI).

Designed for reproducibility, extensibility, and transferability to other regions.

## Deep Learning Forecasting

## Repository Structure

```
/data/              # Input data (scripts)
/models/            # Deep learning training and testing scripts
 # Outputs (maps, plots, metrics)
README.md
requirements.txt

## Installation

Clone the repository:
```bash
git clone https://github.com/recherchemse/SMARTSDG.git
cd smart_sdg_composite_drought_index
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Compute drought indicators

Run scripts to generate SPI, SPEI, SMA, NDVI-A, and LST-A:

```bash
python Data/compute_indicators.py
```

### 2. Generate CDI maps

Run the CDI construction pipeline:

```bash
python scripts/build_cdi.py
```

### 3. Train deep learning models

Train and evaluate models using:

```bash
python train_model.py --model TimeFormer
```

### 4. Forecast CDI

Forecast CDI for future months:

```bash
python forecast.py --model TimeFormer
```
## Results

- The CDI dataset shows strong **spatio-temporal drought variability** across Tunisia, with severe impacts particularly in **central and southern regions**.
- Validation against ground rainfall stations shows strong agreement between station-based SPI and CHIRPS-derived SPI.
- 
## Citation

If you use this repository, please cite:

```bibtex
@article{paper,
  title={A Novel Composite Drought Index from Satellite Imagery and Its Forecasting using Deep Learning Models : Case Study for Tunisia 2000-2025},
  author={Ben Othmen Dhouha, Oueslati Fedi, Chouikhi Farah, Ben Abbes Ali, Rhif Manel, Baltin Hanen, Farah Mohamed and Farah Imed Riadh},
  year={2026}
}
```

## License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration, please contact:

- **Name:** [Pr. Riadh Farah]
- **Email:** [recherche@mse.uma.tn]
