# Composite Drought Index Dataset for Tunisia (2000–2025)

This repository provides the source code and scripts used to develop a high-resolution (1 km) **Composite Drought Index (CDI)** dataset for Tunisia for the period 2000–2025, and to forecast drought dynamics using deep learning models.

The CDI integrates multi-source remote sensing and reanalysis datasets (CHIRPS, ERA5-Land, and MODIS) to capture meteorological, agricultural, and thermal drought conditions.


## Paper Reference

**Title:** A Novel Composite Drought Index from Satellite Imagery and Its Forecasting using Deep Learning Models : Case Study for Tunisia 2000-2025

> If you use this dataset or code, please cite the associated paper.

## Overview

The **Composite Drought Index (CDI)** is built using five drought-related indicators:

| Indicator | Description |
|-----------|-------------|
| **SPI-1** | Standardized Precipitation Index |
| **SPEI-1** | Standardized Precipitation Evapotranspiration Index |
| **SMA** | Soil Moisture Anomaly |
| **NDVI-A** | NDVI Anomaly |
| **LST-A** | Land Surface Temperature Anomaly |

These indicators are combined using a **logic-based cause–effect framework** to classify drought conditions into six categories:

- 🟢 **Normal**
- 🟡 **Watch**
- 🟠 **Warning**
- 🔴 **Alert-1**
- 🔴 **Alert-2**
- ⚫ **Urgency**

The dataset highlights major drought episodes in Tunisia, including **2002–2003**, **2016–2018**, and **2020–2024**.

## Data Sources

The CDI dataset is derived from:

| Source | Variable | Resolution | Frequency |
|--------|----------|------------|-----------|
| **CHIRPS v2** | Precipitation | ~5 km | Monthly |
| **ERA5-Land** | Temperature, Evaporation, Soil Moisture | ~9 km | Monthly |
| **MODIS MOD13A3** | NDVI | 1 km | Monthly |
| **MODIS MOD11A1** | Land Surface Temperature | 1 km | Daily → Monthly |

> All datasets were processed and resampled to a common **1 km resolution**.
> 
## Deep Learning Forecasting

Several deep learning architectures were tested for CDI forecasting:

- TimeFormer ⭐ *(best overall performance)*
- SSSLN
- Transformer
- CNN-LSTM / CNN-GRU
- TCN
- LSTM / GRU
- ConvLSTM
- NHITS
- TiDE
- iTransformer
## Repository Structure

```
/data/              # Input data (scripts)
/models/            # Deep learning training and testing scripts
/Visualisations/    # Outputs (maps, plots, metrics)
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
