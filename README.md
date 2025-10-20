# 🌲 Biomass Estimation with Feature Selection

This repository implements a **Random Forest–based biomass regression workflow**  
using multi-source remote sensing data and interpretable feature selection.

---

## 📂 Project Structure
```
project_root/
│
├── main_train_biomass.py    # Train model and visualize results (select feature set)
├── shap_utils.py            # SHAP value computation and visualization
├── plot_figure.py           # Regression scatter plotting utility
├── data/
│   ├── wenchang_X.npy, wenchang_y.npy
│   ├── gzl_X.npy, gzl_y.npy
│   └── (original .npy inputs)
└── outputs/
    ├── prediction_scatter_*.png
    └── shap_summary_*.png
```

---

## 🧩 Dataset Description

Each input sample is a **12-dimensional feature vector** composed of  
optical bands, SAR polarimetric channels, and structural height attributes.

| Index | Feature Name | Description |
|:--:|:--|:--|
| 0 | R | Optical red reflectance |
| 1 | G | Optical green reflectance |
| 2 | B | Optical blue reflectance |
| 3 | C-HH | C-band SAR, HH polarization |
| 4 | C-HV | C-band SAR, HV polarization |
| 5 | C-VV | C-band SAR, VV polarization |
| 6 | L-HH | L-band SAR, HH polarization |
| 7 | L-HV | L-band SAR, HV polarization |
| 8 | L-VV | L-band SAR, VV polarization |
| 9 | H | Total tree height derived from optical depth estimation |
| 10 | hc | Canopy height (height of canopy top above trunk) |
| 11 | ht | Trunk height (height of main stem or bole) |

The target value (**y**) corresponds to *above-ground biomass (AGB)*  
derived from LiDAR or field calibration data.

---

## ⚙️ Feature Selection and Training

The script `main_train_biomass.py` allows you to evaluate  
how structural height features (H, hc, ht) influence biomass estimation.

To run, specify both the **region** and **feature selection mode**:
```python
main(region="wenchang", mode="full")
```

### Available Modes

| Mode | Description | Kept Features | Feature Count |
|:--|:--|:--|:--:|
| "full" | Use all features | R–B, C-, L-band channels, H, hc, ht | 12 |
| "no_H" | Remove total height H | R–B, C-, L-band, hc, ht | 11 |
| "no_HcHt" | Remove canopy & trunk heights | R–B, C-, L-band, H | 10 |
| "no_all_heights" | Remove all height features | Only R–B, C-, L-band | 9 |

Each run automatically:
1. Splits the data into 80% training / 20% testing;
2. Trains a **Random Forest Regressor** (`max_depth=10`, `n_estimators=100`);
3. Evaluates **R**, **RMSE**, **MAE**;
4. Plots:
   - `prediction_scatter_<mode>.png` — prediction vs ground truth;
   - `shap_summary_<mode>.png` — SHAP feature importance summary.

---

## 🧠 SHAP Interpretation

The **SHAP summary plots** show the relative contribution of each feature  
to biomass prediction, enabling physical interpretability across spectral,  
SAR, and structural dimensions.

> Red points indicate higher feature values, blue indicate lower values.  
> Horizontal ranking reflects the importance learned by the model.

---

## 🚀 Usage Example

1️⃣ **Prepare dataset**
```bash
python prepare_dataset.py
```

2️⃣ **Run feature selection experiment**
```bash
# Full feature set
python main_train_biomass.py --region wenchang --mode full

# Without total height (H)
python main_train_biomass.py --region wenchang --mode no_H

# Without canopy and trunk heights
python main_train_biomass.py --region wenchang --mode no_HcHt

# Without any height-related features
python main_train_biomass.py --region wenchang --mode no_all_heights
```

3️⃣ **Results**  
All results are automatically saved in:
```
outputs/<region>/
```

---

## 📈 Interpretation Goal

This framework enables comparative analysis of structural and radiometric contributions to biomass estimation, revealing:

- The marginal gain provided by **total height (H)**;
- The synergistic effect of **canopy (hc)** and **trunk (ht)** components;
- How **pure spectral–SAR features** perform when structural information is excluded.

---

## 🏷️ Citation
in preparation.
