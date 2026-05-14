# Model Card: Kenya Crop Yield Predictor

**Model ID:** `kenya-crop-yield-v1`
**Author:** Victor Chogo — AISIP Cohort 1, Pathway 4: AI Engineering
**Date:** May 2026
**Version:** 1.0

---

## Model Description

A supervised machine learning regression model that predicts agricultural crop yield
(measured in kg per hectare) for smallholder farms in Kenya. The model ingests climate,
soil, and agronomic input features to produce a yield estimate that can support farm
planning, extension advisory services, and food security monitoring.

**Primary use:** Crop yield estimation before harvest for Kenyan smallholder farmers.

---

## Intended Use

### Who Should Use This Model
- Agricultural extension officers advising Kenyan smallholder farmers
- NGOs and government agencies planning food-security interventions
- Agri-tech startups building tools for East African smallholder farmers
- Researchers studying climate-yield relationships in sub-Saharan Africa

### Who Should NOT Use This Model
- Commodity traders making financial decisions (too much uncertainty)
- Large-scale commercial farms (model is tuned for smallholder conditions)
- Regions outside Kenya without re-training on local data
- Policy decisions without expert review — this is a decision-support tool, not a policy tool

---

## Training Data

| Detail | Value |
|--------|-------|
| Source | Synthetic dataset generated from Kenya FAOSTAT statistics and KARI research reports |
| Size | 2,500 farm records |
| Coverage | 2000–2023, 7 Kenyan regions, 6 crop types |
| Target variable | `yield_kg_per_ha` (kg of crop per hectare) |

### Crops Covered
Maize, Wheat, Beans, Sorghum, Tea, Coffee

### Regions Covered
Rift Valley, Nyanza, Western, Central, Eastern, Coast, North Eastern

---

## Features

| Feature | Description | Type |
|---------|-------------|------|
| `rainfall_mm` | Annual rainfall in millimetres | Numeric |
| `temperature_celsius` | Mean annual temperature | Numeric |
| `fertilizer_kg_per_ha` | Fertilizer applied per hectare | Numeric |
| `pesticide_kg_per_ha` | Pesticide applied per hectare | Numeric |
| `farm_size_ha` | Farm area in hectares | Numeric |
| `elevation_m` | Elevation in metres above sea level | Numeric |
| `irrigation` | Whether irrigation was used (0/1) | Binary |
| `soil_type` | Soil classification | Categorical |
| `region` | Kenyan administrative region | Categorical |
| `crop_type` | Crop being grown | Categorical |
| `year` | Planting year (captures climate trends) | Numeric |
| `temp_deviation` | Absolute deviation from 22°C optimal | Engineered |
| `rain_fert_interaction` | Rainfall × Fertilizer interaction term | Engineered |
| `log_farm_size` | Log-transformed farm size | Engineered |
| `rainfall_category` | Binned rainfall category | Engineered |

---

## Model Performance

| Model | RMSE (kg/ha) | MAE (kg/ha) | R² | CV R² |
|-------|-------------|------------|-----|-------|
| Linear Regression | ~650 | ~480 | ~0.72 | ~0.71 |
| Ridge Regression | ~640 | ~470 | ~0.73 | ~0.72 |
| Random Forest | ~380 | ~270 | ~0.90 | ~0.89 |
| Gradient Boosting | ~360 | ~255 | ~0.91 | ~0.91 |
| **XGBoost** | **~340** | **~240** | **~0.93** | **~0.92** |

*Exact values generated during training. See `models/model_meta.json`.*

**Best Model:** XGBoost (lowest RMSE, highest R²)

**Evaluation metrics explained:**
- **RMSE** (Root Mean Squared Error): Average prediction error in kg/ha
- **MAE** (Mean Absolute Error): Median prediction error in kg/ha
- **R²**: Proportion of yield variance explained (1.0 = perfect)
- **CV R²**: Cross-validated R² — guards against overfitting

---

## Limitations & Risks

### Known Limitations
1. **Synthetic training data** — modelled on real statistics but not actual farm records. Model may underfit unusual growing conditions not captured in the generation process.
2. **No pest/disease modelling** — outbreak events can cut yields by 30–80%, and the model cannot predict these.
3. **No market/input price effects** — yield is a biological quantity; farm income depends on many additional factors.
4. **Limited temporal coverage** — climate variability post-2023 is not reflected.
5. **Coarse regional granularity** — predictions use region-level data, missing micro-climate variation within counties.

### Ethical Risks
- **Bias against marginalized farmers:** Model may be less accurate for pastoralist communities in North Eastern Kenya or coastal fishing communities who practice mixed livelihoods.
- **Data sovereignty:** In a production deployment, farm-level data must remain under farmer and community control; no data should be shared with insurance or credit providers without explicit consent.
- **Access inequality:** A digital tool benefits literate, smartphone-owning farmers more than others — extension officers must bridge this gap.
- **Over-reliance:** Farmers may ignore local knowledge in favour of model predictions. The tool is decision-support, not a replacement for agronomic expertise.

---

## Recommendations for Responsible Deployment

1. Partner with Kenya Agricultural and Livestock Research Organization (KALRO) to validate predictions against actual farm records before large-scale use.
2. Build an SMS/USSD interface in addition to the web app to reach farmers without smartphones.
3. Include Kiswahili and Kikuyu language options in the UI.
4. Implement a feedback loop: collect actual yields from users to continuously retrain the model.
5. Display confidence intervals alongside predictions — never present a single number as certainty.

---

## Citation

```
Victor Chogo (2026). Kenya Crop Yield Predictor v1.0.
Africa AI Hub AISIP Cohort 1 — Pathway 4: AI Engineering Capstone.
```
