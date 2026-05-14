# Kenya Crop Yield Prediction

**AISIP Cohort 1 — Pathway 4: AI Engineering Capstone**
**Africa AI Hub | Victor Chogo | May 2026**

---

## Problem Statement

Kenya's 7 million+ smallholder farming households produce 75% of the country's food but
have almost no access to data-driven yield forecasting tools. Without yield estimates, farmers
cannot plan storage, negotiate fair prices, or make informed decisions about fertilizer
investment. This project builds a machine learning model that predicts crop yield (kg/ha)
before harvest, using climate, soil, and agronomic inputs available to any farmer.

---

## Live Demo

> **Deployed app:** [https://kenya-crops-yield-predictiongit-5wld8bak9kdguwcqegcbmb.streamlit.app/](https://kenya-crops-yield-prediction-gbouimdy3e87gtaair8plc.streamlit.app/)
> **GitHub:** https://github.com/vikkirkobane/kenya-crops-yield-prediction
> Run locally: `streamlit run app.py`

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/vikkirkobane/kenya-crops-yield-prediction
cd kenya-crops-yield-prediction

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data/generate_data.py

# Train models (creates models/ and plots/)
python train_models.py

# Launch the Streamlit app
streamlit run app.py
```

---

## Project Structure

```
kenya-crop-yield-prediction/
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   └── crop_yield_kenya.csv   # Generated dataset (2,500 records)
├── models/                    # Saved model artifacts (auto-created)
│   ├── best_model.pkl
│   ├── model_meta.json
│   └── le_*.pkl               # Label encoders
├── plots/                     # Visualisations (auto-created)
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── yield_distributions.png
├── app.py                     # Streamlit web application
├── train_models.py            # Full training pipeline
├── model_card.md              # Model documentation
├── requirements.txt
└── README.md
```

---

## Dataset

| Field | Description |
|-------|-------------|
| `crop_type` | Maize, Wheat, Beans, Sorghum, Tea, Coffee |
| `region` | 7 Kenyan administrative regions |
| `rainfall_mm` | Annual rainfall |
| `temperature_celsius` | Mean annual temperature |
| `fertilizer_kg_per_ha` | Fertilizer application rate |
| `pesticide_kg_per_ha` | Pesticide application rate |
| `farm_size_ha` | Farm area |
| `elevation_m` | Elevation above sea level |
| `soil_type` | volcanic / loam / clay / sandy |
| `irrigation` | 0 or 1 |
| `year` | 2000–2023 |
| `yield_kg_per_ha` | **Target variable** |

**Source:** Synthetic data generated from FAOSTAT statistics, Kenya National Bureau of Statistics
agricultural surveys, and Kenya Meteorological Department climate norms.

---

## Models & Results

Five models were trained and compared:

| Model | RMSE (kg/ha) | MAE (kg/ha) | R² | CV R² |
|-------|-------------|------------|-----|-------|
| Linear Regression | 1983 | 1236 | 0.261 | 0.299 |
| Ridge Regression | 1982 | 1234 | 0.262 | 0.300 |
| Random Forest | 348 | 164 | 0.977 | 0.954 |
| XGBoost | 327 | 173 | 0.980 | 0.960 |
| **Gradient Boosting** | **294** | **162** | **0.984** | **0.960** |

**Best model:** Gradient Boosting — lowest RMSE (294 kg/ha) and highest R² (0.984).

### Feature Engineering Applied
- `temp_deviation` — absolute deviation from 22°C optimal temperature
- `rain_fert_interaction` — rainfall × fertilizer synergy term
- `log_farm_size` — log-transform of right-skewed farm size
- `fert_per_farm_ha` — fertilizer intensity ratio
- `rainfall_category` — binned rainfall (low / moderate / high / very_high)
- `decade` — decade grouping to capture climate trends

---

## Web Application

The Streamlit app (`app.py`) provides:
- **Predict Yield tab:** Input farm parameters and get instant yield predictions with comparison to Kenya national averages
- **Explore Data tab:** Browse and filter the training dataset
- **About tab:** Project background, model details, limitations, and responsible use guidance

---

## Ethical Considerations

See [`model_card.md`](model_card.md) for full details. Key points:
- Training data is synthetic — validate against real farm records before deployment
- Model is less accurate for North Eastern Kenya (sparse data region)
- Farm data must remain under farmer control — no sharing with lenders/insurers without consent
- This is a decision-support tool, not a replacement for agronomic expertise

---

## Roadmap (Advanced Track / Future Work)

- [ ] Replace synthetic data with real FAOSTAT + KNBS records
- [ ] Add LSTM for time-series yield forecasting using historical weather sequences
- [ ] SMS/USSD interface for farmers without smartphones
- [ ] Kiswahili language support
- [ ] Integrate satellite NDVI data as an additional feature

---

## License

MIT License — see LICENSE file.

---

*Africa AI Hub | AI Skills Immersion Programme (AISIP) Cohort 1 | Pathway 4: AI Engineering*
