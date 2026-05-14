"""
Generate a realistic synthetic dataset for Kenyan crop yield prediction.
Based on Kenya FAOSTAT data, Kenya Agricultural Research Institute (KARI) reports,
and peer-reviewed literature on smallholder agriculture in East Africa.
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

CROPS = {
    "Maize":   {"base_yield": 1800, "optimal_rain": 800,  "optimal_temp": 22, "rain_sensitivity": 0.8},
    "Wheat":   {"base_yield": 2500, "optimal_rain": 650,  "optimal_temp": 18, "rain_sensitivity": 0.6},
    "Beans":   {"base_yield": 900,  "optimal_rain": 700,  "optimal_temp": 20, "rain_sensitivity": 0.7},
    "Sorghum": {"base_yield": 1200, "optimal_rain": 500,  "optimal_temp": 25, "rain_sensitivity": 0.5},
    "Tea":     {"base_yield": 8000, "optimal_rain": 1400, "optimal_temp": 17, "rain_sensitivity": 0.9},
    "Coffee":  {"base_yield": 900,  "optimal_rain": 1200, "optimal_temp": 20, "rain_sensitivity": 0.85},
}

REGIONS = {
    "Rift Valley": {"rain_mean": 900,  "rain_std": 200, "temp_mean": 19, "temp_std": 2,  "soil_types": ["volcanic", "loam"],      "elevation_range": (1500, 2800)},
    "Nyanza":      {"rain_mean": 1400, "rain_std": 300, "temp_mean": 23, "temp_std": 2,  "soil_types": ["clay", "loam"],          "elevation_range": (1100, 1800)},
    "Western":     {"rain_mean": 1600, "rain_std": 250, "temp_mean": 22, "temp_std": 2,  "soil_types": ["loam", "clay"],          "elevation_range": (1200, 2000)},
    "Central":     {"rain_mean": 1200, "rain_std": 200, "temp_mean": 17, "temp_std": 2,  "soil_types": ["volcanic", "loam"],      "elevation_range": (1500, 3000)},
    "Eastern":     {"rain_mean": 550,  "rain_std": 150, "temp_mean": 24, "temp_std": 3,  "soil_types": ["sandy", "clay"],         "elevation_range": (500, 1500)},
    "Coast":       {"rain_mean": 1000, "rain_std": 300, "temp_mean": 28, "temp_std": 2,  "soil_types": ["sandy", "loam"],         "elevation_range": (0, 600)},
    "North Eastern":{"rain_mean": 280, "rain_std": 80,  "temp_mean": 30, "temp_std": 3,  "soil_types": ["sandy", "clay"],         "elevation_range": (150, 600)},
}

SOIL_QUALITY = {"volcanic": 0.95, "loam": 0.80, "clay": 0.65, "sandy": 0.45}

CROP_REGION_FIT = {
    ("Maize", "Rift Valley"): 1.10, ("Maize", "Nyanza"): 1.05, ("Maize", "Western"): 1.05,
    ("Maize", "Central"): 0.90,     ("Maize", "Eastern"): 0.75, ("Maize", "Coast"): 0.70,
    ("Maize", "North Eastern"): 0.50,
    ("Wheat", "Rift Valley"): 1.15, ("Wheat", "Central"): 1.10, ("Wheat", "Nyanza"): 0.85,
    ("Wheat", "Western"): 0.80,     ("Wheat", "Eastern"): 0.60, ("Wheat", "Coast"): 0.50,
    ("Wheat", "North Eastern"): 0.35,
    ("Beans", "Rift Valley"): 1.05, ("Beans", "Nyanza"): 1.00, ("Beans", "Western"): 1.00,
    ("Beans", "Central"): 1.05,     ("Beans", "Eastern"): 0.75, ("Beans", "Coast"): 0.65,
    ("Beans", "North Eastern"): 0.45,
    ("Sorghum", "Eastern"): 1.20,   ("Sorghum", "North Eastern"): 1.10, ("Sorghum", "Nyanza"): 1.00,
    ("Sorghum", "Rift Valley"): 0.85, ("Sorghum", "Western"): 0.80, ("Sorghum", "Coast"): 0.75,
    ("Sorghum", "Central"): 0.70,
    ("Tea", "Central"): 1.20,       ("Tea", "Rift Valley"): 1.10, ("Tea", "Western"): 1.05,
    ("Tea", "Nyanza"): 0.90,        ("Tea", "Eastern"): 0.40,    ("Tea", "Coast"): 0.30,
    ("Tea", "North Eastern"): 0.10,
    ("Coffee", "Central"): 1.15,    ("Coffee", "Rift Valley"): 1.05, ("Coffee", "Nyanza"): 0.95,
    ("Coffee", "Western"): 0.90,    ("Coffee", "Eastern"): 0.65, ("Coffee", "Coast"): 0.55,
    ("Coffee", "North Eastern"): 0.30,
}

records = []
n_samples = 2500

for _ in range(n_samples):
    region = random.choice(list(REGIONS.keys()))
    r = REGIONS[region]

    crop = random.choice(list(CROPS.keys()))
    c = CROPS[crop]

    year = random.randint(2000, 2023)
    rainfall = max(50, np.random.normal(r["rain_mean"], r["rain_std"]))
    temperature = max(10, np.random.normal(r["temp_mean"], r["temp_std"]))
    soil_type = random.choice(r["soil_types"])
    elevation = random.randint(*r["elevation_range"])

    fertilizer = max(0, np.random.normal(40, 20))
    pesticide = max(0, np.random.normal(2.5, 1.5))
    farm_size = max(0.1, np.random.lognormal(0.3, 0.7))
    irrigation = 1 if (random.random() < 0.18 or region in ["Rift Valley", "Central"]) else 0

    rain_effect = 1.0 - 0.0004 * (rainfall - c["optimal_rain"]) ** 2 / (c["optimal_rain"] ** 2) * 300
    rain_effect = max(0.3, min(1.4, rain_effect))

    temp_effect = 1.0 - 0.05 * abs(temperature - c["optimal_temp"])
    temp_effect = max(0.4, min(1.2, temp_effect))

    fert_effect = 1.0 + 0.005 * min(fertilizer, 80)
    pest_effect = 1.0 + 0.04 * min(pesticide, 10)
    soil_effect = SOIL_QUALITY[soil_type]
    region_fit = CROP_REGION_FIT.get((crop, region), 0.80)
    irrig_effect = 1.25 if irrigation else 1.0

    climate_trend = 1.0 - 0.003 * max(0, year - 2005)

    yield_kg_ha = (
        c["base_yield"]
        * rain_effect
        * temp_effect
        * fert_effect
        * pest_effect
        * soil_effect
        * region_fit
        * irrig_effect
        * climate_trend
        * np.random.normal(1.0, 0.10)
    )
    yield_kg_ha = max(100, round(yield_kg_ha, 1))

    records.append({
        "year": year,
        "region": region,
        "crop_type": crop,
        "rainfall_mm": round(rainfall, 1),
        "temperature_celsius": round(temperature, 1),
        "fertilizer_kg_per_ha": round(fertilizer, 1),
        "pesticide_kg_per_ha": round(pesticide, 2),
        "farm_size_ha": round(farm_size, 2),
        "soil_type": soil_type,
        "elevation_m": elevation,
        "irrigation": irrigation,
        "yield_kg_per_ha": yield_kg_ha,
    })

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("C:/Users/victo/Desktop/Gemini Projects/ai-dev-africa/capstone/data/crop_yield_kenya.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df.shape[1]} columns")
print(df.describe())
