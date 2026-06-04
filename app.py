"""
Streamlit web application for Kenyan Crop Yield Prediction.
Supports all 6 major crops across 7 Kenyan regions.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── Load artifacts ───────────────────────────────────────────────────────────
MODELS_DIR = "models"

@st.cache_resource
def load_metadata_and_encoders():
    with open(f"{MODELS_DIR}/model_meta.json") as f:
        meta = json.load(f)
    encoders = {col: joblib.load(f"{MODELS_DIR}/le_{col}.pkl") for col in meta["categorical_cols"]}
    return meta, encoders

@st.cache_resource
def load_model(model_name):
    filename = model_name.replace(" ", "_").lower() + ".pkl"
    return joblib.load(f"{MODELS_DIR}/{filename}")

# Global fallback placeholders
active_model = None
selected_model_name = "Gradient Boosting"

try:
    meta, encoders = load_metadata_and_encoders()
    # Verify that all models are available and loadable (since they are git-ignored and must be trained on-the-fly)
    for model_name in meta["results"].keys():
        _ = load_model(model_name)
    model_loaded = True
    load_error = None
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────
def rainfall_category(mm):
    if mm < 400:
        return "low"
    elif mm < 800:
        return "moderate"
    elif mm < 1200:
        return "high"
    return "very_high"

def safe_encode(encoder, value):
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    return 0

def build_features(inputs, meta, encoders):
    rain = inputs["rainfall_mm"]
    temp = inputs["temperature_celsius"]
    fert = inputs["fertilizer_kg_per_ha"]
    farm = inputs["farm_size_ha"]
    year = inputs["year"]

    row = {
        "year": year,
        "rainfall_mm": rain,
        "temperature_celsius": temp,
        "fertilizer_kg_per_ha": fert,
        "pesticide_kg_per_ha": inputs["pesticide_kg_per_ha"],
        "farm_size_ha": farm,
        "elevation_m": inputs["elevation_m"],
        "irrigation": inputs["irrigation"],
        "temp_deviation": abs(temp - 22),
        "fert_per_farm_ha": fert / (farm + 0.1),
        "rain_fert_interaction": rain * fert / 1000,
        "log_farm_size": np.log1p(farm),
        "decade": (year // 10) * 10,
        "region_enc": safe_encode(encoders["region"], inputs["region"]),
        "crop_type_enc": safe_encode(encoders["crop_type"], inputs["crop_type"]),
        "soil_type_enc": safe_encode(encoders["soil_type"], inputs["soil_type"]),
        "rainfall_category_enc": safe_encode(encoders["rainfall_category"], rainfall_category(rain)),
    }
    return pd.DataFrame([row])[meta["features"]]

# ── Sidebar: model performance ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Flag_of_Kenya.svg", width=80)
    st.title("Kenya Crop Yield Predictor")
    st.caption("AI Skills Immersion Programme — Cohort 1 Capstone")
    st.divider()
    if model_loaded:
        st.subheader("Select Active Model")
        model_options = list(meta["results"].keys())
        default_index = model_options.index(meta["best_model"]) if meta["best_model"] in model_options else 0
        selected_model_name = st.selectbox(
            "Choose active model for predictions:",
            options=model_options,
            index=default_index
        )
        try:
            active_model = load_model(selected_model_name)
            st.success(f"**Active Model:** {selected_model_name}")
        except Exception as e:
            st.error(f"Failed to load {selected_model_name}: {e}. Fallback to best model.")
            active_model = load_model(meta["best_model"])
            selected_model_name = meta["best_model"]

        st.divider()
        st.subheader("Model Performance")
        results = meta["results"]
        perf_df = pd.DataFrame([
            {"Model": k, "RMSE": v["RMSE"], "MAE": v["MAE"], "R²": v["R2"]}
            for k, v in results.items()
        ]).sort_values("R²", ascending=False)
        st.dataframe(perf_df, hide_index=True, use_container_width=True)
    else:
        st.subheader("Model Performance")
        st.error(f"Model not loaded: {load_error}")
        st.info("Run `python train_models.py` first.")
    st.divider()
    st.caption("Built for the Africa AI Hub AISIP Cohort 1\nCapstone — Pathway 4: AI Engineering")

# ── Main content ──────────────────────────────────────────────────────────────
st.title("🌾 Kenyan Smallholder Crop Yield Predictor")
st.markdown(
    """
    Predict expected crop yield (kg/ha) based on your farm's location, crop type,
    climate conditions, and farming practices. Designed for Kenyan smallholder farmers
    and agricultural extension officers.
    """
)

tab_predict, tab_explore, tab_about = st.tabs(["Predict Yield", "Explore Data", "About"])

# ── TAB 1: PREDICTION ─────────────────────────────────────────────────────────
with tab_predict:
    if not model_loaded:
        st.warning("⚠️ Model files are not compatible with the current environment. Auto-retraining the models now to ensure compatibility...")
        with st.spinner("Running the model training pipeline (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost)... This can take up to 2 minutes on the cloud."):
            try:
                import sys
                import subprocess
                # Run train_models.py in the background and capture output
                result = subprocess.run([sys.executable, "train_models.py", "--fast"], capture_output=True, text=True, check=True)
                # Clear function caches to reload the newly trained models
                load_metadata_and_encoders.clear()
                load_model.clear()
                meta, encoders = load_metadata_and_encoders()
                _ = load_model(meta["best_model"])
                st.success("✅ Models retrained and loaded successfully!")
                st.rerun()
            except Exception as retrain_error:
                st.error(f"Failed to retrain models automatically: {retrain_error}")
                if 'result' in locals() and hasattr(result, 'stderr') and hasattr(result, 'stdout'):
                    st.text_area("Training logs (for debugging):", result.stderr + "\n" + result.stdout)
                st.stop()

    st.subheader("Enter Farm & Climate Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Farm Location & Crop**")
        crop = st.selectbox("Crop Type", ["Maize", "Wheat", "Beans", "Sorghum", "Tea", "Coffee"])
        region = st.selectbox("Region", [
            "Rift Valley", "Nyanza", "Western", "Central", "Eastern", "Coast", "North Eastern"
        ])
        soil = st.selectbox("Soil Type", ["volcanic", "loam", "clay", "sandy"])
        elevation = st.number_input("Elevation (m above sea level)", 0, 4000, 1500, step=50)

    with col2:
        st.markdown("**Climate Conditions**")
        rainfall = st.slider("Annual Rainfall (mm)", 100, 2500, 800)
        temperature = st.slider("Mean Temperature (°C)", 8, 38, 20)
        year = st.number_input("Year", 2000, 2030, 2024)

    with col3:
        st.markdown("**Farming Practices**")
        farm_size = st.number_input("Farm Size (ha)", 0.1, 50.0, 1.5, step=0.1)
        fertilizer = st.slider("Fertilizer Applied (kg/ha)", 0, 200, 40)
        pesticide = st.slider("Pesticide Applied (kg/ha)", 0.0, 15.0, 2.5, step=0.1)
        irrigation = st.toggle("Irrigation used?", value=False)

    st.divider()

    if st.button("Predict Yield", type="primary", use_container_width=True):
        inputs = {
            "crop_type": crop,
            "region": region,
            "soil_type": soil,
            "elevation_m": elevation,
            "rainfall_mm": rainfall,
            "temperature_celsius": temperature,
            "year": year,
            "farm_size_ha": farm_size,
            "fertilizer_kg_per_ha": fertilizer,
            "pesticide_kg_per_ha": pesticide,
            "irrigation": int(irrigation),
        }

        if active_model is None:
            st.error("No active model loaded. Please check model files or train models first.")
            st.stop()

        X_input = build_features(inputs, meta, encoders)
        prediction = active_model.predict(X_input)[0]
        total_yield = prediction * farm_size

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Predicted Yield", f"{prediction:,.0f} kg/ha")
        with res_col2:
            st.metric("Total Farm Yield", f"{total_yield:,.0f} kg", help=f"For your {farm_size} ha farm")
        with res_col3:
            rain_cat = rainfall_category(rainfall)
            if prediction > 2000:
                rating = "High ✅"
            elif prediction > 1000:
                rating = "Moderate 🟡"
            else:
                rating = "Low ⚠️"
            st.metric("Yield Rating", rating)

        st.info(
            f"**Interpretation:** A {crop} farm in {region} with {rainfall} mm rainfall, "
            f"{temperature}°C temperature, and {fertilizer} kg/ha fertilizer is predicted to yield "
            f"**{prediction:,.0f} kg/ha**. "
            + ("Irrigation boosts this estimate by ~25%. " if irrigation else "")
            + f"Model: {selected_model_name} (R² = {meta['results'][selected_model_name]['R2']:.3f})"
        )

        # Rainfall benchmark
        nat_avg = {"Maize": 1800, "Wheat": 2500, "Beans": 900, "Sorghum": 1200, "Tea": 8000, "Coffee": 900}
        nat_mean = nat_avg.get(crop, 1500)
        pct_diff = (prediction - nat_mean) / nat_mean * 100
        if pct_diff >= 0:
            st.success(f"📈 Your predicted yield is **{pct_diff:.1f}% above** Kenya's national average for {crop} ({nat_mean:,} kg/ha).")
        else:
            st.warning(f"📉 Your predicted yield is **{abs(pct_diff):.1f}% below** Kenya's national average for {crop} ({nat_mean:,} kg/ha). Consider increasing fertilizer or checking irrigation options.")

# ── TAB 2: EXPLORE DATA ───────────────────────────────────────────────────────
with tab_explore:
    st.subheader("Dataset Exploration")

    try:
        df = pd.read_csv("data/crop_yield_kenya.csv")
        st.write(f"Dataset: {len(df):,} farm records from 2000–2023 across 7 Kenyan regions.")

        c1, c2 = st.columns(2)
        with c1:
            selected_crop = st.selectbox("Filter by Crop", ["All"] + sorted(df["crop_type"].unique().tolist()))
        with c2:
            selected_region = st.selectbox("Filter by Region", ["All"] + sorted(df["region"].unique().tolist()))

        dff = df.copy()
        if selected_crop != "All":
            dff = dff[dff["crop_type"] == selected_crop]
        if selected_region != "All":
            dff = dff[dff["region"] == selected_region]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records", f"{len(dff):,}")
        m2.metric("Avg Yield", f"{dff['yield_kg_per_ha'].mean():,.0f} kg/ha")
        m3.metric("Max Yield", f"{dff['yield_kg_per_ha'].max():,.0f} kg/ha")
        m4.metric("Min Yield", f"{dff['yield_kg_per_ha'].min():,.0f} kg/ha")

        st.markdown("**Sample Records**")
        st.dataframe(dff.head(20), use_container_width=True)

    except FileNotFoundError:
        st.info("Dataset not found. Run `python data/generate_data.py` to create it.")

# ── TAB 3: ABOUT ──────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About This Project")
    st.markdown("""
    ### Kenya Crop Yield Prediction — AI Capstone

    **Problem:** Kenyan smallholder farmers — who produce 75% of Kenya's food supply — lack
    access to data-driven tools to estimate their yields before harvest. This leads to poor
    planning, food insecurity, and financial losses for over 7 million farming households.

    **Solution:** A machine learning model trained on climate, soil, and agronomic data to
    predict crop yield (kg/ha) for 6 major crops across 7 Kenyan regions.

    ---

    #### Models Trained
    | Model | Description |
    |-------|-------------|
    | Linear Regression | Baseline linear model |
    | Ridge Regression | Regularized linear model |
    | Random Forest | Ensemble of decision trees |
    | Gradient Boosting | Sequential boosting ensemble |
    | XGBoost | Optimized gradient boosting |

    #### Key Features
    - Rainfall (mm), Temperature (°C), Elevation (m)
    - Fertilizer & pesticide application rates
    - Soil type, irrigation status
    - Region & crop type (encoded)
    - Engineered: rainfall category, temp deviation, fert efficiency

    #### Limitations
    - Trained on synthetic data modelled from real Kenyan agricultural statistics
    - Does not account for pest/disease outbreaks or market shocks
    - Predictions are estimates — field conditions always vary

    #### Data Sources (for production dataset)
    - FAOSTAT (FAO crop production statistics)
    - Kenya National Bureau of Statistics (KNBS) agricultural surveys
    - Kenya Meteorological Department rainfall records
    - Kenya Soil Survey data

    ---
    *Africa AI Hub — AI Skills Immersion Programme (AISIP) Cohort 1*
    *Pathway 4: AI Engineering Capstone*
    """)
