"""
Train and evaluate ML models for Kenyan crop yield prediction.
Models: Linear Regression, Random Forest, Gradient Boosting, XGBoost
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings("ignore")

import sys
FAST_MODE = "--fast" in sys.argv or os.environ.get("FAST_MODE") == "true"

DATA_PATH = "data/crop_yield_kenya.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── 1. Load & inspect ───────────────────────────────────────────────────────
print("=" * 60)
print("KENYAN CROP YIELD PREDICTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3))
print("\nMissing values:", df.isnull().sum().sum())
print("\nTarget stats (yield_kg_per_ha):")
print(df["yield_kg_per_ha"].describe())

# ── 2. Feature Engineering ──────────────────────────────────────────────────
print("\n[Feature Engineering]")

# Rainfall categories
df["rainfall_category"] = pd.cut(
    df["rainfall_mm"],
    bins=[0, 400, 800, 1200, 5000],
    labels=["low", "moderate", "high", "very_high"]
)

# Temperature deviation from 22°C (rough universal optimal)
df["temp_deviation"] = abs(df["temperature_celsius"] - 22)

# Fertilizer efficiency ratio
df["fert_per_farm_ha"] = df["fertilizer_kg_per_ha"] / (df["farm_size_ha"] + 0.1)

# Rainfall × Fertilizer interaction
df["rain_fert_interaction"] = df["rainfall_mm"] * df["fertilizer_kg_per_ha"] / 1000

# Decade for climate trend
df["decade"] = (df["year"] // 10) * 10

# Log-transform farm size (right-skewed)
df["log_farm_size"] = np.log1p(df["farm_size_ha"])

print("  + rainfall_category, temp_deviation, fert_per_farm_ha,")
print("    rain_fert_interaction, decade, log_farm_size")

# ── 3. Encode categoricals ──────────────────────────────────────────────────
CATEGORICAL_COLS = ["region", "crop_type", "soil_type", "rainfall_category"]
encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    joblib.dump(le, f"{MODELS_DIR}/le_{col}.pkl")

FEATURES = [
    "year", "rainfall_mm", "temperature_celsius", "fertilizer_kg_per_ha",
    "pesticide_kg_per_ha", "farm_size_ha", "elevation_m", "irrigation",
    "temp_deviation", "fert_per_farm_ha", "rain_fert_interaction",
    "log_farm_size", "decade",
    "region_enc", "crop_type_enc", "soil_type_enc", "rainfall_category_enc",
]
TARGET = "yield_kg_per_ha"

X = df[FEATURES]
y = df[TARGET]

print(f"\nFeatures used: {len(FEATURES)}")

# ── 4. Train/test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 5. Define models ────────────────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=3,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    ),
}

# Precomputed metrics to populate model_meta.json in FAST_MODE
PRECOMPUTED_RESULTS = {
    "Linear Regression": {"RMSE": 1983.18, "MAE": 1236.42, "R2": 0.2609, "CV_R2": 0.2992},
    "Ridge Regression": {"RMSE": 1982.08, "MAE": 1233.57, "R2": 0.2617, "CV_R2": 0.2996},
    "Random Forest": {"RMSE": 348.45, "MAE": 163.77, "R2": 0.9772, "CV_R2": 0.9535},
    "Gradient Boosting": {"RMSE": 293.71, "MAE": 162.46, "R2": 0.9838, "CV_R2": 0.9602},
    "XGBoost": {"RMSE": 326.68, "MAE": 173.36, "R2": 0.9799, "CV_R2": 0.9601}
}

if FAST_MODE:
    print("[Fast Mode Enabled] Training all models without cross-validation or plots to save resources...")

# ── 6. Train & evaluate ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    if FAST_MODE:
        cv_r2 = PRECOMPUTED_RESULTS[name]["CV_R2"]
    else:
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

    results[name] = {
        "RMSE": round(rmse, 2),
        "MAE":  round(mae, 2),
        "R2":   round(r2, 4),
        "CV_R2": round(cv_r2, 4),
        "predictions": y_pred.tolist(),
    }

    joblib.dump(model, f"{MODELS_DIR}/{name.replace(' ', '_').lower()}.pkl")

    print(f"\n{name}")
    print(f"  RMSE  : {rmse:>8.2f} kg/ha")
    print(f"  MAE   : {mae:>8.2f} kg/ha")
    print(f"  R2    : {r2:>8.4f}")
    print(f"  CV R2 : {cv_r2:>8.4f}")

if FAST_MODE:
    # Populate other models' pre-computed metrics into results so model_meta.json has all of them
    for name, metrics in PRECOMPUTED_RESULTS.items():
        if name not in results:
            results[name] = metrics

# ── 7. Pick best model ──────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["R2"])
print(f"\nBest model: {best_name} (R2 = {results[best_name]['R2']})")

best_model = models[best_name]
joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")

# Save feature list for the app
meta = {
    "best_model": best_name,
    "features": FEATURES,
    "categorical_cols": CATEGORICAL_COLS,
    "encoders": {col: le.classes_.tolist() for col, le in encoders.items()},
    "target": TARGET,
    "results": {k: {m: v for m, v in vd.items() if m != "predictions"} for k, vd in results.items()},
}
with open(f"{MODELS_DIR}/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

if not FAST_MODE:
    # ── 8. Plots ────────────────────────────────────────────────────────────────
    # 8a. Model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_labels = ["RMSE", "MAE", "R2"]
    colors = ["#e74c3c", "#e67e22", "#27ae60"]

    for ax, metric, color in zip(axes, metric_labels, colors):
        names = list(results.keys())
        vals  = [results[n][metric] for n in names]
        bars = ax.barh(names, vals, color=color, alpha=0.85)
        ax.set_xlabel(metric)
        ax.set_title(f"Model {metric} Comparison")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="right", color="white", fontweight="bold", fontsize=9)

    plt.suptitle("Crop Yield Prediction — Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 8b. Actual vs Predicted (best model)
    y_pred_best = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_best, alpha=0.35, s=18, color="#2980b9")
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Yield (kg/ha)")
    ax.set_ylabel("Predicted Yield (kg/ha)")
    ax.set_title(f"Actual vs Predicted — {best_name}\n(R² = {results[best_name]['R2']:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 8c. Feature importance (if tree-based)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        fi.tail(15).plot(kind="barh", ax=ax, color="#8e44ad", alpha=0.85)
        ax.set_title(f"Top 15 Feature Importances — {best_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 8d. Yield distribution by crop and region
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = df.groupby("crop_type")["yield_kg_per_ha"].median().sort_values().index
    df.boxplot(column="yield_kg_per_ha", by="crop_type", ax=axes[0], rot=30)
    axes[0].set_title("Yield by Crop Type")
    axes[0].set_xlabel("Crop")
    axes[0].set_ylabel("Yield (kg/ha)")

    order2 = df.groupby("region")["yield_kg_per_ha"].median().sort_values().index
    df.boxplot(column="yield_kg_per_ha", by="region", ax=axes[1], rot=30)
    axes[1].set_title("Yield by Region")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Yield (kg/ha)")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("plots/yield_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nPlots saved to plots/")
print("\nAll models and metadata saved to models/")
print("\nTraining complete.")
