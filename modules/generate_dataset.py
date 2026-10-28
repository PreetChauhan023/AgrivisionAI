"""
Module: generate_dataset.py
Purpose: Generate a synthetic agricultural dataset with 10,000 rows and 30 columns.
This dataset mimics real-world crop health monitoring scenarios.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# Constants / Lookup Tables
# ─────────────────────────────────────────────
CROP_TYPES     = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane", "Potato", "Tomato"]
SOIL_TYPES     = ["Loamy", "Sandy", "Clay", "Silt", "Peaty", "Chalky", "Saline"]
LEAF_COLORS    = ["Green", "Yellow", "Brown", "Pale Green", "Dark Green", "Reddish"]
DISEASE_TYPES  = ["Blight", "Rust", "Mildew", "Mosaic Virus", "Root Rot", "Leaf Spot", "Healthy"]
GROWTH_STAGES  = ["Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"]
REGIONS        = ["North", "South", "East", "West", "Central"]
SEASONS        = ["Kharif", "Rabi", "Zaid", "Summer", "Winter"]
WATER_QUALITY  = ["Good", "Moderate", "Poor"]
AIR_QUALITY    = ["Clean", "Moderate", "Polluted"]
FERTILIZERS    = ["Urea", "DAP", "NPK", "Organic", "Compost", "None"]
SEVERITY_LEVELS= ["None", "Low", "Moderate", "High", "Severe"]
STRESS_LEVELS  = ["None", "Mild", "Moderate", "High"]


def generate_dataset(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic agricultural dataset.

    Parameters
    ----------
    n_rows : int   – Number of rows to generate (default 10 000)
    seed   : int   – Random seed for reproducibility

    Returns
    -------
    pd.DataFrame   – Fully formed dataset
    """
    rng = np.random.default_rng(seed)

    # ── Categorical columns ──────────────────────────────────────────────────
    crop_type      = rng.choice(CROP_TYPES,     n_rows)
    soil_type      = rng.choice(SOIL_TYPES,     n_rows)
    leaf_color     = rng.choice(LEAF_COLORS,    n_rows)
    disease_type   = rng.choice(DISEASE_TYPES,  n_rows)
    growth_stage   = rng.choice(GROWTH_STAGES,  n_rows)
    region         = rng.choice(REGIONS,        n_rows)
    season         = rng.choice(SEASONS,        n_rows)
    water_quality  = rng.choice(WATER_QUALITY,  n_rows)
    air_quality    = rng.choice(AIR_QUALITY,    n_rows)
    fertilizer     = rng.choice(FERTILIZERS,    n_rows)
    severity       = rng.choice(SEVERITY_LEVELS,n_rows)
    stress_level   = rng.choice(STRESS_LEVELS,  n_rows)
    pest_presence  = rng.choice(["Yes", "No"],  n_rows)

    # ── Continuous / numerical columns ──────────────────────────────────────
    temperature    = rng.uniform(15,  45,  n_rows).round(2)   # °C
    humidity       = rng.uniform(30,  95,  n_rows).round(2)   # %
    rainfall       = rng.uniform(200, 2000,n_rows).round(2)   # mm/year
    ph_level       = rng.uniform(4.5, 8.5, n_rows).round(2)
    nitrogen       = rng.uniform(10,  150, n_rows).round(2)   # kg/ha
    phosphorus     = rng.uniform(5,   100, n_rows).round(2)
    potassium      = rng.uniform(10,  200, n_rows).round(2)
    leaf_spots     = rng.integers(0,  20,  n_rows)
    irrigation_lvl = rng.uniform(0,   100, n_rows).round(2)   # %
    sunlight_hours = rng.uniform(4,   14,  n_rows).round(2)   # hrs/day
    soil_moisture  = rng.uniform(10,  80,  n_rows).round(2)   # %
    ndvi_index     = rng.uniform(0.1, 0.9, n_rows).round(3)
    leaf_area      = rng.uniform(5,   60,  n_rows).round(2)   # cm²
    chlorophyll    = rng.uniform(10,  80,  n_rows).round(2)   # SPAD units

    # ── Yield (target-adjacent) – derived with some noise ───────────────────
    # Higher nutrients, NDVI, and rainfall → higher yield
    yield_ = (
        0.3 * ndvi_index * 100
        + 0.2 * (rainfall / 200)
        + 0.1 * chlorophyll
        + 0.15 * nitrogen / 10
        + rng.uniform(-5, 5, n_rows)
    ).clip(5, 60).round(2)

    # ── Disease label (target) ───────────────────────────────────────────────
    # Crops are "Healthy" when NDVI is high, leaf_spots are low, and pH is balanced
    health_score = (
        ndvi_index * 100
        - leaf_spots * 2
        - (ph_level - 6.5).clip(0) * 5        # penalty for high pH
        + chlorophyll * 0.3
        - (leaf_color == "Yellow").astype(int) * 10
        - (disease_type != "Healthy").astype(int) * 20
    )
    # Healthy if score > 50 (approximately 60 % healthy in dataset)
    healthy = (health_score > 50).astype(int)
    disease_label = np.where(healthy == 1, "Healthy", disease_type)

    # ── Assemble DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "Crop_Type"        : crop_type,
        "Soil_Type"        : soil_type,
        "Temperature"      : temperature,
        "Humidity"         : humidity,
        "Rainfall"         : rainfall,
        "pH_Level"         : ph_level,
        "Nitrogen"         : nitrogen,
        "Phosphorus"       : phosphorus,
        "Potassium"        : potassium,
        "Leaf_Color"       : leaf_color,
        "Leaf_Spots"       : leaf_spots,
        "Disease_Type"     : disease_type,
        "Disease_Severity" : severity,
        "Irrigation_Level" : irrigation_lvl,
        "Sunlight_Hours"   : sunlight_hours,
        "Growth_Stage"     : growth_stage,
        "Pest_Presence"    : pest_presence,
        "Fertilizer_Used"  : fertilizer,
        "Yield"            : yield_,
        "Region"           : region,
        "Season"           : season,
        "Water_Quality"    : water_quality,
        "Air_Quality"      : air_quality,
        "Soil_Moisture"    : soil_moisture,
        "NDVI_Index"       : ndvi_index,
        "Stress_Level"     : stress_level,
        "Leaf_Area"        : leaf_area,
        "Chlorophyll_Level": chlorophyll,
        "Disease_Label"    : disease_label,
        "Healthy"          : healthy,
    })

    return df


def load_or_generate(path: str = "data/agri_dataset.csv") -> pd.DataFrame:
    """
    Load dataset from CSV if it exists; otherwise generate and save it.

    Parameters
    ----------
    path : str – Relative path (from project root) where the CSV is stored.

    Returns
    -------
    pd.DataFrame
    """
    if os.path.exists(path):
        return pd.read_csv(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_dataset()
    df.to_csv(path, index=False)
    return df
