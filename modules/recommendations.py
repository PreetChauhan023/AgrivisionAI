"""
Module: recommendations.py
Purpose: Rule-based recommendation engine for fertilizer, irrigation,
         and crop suggestions based on soil and environmental inputs.
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Fertilizer Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def fertilizer_recommendation(nitrogen: float, phosphorus: float,
                               potassium: float, crop: str) -> dict:
    """
    Suggest fertilizer type and application rate based on nutrient levels.

    Returns
    -------
    dict with keys: primary, secondary, rate_kg_ha, notes
    """
    recs = []

    # Nitrogen deficiency
    if nitrogen < 40:
        recs.append({
            "nutrient": "Nitrogen (N)",
            "level"   : "Low",
            "product" : "Urea (46-0-0)",
            "rate"    : f"{int((40 - nitrogen) * 2.2)} kg/ha",
            "note"    : "Apply in split doses: 50% at sowing, 50% at 30 days.",
        })
    elif nitrogen > 120:
        recs.append({
            "nutrient": "Nitrogen (N)",
            "level"   : "High",
            "product" : "Reduce Nitrogen inputs",
            "rate"    : "0 kg/ha",
            "note"    : "Excess N causes lodging and pest attraction. Skip top-dressing.",
        })
    else:
        recs.append({
            "nutrient": "Nitrogen (N)",
            "level"   : "Optimal",
            "product" : "Maintain current schedule",
            "rate"    : "As per schedule",
            "note"    : "Nitrogen levels are adequate.",
        })

    # Phosphorus deficiency
    if phosphorus < 20:
        recs.append({
            "nutrient": "Phosphorus (P)",
            "level"   : "Low",
            "product" : "DAP (18-46-0)",
            "rate"    : f"{int((20 - phosphorus) * 2.5)} kg/ha",
            "note"    : "Apply at basal dose before sowing for best uptake.",
        })
    elif phosphorus > 80:
        recs.append({
            "nutrient": "Phosphorus (P)",
            "level"   : "High",
            "product" : "Reduce P inputs",
            "rate"    : "0 kg/ha",
            "note"    : "High P can lock out Zinc and Iron. Monitor micronutrient status.",
        })
    else:
        recs.append({
            "nutrient": "Phosphorus (P)",
            "level"   : "Optimal",
            "product" : "Maintain current schedule",
            "rate"    : "As per schedule",
            "note"    : "Phosphorus levels are adequate.",
        })

    # Potassium deficiency
    if potassium < 50:
        recs.append({
            "nutrient": "Potassium (K)",
            "level"   : "Low",
            "product" : "MOP (0-0-60) or SOP (0-0-50)",
            "rate"    : f"{int((50 - potassium) * 1.67)} kg/ha",
            "note"    : "Potassium improves drought resistance and grain quality.",
        })
    elif potassium > 160:
        recs.append({
            "nutrient": "Potassium (K)",
            "level"   : "High",
            "product" : "Reduce K inputs",
            "rate"    : "0 kg/ha",
            "note"    : "High K may antagonise Magnesium and Calcium uptake.",
        })
    else:
        recs.append({
            "nutrient": "Potassium (K)",
            "level"   : "Optimal",
            "product" : "Maintain current schedule",
            "rate"    : "As per schedule",
            "note"    : "Potassium levels are adequate.",
        })

    return {"recommendations": recs, "crop": crop}


# ─────────────────────────────────────────────────────────────────────────────
# Irrigation Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def irrigation_recommendation(soil_moisture: float, temperature: float,
                               rainfall: float, crop: str,
                               growth_stage: str) -> dict:
    """
    Suggest irrigation frequency and volume based on current conditions.

    Returns
    -------
    dict with schedule, method, notes
    """
    # Daily evapotranspiration estimate (simplified Penman)
    et_factor = 0.0023 * (temperature + 17.8) * (temperature - 0.5) ** 0.5
    et_factor = max(et_factor, 1.0)

    if soil_moisture < 30:
        urgency  = "🔴 Immediate irrigation required"
        freq     = "Every 1–2 days"
        volume   = "40–50 mm"
    elif soil_moisture < 50:
        urgency  = "🟡 Irrigation recommended soon"
        freq     = "Every 3–4 days"
        volume   = "25–35 mm"
    else:
        urgency  = "🟢 Adequate soil moisture"
        freq     = "Every 6–7 days"
        volume   = "15–20 mm"

    # Adjust for growth stage
    stage_mult = {
        "Germination" : 0.8,
        "Seedling"    : 0.9,
        "Vegetative"  : 1.0,
        "Flowering"   : 1.3,   # Critical period
        "Fruiting"    : 1.2,
        "Harvest"     : 0.6,
    }
    multiplier = stage_mult.get(growth_stage, 1.0)

    method_map = {
        "Cotton"   : "Drip irrigation recommended — saves 30% water",
        "Rice"     : "Flood irrigation standard; consider AWD (Alternate Wetting & Drying)",
        "Wheat"    : "Furrow or sprinkler irrigation",
        "Tomato"   : "Drip irrigation preferred to avoid foliar diseases",
        "Sugarcane": "Furrow or subsurface drip irrigation",
        "default"  : "Sprinkler or furrow irrigation",
    }
    method = method_map.get(crop, method_map["default"])

    return {
        "urgency"      : urgency,
        "frequency"    : freq,
        "volume"       : volume,
        "method"       : method,
        "et_estimate"  : round(et_factor * multiplier, 2),
        "stage_factor" : multiplier,
        "notes"        : (
            f"Growth stage '{growth_stage}' factor: {multiplier}. "
            f"Adjust volume by ×{multiplier} vs baseline."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Crop Suggestion
# ─────────────────────────────────────────────────────────────────────────────

def crop_suggestion(ph: float, temperature: float, rainfall: float,
                    soil_type: str, season: str) -> list[dict]:
    """
    Recommend the top 3 most suitable crops based on environmental parameters.

    Returns
    -------
    list of dicts with keys: crop, suitability_score (%), rationale
    """
    # Define crop preference ranges
    CROP_PROFILES = {
        "Wheat"    : {"ph": (6.0,7.5), "temp": (15,25), "rain": (300, 900),
                       "soils": ["Loamy","Clay","Silt"],
                       "seasons": ["Rabi","Winter"]},
        "Rice"     : {"ph": (5.5,7.0), "temp": (22,35), "rain": (800,2000),
                       "soils": ["Clay","Silt","Loamy"],
                       "seasons": ["Kharif","Summer"]},
        "Maize"    : {"ph": (5.8,7.0), "temp": (20,30), "rain": (500,1200),
                       "soils": ["Loamy","Sandy","Silt"],
                       "seasons": ["Kharif","Zaid","Summer"]},
        "Soybean"  : {"ph": (6.0,7.0), "temp": (20,30), "rain": (600,1500),
                       "soils": ["Loamy","Clay"],
                       "seasons": ["Kharif","Summer"]},
        "Cotton"   : {"ph": (6.0,7.5), "temp": (25,40), "rain": (500,1000),
                       "soils": ["Sandy","Loamy","Clay"],
                       "seasons": ["Kharif","Summer"]},
        "Sugarcane": {"ph": (6.0,7.5), "temp": (25,38), "rain": (1000,2000),
                       "soils": ["Loamy","Clay"],
                       "seasons": ["Kharif","Zaid"]},
        "Potato"   : {"ph": (5.0,6.5), "temp": (15,25), "rain": (400, 800),
                       "soils": ["Sandy","Loamy"],
                       "seasons": ["Rabi","Winter"]},
        "Tomato"   : {"ph": (5.5,7.0), "temp": (20,30), "rain": (400, 800),
                       "soils": ["Sandy","Loamy","Silt"],
                       "seasons": ["Rabi","Zaid","Winter"]},
    }

    scores = []
    for crop, profile in CROP_PROFILES.items():
        score = 0
        rationale_parts = []

        # pH match
        ph_lo, ph_hi = profile["ph"]
        if ph_lo <= ph <= ph_hi:
            score += 30
            rationale_parts.append(f"pH {ph} is ideal")
        elif abs(ph - (ph_lo + ph_hi) / 2) < 1.5:
            score += 15
            rationale_parts.append(f"pH {ph} is borderline")
        else:
            rationale_parts.append(f"pH {ph} is outside preferred range")

        # Temperature match
        t_lo, t_hi = profile["temp"]
        if t_lo <= temperature <= t_hi:
            score += 25
            rationale_parts.append(f"temperature {temperature}°C is suitable")
        elif abs(temperature - (t_lo + t_hi) / 2) < 8:
            score += 10

        # Rainfall match
        r_lo, r_hi = profile["rain"]
        if r_lo <= rainfall <= r_hi:
            score += 20
            rationale_parts.append(f"rainfall {rainfall} mm is adequate")
        elif rainfall < r_lo * 0.6:
            score -= 10
        elif rainfall > r_hi * 1.4:
            score -= 10

        # Soil type match
        if soil_type in profile["soils"]:
            score += 15
            rationale_parts.append(f"{soil_type} soil is compatible")

        # Season match
        if season in profile["seasons"]:
            score += 10
            rationale_parts.append(f"{season} season is recommended")

        scores.append({
            "crop"             : crop,
            "suitability_score": max(0, min(100, score)),
            "rationale"        : "; ".join(rationale_parts) if rationale_parts else "General conditions apply",
        })

    scores.sort(key=lambda x: x["suitability_score"], reverse=True)
    return scores[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Soil Health Score
# ─────────────────────────────────────────────────────────────────────────────

def soil_health_score(ph: float, nitrogen: float, phosphorus: float,
                       potassium: float, moisture: float,
                       organic_carbon: float = None) -> dict:
    """
    Compute an aggregate soil health score (0–100).

    Returns
    -------
    dict with overall_score, grade, sub_scores, advice
    """
    sub = {}

    # pH score: optimal 6.0–7.0
    if 6.0 <= ph <= 7.0:
        sub["pH"] = 100
    elif 5.5 <= ph <= 7.5:
        sub["pH"] = 75
    elif 5.0 <= ph <= 8.0:
        sub["pH"] = 50
    else:
        sub["pH"] = 20

    # Nitrogen score: optimal 60–120
    sub["Nitrogen"]   = min(100, max(0, int(100 - abs(nitrogen - 80) * 1.2)))

    # Phosphorus score: optimal 30–70
    sub["Phosphorus"] = min(100, max(0, int(100 - abs(phosphorus - 50) * 1.5)))

    # Potassium score: optimal 80–140
    sub["Potassium"]  = min(100, max(0, int(100 - abs(potassium - 110) * 0.8)))

    # Moisture score: optimal 40–65
    sub["Moisture"]   = min(100, max(0, int(100 - abs(moisture - 52.5) * 2)))

    overall = int(sum(sub.values()) / len(sub))

    if overall >= 80:
        grade  = "A – Excellent 🟢"
        advice = "Soil is in prime condition. Maintain current practices."
    elif overall >= 60:
        grade  = "B – Good 🟡"
        advice = "Minor adjustments needed. Review lowest sub-scores above."
    elif overall >= 40:
        grade  = "C – Fair 🟠"
        advice = "Soil needs attention. Apply targeted amendments and organic matter."
    else:
        grade  = "D – Poor 🔴"
        advice = "Significant remediation required. Consider soil testing and deep conditioning."

    return {
        "overall_score": overall,
        "grade"        : grade,
        "sub_scores"   : sub,
        "advice"       : advice,
    }
