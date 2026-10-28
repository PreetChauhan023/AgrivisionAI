"""
Page: page_recommendations.py
Purpose: Integrated recommendation engine — fertilizer, irrigation, crop suggestions.
"""

import streamlit as st
import pandas as pd
from modules.recommendations import (
    fertilizer_recommendation,
    irrigation_recommendation,
    crop_suggestion,
)


CROP_TYPES    = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane", "Potato", "Tomato"]
GROWTH_STAGES = ["Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"]
SOIL_TYPES    = ["Loamy", "Sandy", "Clay", "Silt", "Peaty", "Chalky", "Saline"]
SEASONS       = ["Kharif", "Rabi", "Zaid", "Summer", "Winter"]


def show(df: pd.DataFrame) -> None:
    st.title("💡 Recommendation System")
    st.markdown(
        "Get personalised recommendations for **fertilizers**, **irrigation scheduling**, "
        "and **crop selection** based on your farm conditions."
    )
    st.markdown("---")

    tab_fert, tab_irrig, tab_crop = st.tabs([
        "🧪 Fertilizer Recommendations",
        "💧 Irrigation Plan",
        "🌱 Crop Suggestions",
    ])

    # ── FERTILIZER ───────────────────────────────────────────────────────────
    with tab_fert:
        st.subheader("🧪 Fertilizer Recommendation Engine")
        with st.form("fert_form"):
            c1, c2 = st.columns(2)
            with c1:
                crop    = st.selectbox("Crop Type", CROP_TYPES, key="fert_crop")
                n_val   = st.slider("Current Nitrogen (kg/ha)",   10.0, 150.0, 55.0)
            with c2:
                p_val   = st.slider("Current Phosphorus (kg/ha)",  5.0, 100.0, 25.0)
                k_val   = st.slider("Current Potassium (kg/ha)",  10.0, 200.0, 70.0)
            submitted_f = st.form_submit_button("🔍 Get Fertilizer Plan", use_container_width=True)

        if submitted_f:
            result = fertilizer_recommendation(n_val, p_val, k_val, crop)
            st.markdown("---")
            st.subheader(f"📋 Fertilizer Plan for {result['crop']}")
            for rec in result["recommendations"]:
                colour = {"Low": "🔵", "High": "🔴", "Optimal": "🟢"}.get(rec["level"], "⚪")
                with st.expander(f"{colour} {rec['nutrient']} — {rec['level']}", expanded=True):
                    st.markdown(f"**Product:** {rec['product']}")
                    st.markdown(f"**Application Rate:** {rec['rate']}")
                    st.markdown(f"**Note:** {rec['note']}")

    # ── IRRIGATION ───────────────────────────────────────────────────────────
    with tab_irrig:
        st.subheader("💧 Smart Irrigation Scheduling")
        with st.form("irrig_form"):
            c1, c2 = st.columns(2)
            with c1:
                crop_i   = st.selectbox("Crop Type",     CROP_TYPES, key="irrig_crop")
                stage_i  = st.selectbox("Growth Stage",  GROWTH_STAGES)
                moisture = st.slider("Soil Moisture (%)", 10.0, 80.0, 35.0)
            with c2:
                temp_i   = st.slider("Temperature (°C)", 15.0, 45.0, 30.0)
                rain_i   = st.slider("Recent Rainfall (mm)", 0.0, 200.0, 20.0)
            submitted_i = st.form_submit_button("💧 Get Irrigation Plan", use_container_width=True)

        if submitted_i:
            result = irrigation_recommendation(moisture, temp_i, rain_i, crop_i, stage_i)
            st.markdown("---")
            st.subheader(f"📋 Irrigation Plan for {crop_i} — {stage_i}")

            c1, c2, c3 = st.columns(3)
            c1.info(f"**Status**\n\n{result['urgency']}")
            c2.info(f"**Frequency**\n\n{result['frequency']}")
            c3.info(f"**Volume/Application**\n\n{result['volume']}")

            st.markdown(f"**🚜 Recommended Method:** {result['method']}")
            st.markdown(f"**💦 Daily ET Estimate:** {result['et_estimate']} mm/day")
            st.markdown(f"**📝 Notes:** {result['notes']}")

    # ── CROP SUGGESTION ──────────────────────────────────────────────────────
    with tab_crop:
        st.subheader("🌱 Crop Suitability Analysis")
        with st.form("crop_form"):
            c1, c2 = st.columns(2)
            with c1:
                ph_c    = st.slider("Soil pH",               4.0, 9.0, 6.5, 0.1)
                temp_c  = st.slider("Average Temperature (°C)", 10.0, 45.0, 26.0)
            with c2:
                rain_c  = st.slider("Annual Rainfall (mm)", 100.0, 2500.0, 800.0)
                soil_c  = st.selectbox("Soil Type", SOIL_TYPES)
                season_c= st.selectbox("Season",    SEASONS)
            submitted_c = st.form_submit_button("🌿 Find Best Crops", use_container_width=True)

        if submitted_c:
            suggestions = crop_suggestion(ph_c, temp_c, rain_c, soil_c, season_c)
            st.markdown("---")
            st.subheader("🏆 Top 3 Crop Recommendations")

            medals = ["🥇", "🥈", "🥉"]
            for i, s in enumerate(suggestions):
                with st.container():
                    col_m, col_info = st.columns([1, 4])
                    col_m.markdown(f"## {medals[i]}")
                    with col_info:
                        st.markdown(f"### {s['crop']}")
                        st.progress(s["suitability_score"])
                        st.markdown(
                            f"**Suitability Score:** {s['suitability_score']}%  \n"
                            f"**Rationale:** {s['rationale']}"
                        )
                    st.markdown("---")

            # Download
            report = pd.DataFrame([
                {"Rank": i+1, "Crop": s["crop"],
                 "Suitability_%": s["suitability_score"],
                 "Rationale": s["rationale"]}
                for i, s in enumerate(suggestions)
            ])
            st.download_button(
                "⬇️ Download Crop Suggestions (CSV)",
                data=report.to_csv(index=False).encode(),
                file_name="crop_suggestions.csv",
                mime="text/csv",
            )
