"""
Page: page_soil_analysis.py
Purpose: Soil health scoring and analysis with parameter inputs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.recommendations import soil_health_score


def show(df: pd.DataFrame) -> None:
    st.title("🧪 Soil Analysis")
    st.markdown(
        "Enter your soil parameters to receive a comprehensive **Soil Health Score** "
        "and targeted improvement recommendations."
    )
    st.markdown("---")

    with st.form("soil_form"):
        st.subheader("🌍 Soil Parameter Input")
        c1, c2 = st.columns(2)

        with c1:
            ph          = st.slider("Soil pH Level",            4.0, 9.0, 6.5, 0.1)
            nitrogen    = st.slider("Nitrogen Content (kg/ha)", 10.0, 150.0, 75.0, 1.0)
            phosphorus  = st.slider("Phosphorus (kg/ha)",        5.0, 100.0, 40.0, 1.0)
            potassium   = st.slider("Potassium (kg/ha)",        10.0, 200.0, 100.0, 1.0)

        with c2:
            moisture    = st.slider("Soil Moisture (%)",        10.0,  80.0, 45.0, 1.0)
            soil_type   = st.selectbox("Soil Type",
                                        ["Loamy","Sandy","Clay","Silt","Peaty","Chalky","Saline"])
            region      = st.selectbox("Region", ["North","South","East","West","Central"])
            season      = st.selectbox("Current Season",
                                        ["Kharif","Rabi","Zaid","Summer","Winter"])

        submitted = st.form_submit_button("🔍 Analyse Soil Health", use_container_width=True)

    if submitted:
        result = soil_health_score(ph, nitrogen, phosphorus, potassium, moisture)

        st.markdown("---")
        st.subheader("📊 Soil Health Analysis Results")

        col1, col2 = st.columns([1, 2])

        with col1:
            score = result["overall_score"]
            grade = result["grade"]
            advice = result["advice"]

            st.metric("Overall Soil Health Score", f"{score} / 100")
            st.markdown(f"**Grade:** {grade}")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = score,
                title = {"text": "Soil Health"},
                gauge = {
                    "axis": {"range": [0, 100]},
                    "bar" : {"color": _score_colour(score)},
                    "steps": [
                        {"range": [0,  40], "color": "#ffcccc"},
                        {"range": [40, 60], "color": "#fff3cc"},
                        {"range": [60, 80], "color": "#d4edda"},
                        {"range": [80,100], "color": "#c3e6cb"},
                    ],
                    "threshold": {
                        "line" : {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.subheader("📋 Sub-Scores")
            sub_scores = result["sub_scores"]

            for param, val in sub_scores.items():
                col_lbl, col_bar = st.columns([1, 3])
                col_lbl.write(f"**{param}**")
                col_bar.progress(val)

            st.markdown(f"---\n**💡 Advice:** {advice}")

        st.markdown("---")

        # ── Parameter Summary Table ──────────────────────────────────────────
        st.subheader("📐 Parameter Interpretation")
        rows = [
            {"Parameter": "pH Level",    "Your Value": ph,
             "Ideal Range": "6.0 – 7.0",
             "Status": _status(6.0, 7.0, ph)},
            {"Parameter": "Nitrogen",    "Your Value": nitrogen,
             "Ideal Range": "60 – 120 kg/ha",
             "Status": _status(60, 120, nitrogen)},
            {"Parameter": "Phosphorus",  "Your Value": phosphorus,
             "Ideal Range": "30 – 70 kg/ha",
             "Status": _status(30, 70, phosphorus)},
            {"Parameter": "Potassium",   "Your Value": potassium,
             "Ideal Range": "80 – 140 kg/ha",
             "Status": _status(80, 140, potassium)},
            {"Parameter": "Moisture",    "Your Value": moisture,
             "Ideal Range": "40 – 65 %",
             "Status": _status(40, 65, moisture)},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Soil-Type Tips ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🌍 Soil-Type Specific Tips")
        tips = _soil_type_tips(soil_type)
        for tip in tips:
            st.markdown(f"- {tip}")

        # Download
        report = pd.DataFrame([{
            "Soil_Type"         : soil_type,
            "Region"            : region,
            "Season"            : season,
            "pH_Level"          : ph,
            "Nitrogen"          : nitrogen,
            "Phosphorus"        : phosphorus,
            "Potassium"         : potassium,
            "Moisture"          : moisture,
            "Overall_Score"     : score,
            "Grade"             : grade,
            **{f"SubScore_{k}": v for k, v in sub_scores.items()},
            "Advice"            : advice,
        }])
        st.download_button(
            "⬇️ Download Soil Analysis Report (CSV)",
            data=report.to_csv(index=False).encode(),
            file_name="soil_analysis_report.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _status(lo: float, hi: float, val: float) -> str:
    if lo <= val <= hi:
        return "✅ Optimal"
    elif val < lo:
        return "🔵 Low"
    else:
        return "🔴 High"


def _score_colour(score: int) -> str:
    if score >= 80:  return "#27ae60"
    if score >= 60:  return "#f39c12"
    if score >= 40:  return "#e67e22"
    return "#e74c3c"


def _soil_type_tips(soil_type: str) -> list[str]:
    tips_map = {
        "Loamy"  : ["Ideal for most crops — maintain organic matter above 3%.",
                     "Apply compost annually to preserve structure.",
                     "Well-draining; moderate irrigation suitable."],
        "Sandy"  : ["Leaches nutrients quickly — split fertilizer doses.",
                     "Add organic matter (compost) to improve water retention.",
                     "Irrigate more frequently with smaller volumes."],
        "Clay"   : ["Prone to compaction — avoid heavy machinery when wet.",
                     "Add gypsum (calcium sulphate) to improve structure.",
                     "Ensure drainage channels; avoid waterlogging."],
        "Silt"   : ["Good fertility but compacts easily.",
                     "Minimise tillage; use cover crops to prevent erosion.",
                     "Moderate irrigation; test drainage regularly."],
        "Peaty"  : ["Naturally acidic — lime application may be needed.",
                     "High organic matter but low in minerals; supplement NPK.",
                     "Good moisture retention; monitor waterlogging."],
        "Chalky" : ["Naturally alkaline (high pH) — may need acidifying agents.",
                     "Iron and Manganese may be deficient; use chelated micronutrients.",
                     "Shallow soil layer — select deep-rooted crops carefully."],
        "Saline" : ["Flush salts with deep irrigation periodically.",
                     "Use salt-tolerant crop varieties (e.g., Barley, Sugar Beet).",
                     "Improve drainage to prevent salt accumulation."],
    }
    return tips_map.get(soil_type, ["Consult a local agronomist for soil-specific advice."])
