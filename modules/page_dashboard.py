"""
Page: page_dashboard.py
Purpose: Interactive visual dashboard with Plotly charts.
"""

import streamlit as st
import pandas as pd
from modules.charts import (
    crop_distribution_chart, disease_frequency_chart,
    temp_vs_yield_chart, nutrients_vs_health_chart,
    ndvi_by_crop_chart, season_health_chart,
    rainfall_yield_chart, soil_moisture_heatmap,
)


def show(df: pd.DataFrame) -> None:
    st.title("📊 Interactive Dashboard")
    st.markdown("Visual exploration of crop health patterns, disease distribution, and key agronomic indicators.")
    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("🔎 Global Dashboard Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            crop_sel = st.multiselect("Crop Type", sorted(df["Crop_Type"].unique()), default=[])
        with col2:
            region_sel = st.multiselect("Region", sorted(df["Region"].unique()), default=[])
        with col3:
            season_sel = st.multiselect("Season", sorted(df["Season"].unique()), default=[])

    dff = df.copy()
    if crop_sel:
        dff = dff[dff["Crop_Type"].isin(crop_sel)]
    if region_sel:
        dff = dff[dff["Region"].isin(region_sel)]
    if season_sel:
        dff = dff[dff["Season"].isin(season_sel)]

    st.caption(f"Showing **{len(dff):,}** records after filters.")
    st.markdown("---")

    # ── Row 1: Crop & Disease distribution ───────────────────────────────────
    st.subheader("🌾 Crop & Disease Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(crop_distribution_chart(dff), use_container_width=True)
    with col_b:
        st.plotly_chart(disease_frequency_chart(dff), use_container_width=True)

    st.markdown("---")

    # ── Row 2: Temperature vs Yield ──────────────────────────────────────────
    st.subheader("🌡️ Environmental Impact on Yield")
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(temp_vs_yield_chart(dff), use_container_width=True)
    with col_d:
        st.plotly_chart(rainfall_yield_chart(dff), use_container_width=True)

    st.markdown("---")

    # ── Row 3: Nutrients vs health ───────────────────────────────────────────
    st.subheader("🧪 Nutrient Profiles & Health")
    st.plotly_chart(nutrients_vs_health_chart(dff), use_container_width=True)

    st.markdown("---")

    # ── Row 4: NDVI & Season ─────────────────────────────────────────────────
    st.subheader("🌿 NDVI & Seasonal Patterns")
    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(ndvi_by_crop_chart(dff), use_container_width=True)
    with col_f:
        st.plotly_chart(season_health_chart(dff), use_container_width=True)

    st.markdown("---")

    # ── Row 5: Soil Moisture Heatmap ─────────────────────────────────────────
    st.subheader("💧 Soil Moisture Heatmap")
    st.plotly_chart(soil_moisture_heatmap(dff), use_container_width=True)
