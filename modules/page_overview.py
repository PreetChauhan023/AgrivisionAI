"""
Page: overview.py
Purpose: Render the Overview / landing tab with project description,
         dataset summary statistics, and key headline metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def show(df: pd.DataFrame) -> None:
    """Render the Enhanced Data Intelligence Overview page."""

    # ── Header Section ───────────────────────────────────────────────────────
    st.title("🌾 AgriVision Data Intelligence")
    st.markdown(
        """
        Welcome to the **AgriVision AI** analytics core. This dashboard provides a 360-degree
        view of our 10,000-record agricultural dataset, uncovering patterns in crop health,
        yield drivers, and environmental impacts.
        """
    )

    # ── Executive Metrics ────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)

    healthy_pct = (df["Healthy"] == 1).mean() * 100
    avg_yield   = df["Yield"].mean()
    pest_idx    = (df["Pest_Presence"] == "Yes").mean() * 100
    avg_moisture= df["Soil_Moisture"].mean()

    c1.metric("🏥 Global Health", f"{healthy_pct:.1f}%", help="Percentage of crops classified as 'Healthy'")
    c2.metric("🌾 Average Yield", f"{avg_yield:.1f} t/ha", help="Mean harvest productivity across all records")
    c3.metric("🐛 Pest Pressure", f"{pest_idx:.1f}%", help="Prevalence of pest detections", delta_color="inverse")
    c4.metric("💧 Soil Moisture", f"{avg_moisture:.1f}%", help="Average moisture levels across all sensors")

    # ── Main Dashboard Tabs ──────────────────────────────────────────────────
    st.markdown("---")
    tab_dist, tab_env, tab_rel, tab_raw = st.tabs([
        "🌍 Distribution & Landscape",
        "🌦️ Environmental Profile",
        "🔗 Relationships & Drivers",
        "📋 Data Browser & Schema"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: Distribution & Landscape
    # ─────────────────────────────────────────────────────────────────────────
    with tab_dist:
        col1, col2 = st.columns(2)

        with col1:
            # Crop Type Distribution
            crop_counts = df["Crop_Type"].value_counts().reset_index()
            crop_counts.columns = ["Crop", "Count"]
            fig_crop = px.pie(
                crop_counts, names="Crop", values="Count",
                title="Crop Portfolio Mix",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_crop.update_layout(height=450)
            st.plotly_chart(fig_crop, use_container_width=True)

        with col2:
            # Regional Distribution by Health
            reg_health = df.groupby(["Region", "Healthy"]).size().reset_index(name="Count")
            reg_health["Status"] = reg_health["Healthy"].map({1: "Healthy", 0: "Unhealthy"})
            fig_reg = px.bar(
                reg_health, x="Region", y="Count", color="Status",
                title="Regional Health Distribution",
                barmode="group",
                color_discrete_map={"Healthy": "#27ae60", "Unhealthy": "#e74c3c"}
            )
            fig_reg.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown("---")
        # Disease Treemap
        disease_data = df[df["Disease_Label"] != "Healthy"]["Disease_Label"].value_counts().reset_index()
        disease_data.columns = ["Disease", "Count"]
        fig_tree = px.treemap(
            disease_data, path=["Disease"], values="Count",
            title="Disease Prevalence Hierarchy",
            color="Count", color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: Environmental Profile
    # ─────────────────────────────────────────────────────────────────────────
    with tab_env:
        st.subheader("Sensory Data Distributions")
        env_cols = ["Temperature", "Humidity", "Rainfall", "pH_Level"]
        
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            fig_temp = px.histogram(df, x="Temperature", color="Crop_Type", 
                                    marginal="box", title="Temperature Variation vs Crop Type",
                                    opacity=0.7)
            fig_temp.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_temp, use_container_width=True)

            fig_rain = px.histogram(df, x="Rainfall", nbins=50, title="Rainfall Density Plot",
                                    color_discrete_sequence=["#3498db"])
            fig_rain.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rain, use_container_width=True)

        with col_e2:
            fig_hum = px.histogram(df, x="Humidity", color="Region",
                                   marginal="rug", title="Humidity Levels by Region",
                                   opacity=0.7)
            fig_hum.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hum, use_container_width=True)

            fig_ph = px.box(df, x="Soil_Type", y="pH_Level", color="Soil_Type",
                            title="pH Level Variation by Soil Type")
            fig_ph.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ph, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: Relationships & Drivers
    # ─────────────────────────────────────────────────────────────────────────
    with tab_rel:
        st.subheader("Feature Correlations & Yield Drivers")
        
        # Heatmap
        c1, c2 = st.columns([1.5, 1])
        with c1:
            num_df = df.select_dtypes(include="number").drop(columns=["Healthy"], errors="ignore")
            corr = num_df.corr().round(2)
            fig_corr = px.imshow(
                corr, text_auto=True, 
                title="Cross-Feature Correlation Matrix",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with c2:
            st.info(
                "**Correlation Insight**\n\n"
                "- Strong positive correlation between **NDVI Index** and **Yield**.\n"
                "- **Chlorophyll Levels** are linked with higher **Nitrogen** levels.\n"
                "- Inverse relationship often seen between **Temperature** and **Soil Moisture**."
            )
            
            # Key Relationship Scatter
            fig_scat = px.scatter(
                df.sample(1000), x="NDVI_Index", y="Yield", color="Crop_Type",
                size="Soil_Moisture", hover_data=["Region"],
                title="NDVI vs Yield (Sample of 1,000)",
                trendline="ols"
            )
            fig_scat.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scat, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: Data Browser & Schema
    # ─────────────────────────────────────────────────────────────────────────
    with tab_raw:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Dataset Schema & Stats")
        
        col_sm1, col_sm2 = st.columns(2)
        with col_sm1:
            st.write("**Numerical Summary**")
            st.dataframe(df.describe().T.round(2), use_container_width=True)
        
        with col_sm2:
            st.write("**Categorical Profile**")
            cat_cols = df.select_dtypes(include="object").columns
            cat_info = []
            for col in cat_cols:
                top_val = df[col].mode()[0]
                cat_info.append({
                    "Feature": col,
                    "Unique": df[col].nunique(),
                    "Mode": top_val,
                    "Mode Frequency": (df[col] == top_val).sum()
                })
            st.dataframe(pd.DataFrame(cat_info), use_container_width=True, hide_index=True)

    # ── Final Analysis Narrative ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Automated Insights")
    
    # Simple logic for "AI" insights
    top_yield_crop = df.groupby("Crop_Type")["Yield"].mean().idxmax()
    best_region = df.groupby("Region")["Healthy"].mean().idxmax()
    most_stressed_crop = df[df["Stress_Level"] == "High"]["Crop_Type"].value_counts().idxmax()
    
    ins_col1, ins_col2, ins_col3 = st.columns(3)
    ins_col1.success(f"🏆 **Highest Yielding Crop**: `{top_yield_crop}`")
    ins_col2.success(f"🌟 **Healthiest Region**: `{best_region}`")
    ins_col3.warning(f"⚠️ **Most Stressed Crop**: `{most_stressed_crop}`")
    
    st.markdown(
        f"""
        Based on the current analysis, `{best_region}` shows the highest percentage of healthy crops, 
        making it a prime area for scaling operations. Meanwhile, `{top_yield_crop}` remains the most 
        productive across all soil types. Close monitoring of `{most_stressed_crop}` is advised due to 
        elevated environmental stress levels detected.
        """
    )

