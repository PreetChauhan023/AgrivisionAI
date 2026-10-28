"""
Page: page_dataset.py
Purpose: Dataset Explorer tab — browse data, view info, nulls, and statistics.
"""

import streamlit as st
import pandas as pd
import io


def show(df: pd.DataFrame) -> None:
    """Render the Dataset Explorer page."""
    st.title("🗂️ Dataset Explorer")
    st.markdown("Explore the raw synthetic agricultural dataset used to train all models.")
    st.markdown("---")

    # ── Shape & Preview ──────────────────────────────────────────────────────
    st.subheader("📄 Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Memory",  f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Filters
    with st.expander("🔎 Filter Data", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            crop_filter = st.multiselect(
                "Filter by Crop Type",
                options=sorted(df["Crop_Type"].unique()),
                default=[],
            )
        with col_f2:
            disease_filter = st.multiselect(
                "Filter by Disease Label",
                options=sorted(df["Disease_Label"].unique()),
                default=[],
            )
        n_rows = st.slider("Rows to display", min_value=10, max_value=500,
                            value=50, step=10)

    filtered = df.copy()
    if crop_filter:
        filtered = filtered[filtered["Crop_Type"].isin(crop_filter)]
    if disease_filter:
        filtered = filtered[filtered["Disease_Label"].isin(disease_filter)]

    st.dataframe(filtered.head(n_rows), use_container_width=True)

    # Download filtered data
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="agri_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

    # ── Null Values ──────────────────────────────────────────────────────────
    st.subheader("🕳️ Missing Values")
    null_df = pd.DataFrame({
        "Column"       : df.columns,
        "Null Count"   : df.isnull().sum().values,
        "Null %"       : (df.isnull().mean().values * 100).round(2),
        "Data Type"    : df.dtypes.values.astype(str),
    })
    null_df = null_df.sort_values("Null Count", ascending=False)
    st.dataframe(null_df, use_container_width=True, hide_index=True)

    max_null = null_df["Null Count"].max()
    if max_null == 0:
        st.success("✅ No missing values found in the dataset.")
    else:
        st.warning(f"⚠️ Some columns have missing values. Max: {max_null}")

    st.markdown("---")

    # ── Descriptive Statistics ───────────────────────────────────────────────
    st.subheader("📊 Descriptive Statistics")
    tab_num, tab_cat = st.tabs(["🔢 Numerical", "🔤 Categorical"])

    with tab_num:
        num_desc = df.select_dtypes(include="number").describe().T
        num_desc["range"] = num_desc["max"] - num_desc["min"]
        st.dataframe(num_desc.round(3), use_container_width=True)

    with tab_cat:
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        selected_cat = st.selectbox("Select a categorical column:", cat_cols)
        vc = df[selected_cat].value_counts().reset_index()
        vc.columns = [selected_cat, "Count"]
        vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
        st.dataframe(vc, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Column Info ──────────────────────────────────────────────────────────
    st.subheader("ℹ️ Column Information")

    COLUMN_DESC = {
        "Crop_Type"        : "Type of crop being grown",
        "Soil_Type"        : "Soil classification (Loamy, Sandy, Clay, etc.)",
        "Temperature"      : "Ambient temperature in °C",
        "Humidity"         : "Relative humidity (%)",
        "Rainfall"         : "Annual rainfall in mm",
        "pH_Level"         : "Soil pH level (4.5 – 8.5)",
        "Nitrogen"         : "Soil nitrogen content (kg/ha)",
        "Phosphorus"       : "Soil phosphorus content (kg/ha)",
        "Potassium"        : "Soil potassium content (kg/ha)",
        "Leaf_Color"       : "Predominant leaf colour observed",
        "Leaf_Spots"       : "Number of visible disease spots on leaves",
        "Disease_Type"     : "Raw disease type label",
        "Disease_Severity" : "Severity of observed disease",
        "Irrigation_Level" : "Irrigation coverage (%)",
        "Sunlight_Hours"   : "Average daily sunlight hours",
        "Growth_Stage"     : "Current stage of crop growth",
        "Pest_Presence"    : "Whether pests are present (Yes/No)",
        "Fertilizer_Used"  : "Fertilizer type applied",
        "Yield"            : "Estimated crop yield (t/ha)",
        "Region"           : "Geographic region of the farm",
        "Season"           : "Agricultural season",
        "Water_Quality"    : "Quality of irrigation water",
        "Air_Quality"      : "Ambient air quality",
        "Soil_Moisture"    : "Volumetric soil moisture (%)",
        "NDVI_Index"       : "Normalised Difference Vegetation Index (0.1–0.9)",
        "Stress_Level"     : "Observed plant stress level",
        "Leaf_Area"        : "Leaf area index (cm²)",
        "Chlorophyll_Level": "Chlorophyll content (SPAD units)",
        "Disease_Label"    : "🎯 Target: Disease name or 'Healthy'",
        "Healthy"          : "🎯 Target: Binary (1=Healthy, 0=Unhealthy)",
    }

    info_df = pd.DataFrame([
        {"Column": col, "Type": str(df[col].dtype), "Description": COLUMN_DESC.get(col, "—")}
        for col in df.columns
    ])
    st.dataframe(info_df, use_container_width=True, hide_index=True)
