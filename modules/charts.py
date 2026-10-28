"""
Module: charts.py
Purpose: All Plotly chart factory functions used in the Dashboard tab.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLOUR_SEQ = px.colors.qualitative.Set2


def crop_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart: count of each Crop_Type in the dataset."""
    counts = df["Crop_Type"].value_counts().reset_index()
    counts.columns = ["Crop Type", "Count"]
    fig = px.bar(
        counts, x="Crop Type", y="Count",
        color="Crop Type", color_discrete_sequence=COLOUR_SEQ,
        title="🌾 Crop Type Distribution",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    return fig


def disease_frequency_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: frequency of each Disease_Label."""
    counts = df["Disease_Label"].value_counts().reset_index()
    counts.columns = ["Disease", "Count"]
    fig = px.bar(
        counts, y="Disease", x="Count",
        orientation="h", color="Disease",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="🦠 Disease Frequency",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    return fig


def temp_vs_yield_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: Temperature vs Yield, coloured by Crop_Type."""
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Temperature", y="Yield",
        color="Crop_Type", opacity=0.6,
        color_discrete_sequence=COLOUR_SEQ,
        title="🌡️ Temperature vs Yield",
        labels={"Temperature": "Temperature (°C)", "Yield": "Yield (t/ha)"},
        trendline="lowess",
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    return fig


def nutrients_vs_health_chart(df: pd.DataFrame) -> go.Figure:
    """Box plots: Nitrogen / Phosphorus / Potassium by Healthy status."""
    fig = make_subplots(rows=1, cols=3,
                         subplot_titles=["Nitrogen", "Phosphorus", "Potassium"])

    for i, nutrient in enumerate(["Nitrogen", "Phosphorus", "Potassium"], start=1):
        for label, colour in [("Healthy", "#2ecc71"), ("Unhealthy", "#e74c3c")]:
            subset = df[df["Healthy"] == (1 if label == "Healthy" else 0)][nutrient]
            fig.add_trace(
                go.Box(y=subset, name=label, marker_color=colour,
                       showlegend=(i == 1)),
                row=1, col=i,
            )

    fig.update_layout(
        title_text="🧪 Soil Nutrients vs Crop Health",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    return fig


def ndvi_by_crop_chart(df: pd.DataFrame) -> go.Figure:
    """Violin chart: NDVI Index distribution per Crop_Type."""
    fig = px.violin(
        df, x="Crop_Type", y="NDVI_Index",
        color="Crop_Type", box=True,
        color_discrete_sequence=COLOUR_SEQ,
        title="🌿 NDVI Index by Crop Type",
    )
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    return fig


def season_health_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: percentage of healthy crops per season."""
    grp = (
        df.groupby(["Season", "Healthy"])
          .size()
          .reset_index(name="Count")
    )
    grp["Status"] = grp["Healthy"].map({1: "Healthy", 0: "Unhealthy"})
    fig = px.bar(
        grp, x="Season", y="Count", color="Status",
        barmode="group",
        color_discrete_map={"Healthy": "#27ae60", "Unhealthy": "#e74c3c"},
        title="📅 Crop Health by Season",
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    return fig


def rainfall_yield_chart(df: pd.DataFrame) -> go.Figure:
    """Hexbin-style density scatter: Rainfall vs Yield."""
    sample = df.sample(min(2000, len(df)), random_state=7)
    fig = px.density_contour(
        sample, x="Rainfall", y="Yield", color="Crop_Type",
        title="🌧️ Rainfall vs Yield Density",
        labels={"Rainfall": "Annual Rainfall (mm)", "Yield": "Yield (t/ha)"},
    )
    fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    return fig


def soil_moisture_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: mean Soil Moisture across Region × Season."""
    pivot = (
        df.groupby(["Region", "Season"])["Soil_Moisture"]
          .mean()
          .round(1)
          .reset_index()
          .pivot(index="Region", columns="Season", values="Soil_Moisture")
    )
    fig = px.imshow(
        pivot, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title="💧 Mean Soil Moisture (%) — Region × Season",
    )
    return fig
