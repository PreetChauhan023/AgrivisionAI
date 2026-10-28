"""
app.py — AgriVision AI: Intelligent Crop Health Monitoring System
─────────────────────────────────────────────────────────────────
Entry point for the Streamlit application.

Run with:
    streamlit run app.py

Requires:
    pip install -r requirements.txt
"""
import torch
import torchvision
import streamlit as st
from streamlit_option_menu import option_menu

# ── Page configuration (must be FIRST Streamlit call) ───────────────────────
st.set_page_config(
    page_title   = "AgriVision AI",
    page_icon    = "🌾",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── Internal module imports ──────────────────────────────────────────────────
from modules.auth              import show_auth_page
from modules.generate_dataset  import load_or_generate
from modules.models            import train_all_models

# Page renderers
from modules.page_overview             import show as show_overview
from modules.page_dataset              import show as show_dataset
from modules.page_dashboard            import show as show_dashboard
from modules.page_health_prediction    import show as show_health
from modules.page_disease_classification import show as show_disease
from modules.page_image_detection      import show as show_image
from modules.page_yield_prediction     import show as show_yield
from modules.page_soil_analysis        import show as show_soil
from modules.page_recommendations      import show as show_recs

from modules.page_smart_vision         import show as show_smart_vision
from modules.page_market_trends        import show as show_market
from modules.page_nutrient_advisor     import show as show_nutrient
from modules.page_knowledge_hub        import show as show_knowledge_hub
from modules.page_profile_settings     import show_profile, show_settings


# ─────────────────────────────────────────────────────────────────────────────
# Session State Defaults
# ─────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Gate
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth_page()
    st.stop()   # Do NOT render the rest of the app until logged in


# ─────────────────────────────────────────────────────────────────────────────
# Load Dataset & Train Models (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="🌱 Generating / loading agricultural dataset…")
def get_dataset():
    return load_or_generate("data/agri_dataset.csv")


df     = get_dataset()
models = train_all_models(df)   # cached via @st.cache_resource inside models.py


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 10px 0 5px 0;'>
            <span style='font-size:40px;'>🌾</span><br/>
            <span style='font-weight:bold; font-size:16px; color:#27ae60;'>AgriVision AI</span><br/>
            <span style='font-size:11px; color:#888;'>Crop Health Monitoring</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"👤 **{st.session_state.username}**")
    st.markdown("---")

    selected = option_menu(
        menu_title  = None,
        options     = [
            "Overview",
            "Knowledge Hub",
            "Dataset Explorer",
            "Dashboard",
            "Crop Health Prediction",
            "Disease Classification",
            "AI Smart Vision Health",
            "Yield Prediction",
            "Nutrient Advisor",
            "Market Price Trends",
            "Soil Analysis",
            "Recommendation System",
            "Profile",
            "Settings",
        ],
        icons       = [
            "house",
            "book",
            "table",
            "bar-chart-line",
            "heart-pulse",
            "bug",
            "camera",
            "graph-up-arrow",
            "patch-check",
            "cash",
            "layers",
            "lightbulb",
            "person-circle",
            "gear",
        ],
        default_index = 0,
        styles = {
            "container"     : {"padding": "0px", "background-color": "transparent"},
            "icon"          : {"font-size": "15px"},
            "nav-link"      : {
                "font-size"    : "13px",
                "text-align"   : "left",
                "margin"       : "2px 0px",
                "padding"      : "8px 12px",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background-color": "#27ae60",
                "color"           : "white",
                "font-weight"     : "600",
            },
        },
    )




# ─────────────────────────────────────────────────────────────────────────────
# Page Router
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Overview":
    show_overview(df)

elif selected == "Knowledge Hub":
    show_knowledge_hub()

elif selected == "Dataset Explorer":
    show_dataset(df)

elif selected == "Dashboard":
    show_dashboard(df)

elif selected == "Crop Health Prediction":
    show_health(df, models)

elif selected == "Disease Classification":
    show_disease(df, models)

elif selected == "AI Smart Vision Health":
    show_smart_vision()

elif selected == "Yield Prediction":
    show_yield(df, models)

elif selected == "Nutrient Advisor":
    show_nutrient(df)

elif selected == "Market Price Trends":
    show_market(df)

elif selected == "Soil Analysis":
    show_soil(df)

elif selected == "Recommendation System":
    show_recs(df)


elif selected == "Profile":
    show_profile()

elif selected == "Settings":
    show_settings()
