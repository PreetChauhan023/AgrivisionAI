"""
Pages: page_profile.py & page_settings.py
Purpose: User profile display and app settings (theme toggle, logout).
"""

import streamlit as st
from modules.auth import get_user_info


# ─────────────────────────────────────────────────────────────────────────────
# Profile Page
# ─────────────────────────────────────────────────────────────────────────────

def show_profile() -> None:
    """Render the user profile page."""
    st.title("👤 User Profile")
    st.markdown("---")

    username = st.session_state.get("username", "Unknown")
    info     = get_user_info(username)

    col1, col2 = st.columns([1, 2])
    with col1:
        # Avatar placeholder
        st.markdown(
            f"""
            <div style='text-align:center; padding:20px;
                        background:#f0f8f0; border-radius:12px;
                        border: 2px solid #27ae60;'>
                <div style='font-size:80px;'>🌾</div>
                <div style='font-size:22px; font-weight:bold; color:#27ae60;'>{username}</div>
                <div style='color:#888; font-size:14px;'>AgriVision AI User</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Account Information")
        st.markdown(f"- 👤 **Username:** `{info.get('username', username)}`")
        st.markdown(f"- 📧 **Email:** `{info.get('email', 'N/A')}`")
        st.markdown(f"- 🔐 **Account Status:** Active ✅")
        st.markdown("---")
        st.subheader("Session Information")
        st.markdown(f"- 🌿 **Current Session:** Active")
        st.markdown(f"- 🖥️ **App Version:** AgriVision AI v1.0")
        st.markdown(f"- 🧪 **Models Loaded:** Logistic Regression | Random Forest | Linear Regression")

    st.markdown("---")
    st.subheader("📊 Platform Features Available to You")

    features = [
        ("🩺", "Crop Health Prediction",   "Logistic Regression binary classifier"),
        ("🦠", "Disease Classification",    "Random Forest 7-class disease detector"),
        ("📷", "Image Disease Detection",   "OpenCV visual analysis pipeline"),
        ("📈", "Yield Prediction",          "Multiple Linear Regression estimator"),
        ("🧪", "Soil Analysis",             "Rule-based soil health scoring"),
        ("💡", "Recommendation System",     "Fertilizer, irrigation & crop engine"),
        ("⚖️", "Model Comparison",          "Side-by-side model benchmarking"),
    ]
    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate(features):
        with cols[i % 3]:
            st.info(f"**{icon} {name}**\n\n{desc}")


# ─────────────────────────────────────────────────────────────────────────────
# Settings Page
# ─────────────────────────────────────────────────────────────────────────────

def show_settings() -> None:
    """Render the settings page."""
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("🎨 Display Preferences")

    # Theme toggle (informational — Streamlit theme is set in config.toml)
    theme_choice = st.radio(
        "Preferred Theme",
        ["🌞 Light Mode", "🌙 Dark Mode"],
        index=0 if st.session_state.get("theme", "light") == "light" else 1,
        horizontal=True,
    )
    if "Light" in theme_choice:
        st.session_state["theme"] = "light"
    else:
        st.session_state["theme"] = "dark"

    st.info(
        "💡 **Note:** To apply a persistent dark/light theme, edit `.streamlit/config.toml`:\n\n"
        "```toml\n[theme]\nbase = 'dark'  # or 'light'\n```"
    )

    st.markdown("---")
    st.subheader("🧠 Model Settings")

    show_confidence = st.toggle(
        "Show prediction confidence scores",
        value=st.session_state.get("show_confidence", True),
    )
    st.session_state["show_confidence"] = show_confidence

    show_tips = st.toggle(
        "Show agronomic tips alongside predictions",
        value=st.session_state.get("show_tips", True),
    )
    st.session_state["show_tips"] = show_tips

    st.markdown("---")
    st.subheader("📁 Data Settings")

    st.info(
        "The application uses a **synthetic dataset** of 10,000 rows stored at "
        "`data/agri_dataset.csv`. User accounts are stored at `data/users.csv`."
    )

    st.markdown("---")
    st.subheader("🔐 Account Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            _logout()
    with col2:
        st.button("🔄 Reload Dataset", use_container_width=True,
                   on_click=_clear_dataset_cache)

    st.markdown("---")
    st.caption("AgriVision AI v1.0 | Built with Streamlit & Scikit-learn | 🌾 Crop Intelligence Platform")


def _logout() -> None:
    """Clear session state and rerun to show login page."""
    for key in ["logged_in", "username", "theme", "show_confidence", "show_tips"]:
        st.session_state.pop(key, None)
    st.rerun()


def _clear_dataset_cache() -> None:
    """Clear Streamlit resource / data caches so dataset reloads."""
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Cache cleared. Dataset will reload on next page visit.")
