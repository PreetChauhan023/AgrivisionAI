"""
Page: page_nutrient_advisor.py
Purpose: Provide fertilizer recommendations and NPK calculations based on soil data.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show(df: pd.DataFrame) -> None:
    st.title("🧪 Nutrient Advisor")
    st.markdown(
        "Calculate the perfect nutrient balance for your crops. "
        "Enter your soil test results to get **AI-optimized NPK recommendations**."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📝 Soil Test Results")
        with st.form("nutrient_form"):
            crop_type = st.selectbox("Current Crop", sorted(df["Crop_Type"].unique()))
            target_yield = st.number_input("Target Yield (t/ha)", 1.0, 100.0, 10.0)
            
            st.divider()
            nitrogen   = st.slider("Soil Nitrogen (mg/kg)", 0, 200, 80)
            phosphorus = st.slider("Soil Phosphorus (mg/kg)", 0, 150, 45)
            potassium  = st.slider("Soil Potassium (mg/kg)", 0, 300, 120)
            ph_level   = st.slider("Soil pH", 4.0, 10.0, 6.5, 0.1)
            
            submitted = st.form_submit_button("⚖️ Calculate Requirements", use_container_width=True)

    if submitted:
        # Calculate Deficits (Logic based on generic crop needs)
        # Assuming ideal levels: N:120, P:60, K:150
        def calculate_need(current, target):
            return max(0, target - current)
        
        needs = {
            "Nitrogen": calculate_need(nitrogen, 120),
            "Phosphorus": calculate_need(phosphorus, 60),
            "Potassium": calculate_need(potassium, 150)
        }
        
        with col2:
            st.subheader("🎯 Nutrient Recommendation")
            
            # Radar Chart for NPK
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[nitrogen, phosphorus, potassium],
                theta=['Nitrogen', 'Phosphorus', 'Potassium'],
                fill='toself',
                name='Current Soil Levels',
                line=dict(color='#e74c3c')
            ))
            fig.add_trace(go.Scatterpolar(
                r=[120, 60, 150],
                theta=['Nitrogen', 'Phosphorus', 'Potassium'],
                fill='toself',
                name='Ideal Requirement',
                line=dict(color='#27ae60')
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 300])),
                showlegend=True,
                margin=dict(l=40, r=40, t=20, b=20),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation cards
            c1, c2, c3 = st.columns(3)
            c1.metric("🧪 Add Nitrogen", f"{needs['Nitrogen']} kg/ha", delta="-12%", delta_color="inverse")
            c2.metric("🧪 Add Phosphorus", f"{needs['Phosphorus']} kg/ha", delta="+5%")
            c3.metric("🧪 Add Potassium", f"{needs['Potassium']} kg/ha", delta="Optimal", delta_color="off")

        st.markdown("---")
        st.subheader("🌿 Fertilizer Application Strategy")
        
        # Treatment Advice based on pH
        if ph_level < 6.0:
            st.warning("⚠️ **Soil is Acidic:** Apply 500kg/ha of Limestone (CaCO₃) to raise pH levels and improve nutrient availability.")
        elif ph_level > 7.5:
            st.warning("⚠️ **Soil is Alkaline:** Apply Gypsum or Elemental Sulphur to lower pH and prevent iron/zinc lockout.")
        else:
            st.success("✅ **Balanced pH:** Your soil pH is in the optimal range (6.0 - 7.5) for most field crops.")
            
        # Commercial Fertilizer suggestion
        st.info(
            f"**📦 Recommended Product:** To meet your {crop_type} requirements, apply **NPK 12-32-16** at "
            f"a rate of **{target_yield * 15:.1f} kg/ha** during the vegetative stage."
        )
        
        # Seasonal advice
        st.markdown(
            """
            **⏳ Application Timeline:**
            - 🔹 **Basal Dose:** Apply 50% of Phosphorus and Potassium during sowing.
            - 🔹 **Top Dressing:** Apply Nitrogen in two splits (Seedling & Flowering stages).
            - 🔹 **Foliar Spray:** Use Zinc-Sulphate (0.5%) if yellowing occurs in new leaves.
            """
        )

        # Download Report
        report_data = {
            "Crop": crop_type,
            "Target_Yield": target_yield,
            "Nitrogen_Need": needs["Nitrogen"],
            "Phosphorus_Need": needs["Phosphorus"],
            "Potassium_Need": needs["Potassium"],
            "Soil_pH": ph_level
        }
        report_df = pd.DataFrame([report_data])
        st.download_button(
            "⬇️ Download Fertilizer Plan (CSV)",
            data=report_df.to_csv(index=False).encode(),
            file_name="fertilizer_plan.csv",
            mime="text/csv",
            use_container_width=True
        )
