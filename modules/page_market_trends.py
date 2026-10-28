"""
Page: page_market_trends.py
Purpose: Display interactive market price trends and forecasts for various crops.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show(df: pd.DataFrame) -> None:
    st.title("₹ Market Price Trends")
    st.markdown(
        "Monitor global and local commodity prices. Use our **AI-driven forecasting** "
        "to decide the best time to harvest and sell your crops."
    )
    st.markdown("---")

    crop_list = sorted(df["Crop_Type"].unique())
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Market Settings")
        selected_crop = st.selectbox("Select Crop", crop_list)
        market_type   = st.radio("Price Index", ["National Average", "Export Market", "Local Mandi"])
        forecast_days = st.slider("Forecast Period (Days)", 7, 90, 30)
        
        st.info(f"**Current Status:** {selected_crop} prices are currently trending **upwards** by 2.4% this week.")
        
    with col2:
        st.subheader(f"📈 {selected_crop} Price Forecast ({market_type})")
        
        # Generate synthetic price data
        dates = [datetime.now() - timedelta(days=i) for i in range(120, 0, -1)]
        base_price = {"Wheat": 22000, "Rice": 34000, "Maize": 19000, "Soybean": 45000, 
                      "Cotton": 60000, "Sugarcane": 3000, "Potato": 12000, "Tomato": 18000}.get(selected_crop, 20000)
        
        # Historical Data
        np.random.seed(42)
        history_prices = base_price + np.cumsum(np.random.normal(0, 50, 120))
        
        # Forecast Data
        forecast_dates = [datetime.now() + timedelta(days=i) for i in range(forecast_days)]
        last_price = history_prices[-1]
        forecast_prices = last_price + np.cumsum(np.random.normal(5, 30, forecast_days))
        
        # Plotly Graph
        fig = go.Figure()
        
        # Historical Line
        fig.add_trace(go.Scatter(
            x=dates, y=history_prices,
            mode='lines',
            name='Historical Price',
            line=dict(color='#27ae60', width=2)
        ))
        
        # Forecast Line
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_prices,
            mode='lines',
            name='AI Forecast',
            line=dict(color='#e67e22', width=3, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=list(forecast_prices + 100) + list(forecast_prices - 100)[::-1],
            fill='toself',
            fillcolor='rgba(230, 126, 34, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title=f"Price (₹/tonne)",
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Market comparison table
    st.subheader("🌍 Regional Market Comparison")
    
    market_data = pd.DataFrame({
        "Market Region": ["North Hub", "South Terminal", "Central Mandi", "East Port", "West Coast"],
        "Current Price": history_prices[-1] + np.random.normal(0, 150, 5),
        "Demand Score": np.random.randint(60, 95, 5),
        "Logistic Cost": np.random.randint(5, 25, 5).astype(float)
    })
    
    market_data["Net Profit (Est)"] = market_data["Current Price"] - (market_data["Current Price"] * 0.1) - (market_data["Logistic Cost"] * 10)
    
    # Display table with formatting
    def color_demand(val):
        color = 'green' if val > 80 else 'orange' if val > 70 else 'red'
        return f'color: {color}'

    st.table(market_data.style.format({
        "Current Price": "₹{:,.2f}",
        "Net Profit (Est)": "₹{:,.2f}",
        "Logistic Cost": "{:.1f}%"
    }).map(color_demand, subset=['Demand Score']))
