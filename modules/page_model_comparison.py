"""
Page: page_model_comparison.py
Purpose: Compare all three ML models — Logistic Regression, Random Forest,
         and Linear Regression — with metrics and plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix


def show(df: pd.DataFrame, models: dict) -> None:
    st.title("⚖️ Model Comparison")
    st.markdown(
        "Side-by-side evaluation of all three machine learning models trained on the AgriVision dataset."
    )
    st.markdown("---")

    lr_m  = models["lr_metrics"]
    rf_m  = models["rf_metrics"]
    lin_m = models["linreg_metrics"]

    # ── Summary Table ────────────────────────────────────────────────────────
    st.subheader("📊 Model Performance Summary")

    summary = pd.DataFrame([
        {
            "Model"      : "Logistic Regression",
            "Task"       : "Health Classification (Binary)",
            "Metric"     : "Accuracy",
            "Score"      : f"{lr_m['accuracy']*100:.2f}%",
            "Notes"      : "Predicts Healthy vs Unhealthy",
        },
        {
            "Model"      : "Random Forest",
            "Task"       : "Disease Classification (Multi-class)",
            "Metric"     : "Accuracy",
            "Score"      : f"{rf_m['accuracy']*100:.2f}%",
            "Notes"      : "Classifies disease type (7 classes)",
        },
        {
            "Model"      : "Linear Regression",
            "Task"       : "Yield Prediction (Regression)",
            "Metric"     : "R² Score",
            "Score"      : f"{lin_m['r2']:.4f}",
            "Notes"      : f"RMSE: {lin_m['rmse']:.3f} t/ha",
        },
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Accuracy Comparison Bar ───────────────────────────────────────────────
    st.subheader("📈 Accuracy / Score Comparison")
    scores_df = pd.DataFrame({
        "Model"  : ["Logistic Regression\n(Accuracy)", "Random Forest\n(Accuracy)", "Linear Regression\n(R²)"],
        "Score"  : [lr_m["accuracy"], rf_m["accuracy"], lin_m["r2"]],
        "Color"  : ["#3498db", "#27ae60", "#e67e22"],
    })
    fig_bar = go.Figure(go.Bar(
        x=scores_df["Model"], y=scores_df["Score"],
        marker_color=scores_df["Color"],
        text=[f"{v:.3f}" for v in scores_df["Score"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Model Score Comparison",
        yaxis=dict(range=[0, 1.1]),
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Detailed Reports ──────────────────────────────────────────────────────
    tab_lr, tab_rf, tab_lin = st.tabs([
        "🔵 Logistic Regression",
        "🟢 Random Forest",
        "🟠 Linear Regression",
    ])

    # ── Logistic Regression ──────────────────────────────────────────────────
    with tab_lr:
        st.subheader("Logistic Regression — Binary Health Classification")
        st.metric("Test Accuracy", f"{lr_m['accuracy']*100:.2f}%")

        # Classification report
        report = lr_m["report"]
        rpt_df = pd.DataFrame(report).T
        rpt_df = rpt_df.drop(columns=["support"], errors="ignore")
        st.markdown("**Classification Report:**")
        st.dataframe(rpt_df.round(3), use_container_width=True)

        # Confusion matrix
        st.markdown("**Confusion Matrix:**")
        X_te  = models["X_te"]
        yh_te = models["yh_te"]
        yh_pr = models["lr_model"].predict(X_te)
        cm    = confusion_matrix(yh_te, yh_pr)
        fig_cm = _confusion_matrix_fig(cm, ["Unhealthy", "Healthy"],
                                        "Logistic Regression Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Random Forest ────────────────────────────────────────────────────────
    with tab_rf:
        st.subheader("Random Forest — Disease Classification")
        st.metric("Test Accuracy", f"{rf_m['accuracy']*100:.2f}%")

        report = rf_m["report"]
        rpt_df = pd.DataFrame(report).T
        rpt_df = rpt_df.drop(columns=["support"], errors="ignore")
        st.markdown("**Classification Report:**")
        st.dataframe(rpt_df.round(3), use_container_width=True)

        # Feature importance
        st.markdown("**Top 15 Feature Importances:**")
        importances = models["rf_model"].feature_importances_
        feat_names  = models["feature_cols"]
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(15)
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale="Greens",
            title="Random Forest Feature Importances",
        )
        fig_imp.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Linear Regression ────────────────────────────────────────────────────
    with tab_lin:
        st.subheader("Linear Regression — Yield Prediction")
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{lin_m['r2']:.4f}")
        c2.metric("RMSE", f"{lin_m['rmse']:.3f} t/ha")

        # Actual vs Predicted scatter
        X_te  = models["X_te"]
        yy_te = models["yy_te"]
        yy_pr = models["linreg_model"].predict(X_te)
        # Sample 500 for readability
        idx = np.random.choice(len(yy_te), min(500, len(yy_te)), replace=False)
        fig_scatter = px.scatter(
            x=yy_te[idx], y=yy_pr[idx],
            labels={"x": "Actual Yield (t/ha)", "y": "Predicted Yield (t/ha)"},
            title="Actual vs Predicted Yield",
            opacity=0.6,
        )
        # Perfect prediction line
        lim = [float(min(yy_te.min(), yy_pr.min())),
               float(max(yy_te.max(), yy_pr.max()))]
        fig_scatter.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="red", dash="dash"), name="Perfect Fit"
        ))
        fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Residual plot
        residuals = yy_te[idx] - yy_pr[idx]
        fig_resid = px.histogram(
            x=residuals, nbins=40,
            title="Residual Distribution",
            labels={"x": "Residual (Actual − Predicted)"},
            color_discrete_sequence=["#e67e22"],
        )
        fig_resid.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_resid, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_matrix_fig(cm: np.ndarray, labels: list[str], title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
    )
    return fig
