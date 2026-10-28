"""Advanced, explainable crop yield prediction using a standalone trained pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.yield_engine import train_yield_engine, estimate, local_perturbation, FEATURE_SET
from modules.yield_shap import explain_yield_shap

@st.cache_resource(show_spinner='Training and validating advanced yield models…')
def _cached_engine(df: pd.DataFrame):
    return train_yield_engine(df)


def _defaults(df: pd.DataFrame) -> dict:
    defaults = {}
    for col in FEATURE_SET:
        s = df[col].dropna() if col in df else pd.Series(dtype=object)
        if len(s) == 0:
            defaults[col] = None
        elif pd.api.types.is_numeric_dtype(s):
            defaults[col] = float(s.median())
        else:
            defaults[col] = str(s.mode().iloc[0])
    return defaults


def _input(df: pd.DataFrame) -> dict:
    values = _defaults(df)
    main = ['Crop_Type','Soil_Type','Season','Region','Nitrogen','Phosphorus',
            'Potassium','Rainfall','Irrigation_Level','Sunlight_Hours','Temperature',
            'Humidity','pH_Level','Soil_Moisture']
    st.caption('Unspecified inputs use training-dataset medians or most frequent categories. '
               'Review advanced inputs for meaningful estimates.')
    with st.form('advanced_yield_form'):
        a,b = st.columns(2)
        for i, col in enumerate(FEATURE_SET):
            container = a if i%2==0 else b
            # All features are editable: advanced group is visually separated below the form.
            with container:
                label = col.replace('_',' ')
                series = df[col].dropna() if col in df else pd.Series(dtype=object)
                if pd.api.types.is_numeric_dtype(series):
                    proposed = values.get(col)
                    values[col] = st.number_input(label, value=float(proposed) if proposed is not None else 0.0,
                                                  format='%.3f', key='yield_'+col)
                else:
                    choices = sorted({str(v) for v in series.astype(str) if str(v) != 'nan'})
                    if not choices: choices=['Missing']
                    default = str(values.get(col))
                    values[col] = st.selectbox(label, choices, index=choices.index(default) if default in choices else 0,
                                               key='yield_'+col)
        sent = st.form_submit_button('Analyze crop yield', type='primary', use_container_width=True)
    return values if sent else None


def show(df: pd.DataFrame, models: dict) -> None:
    st.title('🌾 Advanced Crop Yield Intelligence')
    st.markdown('Model selection · 90% prediction intervals · sensitivity explanations · scenario analysis')
    st.info('Research prototype trained on synthetic agricultural data. Results are not validated field forecasts. '
            'Scenario changes are associations learned by the model, not proof of causal benefit.')
    try:
        engine = _cached_engine(df)
    except Exception as exc:
        st.error(f'Yield-model training failed: {exc}')
        return
    selected = engine['selected']
    with st.expander('📋 Model quality and training method', expanded=False):
        met=engine['evaluation']
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Test MAE',f"{met['MAE']:.2f} t/ha")
        c2.metric('Test RMSE',f"{met['RMSE']:.2f} t/ha")
        c3.metric('Test R²',f"{met['R2']:.3f}")
        c4.metric('Interval coverage',f"{met['PI_coverage']:.1%}")
        st.write('Selected by 3-fold cross-validation MAE on training-only records:', selected)
        st.dataframe(engine['comparison'].round(3),hide_index=True,use_container_width=True)
        st.caption('Holdout test: 20%; calibration: 16%; model fit: 64%. '
                   'Conformal intervals target 90% *marginal* coverage under exchangeability.')
    values = _input(df)
    if values is not None:
        st.session_state['yield_input_row'] = values
    values = st.session_state.get('yield_input_row')
    if values is None:
        return
    row = pd.DataFrame([values])
    try:
        forecast = estimate(engine, row).iloc[0]
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')
        return
    st.divider()
    st.subheader('📍 Yield forecast')
    c1,c2,c3=st.columns(3)
    c1.metric('Predicted yield',f"{forecast['Prediction']:.2f} t/ha")
    c2.metric('90% lower bound',f"{forecast['Lower90']:.2f} t/ha")
    c3.metric('90% upper bound',f"{forecast['Upper90']:.2f} t/ha")
    st.caption(f'Selected estimator: {selected}. The interval is calibrated from held-out residuals; '
               'its coverage is not guaranteed for an individual field.')
    tab_explain,tab_scenario,tab_evaluate,tab_export=st.tabs([
        '🧠 Explanation','🧪 What-if scenarios','📈 Experimental results','📥 Export'])
    with tab_explain:
        st.subheader('SHAP — feature contributions')
        st.markdown('SHAP attributes the selected model’s prediction to its input features. '
                    'Positive values raise the estimated yield relative to the SHAP baseline; '
                    'negative values lower it. The units are the yield target’s units.')
        if st.button('Calculate SHAP explanation', key='yield_shap_button'):
            with st.spinner('Calculating SHAP contributions (may take a moment on CPU)…'):
                try:
                    explanation = explain_yield_shap(engine, row)
                    st.session_state['yield_shap_result'] = explanation
                    st.session_state['yield_shap_input'] = repr(sorted(values.items()))
                except Exception as exc:
                    st.error(f'SHAP calculation failed: {exc}')
        explanation = st.session_state.get('yield_shap_result')
        same_input = st.session_state.get('yield_shap_input') == repr(sorted(values.items()))
        if explanation is not None and same_input:
            explanation_table = explanation['contributions']
            k1, k2, k3 = st.columns(3)
            k1.metric('SHAP baseline', f"{explanation['base_value']:.2f}")
            k2.metric('Model prediction', f"{explanation['prediction']:.2f}")
            k3.metric('Training background', str(explanation['background_size']))
            top = explanation_table.head(15).sort_values('SHAP')
            chart = px.bar(top, x='SHAP', y='Feature', orientation='h',
                           color='SHAP', color_continuous_scale='RdBu',
                           color_continuous_midpoint=0,
                           title='Top 15 SHAP contributions (original feature groups)')
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(explanation_table[['Feature','SHAP']].round(4),
                         hide_index=True, use_container_width=True)
            st.caption('One-hot category contributions are grouped into their original input feature. '
                       'Derived features are shown separately. Baseline + all SHAP contributions '
                       'reconstruct the raw, unclipped model prediction; displayed yield may be '
                       'clipped to zero. Explanations are associative, not causal.')
        else:
            st.info('Click Calculate SHAP explanation to compute genuine model attributions for this input.')
        with st.expander('Additional analysis: reference-replacement sensitivity'):
            sensitivity = local_perturbation(engine, row, engine['X_fit'])
            if not sensitivity.empty:
                fig=px.bar(sensitivity.sort_values('Impact'), x='Impact', y='Feature', orientation='h',
                           labels={'Impact':'Prediction change (t/ha)'})
                st.plotly_chart(fig,use_container_width=True)
                st.dataframe(sensitivity[['Feature','Impact','Reference']].round(3),
                             hide_index=True,use_container_width=True)
            st.caption('This sensitivity analysis is separate from SHAP and does not estimate causal effects.')
    with tab_scenario:
        st.subheader('Compare agricultural input scenarios')
        st.write('Adjust model inputs below to compare hypothetical predictions.')
        scenarios={'Baseline':row.copy()}
        with st.form('yield_scenario_form'):
            rain=st.number_input('Scenario rainfall (mm)',min_value=0.,value=float(values['Rainfall']),step=25.)
            irri=st.number_input('Scenario irrigation (%)',min_value=0.,max_value=100.,
                                 value=float(values['Irrigation_Level']),step=5.)
            nitrogen=st.number_input('Scenario nitrogen (kg/ha)',min_value=0.,
                                     value=float(values['Nitrogen']),step=10.)
            compare=st.form_submit_button('Compare scenario')
        if compare:
            changed=row.copy()
            changed.loc[0,'Rainfall']=rain
            changed.loc[0,'Irrigation_Level']=irri
            changed.loc[0,'Nitrogen']=nitrogen
            scenarios['Adjusted']=changed
            table=[]
            for name,frame in scenarios.items():
                data=estimate(engine,frame).iloc[0].to_dict()
                table.append({'Scenario':name,**data})
            result=pd.DataFrame(table)
            st.dataframe(result.round(2),hide_index=True,use_container_width=True)
            fig=px.bar(result,x='Scenario',y='Prediction',title='Model-based predicted yield comparison')
            st.plotly_chart(fig,use_container_width=True)
            st.caption('Changes are model outputs, not predictions of the outcome of real interventions.')
    with tab_evaluate:
        observed=pd.DataFrame({'Observed':engine['y_test'],'Predicted':engine['y_pred']})
        fig=px.scatter(observed,x='Observed',y='Predicted',opacity=.5,
                       title='Held-out actual vs predicted yield')
        st.plotly_chart(fig,use_container_width=True)
        observed['Residual']=observed['Observed']-observed['Predicted']
        st.plotly_chart(px.histogram(observed,x='Residual',nbins=35,
                                    title='Held-out prediction residuals'),use_container_width=True)
        if 'ablation' in engine:
            st.subheader('Feature engineering ablation (held-out comparison)')
            st.dataframe(engine['ablation'].round(3),hide_index=True,use_container_width=True)
            st.caption('The two configurations each select an estimator via cross-validation; '
                       'do not interpret small differences as statistically significant.')
    with tab_export:
        output=row.copy()
        for key,value in forecast.items(): output[key]=value
        output['Selected_Model']=selected
        st.download_button('Download prediction report (CSV)',output.to_csv(index=False),
                           file_name='agrivision_yield_report.csv',mime='text/csv')
