"""
Page: page_disease_classification.py
Purpose: Random Forest multi-class disease classification.
"""

import streamlit as st
import pandas as pd


CROP_TYPES    = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane", "Potato", "Tomato"]
SOIL_TYPES    = ["Loamy", "Sandy", "Clay", "Silt", "Peaty", "Chalky", "Saline"]
LEAF_COLORS   = ["Green", "Yellow", "Brown", "Pale Green", "Dark Green", "Reddish"]
GROWTH_STAGES = ["Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"]
REGIONS       = ["North", "South", "East", "West", "Central"]
SEASONS       = ["Kharif", "Rabi", "Zaid", "Summer", "Winter"]
FERTILIZERS   = ["Urea", "DAP", "NPK", "Organic", "Compost", "None"]
WATER_Q       = ["Good", "Moderate", "Poor"]
AIR_Q         = ["Clean", "Moderate", "Polluted"]
STRESS_L      = ["None", "Mild", "Moderate", "High"]
SEVERITY_L    = ["None", "Low", "Moderate", "High", "Severe"]

DISEASE_EMOJIS = {
    "Blight"      : "🟤",
    "Rust"        : "🟠",
    "Mildew"      : "⚪",
    "Mosaic Virus": "🟡",
    "Root Rot"    : "⚫",
    "Leaf Spot"   : "🟣",
    "Healthy"     : "🟢",
}

DISEASE_TIPS = {
    "Blight"      : "Apply copper-based fungicide. Ensure proper field drainage.",
    "Rust"        : "Use Propiconazole fungicide. Remove and burn infected leaves.",
    "Mildew"      : "Spray diluted neem oil. Improve canopy air circulation.",
    "Mosaic Virus": "No chemical cure. Remove infected plants. Control aphid vectors.",
    "Root Rot"    : "Treat with Trichoderma biofungicide. Avoid waterlogging.",
    "Leaf Spot"   : "Apply Chlorothalonil. Remove heavily infected leaves.",
    "Healthy"     : "No disease detected. Maintain regular crop management practices.",
}


def show(df: pd.DataFrame, models: dict) -> None:
    st.title("🦠 Disease Classification")
    st.markdown(
        "Uses a **Random Forest Classifier** trained on 10,000 records to predict "
        "the specific disease affecting your crop."
    )
    st.markdown("---")

    with st.form("disease_form"):
        st.subheader("📋 Disease Symptoms & Environmental Triggers")
        c1, c2 = st.columns(2)

        with c1:
            crop_type      = st.selectbox("Crop Type",        CROP_TYPES)
            leaf_color     = st.selectbox("Leaf Color",       LEAF_COLORS)
            leaf_spots     = st.slider("Visible Leaf Spots",  0,    20,    8)
            severity       = st.selectbox("Disease Severity", SEVERITY_L)
            growth_stage   = st.selectbox("Growth Stage",     GROWTH_STAGES)

        with c2:
            temperature    = st.slider("Temperature (°C)",    15.0, 45.0, 28.0)
            humidity       = st.slider("Humidity (%)",        30.0, 95.0, 70.0)
            rainfall       = st.slider("Rainfall (mm)",      200.0, 2000.0, 700.0)
            pest_presence  = st.selectbox("Pest Presence",    ["Yes","No"])
            stress_level   = st.selectbox("Stress Level",  STRESS_L)

        with st.expander("🛠️ Advanced Environmental Filters (Defaults)"):
            c3, c4 = st.columns(2)
            with c3:
                water_quality  = st.selectbox("Water Quality", WATER_Q, index=1)
                air_quality    = st.selectbox("Air Quality",   AIR_Q, index=1)
            with c4:
                soil_type      = st.selectbox("Soil Type",     SOIL_TYPES, index=0)
                region         = st.selectbox("Region",        REGIONS, index=0)

        submitted = st.form_submit_button("🔬 Classify Disease", use_container_width=True)

    if submitted:

        input_df = pd.DataFrame([{
            "Crop_Type"        : crop_type,
            "Soil_Type"        : soil_type,
            "Temperature"      : temperature,
            "Humidity"         : humidity,
            "Rainfall"         : rainfall,
            "pH_Level"         : 6.2,     # Default
            "Nitrogen"         : 70.0,    # Default
            "Phosphorus"       : 35.0,    # Default
            "Potassium"        : 80.0,    # Default
            "Leaf_Color"       : leaf_color,
            "Leaf_Spots"       : leaf_spots,
            "Disease_Severity" : severity,
            "Irrigation_Level" : 55.0,    # Default
            "Sunlight_Hours"   : 7.0,     # Default
            "Growth_Stage"     : growth_stage,
            "Pest_Presence"    : pest_presence,
            "Fertilizer_Used"  : "NPK",   # Default
            "Region"           : region,
            "Season"           : "Rabi",  # Default
            "Water_Quality"    : water_quality,
            "Air_Quality"      : air_quality,
            "Soil_Moisture"    : 40.0,    # Default
            "NDVI_Index"       : 0.45,    # Default
            "Stress_Level"     : stress_level,
            "Leaf_Area"        : 25.0,    # Default
            "Chlorophyll_Level": 38.0,    # Default
        }])

        st.session_state['disease_last_input'] = input_df.copy()

    if 'disease_last_input' in st.session_state:
        input_df = st.session_state['disease_last_input'].copy()
        st.markdown('---')
        st.subheader('🔬 Detailed tabular disease assessment')
        prediction, score = _show_prediction_shap(df, models, input_df, 'disease')
        st.subheader('🧑‍🌾 Interpreting the next steps')
        st.info('The Random Forest predicts a broad dataset label from entered features; it is not a confirmed plant-pathology diagnosis. Verify visible symptoms and, where necessary, obtain agronomist or laboratory confirmation before treatment.')
        st.caption('This tabular disease model is separate from the potato-image MobileNetV3 Smart Vision classifier. Synthetic-data performance should not be interpreted as field accuracy.')


# Self-contained prediction + SHAP utilities (no extra project modules).
def _predict_proba_for_row(bundle, row, task):
    import numpy as np
    from modules.models import _prepare_inference
    model = bundle['lr_model'] if task == 'health' else bundle['rf_model']
    X = row[list(bundle['feature_cols'])].copy() if hasattr(model, 'named_steps') else _prepare_inference(bundle, row)
    raw = np.asarray(model.predict_proba(X)[0], dtype=float)
    if task == 'health':
        names = ['Healthy' if int(c) == 1 else 'Unhealthy' for c in model.classes_]
    else:
        names = list(bundle['le_disease'].inverse_transform(np.asarray(model.classes_, dtype=int)))
    return model, names, raw


def _shap_for_row(bundle, dataset, row, task, class_name, background_size=16):
    import numpy as np
    import pandas as pd
    import shap
    model, classes, _ = _predict_proba_for_row(bundle, row, task)
    selected_idx = classes.index(class_name)
    cols = list(bundle['feature_cols'])
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(dataset[c])]
    values_map = {}
    bg = dataset[cols].sample(n=min(background_size, len(dataset)), random_state=42).copy()
    sample = row[cols].copy()
    for c in cols:
        if c in categorical:
            mode = dataset[c].dropna().mode()
            fill = mode.iloc[0] if not mode.empty else 'Unknown'
            bg[c] = bg[c].fillna(fill).astype(str)
            sample[c] = sample[c].fillna(fill).astype(str)
            values_map[c] = list(pd.concat([bg[c], sample[c]], ignore_index=True).unique())
        else:
            median = pd.to_numeric(dataset[c], errors='coerce').median()
            if pd.isna(median): median = 0.0
            bg[c] = pd.to_numeric(bg[c], errors='coerce').fillna(median)
            sample[c] = pd.to_numeric(sample[c], errors='coerce').fillna(median)
    def encode(frame):
        arr = np.zeros((len(frame), len(cols)), dtype=float)
        for i, c in enumerate(cols):
            if c in values_map:
                lookup = {str(v): idx for idx, v in enumerate(values_map[c])}
                arr[:, i] = frame[c].astype(str).map(lookup).to_numpy(float)
            else:
                arr[:, i] = frame[c].to_numpy(float)
        return arr
    def class_score(z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        frame = pd.DataFrame(index=range(len(z)))
        for i, c in enumerate(cols):
            if c in values_map:
                opts = values_map[c]
                frame[c] = [opts[int(np.clip(round(v), 0, len(opts)-1))] for v in z[:, i]]
            else:
                frame[c] = z[:, i]
        model_here, labels, p = _predict_proba_for_batch(bundle, frame[cols], task)
        return p[:, labels.index(class_name)]
    bg_enc, x_enc = encode(bg), encode(sample)
    explainer = shap.Explainer(class_score, shap.maskers.Independent(bg_enc),
                              algorithm='permutation', feature_names=cols)
    exp = explainer(x_enc, max_evals=max(2*len(cols)+1, 61), silent=True)
    attrib = np.asarray(exp.values).reshape(-1)
    baseline = float(np.asarray(exp.base_values).reshape(-1)[0])
    actual = float(class_score(x_enc)[0])
    if abs(baseline + attrib.sum() - actual) > 0.02:
        raise RuntimeError('SHAP additivity check failed; explanation is not reliable.')
    table = pd.DataFrame({
        'Feature': [c.replace('_', ' ') for c in cols],
        'SHAP (percentage points)': 100*attrib,
        'Input value': [str(sample[c].iloc[0]) for c in cols]
    })
    table = table.reindex(table['SHAP (percentage points)'].abs().sort_values(ascending=False).index)
    return baseline, actual, table.reset_index(drop=True)


def _predict_proba_for_batch(bundle, rows, task):
    import numpy as np
    from modules.models import _prepare_inference
    model = bundle['lr_model'] if task == 'health' else bundle['rf_model']
    if hasattr(model, 'named_steps'):
        transformed = rows[list(bundle['feature_cols'])]
    else:
        # Legacy preprocessing encodes only the first row per call; process
        # every SHAP perturbation separately to avoid repeated predictions.
        transformed = np.vstack([_prepare_inference(bundle, rows.iloc[[i]])
                                 for i in range(len(rows))])
    proba = np.asarray(model.predict_proba(transformed), dtype=float)
    if task == 'health':
        labels = ['Healthy' if int(c) == 1 else 'Unhealthy' for c in model.classes_]
    else:
        labels = list(bundle['le_disease'].inverse_transform(np.asarray(model.classes_, dtype=int)))
    return model, labels, proba


def _show_prediction_shap(dataset, bundle, row, task):
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    _, labels, scores = _predict_proba_for_row(bundle, row, task)
    ordering = sorted(range(len(scores)), key=lambda i: -scores[i])
    prediction = labels[ordering[0]]
    score = float(scores[ordering[0]])
    st.subheader('Prediction')
    left, right = st.columns([2, 1])
    left.markdown(f'### {prediction}')
    right.metric('Model class score', f'{score:.2%}')
    st.caption('Model class scores are not independently calibrated probabilities.')
    prediction_tab, shap_tab = st.tabs(['📊 Prediction output', '🧠 SHAP explanation'])
    with prediction_tab:
        table = pd.DataFrame({'Class':labels, 'Model score':scores}).sort_values('Model score')
        fig = px.bar(table, x='Model score', y='Class', orientation='h', text='Model score')
        fig.update_traces(texttemplate='%{x:.1%}', textposition='outside')
        fig.update_xaxes(tickformat='.0%', range=[0, 1.12])
        fig.update_layout(height=max(240, len(labels)*55+100), margin=dict(l=4,r=15,t=5,b=5))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(table.assign(**{'Model score':lambda d:d['Model score'].map(lambda v:f'{v:.2%}')}),
                     hide_index=True, use_container_width=True)
    with shap_tab:
        st.write(f'SHAP explanation of the **{prediction}** class score for this observation.')
        st.caption('Values indicate model contributions relative to sampled background observations, not real-world causes.')
        try:
            with st.spinner('Calculating SHAP values…'):
                base, actual, report = _shap_for_row(bundle, dataset, row, task, prediction)
            c1,c2=st.columns(2)
            c1.metric('Background class score', f'{base:.2%}')
            c2.metric('Explained class score', f'{actual:.2%}')
            view = report.head(15).sort_values('SHAP (percentage points)')
            fig = px.bar(view, x='SHAP (percentage points)', y='Feature', orientation='h',
                         color='SHAP (percentage points)', color_continuous_scale='RdBu',
                         color_continuous_midpoint=0)
            fig.update_layout(height=max(340, len(view)*32+100))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(report.round(3), hide_index=True, use_container_width=True)
        except ImportError:
            st.error('Install SHAP: python -m pip install shap')
        except Exception as e:
            st.error(f'SHAP explanation failed: {e}')
    export = row.copy()
    export['Prediction'] = prediction
    export['Model_score'] = score
    st.download_button('Download prediction CSV', export.to_csv(index=False).encode('utf-8'),
                       file_name=f'{task}_prediction.csv', mime='text/csv', key=f'{task}_csv')
    return prediction, score
