"""Shared, read-only interpretability helpers for AgriVision's tabular classifiers.

Supports both legacy StandardScaler-based models.py and the newer sklearn Pipeline
version. All analyses describe model behavior, not agricultural causal effects.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def _model_input(bundle, row):
    model = bundle.get('lr_model')
    if hasattr(model, 'named_steps'):
        missing = set(bundle['feature_cols']) - set(row.columns)
        if missing:
            raise ValueError('Missing required feature columns: ' + ', '.join(sorted(missing)))
        return row[bundle['feature_cols']].copy()
    # Old implementation's same transformation, using its shared inference helper.
    from modules.models import _prepare_inference
    return _prepare_inference(bundle, row)


def get_probability_report(bundle, row, task):
    model = bundle['lr_model'] if task == 'health' else bundle['rf_model']
    X = _model_input(bundle, row)
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if task == 'health':
        labels = [('Healthy' if int(c) == 1 else 'Unhealthy') for c in classes]
    else:
        labels = list(bundle['le_disease'].inverse_transform(np.asarray(classes, dtype=int)))
    probabilities = pd.DataFrame({'Class': labels, 'Model score': probabilities})
    probabilities = probabilities.sort_values('Model score', ascending=False).reset_index(drop=True)
    return model, probabilities


def score_chart(probs):
    fig = px.bar(probs.sort_values('Model score'), x='Model score', y='Class',
                 orientation='h', text='Model score', color='Class')
    fig.update_traces(texttemplate='%{x:.1%}', textposition='outside', cliponaxis=False)
    fig.update_xaxes(tickformat='.0%', range=[0, 1.12], title='Model class score')
    fig.update_yaxes(title='')
    fig.update_layout(showlegend=False, margin=dict(l=5,r=8,t=12,b=6), height=max(230,60*len(probs)))
    return fig


def input_profile(df, row, cols):
    """Dataset-relative input positioning; this is not an agronomic suitability score."""
    records=[]
    for col in cols:
        if col not in df or col not in row or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals=pd.to_numeric(df[col],errors='coerce').dropna()
        if len(vals) == 0: continue
        value=float(row[col].iloc[0])
        pct=100*float((vals<=value).mean())
        records.append({'Feature':col.replace('_',' '),'Entered value':value,
                        'Dataset median':float(vals.median()),'Dataset percentile':round(pct,1)})
    return pd.DataFrame(records)


def local_sensitivity(bundle, row, df, task, fields, target_class, max_rows=12):
    """One-at-a-time substitution against the dataset median/mode.

    Changes in model scores are NONCAUSAL and should not be called SHAP.
    """
    _, base = get_probability_report(bundle,row,task)
    lookup=dict(zip(base['Class'],base['Model score']))
    base_score=float(lookup[target_class])
    results=[]
    for field in fields[:max_rows]:
        if field not in row or field not in df: continue
        series=df[field].dropna()
        if series.empty:continue
        if pd.api.types.is_numeric_dtype(series):
            replacement=float(series.median())
        else:
            m=series.mode()
            if m.empty: continue
            replacement=m.iloc[0]
        changed=row.copy()
        if str(changed.at[changed.index[0],field])==str(replacement):continue
        changed.loc[changed.index[0],field]=replacement
        try:
            _, other = get_probability_report(bundle,changed,task)
            repl_score=float(dict(zip(other['Class'],other['Model score']))[target_class])
            results.append({'Feature':field.replace('_',' '),'Score change (pp)':round(100*(base_score-repl_score),2),
                           'Reference value':str(replacement),
                           'Original value':str(row[field].iloc[0])})
        except (ValueError,TypeError,KeyError):
            continue
    if not results:return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Score change (pp)',key=lambda x:x.abs(),ascending=False)


def global_importance(bundle, task, top=12):
    """Gini importance for RF only; original-feature grouped when pipeline is used."""
    if task!='disease': return pd.DataFrame()
    model=bundle['rf_model']
    est=model.named_steps['est'] if hasattr(model,'named_steps') else model
    if not hasattr(est,'feature_importances_'):return pd.DataFrame()
    raw=np.asarray(est.feature_importances_)
    original=bundle['feature_cols']
    if hasattr(model,'named_steps'):
        prep=model.named_steps['prep']
        try:
            names=list(prep.get_feature_names_out())
        except Exception:return pd.DataFrame()
        numerical=list(prep.transformers_[0][2])
        categorical=list(prep.transformers_[1][2])
        grouped={x:0.0 for x in original}
        for n,v in zip(names,raw):
            name=n.split('__',1)[-1]
            matched=name if name in grouped else next((c for c in sorted(categorical,key=len,reverse=True) if name.startswith(c+'_')),None)
            if matched is not None:grouped[matched]+=float(v)
        out=pd.DataFrame({'Feature':list(grouped),'Importance':list(grouped.values())})
    else:
        if len(raw)!=len(original): return pd.DataFrame()
        out=pd.DataFrame({'Feature':original,'Importance':raw})
    return out.nlargest(top,'Importance').assign(Feature=lambda d:d.Feature.str.replace('_',' ',regex=False))


def show_evaluation(bundle,task):
    key='lr_metrics' if task=='health' else 'rf_metrics'
    metrics=bundle.get(key)
    if not metrics:
        st.info('Held-out evaluation metrics are not included in the currently loaded model bundle. No performance figures are assumed.')
        return
    a,b,c=st.columns(3)
    a.metric('Held-out accuracy',f"{metrics['accuracy']:.1%}")
    b.metric('Weighted F1',f"{metrics['f1']:.3f}")
    c.metric('Weighted recall',f"{metrics['recall']:.3f}")
    cm=np.asarray(metrics.get('confusion_matrix',[]))
    if cm.ndim==2 and cm.size:
        names=['Unhealthy','Healthy'] if task=='health' else list(bundle['le_disease'].classes_)
        if len(names)==cm.shape[0]:
            fig=px.imshow(cm,x=names,y=names,text_auto=True,aspect='auto',color_continuous_scale='Blues',labels={'x':'Predicted','y':'Actual','color':'Count'})
            fig.update_layout(height=max(300,45*len(names)+160))
            st.plotly_chart(fig,use_container_width=True)
    st.caption('Metrics are taken from the currently loaded model bundle. They describe a synthetic-data test split, not field performance.')


def show_details(df,bundle,row,task,fields):
    model,prob=get_probability_report(bundle,row,task)
    prediction=str(prob.iloc[0]['Class'])
    score=float(prob.iloc[0]['Model score'])
    a,b=st.columns([2,1])
    a.subheader('Model prediction')
    a.markdown(f"### {prediction}")
    b.metric('Top class score',f'{score:.1%}')
    st.caption('Class scores are model outputs; they have not been demonstrated to be calibrated probabilities.')
    tab1,tab2,tab3,tab4=st.tabs(['📊 Class comparison','🧭 Input profile','🔎 Model interpretation','📈 Model evaluation'])
    with tab1:
        st.plotly_chart(score_chart(prob),use_container_width=True)
        shown=prob.copy();shown['Model score']=shown['Model score'].map(lambda v:f'{v:.2%}')
        st.dataframe(shown,use_container_width=True,hide_index=True)
        if len(prob)>1:
            gap=float(prob.iloc[0]['Model score']-prob.iloc[1]['Model score'])
            st.write(f'**Top-two separation:** {gap:.1%} (difference in model scores)')
    with tab2:
        st.write('How the entered measurements compare to distributions in this dataset:')
        profile=input_profile(df,row,fields)
        if not profile.empty:
            st.dataframe(profile.round(2),use_container_width=True,hide_index=True)
            fig=px.bar(profile,x='Dataset percentile',y='Feature',orientation='h',range_x=[0,100])
            fig.update_layout(height=max(230,len(profile)*33+80),margin=dict(l=8,r=10,t=8,b=8))
            st.plotly_chart(fig,use_container_width=True)
            st.caption('Percentiles compare the input with the synthetic dataset; they are not plant-health thresholds.')
    with tab3:
        st.write('**Local reference-replacement sensitivity**')
        st.caption('Each feature is replaced separately with its dataset median or mode. The chart shows how the predicted-class score changes; it is neither SHAP nor a causal explanation.')
        impacts=local_sensitivity(bundle,row,df,task,fields,prediction)
        if impacts.empty: st.info('No usable feature substitutions were available for this input.')
        else:
            fig=px.bar(impacts.sort_values('Score change (pp)'),x='Score change (pp)',y='Feature',orientation='h',color='Score change (pp)',color_continuous_scale='RdBu')
            fig.update_layout(height=max(250,len(impacts)*35+80),margin=dict(l=8,r=8,t=8,b=8))
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(impacts,use_container_width=True,hide_index=True)
        fi=global_importance(bundle,task)
        if not fi.empty:
            st.write('**Random Forest global feature importance**')
            st.bar_chart(fi.set_index('Feature')['Importance'])
            st.caption('Impurity-based importance summarizes tree-split usage across the training data; it is not a patient/field-specific explanation.')
    with tab4:
        show_evaluation(bundle,task)
    report=row.copy();report['Prediction']=prediction;report['Top_model_score']=score
    st.download_button('⬇️ Download detailed input and prediction (CSV)',report.to_csv(index=False).encode(),file_name=f'{task}_detailed_prediction.csv',mime='text/csv',key=f'{task}_report')
    return prediction, score
