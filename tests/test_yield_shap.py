"""Run with python -m pytest tests/test_yield_shap.py -q"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from modules.yield_engine import features, preprocessing
from modules.yield_shap import explain_yield_shap


def test_contributions_reconstruct_prediction():
    rng = np.random.default_rng(5)
    n = 90
    raw = pd.DataFrame({
        'Yield':rng.normal(4,1,n), 'Crop_Type':['Potato' if i%2 else 'Rice' for i in range(n)],
        'Soil_Type':['Clay' if i%3 else 'Loam' for i in range(n)],
        'Temperature':rng.uniform(15,35,n), 'Rainfall':rng.uniform(30,120,n),
    })
    x = features(raw)
    model = Pipeline([('preprocess',preprocessing(x)),('regressor',Ridge())]).fit(x,raw['Yield'])
    result = explain_yield_shap({'model':model,'engineered':True,'X_fit':x},raw.iloc[[0]], background_size=8)
    assert np.isclose(result['prediction'], result['base_value']+result['contributions']['SHAP'].sum(),atol=1e-4)
    assert 'Crop_Type' in result['contributions']['Feature'].values
