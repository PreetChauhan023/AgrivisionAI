"""Run: python -m pytest tests/test_yield_engine.py -q"""
import pandas as pd
from modules.yield_engine import features, preprocessing, metrics


def test_engineering_no_target():
    data = pd.DataFrame([{'Yield': 99., 'Nitrogen': 10., 'Phosphorus': 5.,
                          'Potassium': 8., 'Rainfall': 100., 'Temperature': 23.,
                          'Soil_Moisture': 50., 'Irrigation_Level': 40.}])
    X = features(data)
    assert 'Yield' not in X
    assert X['NPK_Total'].iloc[0] == 23.
    assert X['Rain_Temp_Interaction'].iloc[0] == 2300.


def test_unseen_categorical():
    train = features(pd.DataFrame([{'Crop_Type': 'Potato', 'Yield':1., 'Rainfall':500.},
                                  {'Crop_Type': 'Rice', 'Yield':2., 'Rainfall':600.}]))
    for col in train.select_dtypes(include='number').columns:
        train[col] = train[col].fillna(1.0)
    proc = preprocessing(train)
    proc.fit(train)
    unseen=features(pd.DataFrame([{'Crop_Type':'NewCrop','Yield':3.}]))
    assert proc.transform(unseen).shape[0] == 1
