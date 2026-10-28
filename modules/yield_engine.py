"""Independent, leakage-safe crop yield research pipeline for AgriVision AI.

Only model scenarios, not causal interventions. The supplied dataset is synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = {'Yield', 'Disease_Label', 'Disease_Type', 'Healthy'}
# Additional features can be disabled to provide an honest ablation comparison.
NUMERIC_SOURCE = ['Temperature','Humidity','Rainfall','pH_Level','Nitrogen','Phosphorus',
                  'Potassium','Leaf_Spots','Irrigation_Level','Sunlight_Hours',
                  'Soil_Moisture','NDVI_Index','Leaf_Area','Chlorophyll_Level']
FEATURE_SET = ['Crop_Type','Soil_Type','Temperature','Humidity','Rainfall','pH_Level',
               'Nitrogen','Phosphorus','Potassium','Leaf_Color','Leaf_Spots',
               'Disease_Severity','Irrigation_Level','Sunlight_Hours','Growth_Stage',
               'Pest_Presence','Fertilizer_Used','Region','Season','Water_Quality',
               'Air_Quality','Soil_Moisture','NDVI_Index','Stress_Level','Leaf_Area',
               'Chlorophyll_Level']
ENGINEERED = ['NPK_Total','N_to_P_Ratio','Rain_Temp_Interaction','Moisture_Irrigation_Interaction']


def features(frame: pd.DataFrame, engineered: bool = True) -> pd.DataFrame:
    """Prepare features deterministically; does not learn statistics from any partition."""
    X = frame.reindex(columns=FEATURE_SET).copy()
    for name in NUMERIC_SOURCE:
        X[name] = pd.to_numeric(X[name], errors='coerce')
    if engineered:
        n, p, k = [X[x] for x in ('Nitrogen','Phosphorus','Potassium')]
        X['NPK_Total'] = n + p + k
        X['N_to_P_Ratio'] = n / (p.abs() + 1.0)
        X['Rain_Temp_Interaction'] = X['Rainfall'] * X['Temperature']
        X['Moisture_Irrigation_Interaction'] = X['Soil_Moisture'] * X['Irrigation_Level']
    for name in X.columns.difference(NUMERIC_SOURCE + (ENGINEERED if engineered else [])):
        X[name] = X[name].astype('string').fillna('Missing').astype(str)
    return X


def preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    nums = list(X.select_dtypes(include='number').columns)
    cats = [x for x in X.columns if x not in nums]
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=False)),
                          ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats),
    ], sparse_threshold=0.0)


def candidates() -> dict:
    result = {
        'Ridge (baseline)': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=110, min_samples_leaf=3,
                                               max_features=0.8, random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=150,
                                  learning_rate=0.07, max_leaf_nodes=20, random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        result['XGBoost'] = XGBRegressor(n_estimators=140, max_depth=4, learning_rate=.06,
                                           subsample=.85, colsample_bytree=.85,
                                           objective='reg:squarederror', random_state=42, n_jobs=2)
    except ImportError:
        pass
    return result


def metrics(y, pred) -> dict:
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred))}


def _fit(frame: pd.DataFrame, engineered: bool, seed: int = 42) -> dict:
    valid = frame.loc[pd.to_numeric(frame['Yield'], errors='coerce').notna()].copy()
    if len(valid) < 100:
        raise ValueError('At least 100 labeled yield records are needed.')
    y = pd.to_numeric(valid['Yield']).astype(float)
    X = features(valid, engineered=engineered)
    # Honest separation: test never determines selected model or interval size.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.20, random_state=seed)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_dev, y_dev, test_size=.20, random_state=seed + 1)
    folds = KFold(n_splits=3, shuffle=True, random_state=seed)
    reports = []
    for name, estimator in candidates().items():
        model = Pipeline([('preprocess', preprocessing(X_fit)), ('regressor', estimator)])
        scores = -cross_val_score(model, X_fit, y_fit, cv=folds,
                                  scoring='neg_mean_absolute_error', n_jobs=1)
        reports.append({'Model':name, 'CV_MAE':float(scores.mean()),
                        'CV_MAE_SD':float(scores.std())})
    comparison = pd.DataFrame(reports).sort_values('CV_MAE').reset_index(drop=True)
    name = str(comparison.iloc[0]['Model'])
    model = Pipeline([('preprocess', preprocessing(X_fit)), ('regressor', candidates()[name])])
    model.fit(X_fit, y_fit)
    # Split conformal: marginal finite-sample coverage under exchangeability.
    errors = np.abs(np.asarray(y_cal) - model.predict(X_cal))
    alpha = .1
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * (1 - alpha))))
    q = float(np.partition(errors, rank-1)[rank-1])
    predictions = model.predict(X_test)
    evaluation = metrics(y_test, predictions)
    evaluation['PI_coverage'] = float(np.mean(np.abs(y_test - predictions) <= q))
    evaluation['PI_width'] = 2*q
    return {'model':model, 'selected':name, 'comparison':comparison,
            'evaluation':evaluation, 'q':q, 'engineered':engineered,
            'y_test':np.asarray(y_test), 'y_pred':predictions,
            'X_test':X_test, 'X_fit':X_fit, 'y_fit':y_fit}


def train_yield_engine(frame: pd.DataFrame, ablation: bool = True) -> dict:
    result = _fit(frame, engineered=True)
    if ablation:
        base = _fit(frame, engineered=False)
        result['ablation'] = pd.DataFrame([
            {'Features':'Original features', **base['evaluation']},
            {'Features':'Original + engineered', **result['evaluation']},
        ])
    return result


def estimate(engine: dict, frame: pd.DataFrame) -> pd.DataFrame:
    X = features(frame, engineered=engine['engineered'])
    value = np.maximum(0., engine['model'].predict(X))
    return pd.DataFrame({'Prediction':value,
                         'Lower90':np.maximum(0., value-engine['q']),
                         'Upper90':value+engine['q']})


def local_perturbation(engine: dict, row: pd.DataFrame, reference: pd.DataFrame,
                       max_features: int = 10) -> pd.DataFrame:
    """Sensitivity to replacing one input with its training median/mode.

    This is *not SHAP* and is not causal feature attribution.
    """
    row = row.iloc[:1].copy()
    baseline = float(estimate(engine, row)['Prediction'].iloc[0])
    deltas = []
    for col in FEATURE_SET:
        modified = row.copy()
        series = reference[col].dropna()
        if len(series) == 0: continue
        if col in NUMERIC_SOURCE:
            replacement = float(pd.to_numeric(series, errors='coerce').median())
        else:
            replacement = series.mode().iloc[0]
        modified.loc[modified.index[0], col] = replacement
        adjusted = float(estimate(engine, modified)['Prediction'].iloc[0])
        deltas.append({'Feature':col, 'Impact':baseline-adjusted,
                       'Reference':str(replacement)})
    return pd.DataFrame(deltas).assign(Magnitude=lambda x: x.Impact.abs()).sort_values(
        'Magnitude', ascending=False).head(max_features)
