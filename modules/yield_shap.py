"""Genuine permutation SHAP for the AgriVision fitted yield pipeline.

Explains raw (unclipped) regression outputs in the model's yield units.
One-hot encoded categories are summed back into their source input feature.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from modules.yield_engine import features


def _column_groups(preprocessor):
    """Map transformed output indices back to original/engineered columns."""
    numeric_pipe, numeric_cols = preprocessor.named_transformers_['num'], list(preprocessor.transformers_[0][2])
    categorical_pipe, categorical_cols = preprocessor.named_transformers_['cat'], list(preprocessor.transformers_[1][2])
    groups = {}
    position = 0
    # Imputer may drop entirely missing numeric fields. Only count emitted columns.
    emitted_numeric = list(numeric_pipe.get_feature_names_out(numeric_cols))
    for name in emitted_numeric:
        groups[name] = [position]
        position += 1
    encoder = categorical_pipe.named_steps['onehot']
    encoded_names = list(encoder.get_feature_names_out(categorical_cols))
    for name in categorical_cols:
        width = sum(encoded_name.startswith(name + "_") for encoded_name in encoded_names)
        groups[name] = list(range(position, position + width))
        position += width
    actual_width = len(preprocessor.get_feature_names_out())
    if position != actual_width:
        raise ValueError(f'SHAP feature mapping mismatch: {position} != {actual_width}')
    return groups


def explain_yield_shap(engine: dict, row: pd.DataFrame, background: pd.DataFrame | None = None,
                       background_size: int = 24, seed: int = 42) -> dict:
    """SHAP permutation explainer, with independent training-only background.

    The attribution is to encoded feature groups, not causal effects. Only call
    this for a small number of single-row predictions: permutation SHAP is costly.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError('Install SHAP: python -m pip install shap') from exc
    if len(row) != 1:
        raise ValueError('Explain exactly one prediction at a time.')
    model = engine['model']
    pre = model.named_steps['preprocess']
    reg = model.named_steps['regressor']
    background = engine['X_fit'] if background is None else background
    if len(background) == 0:
        raise ValueError('SHAP background must have at least one training observation.')
    # Avoid taking background from calibration/test observations.
    background_X = features(background, engineered=engine['engineered'])
    query_X = features(row, engineered=engine['engineered'])
    background_X = background_X.sample(n=min(background_size, len(background_X)), random_state=seed)
    bg = np.asarray(pre.transform(background_X), dtype=np.float64)
    query = np.asarray(pre.transform(query_X), dtype=np.float64)
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    explainer = shap.Explainer(reg.predict, masker, algorithm='permutation')
    # one full forward/reverse permutation, deterministic from numpy seed
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        explanation = explainer(query, max_evals=2 * query.shape[1] + 1)
    finally:
        np.random.set_state(state)
    raw = np.asarray(explanation.values).reshape(-1)
    groups = _column_groups(pre)
    table = pd.DataFrame([{'Feature':name, 'SHAP':float(np.sum(raw[positions]))}
                          for name, positions in groups.items()])
    table['Abs_SHAP'] = table['SHAP'].abs()
    table = table.sort_values('Abs_SHAP', ascending=False).reset_index(drop=True)
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    prediction = float(reg.predict(query)[0])
    reconstructed = base + float(table['SHAP'].sum())
    if not np.isclose(prediction, reconstructed, rtol=1e-4, atol=1e-4):
        raise RuntimeError('SHAP attributions do not reconstruct the model prediction.')
    return {'contributions':table, 'base_value':base, 'prediction':prediction,
            'background_size':len(bg), 'transformed_features':int(query.shape[1])}
