import os
from typing import Tuple, List, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Simple baseline models and utilities

def _prep_xy(df: pd.DataFrame, label_col: str, drop_cols: Tuple[str, ...]):
    # Drop id cols, the main label col, and any other label_* leakage cols
    extra_labels = [c for c in df.columns if c.startswith("label_") and c != label_col]
    drop_all = list(drop_cols) + [label_col] + extra_labels
    y = df[label_col].astype(int)
    X = df.drop(columns=drop_all, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    return X, y


def train_baseline_classifier(
    train_df: pd.DataFrame,
    label_col: str,
    drop_cols: Tuple[str, ...],
    model_name: str,
    calib_method: str = "isotonic",
    balance_classes: bool = False,
):
    X, y = _prep_xy(train_df, label_col, drop_cols)

    base = GradientBoostingClassifier(random_state=42)
    model = CalibratedClassifierCV(base, cv=3, method=calib_method)

    # Optional class balancing via sample weights
    sample_weight = None
    if balance_classes:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        if pos > 0 and neg > 0:
            w_pos = neg / pos
            sample_weight = np.where(y.values == 1, w_pos, 1.0)

    model.fit(X, y, sample_weight=sample_weight)

    # quick metrics on a holdout
    Xtr, Xte, ytr, yte, sw_tr, sw_te = train_test_split(
        X, y, sample_weight if sample_weight is not None else np.ones(len(y)),
        test_size=0.2, random_state=42, stratify=y
    )
    base2 = GradientBoostingClassifier(random_state=42)
    cal2 = CalibratedClassifierCV(base2, cv=3, method=calib_method)
    cal2.fit(Xtr, ytr, sample_weight=sw_tr)
    p = cal2.predict_proba(Xte)[:,1]
    try:
        auc = roc_auc_score(yte, p)
    except Exception:
        auc = float('nan')
    brier = brier_score_loss(yte, p)
    print(f"{model_name} holdout - AUC: {auc:.3f}, Brier: {brier:.3f}, N={len(y)}")

    bundle = {"model": model, "feature_names": list(X.columns)}
    joblib.dump(bundle, os.path.join(MODELS_DIR, f"{model_name}.joblib"))
    return bundle


def _extract_feature_names_from_model(model: Any) -> List[str] | None:
    # Try direct attribute first
    if hasattr(model, "feature_names_in_"):
        try:
            return list(getattr(model, "feature_names_in_"))
        except Exception:
            pass
    # Handle CalibratedClassifierCV structure
    try:
        cc = getattr(model, "calibrated_classifiers_", None)
        if cc and len(cc) > 0:
            # sklearn 1.5+: attribute is 'estimator'
            est = getattr(cc[0], "base_estimator", None) or getattr(cc[0], "estimator", None)
            if est is not None and hasattr(est, "feature_names_in_"):
                return list(getattr(est, "feature_names_in_"))
    except Exception:
        pass
    return None


def predict_proba(model_bundle: Union[Dict[str, Any], Any], df: pd.DataFrame, drop_cols: Tuple[str, ...]):
    # Accept either a bundle or raw model
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
        feat_names: List[str] = list(model_bundle.get("feature_names", []))
    else:
        model = model_bundle
        feat_names = _extract_feature_names_from_model(model) or []

    if not feat_names:
        raise ValueError("Could not determine model feature names. Retrain models or save with feature_names.")

    # Build aligned matrix with exact training features in order
    X = pd.DataFrame(index=df.index)
    for f in feat_names:
        if f in df.columns:
            X[f] = pd.to_numeric(df[f], errors='coerce')
        elif f == 'pa_rolling' and 'pa' in df.columns:
            X[f] = pd.to_numeric(df['pa'], errors='coerce')
        else:
            X[f] = 0.0
    X = X[feat_names].fillna(0)
    return model.predict_proba(X)[:,1]
