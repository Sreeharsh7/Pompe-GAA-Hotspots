"""Classifier definitions, hyperparameter grids and repeated nested cross-validation.

Every preprocessing step (median imputation, standardisation, ADASYN oversampling) sits
inside an imblearn Pipeline, so it is fitted on training folds only and synthetic samples
never reach a validation or test fold.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Search spaces follow the manuscript (Methods 2.4). kNN 'algorithm' and 'leaf_size' only
# change the neighbour-search implementation, not predictions, so they are not searched;
# 'metric' is covered by the Minkowski power p (p=1 Manhattan, p=2 Euclidean).
GRIDS: dict[str, dict | list] = {
    "rf": {
        "n_estimators": [100, 200, 300],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    },
    "lr": [
        {"penalty": ["l1", "l2"], "C": [0.1, 1, 10, 100], "solver": ["liblinear", "saga"], "max_iter": [100, 200, 500]},
    ],
    "knn": {"n_neighbors": [3, 5, 7, 10, 20], "weights": ["uniform", "distance"], "p": [1, 2]},
    "svm": [
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf", "sigmoid"]},
        {"C": [0.1, 1, 10, 100], "kernel": ["poly"], "degree": [3, 4, 5]},
    ],
    "xgb": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 6, 9], "subsample": [0.8, 1.0]},
}


def base_estimator(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "lr":
        return LogisticRegression(random_state=seed)
    if name == "knn":
        return KNeighborsClassifier()
    if name == "svm":
        return SVC(gamma="scale", random_state=seed, max_iter=200_000)
    if name == "xgb":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0)
    raise KeyError(name)


def make_pipeline(name: str, seed: int, params: dict | None = None, probability: bool = False) -> Pipeline:
    est = base_estimator(name, seed)
    if params:
        est.set_params(**params)
    if name == "svm" and probability:
        est.set_params(probability=True)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("adasyn", ADASYN(random_state=seed)),
            ("model", est),
        ]
    )


def _prefixed(grid):
    if isinstance(grid, list):
        return [{f"model__{k}": v for k, v in g.items()} for g in grid]
    return {f"model__{k}": v for k, v in grid.items()}


def tune(name: str, X: np.ndarray, y: np.ndarray, seed: int, inner_folds: int, rf_iter: int, n_jobs: int) -> tuple[dict, float]:
    """Inner-loop hyperparameter search; returns best parameters and inner CV AUC."""
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    pipe = make_pipeline(name, seed)
    grid = _prefixed(GRIDS[name])
    if name == "rf":
        search = RandomizedSearchCV(pipe, grid, n_iter=rf_iter, scoring="roc_auc", cv=cv, random_state=seed, n_jobs=n_jobs, refit=False, error_score=np.nan)
    else:
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=False, error_score=np.nan)
    search.fit(X, y)
    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


def fit_predict(name: str, params: dict, X_tr, y_tr, X_te, seed: int) -> np.ndarray:
    pipe = make_pipeline(name, seed, params, probability=True)
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[str],
    outer_folds: int,
    repeats: int,
    inner_folds: int,
    seed: int,
    rf_iter: int,
    n_jobs: int,
    log=print,
):
    """Repeated stratified nested CV.

    Returns a long OOF table (repeat, residue, model, probability) and a table of the
    hyperparameters chosen in every outer fold.
    """
    Xv, yv = X.values, y.values.astype(int)
    oof_rows, param_rows = [], []
    for r in range(repeats):
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(outer.split(Xv, yv)):
            for m in models:
                best, inner_auc = tune(m, Xv[tr], yv[tr], seed, inner_folds, rf_iter, n_jobs)
                prob = fit_predict(m, best, Xv[tr], yv[tr], Xv[te], seed)
                fold_auc = roc_auc_score(yv[te], prob)
                oof_rows.append(pd.DataFrame({"repeat": r, "fold": f, "model": m, "residue": X.index[te], "y": yv[te], "probability": prob}))
                param_rows.append({"repeat": r, "fold": f, "model": m, "inner_cv_auc": inner_auc, "outer_fold_auc": fold_auc, "params": json.dumps(best, default=str)})
            log(f"repeat {r + 1}/{repeats} fold {f + 1}/{outer_folds} done")
    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(param_rows)


def final_models(X: pd.DataFrame, y: pd.Series, models: list[str], seed: int, inner_folds: int, rf_iter: int, n_jobs: int) -> dict:
    """Tune on all residues and refit; used only for SHAP interpretation, not for evaluation."""
    out = {}
    for m in models:
        best, auc = tune(m, X.values, y.values.astype(int), seed, inner_folds, rf_iter, n_jobs)
        pipe = make_pipeline(m, seed, best, probability=True).fit(X.values, y.values.astype(int))
        out[m] = {"pipeline": pipe, "params": best, "cv_auc": auc}
    return out


def threshold_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score

    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "sensitivity": tp / int((y == 1).sum()),
        "specificity": tn / int((y == 0).sum()),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
    }

