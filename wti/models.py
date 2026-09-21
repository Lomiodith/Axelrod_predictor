"""Model definitions, walk-forward cross-validation and model selection.

The three functions here are what turn a pile of features into a chosen model:

* :func:`build_model_zoo` constructs the candidates, heavily regularised on purpose --
  one-step-ahead returns are mostly noise, and a model that fits noise does worse than
  doing nothing.
* :func:`walk_forward_cv` scores them honestly: train on the past, validate on what
  came next, never the other way round.
* :func:`select_best` picks the winner on validation folds alone, before the hold-out
  is touched.

:func:`fit_all` then refits the whole zoo on the full training window so the final
table can report every model, not just the winner.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wti.config import SEED
from wti.evaluate import Estimator, Predictor, Scorer

__all__ = ["build_model_zoo", "walk_forward_cv", "select_best", "fit_all"]


# =============================================================================
# The candidates
# =============================================================================


def build_model_zoo(task: str, seed: int = SEED) -> dict[str, Estimator]:
    """Return ``{display name: unfitted estimator}`` for ``task``.

    ``task`` is ``"regression"`` (daily, tomorrow's % change) or ``"classification"``
    (hourly, P(up)); anything else is a ``ValueError``. Hyperparameters are the ones
    tuned in the notebooks.

    Only the linear models are wrapped in a ``StandardScaler`` pipeline -- trees split
    on thresholds, so scaling them is wasted work. Because a Pipeline exposes the same
    ``fit`` / ``predict`` interface as a bare estimator, the rest of the package never
    has to know which is which.
    """
    if task == "regression":
        models = {
            "linear (ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1000.0)),
            "random forest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=100,
                max_features=0.5,
                n_jobs=-1,
                random_state=seed,
            ),
            "gradient boosting": HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.01,
                max_leaf_nodes=7,
                min_samples_leaf=200,
                early_stopping=False,
                random_state=seed,
            ),
        }
    elif task == "classification":
        models = {
            "logistic": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.05, max_iter=3000),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=60,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
            "hist_gb": HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
        }
    else:
        raise ValueError(f"Unknown task {task}")
    return models


# =============================================================================
# Walk-forward cross-validation
# =============================================================================


def walk_forward_cv(
    zoo: Mapping[str, Estimator],
    X: pd.DataFrame,
    y: pd.Series,
    scorer: Scorer,
    predict: Predictor,
    n_splits: int = 5,
    gap: int = 0,
) -> pd.DataFrame:
    """Score every model in ``zoo`` on expanding walk-forward folds.

    ``X`` and ``y`` are the training portion only -- ``pipeline.train`` has already set
    the hold-out aside, and this function never sees it. ``TimeSeriesSplit`` then carves
    that training block into ``n_splits`` folds where the validation window always sits
    after the training window, with ``gap`` rows dropped between them so that features
    built from long rolling windows cannot bleed across the boundary.

    ``scorer`` is ``(y_true, y_pred) -> {metric: value}`` and ``predict`` is
    ``(model, X) -> np.ndarray``; both come from :mod:`wti.evaluate`. Taking them as
    arguments is what lets one loop serve both the regression and the classification
    horizon.

    Returns one row per (model, fold) in long format -- nothing averaged, that is
    :func:`select_best`'s job. Columns: ``model``, ``fold``, then one per metric.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    # folds depend only on the shape of X, so every model sees the same ones
    folds = list(splitter.split(X))

    rows = []
    for name, model in zoo.items():
        for fold, (train_rows, validate_rows) in enumerate(folds):
            # clone so the caller's zoo stays unfitted: pipeline.train reuses it.
            estimator = clone(model)
            # the split yields POSITIONS, hence .iloc; slicing y this way keeps its
            # index, which the regression scorer needs to look up prices.
            estimator.fit(X.iloc[train_rows], y.iloc[train_rows])
            pred = predict(estimator, X.iloc[validate_rows])

            metrics = scorer(y.iloc[validate_rows], pred)
            rows.append({"model": name, "fold": fold, **metrics})

    return pd.DataFrame(rows)


# =============================================================================
# Picking the winner
# =============================================================================


def select_best(cv: pd.DataFrame, metric: str, higher_is_better: bool = True) -> str:
    """Name of the model with the best mean ``metric`` across the folds in ``cv``.

    Averaging over folds is deliberate: one lucky fold should not win. Called before
    anyone looks at the test set -- pick on validation, report on hold-out.
    """
    if metric not in cv.columns:
        raise KeyError(
            f"{metric!r} is not a CV column; available: {sorted(cv.columns)}"
        )
    means = cv.groupby("model")[metric].mean()
    return str(means.idxmax() if higher_is_better else means.idxmin())


# =============================================================================
# Final fit
# =============================================================================


def fit_all(
    zoo: Mapping[str, Estimator], X: pd.DataFrame, y: pd.Series
) -> dict[str, Estimator]:
    """Fit every model in the zoo on the full training window.

    Called once, after selection, so the final table can show what every model scores on
    the hold-out -- not just the winner. Cloning keeps the caller's zoo unfitted.
    """
    fitted = {}
    for name, model in zoo.items():
        estimator = clone(model)
        estimator.fit(X, y)
        fitted[name] = estimator
    return fitted
