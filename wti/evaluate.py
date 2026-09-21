"""Splitting, scoring and benchmarks.

Two ideas do most of the work here:

* A **scorer** is a callable ``(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]``.
  It is built by a factory so it can close over the price series it needs to turn a
  predicted % change into dollars. Because ``y_true`` arrives as an indexed Series, the
  scorer can look up the matching closes itself.
* A **predictor** is a callable ``(model, X) -> np.ndarray``. Regression wants
  ``model.predict``; classification wants column 1 of ``model.predict_proba``. Keeping
  this out of the cross-validation loop is what lets one loop serve both problems.

Both are passed into :func:`wti.models.walk_forward_cv`, so that one loop serves both
horizons without ever asking which kind of problem it is looking at.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

from wti.config import Horizon
from wti.features import Dataset

Scorer = Callable[[pd.Series, np.ndarray], dict[str, float]]
Predictor = Callable[[object, pd.DataFrame], np.ndarray]

RANDOM_WALK = "random walk"


class Estimator(Protocol):
    """The slice of the sklearn API this package relies on."""

    def fit(self, X, y): ...
    def predict(self, X): ...


# ----------------------------------------------------------------------- splitting

def time_split(dataset: Dataset, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last ``test_size`` share of rows is the hold-out.

    No shuffling, ever. The test set is not looked at until a model has already been
    chosen on the training set's walk-forward folds.
    """
    split = int(len(dataset) * (1 - test_size))
    frame = dataset.frame
    return frame.iloc[:split], frame.iloc[split:]


# ---------------------------------------------------------------------- predictors

def predict_point(model: Estimator, X: pd.DataFrame) -> np.ndarray:
    """Regression: the predicted % change."""
    return np.asarray(model.predict(X))


def predict_proba_up(model, X: pd.DataFrame) -> np.ndarray:
    """Classification: P(next bar closes up)."""
    return np.asarray(model.predict_proba(X))[:, 1]


def predictor_for(horizon: Horizon) -> Predictor:
    """The right predictor for this horizon's task."""
    return predict_proba_up if horizon.is_classification else predict_point


# ------------------------------------------------------------------------- scorers

def regression_scorer(price: pd.Series) -> Scorer:
    """Score predicted % changes, in dollars as well as in R-squared.

    ``price`` is the close at the moment of each forecast, indexed like the labels.
    ``beats_rw_by`` is the headline number: how much lower the model's RMSE is than the
    random walk's, in percent. Zero means the model adds nothing.
    """

    def score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        close = np.asarray(price.loc[y_true.index], dtype=float)
        y_true_arr, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)

        true_px = close * (1 + y_true_arr)
        pred_px = close * (1 + y_pred)
        rw_px = close                                   # random walk: predicted change 0

        rmse = float(np.sqrt(mean_squared_error(true_px, pred_px)))
        rw_rmse = float(np.sqrt(mean_squared_error(true_px, rw_px)))
        called = y_pred != 0

        return {
            "mae_usd": float(mean_absolute_error(true_px, pred_px)),
            "rmse_usd": rmse,
            "r2_change": float(r2_score(y_true_arr, y_pred)),
            # r2_price is near 0.99 for everything including the random walk: ignore it.
            "r2_price": float(r2_score(true_px, pred_px)),
            "direction_acc": (float(np.mean(np.sign(y_pred[called]) == np.sign(y_true_arr[called])))
                              if called.any() else float("nan")),
            "beats_rw_by": (1 - rmse / rw_rmse) * 100 if rw_rmse else float("nan"),
        }

    return score


def classification_scorer(hard: bool = False) -> Scorer:
    """Score predicted P(up).

    ``hard=True`` for a rule that only emits 0 or 1: log loss and Brier are meaningless
    for those, so they come back as NaN rather than as an infinite penalty.
    """

    def score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        y_true_arr = np.asarray(y_true, dtype=int)
        proba = np.asarray(y_pred, dtype=float)
        label = (proba >= 0.5).astype(int)

        out = {
            "accuracy": float(accuracy_score(y_true_arr, label)),
            "balanced_acc": float(balanced_accuracy_score(y_true_arr, label)),
            "roc_auc": float(roc_auc_score(y_true_arr, proba)),
            "log_loss": float("nan"),
            "brier": float("nan"),
        }
        if not hard:
            out["log_loss"] = float(log_loss(y_true_arr, np.clip(proba, 1e-6, 1 - 1e-6)))
            out["brier"] = float(brier_score_loss(y_true_arr, proba))
        return out

    return score


def scorer_for(horizon: Horizon, dataset: Dataset, frame: pd.DataFrame | None = None) -> Scorer:
    """The right scorer for this horizon, wired to the prices it needs."""
    if horizon.is_classification:
        return classification_scorer()
    source = dataset.frame if frame is None else frame
    return regression_scorer(source[dataset.price])


# ---------------------------------------------------------------------- benchmarks

def regression_benchmark(scorer: Scorer, y_true: pd.Series) -> dict[str, dict[str, float]]:
    """The random walk: tomorrow's close is today's close."""
    return {RANDOM_WALK: scorer(y_true, np.zeros(len(y_true)))}


def classification_baselines(X: pd.DataFrame, y: pd.Series, up_rate: float) -> dict[str, dict[str, float]]:
    """Three rules a model has to beat before it is worth anything.

    ``majority class`` always predicts the training UP rate; the other two are the naive
    momentum and mean-reversion rules on the last bar's return.
    """
    soft = classification_scorer()
    rigid = classification_scorer(hard=True)
    momentum = (X["ret_1"] > 0).astype(float).to_numpy()
    return {
        "majority class": soft(y, np.full(len(y), up_rate)),
        "momentum (follow last hour)": rigid(y, momentum),
        "mean reversion (fade last hour)": rigid(y, 1 - momentum),
    }


def results_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn ``{name: metrics}`` into a readable table."""
    return pd.DataFrame(rows).T


def verdict(horizon: Horizon, table: pd.DataFrame, best: str, n_test: int) -> str:
    """One honest sentence about whether the selected model beat doing nothing.

    This is the line worth reading first. Both notebooks ended here with "no", which is
    the expected answer for one-step-ahead direction on a liquid future.
    """
    if horizon.is_classification:
        naive = table.drop(index=[best]).loc[
            [i for i in table.index if i in
             ("majority class", "momentum (follow last hour)", "mean reversion (fade last hour)")],
            "accuracy"].max()
        edge = table.loc[best, "accuracy"] - naive
        se = float(np.sqrt(0.25 / n_test))      # std error of an accuracy estimate at p=0.5
        good = edge > 2 * se
        return (f"edge over best naive rule {edge:+.4f} (one std error is {se:.4f}) -> "
                + ("worth investigating further" if good
                   else "within noise, no reliable signal yet"))

    beats = table.loc[best, "beats_rw_by"]
    return (f"beats the random walk by {beats:+.2f}% -> "
            + ("a real edge" if beats > 1
               else "no reliable edge; the honest forecast is 'about today's price'"))
