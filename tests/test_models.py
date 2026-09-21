"""Tests for wti/models.py: the zoo, the CV loop and model selection.

The CV tests are the load-bearing ones. A spy estimator records which rows each fold
actually trains and validates on, so they assert the split itself -- that no fold ever
trains on the future, that ``gap`` reaches ``TimeSeriesSplit``, and that the training
window expands. Shape-only assertions would pass on a loop that leaks.

    python -m pytest tests/test_models.py -v
    python -m pytest tests/test_models.py -v -k zoo
    python -m pytest tests/test_models.py -v -k walk_forward
    python -m pytest tests/test_models.py -v -k select
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from wti.models import build_model_zoo, select_best, walk_forward_cv

# ---------------------------------------------------------------- helpers

CALLS: list[tuple[str, int, int]] = []


class SpyEstimator(BaseEstimator):
    """Records the first and last row index it is shown, so folds can be inspected.

    Subclasses BaseEstimator so ``sklearn.base.clone`` works on it; the recording goes
    to a module-level list because clone deep-copies constructor arguments.
    """

    def fit(self, X, y):
        CALLS.append(("fit", int(X.index[0]), int(X.index[-1])))
        self.fitted_ = True
        return self

    def predict(self, X):
        CALLS.append(("predict", int(X.index[0]), int(X.index[-1])))
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])


def toy_xy(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(n)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=index)
    y = pd.Series(rng.normal(size=n), index=index, name="y")
    return X, y


def final_step(estimator):
    """The estimator itself, or the last step if it is a Pipeline."""
    return estimator[-1] if isinstance(estimator, Pipeline) else estimator


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()
    yield


# ============================================================== the model zoo

def test_zoo_regression_keys():
    zoo = build_model_zoo("regression")
    assert set(zoo) == {"linear (ridge)", "random forest", "gradient boosting"}


def test_zoo_classification_keys():
    zoo = build_model_zoo("classification")
    assert set(zoo) == {"logistic", "random_forest", "hist_gb"}


def test_zoo_rejects_unknown_task():
    with pytest.raises(ValueError):
        build_model_zoo("forecasting the weather")


def test_zoo_scales_only_the_linear_models():
    """Trees split on thresholds, so scaling them is wasted work."""
    regression = build_model_zoo("regression")
    assert isinstance(regression["linear (ridge)"], Pipeline)
    assert isinstance(final_step(regression["linear (ridge)"]), Ridge)
    assert not isinstance(regression["random forest"], Pipeline)
    assert not isinstance(regression["gradient boosting"], Pipeline)

    classification = build_model_zoo("classification")
    assert isinstance(classification["logistic"], Pipeline)
    assert isinstance(final_step(classification["logistic"]), LogisticRegression)


def test_zoo_uses_the_tuned_hyperparameters():
    """Strong regularisation is the point: unregularised models fit the noise."""
    regression = build_model_zoo("regression")
    assert final_step(regression["linear (ridge)"]).alpha == 1000.0
    assert regression["random forest"].min_samples_leaf == 100
    assert regression["gradient boosting"].learning_rate == 0.01

    classification = build_model_zoo("classification")
    assert final_step(classification["logistic"]).C == 0.05
    assert classification["random_forest"].min_samples_leaf == 60
    assert classification["hist_gb"].max_leaf_nodes == 15


def test_zoo_returns_unfitted_estimators():
    X, _ = toy_xy(50)
    for model in build_model_zoo("regression").values():
        with pytest.raises(NotFittedError):
            model.predict(X)


def test_zoo_respects_the_seed():
    a = build_model_zoo("regression", seed=7)["random forest"]
    b = build_model_zoo("regression", seed=7)["random forest"]
    c = build_model_zoo("regression", seed=99)["random forest"]
    assert a.random_state == b.random_state == 7
    assert c.random_state == 99


def test_zoo_regression_models_fit_and_predict():
    X, y = toy_xy(200)
    for name, model in build_model_zoo("regression").items():
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (200,), name
        assert np.isfinite(pred).all(), name


def test_zoo_classification_models_give_probabilities():
    X, y = toy_xy(200)
    labels = (y > 0).astype(int)
    for name, model in build_model_zoo("classification").items():
        model.fit(X, labels)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2), name
        assert ((proba >= 0) & (proba <= 1)).all(), name


# ============================================================ walk_forward_cv

def simple_scorer(y_true, y_pred):
    assert isinstance(y_true, pd.Series), "the scorer needs y_true to keep its index"
    return {"score": float(np.mean(y_pred)), "n": float(len(y_true))}


def test_walk_forward_cv_returns_one_row_per_model_per_fold():
    X, y = toy_xy()
    zoo = {"a": SpyEstimator(), "b": SpyEstimator(), "c": SpyEstimator()}
    cv = walk_forward_cv(zoo, X, y, simple_scorer, lambda m, d: m.predict(d), n_splits=5, gap=0)

    assert isinstance(cv, pd.DataFrame)
    assert len(cv) == 15
    assert cv["model"].value_counts().to_dict() == {"a": 5, "b": 5, "c": 5}
    assert sorted(cv["fold"].unique()) == [0, 1, 2, 3, 4]


def test_walk_forward_cv_includes_the_scorer_metrics():
    X, y = toy_xy()
    cv = walk_forward_cv({"a": SpyEstimator()}, X, y, simple_scorer,
                         lambda m, d: m.predict(d), n_splits=3, gap=0)
    assert {"model", "fold", "score", "n"} <= set(cv.columns)
    assert cv["n"].gt(0).all()


def test_walk_forward_cv_uses_the_predict_callable():
    """Classification needs predict_proba, so the loop must not hard-code .predict."""
    X, y = toy_xy()
    sentinel = lambda model, data: np.full(len(data), 7.0)  # noqa: E731
    cv = walk_forward_cv({"a": SpyEstimator()}, X, y, simple_scorer, sentinel,
                         n_splits=3, gap=0)
    assert (cv["score"] == 7.0).all()


def test_walk_forward_cv_never_trains_on_the_future():
    """Every validation block must start after the training block ends."""
    X, y = toy_xy()
    walk_forward_cv({"a": SpyEstimator()}, X, y, simple_scorer,
                    lambda m, d: m.predict(d), n_splits=4, gap=0)

    pairs = [(CALLS[i], CALLS[i + 1]) for i in range(len(CALLS) - 1)
             if CALLS[i][0] == "fit" and CALLS[i + 1][0] == "predict"]
    assert len(pairs) == 4
    for (_, _, train_end), (_, val_start, _) in pairs:
        assert val_start > train_end


@pytest.mark.parametrize("gap", [0, 10, 25])
def test_walk_forward_cv_honours_the_gap(gap):
    """The gap is not cosmetic: rolling features overlap the rows that follow them."""
    X, y = toy_xy()
    walk_forward_cv({"a": SpyEstimator()}, X, y, simple_scorer,
                    lambda m, d: m.predict(d), n_splits=4, gap=gap)

    pairs = [(CALLS[i], CALLS[i + 1]) for i in range(len(CALLS) - 1)
             if CALLS[i][0] == "fit" and CALLS[i + 1][0] == "predict"]
    for (_, _, train_end), (_, val_start, _) in pairs:
        assert val_start - train_end - 1 == gap


def test_walk_forward_cv_training_window_expands():
    X, y = toy_xy()
    walk_forward_cv({"a": SpyEstimator()}, X, y, simple_scorer,
                    lambda m, d: m.predict(d), n_splits=5, gap=0)
    sizes = [end - start for kind, start, end in CALLS if kind == "fit"]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


# ================================================================ select_best

def fake_cv() -> pd.DataFrame:
    return pd.DataFrame({
        "model": ["a", "a", "b", "b", "c", "c"],
        "fold": [0, 1, 0, 1, 0, 1],
        "roc_auc": [0.50, 0.52, 0.60, 0.58, 0.40, 0.42],
        "log_loss": [0.70, 0.69, 0.68, 0.67, 0.75, 0.74],
    })


def test_select_best_takes_the_highest_mean():
    assert select_best(fake_cv(), "roc_auc") == "b"


def test_select_best_can_minimise():
    assert select_best(fake_cv(), "log_loss", higher_is_better=False) == "b"
    assert select_best(fake_cv(), "roc_auc", higher_is_better=False) == "c"


def test_select_best_averages_rather_than_taking_the_best_fold():
    """One lucky fold must not win: 'c' peaks highest but is worst on average."""
    cv = pd.DataFrame({
        "model": ["a", "a", "c", "c"],
        "fold": [0, 1, 0, 1],
        "roc_auc": [0.55, 0.55, 0.99, 0.05],
    })
    assert select_best(cv, "roc_auc") == "a"


def test_select_best_rejects_an_unknown_metric():
    with pytest.raises(KeyError):
        select_best(fake_cv(), "roc_au")
