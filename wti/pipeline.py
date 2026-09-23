"""End-to-end orchestration: train a model, save it, score the latest bar.

This module is the storyline the notebooks told, with the plotting removed and the
printing kept. Each function here maps to one command in :mod:`wti.cli`.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import joblib
import pandas as pd

from wti import backtest
from wti.config import MODEL_DIR, SEED, Horizon
from wti.data import coverage, fresh_panel, load_panel
from wti.evaluate import (classification_baselines, predictor_for,
                          regression_benchmark, results_table, scorer_for, time_split,
                          verdict)
from wti.features import build_dataset, latest_features
from wti.models import build_model_zoo, fit_all, select_best, walk_forward_cv


def _section(title: str) -> None:
    """Print an underlined heading, so a long run stays readable."""
    print(f"\n{title}\n{'-' * len(title)}")


def train(horizon: Horizon, refresh: bool = False, show_coverage: bool = False) -> dict[str, Any]:
    """Fit, validate, select, evaluate on the hold-out, backtest and save.

    Returns the artifact dict that was written to ``models/``.
    """
    _section(f"1. Data ({horizon.name}, {horizon.interval})")
    panel = load_panel(horizon, force=refresh)
    print(f"{len(panel):,} bars  {panel.index.min()} -> {panel.index.max()}")
    if show_coverage:
        print(coverage(panel).to_string())

    dataset = build_dataset(horizon, panel)
    print(f"{len(dataset):,} modelling rows, {len(dataset.features)} features "
          f"({dataset.frame.index.min():%Y-%m-%d} -> {dataset.frame.index.max():%Y-%m-%d})")

    _section("2. Split")
    train_frame, test_frame = time_split(dataset, horizon.test_size)
    X_train, y_train = train_frame[dataset.features], train_frame[dataset.target]
    X_test, y_test = test_frame[dataset.features], test_frame[dataset.target]
    print(f"Train: {len(train_frame):,} rows  "
          f"{train_frame.index.min():%Y-%m-%d} -> {train_frame.index.max():%Y-%m-%d}")
    print(f"Test : {len(test_frame):,} rows  "
          f"{test_frame.index.min():%Y-%m-%d} -> {test_frame.index.max():%Y-%m-%d}")

    scorer = scorer_for(horizon, dataset)
    predict_fn = predictor_for(horizon)

    _section(f"3. Walk-forward CV ({horizon.cv_splits} folds, gap {horizon.cv_gap})")
    zoo = build_model_zoo(horizon.task, seed=SEED)
    cv = walk_forward_cv(zoo, X_train, y_train, scorer, predict_fn,
                         n_splits=horizon.cv_splits, gap=horizon.cv_gap)
    metrics = [c for c in cv.columns if c not in ("model", "fold")]
    print(cv.groupby("model")[metrics].mean().round(4).to_string())

    best_name = select_best(cv, horizon.selection_metric)
    print(f"\nSelected on validation {horizon.selection_metric}: {best_name}")

    _section("4. Hold-out results")
    fitted = fit_all(zoo, X_train, y_train)
    predictions = {name: predict_fn(model, X_test) for name, model in fitted.items()}

    if horizon.is_classification:
        rows = classification_baselines(X_test, y_test, up_rate=float(y_train.mean()))
    else:
        rows = regression_benchmark(scorer, y_test)
    rows.update({name: scorer(y_test, pred) for name, pred in predictions.items()})

    table = results_table(rows)
    print(table.round(4).to_string())
    print(f"\nVerdict: {best_name} {verdict(horizon, table, best_name, len(test_frame))}")

    _section("5. Toy backtest on the hold-out")
    best_pred = pd.Series(predictions[best_name], index=test_frame.index)
    _, summary = backtest.run(horizon, best_pred, test_frame[dataset.ret])
    print(summary.round(4).to_string())
    print(f"(cost: {horizon.cost_bps} bp per position change)")

    if horizon.is_classification:
        print("\nAccuracy by confidence bucket:")
        print(backtest.accuracy_by_confidence(best_pred, y_test).to_string())
    else:
        window = 63  # about three months of trading days
        share = backtest.rolling_win_rate(best_pred, y_test, test_frame[dataset.price], window)
        print(f"\nBeat the random walk in {share * 100:.0f}% of rolling "
              f"{window}-day windows (50% is a coin flip).")

    _section("6. Save")
    artifact = {
        "horizon": horizon.name,
        "task": horizon.task,
        "model_name": best_name,
        "model": fitted[best_name],
        "features": dataset.features,
        "tickers": dict(horizon.tickers),
        "train_end": train_frame.index.max(),
        "test_metrics": table.loc[best_name].to_dict(),
        # typical size of a miss, used for the band around a point forecast
        "resid_std": float((y_test - best_pred).std()) if not horizon.is_classification else None,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, horizon.artifact_path)
    print(f"Saved {horizon.artifact_path}")
    return artifact


def load_artifact(horizon: Horizon) -> dict[str, Any]:
    """Load a saved model, with a message pointing at `train` if there is none."""
    if not horizon.artifact_path.exists():
        raise FileNotFoundError(
            f"no saved model at {horizon.artifact_path}. Run: python -m wti train --horizon {horizon.name}"
        )
    return joblib.load(horizon.artifact_path)


def predict(horizon: Horizon, period: str | None = None) -> dict[str, Any]:
    """Score the most recent bar and describe the next one."""
    artifact = load_artifact(horizon)
    panel = fresh_panel(horizon, period=period)
    row = latest_features(horizon, panel, artifact["features"])
    stamp = row.index[0]
    last_close = float(panel.close.loc[stamp, "wti"])

    if horizon.is_classification:
        p_up = float(artifact["model"].predict_proba(row)[0, 1])
        return {
            "horizon": horizon.name,
            "model": artifact["model_name"],
            "as_of_bar": stamp,
            "last_close": round(last_close, 2),
            "p_up": round(p_up, 4),
            "call": "UP" if p_up >= 0.5 else "DOWN",
        }

    change = float(artifact["model"].predict(row)[0])
    band = artifact["resid_std"] or 0.0
    now = pd.Timestamp.now(tz="America/New_York").date()
    return {
        "horizon": horizon.name,
        "model": artifact["model_name"],
        "as_of": stamp.date(),
        "bar_may_be_partial": stamp.date() >= now,
        "today_close": round(last_close, 2),
        "predicted_change_pct": round(change * 100, 2),
        "predicted_next_close": round(last_close * (1 + change), 2),
        "likely_range": (round(last_close * (1 + change - band), 2),
                         round(last_close * (1 + change + band), 2)),
    }


def backtest_saved(horizon: Horizon) -> pd.DataFrame:
    """Re-run the toy backtest for an already-saved model, without retraining."""
    artifact = load_artifact(horizon)
    dataset = build_dataset(horizon, load_panel(horizon))
    _, test_frame = time_split(dataset, horizon.test_size)

    predict_fn = predictor_for(horizon)
    pred = pd.Series(predict_fn(artifact["model"], test_frame[dataset.features]),
                     index=test_frame.index)
    _, summary = backtest.run(horizon, pred, test_frame[dataset.ret])

    print(f"{artifact['model_name']} on {len(test_frame):,} hold-out bars "
          f"({test_frame.index.min():%Y-%m-%d} -> {test_frame.index.max():%Y-%m-%d})")
    print(summary.round(4).to_string())
    return summary


def show_data(horizon: Horizon, refresh: bool = False) -> pd.DataFrame:
    """Download or load the cache and print a coverage report."""
    panel = load_panel(horizon, force=refresh)
    print(f"{len(panel):,} bars  {panel.index.min()} -> {panel.index.max()}")
    report = coverage(panel)
    print(report.to_string())
    return report
