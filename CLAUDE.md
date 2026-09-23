# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`wti/` is a Python package for forecasting the WTI crude front-month future (`CL=F`, via yfinance). It handles two problems in one pipeline, selected by `--horizon`:

- `daily`: regression. The model predicts tomorrow's % change, and results are reported in dollars against a random walk.
- `hourly`: classification. The model predicts whether the next hourly bar closes up, and results are compared with naive rules.

Both horizons currently show **no edge**. That is the expected and accepted result, not a bug to fix. `README.md` has the numbers, the reasoning, and the "where next" list (event data, curve shape, longer horizons).

The package was ported from `WTI_Direction_Model.ipynb` (hourly) and `WTI_Price_Model.ipynb` (daily) and reproduces their numbers exactly. The notebooks are kept only for their charts and commentary. Change the package, not the notebooks.

## Commands

Run everything from the project root. There is no `pyproject.toml`, so `python -m wti` and the tests depend on the working directory (`pytest.ini` sets `pythonpath = .`).

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest

python -m wti train    --horizon daily     # fit, CV, select, hold-out eval, backtest, save
python -m wti predict  --horizon hourly    # score latest bar with saved model (needs network)
python -m wti backtest --horizon daily     # re-run backtest from saved artifact
python -m wti data     --horizon daily     # cache status / coverage
# flags: --refresh (re-download), --quiet (mute warnings), train --coverage

python -m pytest tests/                                            # full suite (41 tests)
python -m pytest tests/test_models.py::test_select_best_can_minimise   # single test
```

`train` and `backtest` run offline from `data/*.pkl`. Only `predict` and `--refresh` need the network. The tests use synthetic data from `tests/conftest.py` and never touch the network or `data/`.

## Architecture

The flow is `cli.py` (argparse only) → `pipeline.py` (orchestration) → `data` → `features` → `evaluate` / `models` → `backtest`.

- **`Horizon` (config.py)** is a frozen dataclass that holds everything that differs between daily and hourly: tickers, cache and artifact names, split size, CV folds and gap, selection metric, and backtest cost. Other modules branch on `horizon.task` / `horizon.is_classification` rather than on the horizon name. To change a setting, edit `DAILY` / `HOURLY` in `config.py`.
- **`Panel` (data.py)** is the raw multi-ticker download, reshaped and aligned on the WTI calendar. **`Dataset` (features.py)** is the modelling table, holding `frame`, `features`, `target`, `ret` (next-bar return, used by the backtest) and `price` (close at forecast time).
- **Scorer and predictor abstractions (evaluate.py)**:
  - A scorer is `(y_true, y_pred) -> dict[str, float]`, built by a factory so the regression scorer can close over prices and convert % errors into dollars.
  - A predictor is `(model, X) -> ndarray`: `predict` for regression, `predict_proba[:, 1]` for classification.
  - `models.walk_forward_cv` takes both as arguments, so one CV loop serves both tasks without branching. Keep it that way.
- **Two splits.**
  1. `evaluate.time_split` holds out the last `test_size` rows once, in `pipeline.train`.
  2. `walk_forward_cv` then runs `TimeSeriesSplit` with `gap` (5 daily, 48 hourly) on the training block only.

  Model selection uses mean CV scores. The hold-out is scored only after selection.
- **Leakage rule.** A feature at time `t` may use only bars `<= t`. `tests/test_features.py` enforces this by recomputing features on a truncated series and diffing the last row. A new feature goes into one builder in `features.py`, and the leakage tests will catch timing errors. `tests/test_models.py` uses a `SpyEstimator` to assert the fold boundaries themselves.
- **Artifacts.** `pipeline.train` saves a dict with `joblib.dump` to `models/wti_next_day.joblib` / `wti_next_hour.joblib`. Because these are pickles, training and prediction must use the same scikit-learn version. `data/*.pkl` and `models/*.joblib` are rebuildable caches.
- `config.py` sets `LOKY_MAX_CPU_COUNT` before sklearn is imported, to avoid a Windows 11 `wmic` probe. `PROJECT_ROOT` is derived from the package location, so `data/` and `models/` resolve next to `wti/` unless `WTI_DATA_DIR` / `WTI_MODEL_DIR` override them (the container sets `/data`, `/models`).

## Working with this user

- **Learning by stubs.** The user is practising Python on this project. When they ask for new functionality, ask which parts they want to write themselves. For those parts, provide:
  - a fully type-hinted signature;
  - a docstring that states exactly what the function returns, including any tuned constants;
  - `raise NotImplementedError`;
  - a pytest that defines correctness.

  Don't add step-by-step comments. Keep everything around the stub complete and wired up.
- **Docker.** The VM uses **CLI + cron**. `Dockerfile` is a one-shot `python -m wti` image with `predict --horizon daily` as its default command. It was written by the user, so review it rather than rewriting. An HTTP API is out of scope for now. Checks: `hadolint Dockerfile` (lint), plus the `deploy/wti.sh` build → train daily → predict daily flow as the smoke test. `deploy/wti.sh` (build / train / predict / status) and `deploy/crontab` were written by Claude at the user's request (the spec is in the `wti.sh` header). The runtime is `requirements.txt` only, and `requirements-dev.txt` adds pytest (CI uses it). Train inside the container, never copy Windows-built `.joblib` files into it.
- The project is a git repo on `main` with a manual GitHub Actions workflow (`.github/workflows/train-and-predict.yml`).
