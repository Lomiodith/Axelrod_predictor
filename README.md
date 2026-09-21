# WTI crude-oil forecasting

Two forecasting problems on the WTI front-month future (`CL=F`), sharing one pipeline.

| Horizon  | Question                                   | Task           | Data                  |
|----------|--------------------------------------------|----------------|-----------------------|
| `daily`  | What does WTI close at tomorrow?           | regression     | daily bars since 2010 |
| `hourly` | Does the next hourly bar close up or down? | classification | hourly bars, ~2 years |

```bash
pip install -r requirements.txt

python -m wti train    --horizon daily     # fit, validate, backtest, save
python -m wti predict  --horizon daily     # tomorrow's close from the saved model
python -m wti backtest --horizon hourly    # re-run the toy backtest, no retraining
python -m wti data     --horizon daily     # cache status and coverage
python -m wti train --help
```

`--refresh` re-downloads instead of using `data/*.pkl`; `--quiet` mutes library warnings.

## Current results

Both horizons land on **no edge**, which is the expected answer for one-step-ahead
forecasting of a liquid future. The honest number in each case is the comparison against
doing nothing, not the raw accuracy.

```
daily     linear (ridge) selected on validation
          MAE $1.3913 vs random walk $1.3899  ->  beats_rw_by -0.02%
          beat the random walk in 16% of rolling 63-day windows (50% = coin flip)
          backtest: strategy -10.5% vs buy & hold +65.2% over the test years

hourly    hist_gb selected on validation
          accuracy 0.5038, AUC 0.5103
          best naive rule 0.5010  ->  edge +0.0028, one std error is 0.0100
          accuracy does not rise with model confidence, so the probabilities are noise
```

`python -m wti train` prints all of this, ending with a one-line verdict.

## How it fits together

```
wti/
  config.py      Horizon dataclass: tickers, paths, split and CV settings. No logic.
  data.py        Download, cache, reshape into a Panel aligned on the WTI calendar.
  features.py    Feature builders, labels, Dataset assembly. Strictly backward-looking.
  evaluate.py    Splitting, scorers, predictors, baselines, the verdict line.
  models.py      The model zoo, walk-forward CV, model selection, final fit.
  backtest.py    Toy backtest, confidence buckets, rolling win rate.
  pipeline.py    Orchestration: train / predict / backtest_saved / show_data.
  cli.py         argparse only.
tests/           37 tests, all passing
```

Two abstractions carry the whole thing, both defined in `evaluate.py`:

* a **scorer** is `(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]`, built by
  a factory so it can close over the prices it needs to turn a % change into dollars;
* a **predictor** is `(model, X) -> np.ndarray` — `model.predict` for regression, column
  1 of `predict_proba` for classification.

`walk_forward_cv` takes both as arguments, which is why one loop serves both horizons
without ever branching on the task.

### Where the data gets split

Two splits, at two scales — worth keeping straight:

1. **`time_split`** (`evaluate.py`), called once from `pipeline.train`, holds out the
   last 20% of rows. Nothing downstream of it sees the test block until the final table.
2. **`TimeSeriesSplit`** inside `walk_forward_cv` then carves *only the training block*
   into expanding folds, each validating on the stretch that came next, with a `gap`
   (5 rows daily, 48 hourly) so long rolling features cannot bleed across the boundary.

So the first decides what may be learned from at all; the second decides how to rehearse
within it. The model is chosen on fold 0-4 averages, and only then scored on the hold-out.

## Testing

```bash
python -m pytest tests/ -v
```

* `test_features.py` (16) — the leakage guards. They recompute features on a truncated
  series and diff the last row, so any feature that peeks at the future fails the suite.
  That matters because look-ahead is invisible in the metrics: everything just improves.
* `test_models.py` (21) — the zoo, the CV loop and selection. A spy estimator records
  which rows each fold actually trains and validates on, so these assert the split
  itself: no fold trains on the future, `gap` is honoured, the window expands.

All tests run on synthetic data — no network, no dependence on what is in `data/`.

## Notes

* `data/*.pkl` and `models/*.joblib` are caches. Both are rebuildable and gitignored.
* Yahoo caps hourly history at roughly two years, which is the binding constraint on the
  hourly model — it only ever sees one or two regimes.
* `models/wti_next_hour_hist_gb.joblib` is a leftover from the notebook's naming scheme,
  superseded by `wti_next_hour.joblib`. Safe to delete.
* `WTI_Direction_Model.ipynb` and `WTI_Price_Model.ipynb` are the exploratory originals
  this package was ported from, kept for the charts and commentary. The package
  reproduces their numbers exactly; it is the version to run.

## Where this would go next

More features and bigger models are not the gap. Event data is: EIA inventory releases
(Wednesdays 10:30 ET), OPEC+ meetings, FOMC, month-end contract roll. Right now the
models have no idea when news happens and are trying to predict scheduled volatility
from technicals alone. After that: the shape of the futures curve
(contango / backwardation), and a longer horizon — a week or a month — where
fundamentals outweigh one-step noise.

Adding a feature is a change to one function in `features.py`, and the leakage tests
will tell you if the timing is wrong.
