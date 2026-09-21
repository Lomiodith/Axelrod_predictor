"""Command-line interface. Argument parsing only -- the work lives in :mod:`wti.pipeline`."""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from wti.config import HORIZONS

EPILOG = """\
examples:
  python -m wti train --horizon daily            fit, validate, backtest and save
  python -m wti train --horizon hourly --refresh re-download before training
  python -m wti predict --horizon daily          tomorrow's close from the saved model
  python -m wti backtest --horizon hourly        re-run the backtest, no retraining
  python -m wti data --horizon daily             cache status and coverage report
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wti",
        description="WTI crude-oil forecasting: next-day price and next-hour direction.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Flags every subcommand shares. Declared once here and inherited via `parents`, so
    # they are accepted after the subcommand, which is where people actually type them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--horizon", choices=sorted(HORIZONS), default="daily",
                        help="daily = tomorrow's price, hourly = next hour up/down (default: daily)")
    common.add_argument("--quiet", action="store_true", help="suppress library warnings")

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", parents=[common],
                             help="fit, select, evaluate and save a model")
    p_train.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    p_train.add_argument("--coverage", action="store_true", help="print the per-market coverage table")

    p_predict = sub.add_parser("predict", parents=[common],
                               help="score the latest bar with the saved model")
    p_predict.add_argument("--period", default=None,
                           help="yfinance period to pull (default: the horizon's live_period)")

    sub.add_parser("backtest", parents=[common],
                   help="re-run the toy backtest for the saved model")

    p_data = sub.add_parser("data", parents=[common], help="show cache status and data coverage")
    p_data.add_argument("--refresh", action="store_true", help="re-download and overwrite the cache")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        warnings.filterwarnings("ignore")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 80)

    horizon = HORIZONS[args.horizon]

    # Imported here so `--help` stays instant and does not pay for pandas + sklearn.
    from wti import pipeline

    try:
        if args.command == "train":
            pipeline.train(horizon, refresh=args.refresh, show_coverage=args.coverage)
        elif args.command == "predict":
            result = pipeline.predict(horizon, period=args.period)
            width = max(len(k) for k in result)
            for key, value in result.items():
                print(f"{key:<{width}}  {value}")
        elif args.command == "backtest":
            pipeline.backtest_saved(horizon)
        elif args.command == "data":
            pipeline.show_data(horizon, refresh=args.refresh)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    return 0
