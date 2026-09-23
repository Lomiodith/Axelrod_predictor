#!/usr/bin/env bash
# What cron (and you) run on the VM:
#
#   deploy/wti.sh build              build the image as wti-predictor:latest (./Dockerfile)
#   deploy/wti.sh train HORIZON      re-download and retrain HORIZON (daily|hourly)
#   deploy/wti.sh predict HORIZON    score the latest bar and append the result, headed by
#                                    a UTC timestamp line, to ~/wti-logs/predictions-HORIZON.log
#   deploy/wti.sh status             which models exist in the volume, and when they were trained
#
# Every run is `docker run --rm` of the same image with the same two named volumes,
# wti-data -> /data and wti-models -> /models. Anything else prints usage and exits 1.
# Stops at the first error and exits non-zero whenever the container does, so cron can tell.
# The schedule is in deploy/crontab.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash only: keep /data and /models as container paths

IMAGE=wti-predictor:latest
LOG_DIR="$HOME/wti-logs"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "usage: deploy/wti.sh build | train daily|hourly | predict daily|hourly | status" >&2
    exit 1
}

horizon() {
    case ${1:-} in
        daily|hourly) echo "$1" ;;
        *) usage ;;
    esac
}

run() {
    docker run --rm -v wti-data:/data -v wti-models:/models "$@"
}

# Cron can start a predict while a retrain is still writing the artifact (hourly on
# Sunday 06:05), so runs queue behind each other. flock is missing in Git Bash; skip it there.
mkdir -p "$LOG_DIR"
if command -v flock >/dev/null; then
    exec 9>"$LOG_DIR/.lock"
    flock 9
fi

case ${1:-} in
    build)
        [ $# -eq 1 ] || usage
        cd "$PROJECT_ROOT"
        docker build -t "$IMAGE" .
        ;;
    train)
        [ $# -eq 2 ] || usage
        h=$(horizon "$2")
        run "$IMAGE" train --horizon "$h" --refresh --quiet
        ;;
    predict)
        [ $# -eq 2 ] || usage
        h=$(horizon "$2")
        out=$(run "$IMAGE" predict --horizon "$h" --quiet)
        {
            echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "$out"
        } >> "$LOG_DIR/predictions-$h.log"
        echo "$out"
        ;;
    status)
        [ $# -eq 1 ] || usage
        run --entrypoint python "$IMAGE" -c '
import joblib
from wti.config import HORIZONS
for name, h in HORIZONS.items():
    if not h.artifact_path.exists():
        print(f"{name:7} no model at {h.artifact_path}")
        continue
    a = joblib.load(h.artifact_path)
    print(f"{name:7} {a["model_name"]:20} trained {a["trained_at"]} UTC, train split ends {a["train_end"]:%Y-%m-%d}")
'
        ;;
    *)
        usage
        ;;
esac
