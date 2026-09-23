# CLI image: every run is one `python -m wti` command that does its work and exits.
# Cron on the VM starts it (see deploy/). There is no server, port or healthcheck.
#
#   docker build -t wti-predictor .
#   docker run --rm -v wti-models:/models -v wti-data:/data wti-predictor train --horizon daily
#   docker run --rm -v wti-models:/models wti-predictor                # predict --horizon daily
#   hadolint Dockerfile

FROM python:3.12.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data /models \
    && chown 1000 /data /models
USER 1000

COPY wti/ ./wti

VOLUME [ "/data", "/models" ]


ENV WTI_DATA_DIR=/data

ENV WTI_MODEL_DIR=/models

ENTRYPOINT ["python", "-m","wti"]

CMD ["predict", "--horizon", "daily"]
