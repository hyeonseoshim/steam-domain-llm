#!/usr/bin/env bash
set -e

export GAME_TOPIC_SUMMARY_PATH="$PWD/game_topic_summary.json.gz"
export PORT="${PORT:-8000}"

exec python3 -m gunicorn \
  --workers 1 \
  --threads 4 \
  --timeout 120 \
  --bind "0.0.0.0:${PORT}" \
  app:app
