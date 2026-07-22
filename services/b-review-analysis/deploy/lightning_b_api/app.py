#!/usr/bin/env python3

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


APP_DIR = Path(__file__).resolve().parent
SERVICE_DIR = APP_DIR.parent

DEFAULT_DATA_PATH = (
    SERVICE_DIR
    / "data"
    / "predictions"
    / "game_topic_summary"
    / "game_topic_summary.json.gz"
)

DATA_PATH = Path(
    os.environ.get(
        "GAME_TOPIC_SUMMARY_PATH",
        str(DEFAULT_DATA_PATH),
    )
)

app = Flask(__name__)


def load_game_data(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Game topic summary not found: {path}"
        )

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Game topic summary must be a JSON object."
        )

    return data


GAME_DATA = load_game_data(DATA_PATH)

print(
    f"Loaded {len(GAME_DATA):,} games "
    f"from {DATA_PATH}"
)


def format_topics(topics: Any) -> str:
    if not isinstance(topics, list):
        return "없음"

    cleaned = [
        str(topic).strip()
        for topic in topics
        if str(topic).strip()
    ]

    if not cleaned:
        return "없음"

    return ", ".join(cleaned)


def unavailable_response(note: str = "데이터 없음"):
    return jsonify(
        {
            "status": "unavailable",
            "fields": [],
            "note": note,
        }
    ), 200


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, OPTIONS"
    )
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type"
    )
    return response


@app.route("/", methods=["GET", "OPTIONS"])
def root():
    if request.method == "OPTIONS":
        return "", 204

    return jsonify(
        {
            "status": "ok",
            "service": "steam-review-panel-api",
        }
    ), 200


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return "", 204

    return jsonify(
        {
            "status": "ok",
        }
    ), 200


@app.route("/panel", methods=["GET", "OPTIONS"])
def panel():
    if request.method == "OPTIONS":
        return "", 204

    raw_appid = request.args.get(
        "appid",
        default="",
        type=str,
    ).strip()

    if not raw_appid:
        return unavailable_response(
            "유효한 appid가 필요합니다."
        )

    try:
        appid = str(int(raw_appid))
    except ValueError:
        return unavailable_response(
            "유효한 appid가 필요합니다."
        )

    # q와 uid는 통합 인터페이스 호환을 위해 받지만
    # B 파트에서는 사용하지 않습니다.
    _query = request.args.get(
        "q",
        default="",
        type=str,
    )
    _uid = request.args.get(
        "uid",
        default="",
        type=str,
    )

    record = GAME_DATA.get(appid)

    if record is None:
        return unavailable_response()

    positive_topics = format_topics(
        record.get(
            "top_positive_topics",
            [],
        )
    )

    negative_topics = format_topics(
        record.get(
            "top_negative_topics",
            [],
        )
    )

    review_count = int(
        record.get(
            "review_count",
            0,
        )
    )

    return jsonify(
        {
            "status": "ok",
            "fields": [
                {
                    "k": "긍정 토픽",
                    "v": positive_topics,
                },
                {
                    "k": "부정 토픽",
                    "v": negative_topics,
                },
                {
                    "k": "표본 리뷰 수",
                    "v": f"{review_count:,}건",
                },
            ],
            "note": "",
        }
    ), 200


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    print(
        "Unexpected API error:",
        repr(error),
    )

    return jsonify(
        {
            "status": "error",
            "fields": [],
            "note": "서버 처리 오류",
        }
    ), 200


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            "8000",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
