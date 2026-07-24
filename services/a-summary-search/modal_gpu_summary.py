"""Modal 배포 — A 실시간 요약 전용 L4. 검색 GPU와 독립적으로 scale-to-zero."""

from __future__ import annotations

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).parent
HF_CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.17.1", "fastapi", "uvicorn")
    .env({
        # 운영/벤치와 같은 0.5를 유지해 검증된 AOT 캐시 키와 생성 조건을 그대로 재사용한다.
        "VLLM_GPU_UTIL": os.environ.get("VLLM_GPU_UTIL", "0.5"),
        "SUMMARIZER": "vllm",
        "MERGED_MODEL_DIR": "/data/merged-qwen2.5-3b",
        "VLLM_QUANT": os.environ.get("VLLM_QUANT", "fp8"),
        "VLLM_SPEC_MODEL": os.environ.get(
            "VLLM_SPEC_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        ),
        "VLLM_SPEC_TOKENS": os.environ.get("VLLM_SPEC_TOKENS", "1"),
    })
    .add_local_dir(
        str(REPO / "backend"),
        remote_path="/root/backend",
        ignore=["*.npy", "*.pkl", "corpus_*", "*.log", "*.csv"],
    )
    .add_local_file(
        str(REPO / "eval" / "baseline_predict.py"),
        "/root/eval/baseline_predict.py",
    )
)

data_vol = modal.Volume.from_name("steam-part-a-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("steam-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("steam-vllm-cache", create_if_missing=True)
app = modal.App("steam-part-a-summary")


def _link_data() -> None:
    source = pathlib.Path("/data")
    backend = pathlib.Path("/root/backend")
    refs = pathlib.Path("/root/data/references")
    refs.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        destination = refs / path.name if path.name == "gold.jsonl" else backend / path.name
        if not destination.exists():
            destination.symlink_to(path)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    volumes={
        "/data": data_vol,
        HF_CACHE: hf_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    scaledown_window=300,
    timeout=600,
    min_containers=0,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import sys

    _link_data()
    sys.path.insert(0, "/root/backend")
    from summary_backend import app as fastapi_app

    return fastapi_app
