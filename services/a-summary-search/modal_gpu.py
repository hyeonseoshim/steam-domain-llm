"""Modal 배포 — 검색 전용 GPU 백엔드. scale-to-zero L4.

CPU 게이트웨이(modal_app.py)가 검색을 이 앱으로 원격 호출한다. A 실시간 요약은
modal_gpu_summary.py가 별도 L4에서 맡아 두 콜드스타트를 병렬화한다.

배포(서비스 폴더에서):  modal deploy modal_gpu.py
데이터 볼륨은 modal_app 과 공유(steam-part-a-data): DEPLOY.md 참조.
"""

from __future__ import annotations

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).parent          # services/a-summary-search/
HF_CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM 0.17.1 + 0.5B draft K=1: L4 warm batch=1 실측 p50 2.51초.
    # 검색(sentence-transformers)은 vllm 이 고정한 torch 위에서 공존한다.
    .pip_install(
        "vllm==0.17.1",
        "sentence-transformers", "rank-bm25", "kiwipiepy",
        "fastapi", "uvicorn",
    )
    .env({
        "ALLOW_ORIGINS": os.environ.get("ALLOW_ORIGINS", "*"),
        "VLLM_GPU_UTIL": os.environ.get("VLLM_GPU_UTIL", "0.5"),   # vLLM GPU 비율(나머지=검색 모델)
        "SUMMARIZER": "vllm",                                     # gpu_backend: free-VRAM 백프레셔 끔
        # 실시간 요약은 별도 steam-part-a-summary L4에서 병렬 기동한다. 이 앱은 검색 전용.
        "ENABLE_SUMMARY": "0",
        "MERGED_MODEL_DIR": "/data/merged-qwen2.5-3b",           # 병합모델(LoRA 오버헤드 제거)=볼륨
        "VLLM_QUANT": os.environ.get("VLLM_QUANT", "fp8"),       # 온라인 FP8(서빙단계, 저장물 없음). ""=끔
        # 빈 문자열로 재배포하면 즉시 비활성화 가능. K>1은 L4 실측에서 오히려 느렸음.
        "VLLM_SPEC_MODEL": os.environ.get(
            "VLLM_SPEC_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        "VLLM_SPEC_TOKENS": os.environ.get("VLLM_SPEC_TOKENS", "1"),
    })
    # 코드만(평면). 대용량 인덱스·gold 는 볼륨(_link_data 로 심링크). 프론트는 이 앱엔 불필요.
    .add_local_dir(str(REPO / "backend"), remote_path="/root/backend",
                   ignore=["*.npy", "*.pkl", "corpus_*", "*.log", "*.csv"])
    .add_local_file(str(REPO / "eval" / "baseline_predict.py"), "/root/eval/baseline_predict.py")
    .add_local_dir(str(REPO / "train" / "qwen2.5-3b-lora"), remote_path="/root/train/qwen2.5-3b-lora",
                   ignore=["checkpoint-*"])
)

data_vol = modal.Volume.from_name("steam-part-a-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("steam-hf-cache", create_if_missing=True)

app = modal.App("steam-part-a-gpu")


def _link_data() -> None:
    src = pathlib.Path("/data")
    backend = pathlib.Path("/root/backend")
    refs = pathlib.Path("/root/data/references")
    if not src.exists():
        return
    refs.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        dst = (refs / p.name) if p.name == "gold.jsonl" else (backend / p.name)
        if not dst.exists():
            dst.symlink_to(p)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,                # ⚠️ 기본 CPU 적음 → HF/vLLM-eager의 CPU-bound(토큰 루프·디스패치)가 병목.
    #                       Lightning 풀VM(5초) vs Modal 기본(9~24초) 차이의 유력 원인 → 코어 확보.
    volumes={"/data": data_vol, HF_CACHE: hf_vol},
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
    from gpu_backend import app as fastapi_app
    return fastapi_app
