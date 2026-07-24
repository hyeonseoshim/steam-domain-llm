"""Modal 배포 — CPU 게이트웨이(오케스트레이션 + 팀원 파트). GPU 없음, 콜드스타트 초 단위.

검색은 modal_gpu.py, A 요약은 modal_gpu_summary.py의 별도 GPU 앱으로 원격/프록시한다.
이 게이트웨이는 GPU가 자도 팀원 파트(B/C/D)·페이지를 계속 서빙 → 가용성 분리 + 유휴 과금 0.

  검색 GPU: modal deploy modal_gpu.py          → steam-part-a-gpu
  요약 GPU: modal deploy modal_gpu_summary.py  → steam-part-a-summary
  게이트웨이: modal deploy modal_app.py  → steam-part-a (이 파일, CPU)

env(게이트웨이): GPU_BASE_URL · GPU_SUMMARY_URL · PART_{B,C,D}_URL · ALLOW_ORIGINS
※ 프론트(demo_index.html)는 최종적으로 Vercel 정적 호스팅. 지금은 이 앱도 / 로 서빙(테스트).
"""

from __future__ import annotations

import hashlib
import os
import pathlib

import modal

REPO = pathlib.Path(__file__).parent          # services/a-summary-search/
FRONTEND_VERSION = hashlib.sha256(
    (REPO / "frontend" / "index.html").read_bytes()).hexdigest()[:12]
GPU_APP_URL = os.environ.get(
    "GPU_BASE_URL", "https://jslblar080--steam-part-a-gpu-web.modal.run")
GPU_SUMMARY_URL = os.environ.get(
    "GPU_SUMMARY_URL", "https://jslblar080--steam-part-a-summary-web.modal.run")
PART_B_URL = os.environ.get(
    "PART_B_URL", "https://8000-01kxyzf0q91ejvrg04tcrxdzrn.cloudspaces.litng.ai")
PART_C_URL = os.environ.get(
    "PART_C_URL", "https://hunger-treble-banking.ngrok-free.dev")
PART_D_URL = os.environ.get(
    "PART_D_URL", "https://penalty-newspapers-scholarships-clothes.trycloudflare.com")

# 가벼운 이미지 — demo_gateway 는 stdlib, 게이트웨이는 GPU/torch 불필요 → 콜드스타트 빠름.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "uvicorn")
    .env({
        "ALLOW_ORIGINS": os.environ.get("ALLOW_ORIGINS", "*"),
        # 프론트 변경마다 이미지 ID와 /health 버전을 바꿔 운영 반영 여부를 검증할 수 있게 한다.
        "FRONTEND_VERSION": FRONTEND_VERSION,
        "GPU_BASE_URL": GPU_APP_URL,
        "GPU_SUMMARY_URL": GPU_SUMMARY_URL,
        # B는 공개 Lightning 서버 연동 완료. 외부 env를 주면 재배포 없이 설정 출처만 교체 가능.
        "PART_B_URL": PART_B_URL,
        # C·D는 각 담당자의 공개 서버에 연결한다.
        "PART_C_URL": PART_C_URL,
        "PART_D_URL": PART_D_URL,
    })
    # 코드(demo_app·demo_gateway)만 + B 샘플 + 프론트. 대용량 데이터·모델 없음.
    .add_local_dir(str(REPO / "backend"), remote_path="/root/backend",
                   ignore=["*.npy", "*.pkl", "corpus_*", "*.log", "*.csv"])
    .add_local_dir(str(REPO / "frontend"), remote_path="/root/frontend")
)

app = modal.App("steam-part-a")


@app.function(
    image=image,
    scaledown_window=300,
    timeout=180,           # 검색 upstream 120초 + 게이트웨이 처리 여유
    min_containers=0,     # 유휴 0 (콜드스타트 초 단위라 부담 적음)
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root/backend")
    from demo_app import app as fastapi_app     # import 시 게이트웨이 프로바이더 구성(GPU 무관)
    return fastapi_app
