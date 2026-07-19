<h1 align="center">🎮 Steam Domain-Specific LLM</h1>

<h3 align="center">파인튜닝 소형 모델 기반 Steam 게임 검색·추천·분석 서비스</h3>

<p align="center">
  긴 게임 설명, 흩어진 리뷰, 근거가 부족한 추천, 복잡한 검색 조건을<br>
  네 개의 독립적인 도메인 특화 LLM 파이프라인으로 해결합니다.
</p>

<p align="center">
  <a href="https://steam-domain-specific-llm.vercel.app/"><strong>라이브 데모</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Inference-vLLM-7C3AED?style=flat-square" alt="vLLM" />
  <img src="https://img.shields.io/badge/GPU-Modal_L4-111111?style=flat-square" alt="Modal L4" />
  <img src="https://img.shields.io/badge/Frontend-Vercel-000000?style=flat-square&logo=vercel&logoColor=white" alt="Vercel" />
</p>

## 1. 프로젝트 소개

Steam에는 약 14만 개 이상의 게임이 있지만, 사용자가 자신에게 맞는 게임을 찾기까지는 여전히 많은 정보 탐색이 필요합니다.

| 문제 | 사용자 불편 | 프로젝트의 접근 |
|---|---|---|
| 길고 마케팅성 표현이 많은 설명 | 실제 플레이가 무엇인지 빠르게 파악하기 어려움 | 핵심 정보를 고정 스키마로 요약 |
| 수많은 비정형 리뷰 | 장점과 불만의 공통 주제를 직접 읽어야 함 | 감성과 장단점 토픽을 구조화 |
| 추천 근거와 취향 맥락 부족 | 왜 지금 이 게임이 나에게 맞는지 알기 어려움 | 시간에 따라 변하는 성향과 추천 이유 제공 |
| 정형 필터 중심 검색 | “무료 멀티 공포 게임” 같은 요청을 바로 표현하기 어려움 | 자연어를 검색 조건으로 변환해 필터링·재정렬 |

이 프로젝트는 하나의 범용 모델에 모든 기능을 맡기지 않습니다. 같은 Steam 데이터 위에서 팀원 네 명이 각자 하나의 문제를 맡아 **데이터 구축 → 파인튜닝 → 평가 → API 서빙**까지 완주하고, 결과를 하나의 사용자 화면에서 연결합니다.

> 핵심 산출물은 단일 거대 모델이 아니라, 서로 독립적으로 학습·평가·배포할 수 있는 네 개의 도메인 특화 미니 파이프라인입니다.

## 2. 네 개의 도메인 태스크

| 파트 | 담당 | 태스크 | 사용자에게 제공하는 가치 |
|---|---|---|---|
| **A · 설명 요약** | 이정수 | 긴 게임 설명을 `장르 / 핵심플레이 / 특징`으로 요약 | “이 게임에서 무엇을 하는지” 빠르게 파악 |
| **B · 리뷰 분석** | 심현서 | 리뷰의 감성과 16개 장단점 토픽을 분류·집계 | 사람들이 왜 좋아하고 불편해하는지 요약 |
| **C · 동적 추천** | 유재현 | 최근 경험에 따라 변하는 사용자 성향과 추천 이유 생성 | 현재 취향에 맞는 게임과 그 근거 제공 |
| **D · 조건 검색** | 성화섭 | 자연어에서 가격·장르·멀티플레이 등의 조건을 추출 | 복합 조건을 자연어 한 문장으로 검색 |

각 파트의 모델, 데이터 구성과 평가 지표는 태스크 특성에 맞게 독립적으로 선택합니다.

## 3. 사용자 흐름

```text
자연어 질의 입력
→ A 일반 검색 / C 동적 추천 / D 조건 검색 중 모드 선택
→ 최대 30개 게임 검색 및 재정렬
→ 게임 선택
→ A 설명 요약 + B 리뷰 분석 패널 확인
→ C가 추천한 게임은 C 성향 근거 패널도 함께 확인
```

- 검색 결과는 우선 10개를 보여주고 필요할 때 최대 30개까지 펼칩니다.
- D가 추출한 조건과 게임별 충족 근거는 검색 결과에서 바로 확인합니다.
- 패널은 파트별로 독립 로딩되어 한 서버가 느리거나 중단되어도 다른 결과를 먼저 볼 수 있습니다.
- 모든 데이터와 서비스는 Steam 원본 `appid`를 공통 연결 키로 사용합니다.

## 4. 전체 시스템 아키텍처

```mermaid
flowchart LR
    U[사용자] --> F[Vercel Frontend]
    F --> G[Modal CPU Gateway]
    G --> A[Part A<br/>Modal L4 · Search & Summary]
    G --> B[Part B API<br/>Review Panel]
    G --> C[Part C API<br/>Dynamic Search & Panel]
    G --> D[Part D API<br/>Constraint Search]

    A --> G
    B --> G
    C --> G
    D --> G
    G --> F
```

| 계층 | 역할 |
|---|---|
| **Frontend** | 검색 모드 선택, 결과 카드, 통합 게임 패널, 예열·fallback 상태 표시 |
| **CPU Gateway** | 파트별 요청 라우팅, `appid` 기반 결과 결합, 응답 형식 정규화 |
| **Part A/B/C/D API** | 각 담당자가 독립 배포한 모델 서버를 공통 JSON 계약으로 연결 |

게이트웨이는 모델 추론을 직접 수행하지 않는 얇은 오케스트레이션 계층입니다. 각 파트는 서로 다른 서버와 배포 주기를 유지하므로 한 파트의 장애가 전체 서비스 장애로 번지지 않도록 설계했습니다.

## 5. 데이터와 학습 원칙

공통 데이터는 **Steam Dataset 2025: Multi-Modal Gaming Analytics**를 사용합니다.

| 항목 | 내용 |
|---|---|
| 규모 | Steam 앱 239,664개, 이 중 게임 150,279개 · 검색 색인 약 14만 개 · 리뷰 1,048,148개 |
| 주요 데이터 | 게임 설명, 장르·카테고리·가격·플랫폼, 사용자 리뷰, 설명·리뷰 임베딩 |
| 공통 식별자 | Steam `appid` |
| 배포 형식 | PostgreSQL dump, CSV export, 임베딩 패키지, raw JSON sample |
| 라이선스 | CC BY 4.0 |

- 데이터 이해와 스키마: [Steam Dataset 2025 GitHub](https://github.com/vintagedon/steam-dataset-2025)
- 공식 데이터 배포: [Zenodo record 17266923](https://zenodo.org/records/17266923)

파트별 데이터는 중복·노이즈 제거, 형식 정규화, 개인정보 검토, Train/Validation/Test 분할을 거칩니다. 생성 태스크의 참조 답변과 분류·추출 라벨은 규칙 검사, 독립 모델 평가와 사람 검수를 조합해 품질을 관리합니다.

## 6. 학습과 평가 방식

네 파트는 공통적으로 베이스라인과 파인튜닝 모델을 같은 test split에서 비교합니다. 단, 서로 다른 태스크를 하나의 점수로 억지로 묶지 않습니다.

| 태스크 | 주요 평가 관점 |
|---|---|
| A 설명 요약 | ROUGE, BERTScore, 사실성, 3필드 형식 준수율 |
| B 리뷰 분석 | Micro/Macro F1, Exact Match, JSON 형식 준수율 |
| C 동적 추천 | 추천 적합성, 설명의 근거성, 사용자 상태 변화 반영 |
| D 조건 검색 | 조건 추출 정확도, 검색 적합성, 하드 조건 준수율 |

모든 파트가 공통으로 보고하는 것은 다음 세 가지입니다.

1. 베이스라인 대비 파인튜닝 성능 변화
2. 출력 형식·유효성 준수율과 오류 사례
3. 실제 서빙 환경의 응답시간과 자원 사용량

## 7. 검색·서빙 설계

Part A 검색은 약 14만 개 게임 색인에서 키워드와 의미를 함께 사용합니다.

```text
자연어 질의
→ BM25 키워드 검색
→ BGE-M3 의미 검색
→ Reciprocal Rank Fusion
→ BGE Reranker 재정렬
→ Top-K 게임 반환
```

게임 설명 요약은 Qwen2.5-3B 기반 LoRA 모델을 vLLM으로 서빙합니다. 현재 배포는 Modal L4 GPU에서 CUDA Graph와 FP8 온라인 양자화를 사용하며, 요청이 없을 때 GPU를 종료하는 scale-to-zero 방식입니다.

- 검색 요청: 컨테이너 콜드스타트를 고려해 최대 120초 대기
- 패널 요청: 요청당 25초, 예열 상태일 때 해당 패널만 자동 재시도
- 파싱 실패·서버 미제공·예열 중 상태는 화면에서 명시적으로 구분

이 시간은 목표 지연시간이 아니라 장애로 판단하기 전의 최대 허용시간입니다. 웜 상태에서는 가능한 한 빠른 응답을 목표로 합니다.

## 8. 기술 스택

| 영역 | 기술 |
|---|---|
| Data / Training | Python, JSONL, LoRA/QLoRA, Hugging Face Transformers, PEFT |
| Search | BM25, BGE-M3, BGE Reranker, RRF |
| Inference | Qwen2.5-3B, vLLM, CUDA Graph, FP8 |
| API / Gateway | FastAPI, Python standard HTTP client |
| Infrastructure | Modal CPU/GPU, scale-to-zero, persistent Volume |
| Frontend | HTML, CSS, Vanilla JavaScript, Vercel |

## 9. 문서 안내

- [통합 프론트](services/a-summary-search/frontend/index.html): A/B/C/D 통합 사용자 화면
- [CPU 게이트웨이](services/a-summary-search/backend/demo_app.py): 검색·패널 라우팅과 fallback
- [GPU 백엔드](services/a-summary-search/backend/gpu_backend.py): Part A 검색·실시간 요약 API

---

이 프로젝트는 완성된 추천 화면만을 목표로 하지 않습니다. 같은 도메인 데이터에서 서로 다른 태스크를 정의하고, 각자가 모델과 평가 기준을 선택해 실제 API로 배포한 과정 전체를 결과물로 봅니다.
