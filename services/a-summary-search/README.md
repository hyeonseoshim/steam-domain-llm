# Part A — 게임 설명 요약 (Game Description Summarization)

긴 `detailed_description`(HTML·마케팅 범벅)을 **중립적·정보 위주의 짧은 요약**으로 변환하는
소형 LLM을 데이터 구축 → 파인튜닝 → 평가 → 서빙까지 직접 도는 미니 파이프라인.

담당: 이정수 · 데이터: Steam Dataset 2025 (CC BY 4.0, GitHub 5K raw 샘플)

---

## 진행 단계

| 단계 | 산출물 | 상태 |
|---|---|---|
| 1. 데이터 전처리 | `preprocess.py` → `data/processed/` | ✅ 완료 |
| 2. 참조 요약(정답) 생성 | Claude API 생성(`generate_references*.py`) | ✅ 완료 (4,165건) |
| 2b. 참조 요약 품질검사(QC) | `qc_references.py`(규칙) + `review_references.py`(LLM 심사) | ✅ 규칙·심사 완료 |
| 2c. gold 확정 | `build_gold.py` + `fix_early_access.py` → `gold.jsonl` | ✅ 완료 (gold 4,158) |
| 3. 베이스라인 평가 | Zero/Few-shot, ROUGE·BERTScore·형식준수율 | ⬜ |
| 4. QLoRA 파인튜닝 | 4bit, 손실·GPU 기록, 오류분석·2차 | ⬜ |
| 5. 평가·안전성·서빙 | FastAPI `/infer`+`/health`, 모델카드 | ⬜ |

---

## 1단계: 데이터 전처리 (완료)

```bash
python3 part_a/preprocess.py          # 기본 입력 games_5k.json.gz
```

**파이프라인:** load → `success & type=game` 필터 → HTML 정제 → 중복 제거
→ usable 필터(정제 후 ≥200자) → 정제 전후 통계 → train/val/test 분할(80/10/10, seed=42)

**정제 전후 통계 (5K 샘플 실측):**

| 단계 | 레코드 수 |
|---|---|
| 전체 raw 레코드 | 8,711 |
| `success=false` 제외 | −744 |
| `type≠game` 제외 (dlc/demo/music 등) | −2,967 |
| → 필터 후 (진짜 게임) | **5,000** |
| 정제 텍스트 완전중복 제외 | −130 |
| 정제 후 <200자 제외 | −705 |
| → **usable** | **4,165** |
| 분할 | train 3,332 / val 416 / test 417 |

**설명 길이 (문자 수):**

| | median | mean | p90 | p99 | max |
|---|---|---|---|---|---|
| 정제 전(raw HTML) | 1,668 | 2,238 | 4,730 | 8,645 | 30,125 |
| 정제 후(clean) | 877 | 1,099 | 2,089 | 4,332 | 13,030 |

→ 길이 편차가 매우 큼(200자~1.3만자). **파인튜닝 시 입력 절단/청킹 전략 필요** (오류분석 포인트).

### HTML 정제 (`clean_html`)
- 표준 라이브러리 `html.parser`만 사용 (외부 의존성 0 → 어디서든 재현).
- `img/video/script/style/iframe` 등은 **내용까지 제거**, 블록 태그(`<p><br><div>...`)는 줄바꿈, `<li>`는 `- ` 불릿으로.
- 엔티티(`&amp;` 등) 자동 디코딩, 공백/빈 줄 정규화.
- **품질검사:** train 3,332건 중 잔여 HTML 태그 0건 (유일한 1건은 작성자가 본문에 쓴 리터럴 꺾쇠 `<...>` — 오탐).

### 개인정보(PII)
- games 데이터에는 PII 없음(리뷰 쪽만 해당). A파트는 마스킹 대상 없음.

---

## 2단계: 참조 요약(정답) 생성 — 왜/어떻게 (방어 논리)

> "그냥 `short_description` 베낀 거 아냐?" 라는 질문에 대한 답을 문서로 남긴다.

- `short_description`은 **정답이 아니다.** 실측상 마케팅 티저·원문 첫 문장 복붙·초점 불일치가 섞여 품질이 제각각이며,
  A의 목적("마케팅 톤 제거·중립 정보 요약")과 **정반대 톤**인 경우가 많음 → 잘해야 약한 참고(silver).
- 따라서 정답 target = **고정 스키마 `{장르 / 핵심플레이 / 특징}`의 중립 요약**을 별도로 생성한다.
- 생성 방식: **Claude(Opus 4.8) + 구조화 출력(json_schema)** 으로 3필드 포맷을 강제.
  파일럿 40건(`generate_references.py`)으로 프롬프트 검증 → 전체 4,165건은 **Message Batches API**(`generate_references_batch.py`, 표준 대비 −50%)로 생성.
- 방어 포인트: 생성 프롬프트·고정 스키마·품질검사 기준·검수 표본을 모두 기록해 재현·설명 가능하게 한다.

## 2b단계: 품질검사(QC) & 사람 검수 (규칙검사 완료)

```bash
python3 part_a/qc_references.py       # all.jsonl → qc_report / flagged / review_sample
```

silver 4,165건에 **외부 의존성 0**(표준 라이브러리)의 규칙 기반 QC를 돌려, 프롬프트 제약 준수와
"복붙 아님"을 **정량화**한다. 규칙은 실측 오탐(프롬프트가 허용한 고유명사 보존, `2D/3D` 토큰화)에 맞춰 보정.

**QC 결과 (4,165건 실측):**

| 항목 | 값 |
|---|---|
| 하드룰 위반 (스키마 누락·장르 괄호·불릿/이모지·머리말·미번역) | **0건** |
| 소프트룰(검수 후보): 장르 영어병기 9 · 마케팅어 잔존 9 · 과장 길이 10 | 28건 (0.7%) |
| 무결(clean) | 4,137건 (99.3%) |
| **복붙 지표** 요약↔`short_description` char-3gram Jaccard | median **0.0** / p99 0.075 / **max 0.156** / ≥0.5 **0건** |

→ **정답이 `short_description` 복붙이 아님을 수치로 증명**(최대 겹침 0.16). 소프트 플래그는 자동 수정하지 않고
사람 검수로 넘긴다(예: "은하계 **최고의** 피자 가게"는 마케팅 톤이 아니라 **작품 내 설정** — 사전이 후보를 잡고 사람이 확정).

**규칙(하드=수정 대상 / 소프트=검수 우선):**
- H: 3필드 존재·비공백 / 장르 괄호(원어병기) 금지 / 불릿·이모지 금지 / 머리말("이 게임은") 금지 / 한글 지배율 ≥0.40(원문 언어 통째 복사 방지)
- S: 장르 영어병기(RPG·MMO·2D 등 정착 약어 제외) / 마케팅 과장어 사전 매칭 / 필드 과장·과소 길이 / `short_description` 겹침 ≥0.5

**사람 검수(형식용):** 길이 버킷(short/mid/long) **층화 랜덤 표본 60건**(`review_sample.jsonl`, seed=42)에 판정칸
(`장르_ok/핵심플레이_ok/특징_ok/환각_없음/코멘트`)을 두어 사람이 채운다.

## 2b단계(계속): 독립 LLM 심사 → 사람 확정 (심사 완료)

```bash
python3 part_a/review_references.py            # all.jsonl → reviews / review_todo / review_audit
python3 part_a/review_references.py --report-only   # 재채점 없이 티어 집계만
```

규칙 QC는 **형식**만 본다. 내용의 **충실성(환각·누락)**은 규칙으로 못 잡으므로, **생성기(Opus)와 다른 모델(Sonnet 4.6)** 이
**생성 프롬프트를 모른 채 원문↔요약만 대조**해 4축(환각/누락/톤/형식)을 0~2점 채점한다. 자기채점 순환을 끊기 위해
모델을 분리하고, **사람은 저점/불일치만 최종 확정**한다(3중 게이트: 규칙→LLM심사→사람).

**심사 결과 (4,165건, judge=Sonnet 4.6):**

| 축 | 0점 | 1점 | 2점(양호) | mean |
|---|---|---|---|---|
| 환각 | 9 | 915 | 3,241 | 1.78 |
| 누락 | 3 | 682 | 3,480 | 1.84 |
| 톤 | 0 | 5 | 4,160 | **1.99** |
| 형식 | 13 | 552 | 3,600 | 1.86 |

- **톤 mean 1.99** → 마케팅 톤 제거(A의 핵심 목표)가 거의 완벽히 달성됨을 독립 모델이 확인.
- **3티어 분류로 사람 작업량 현실화:** critical(어느 축이든 0점) **23건 전수** + minor 감사 랜덤표본 **60건** = **사람 실작업 83건**, 나머지 **2,850건(68.4%)은 auto-gold**.

**심사가 잡은 실제 결함 패턴(방어 자료):**
1. **비게임 본문 유입** — 원문이 "여름 세일 안내 페이지"·"계정 연동 아이템 안내"·빈 설명인데(전처리 ≥200자 필터 통과), 생성기가 Steam 장르 태그·제목만 보고 플레이/특징을 **지어냄**. → 해당 gold는 drop 대상이자 **전처리 개선 신호**.
2. **"얼리 액세스" 환각** — 원문에 없는 앞서 해보기 문구를 요약이 추가.
3. **심사관도 오류 있음**(예: `Lab Sorters`는 근거는 "문제없음"인데 점수는 0) → 그래서 **사람 확정이 최종 게이트**.

> ⚠️ 심사 v1은 형식 축을 오채점(쉼표로 나열한 복수 장르를 감점 → 3,230건 오탐)했다. 원인은 **심사 루브릭이 생성 스펙("여러 장르는 쉼표로 나열")과 어긋난 것**. 루브릭을 스펙에 맞춰 고쳐 재채점(`reviews_v1_miscalibrated.jsonl` 로 증거 보존). — 독립 심사의 함정과 그 교정 과정 자체가 방어 포인트.

## 2c단계: gold 확정 (완료)

```bash
python3 part_a/build_gold.py            # 검수 결정 반영 → gold.jsonl + gold_{split}.jsonl
python3 part_a/fix_early_access.py      # (선행) 얼리액세스 환각 타겟 재심사 → ea_fixes.jsonl
```

critical 23건을 사람이 확정하고(`build_gold.py` 에 결정 인라인 기록 — 파생데이터 gitignore 대비), 
전체에 적용해 gold 를 만든다.

**critical 23건 사람 확정:**
- **drop 7** — 원문이 게임 설명이 아님(여름세일 안내·계정연동 안내·빈 설명·후원요청문 등)인데 생성기가 Steam 태그·제목으로 플레이를 창작 → 복구 불가, 제외.
- **fix 6** — 진짜 게임인데 원문에 없는 '얼리 액세스' 절만 환각 → 해당 절 제거.
- **keep 10** — 심사관 오탐(장르가 Steam 태그 기반이라 원문엔 없음) 또는 심사관 채점 오류(`Lab Sorters`) → 그대로 채택.

**얼리액세스 환각 타겟 재심사(`fix_early_access.py`):** critical 검증 중 auto-gold 에도 같은 '얼리 액세스' 
상투 문구가 남은 걸 발견. 요약에 EA 문구가 있는 **95건**을 judge(Sonnet)로 "원문이 출시·개발 상태를 
지지하는가(로드맵·후원·에피소드예정 함의 포함)" 재심사 → **지지 64 / 미지지 31**. 미지지 31건만 
**출시상태 절만 제거하고 나머지는 한 글자도 안 바꾼 특징**을 모델이 재작성해 반영. (규칙 정규식으로 
일괄 삭제하면 근거 있는 64건과 마침표 없는 단문까지 훼손 → 그래서 judge 재작성 방식 채택.)

**gold 산출 (silver 4,165 → gold 4,158):**

| 분류 | 건수 |
|---|---|
| drop (게임설명 아님) | 7 |
| fix (critical 얼리액세스) | 6 |
| keep_reviewed (심사관 오탐/오류 확정) | 10 |
| ea_fix (얼리액세스 타겟 재심사) | 31 |
| auto (치명 결함 없음) | 4,111 |
| **gold 합계** | **4,158** |

split: train 3,327 / val 415 / test 416 (drop 7건 제외분 반영). gold 레코드는 `review` 필드로
검수 이력(auto/fix/keep_reviewed/ea_fix)을 남겨 추적 가능.

**minor 티어 감사(`review_audit.jsonl` 60건, `audit_result.json`):** 심사관 level-1 플래그 타당성 검증.
- **결론:** minor 티어는 무해한 "장르=Steam 태그" 오탐(45건, critical keep와 동일 사유)이 지배하고 심사관 판정이 일관됨(형식 v1 같은 체계적 오류 없음) → **minor 1,292건 auto-채택 정당화**. 8건은 심사관이 '양호'라 적고 형식 1을 준 무결 케이스.
- **잔여 결함:** 표본 내 실질 오류 3~4/60 ≈ **5~7%**(전부 level-1 경미: 장르 세부/사소한 서술; 예 STONEBOND 승리조건 반전, CoD 'lean' 음역). EA 언급 2건은 이미 `ea_fix` 처리됨. → gold의 **알려진 잔여 한계**로 모델카드에 기록.

---

## 디렉터리

```
part_a/
├── README.md
├── preprocess.py                 # 1단계 전처리 (완료)
├── generate_references.py        # 2단계 파일럿 생성 (완료)
├── generate_references_batch.py  # 2단계 전체 생성 Batch (완료)
├── qc_references.py              # 2b 규칙 품질검사·검수표본 (완료)
├── review_references.py          # 2b 독립 LLM 심사·티어 분류 (완료)
├── fix_early_access.py           # 2c 얼리액세스 환각 타겟 재심사 (완료)
├── build_gold.py                 # 2c silver→gold 확정·split 조인 (완료)
└── data/                         # (gitignore) 파생 산출물
    ├── processed/
    │   ├── clean.jsonl           # usable 전체 4,165
    │   ├── train/val/test.jsonl  # 3,332 / 416 / 417
    │   └── stats.json            # 정제 전후 통계
    └── references/
        ├── pilot.jsonl           # 파일럿 40
        ├── all.jsonl             # 정답 요약 4,165 (silver)
        ├── qc_report.json        # 규칙 QC 집계·복붙지표
        ├── flagged.jsonl         # 규칙 소프트룰 후보 28
        ├── review_sample.jsonl   # 규칙 QC 형식 검수표본 60
        ├── reviews.jsonl         # LLM 심사 전량 채점 4,165
        ├── review_report.json    # 심사 티어·축별 분포
        ├── review_todo.jsonl     # critical 전수검수 23
        ├── review_audit.jsonl    # minor 감사표본 60
        ├── audit_result.json     # 감사 결론·잔여결함 추정(5~7%)
        ├── reviews_v1_miscalibrated.jsonl  # 심사 v1(형식 오채점) 증거 보존
        ├── ea_fixes.jsonl        # 얼리액세스 미지지 31건 수정 특징
        ├── ea_fix_report.json    # EA 재심사 집계(지지 64/미지지 31)
        ├── gold.jsonl            # ★ 최종 gold 4,158 (input + summary + review)
        ├── gold_{train,val,test}.jsonl  # split별 gold 3,327/415/416
        └── gold_manifest.json    # gold 산출 내역·결정 기록
```

레코드 스키마 (JSONL 한 줄):
- `processed/`: `appid, name, is_free, detailed_description_raw, detailed_description_clean,
  short_description, genres[], categories[], release_date, len_raw, len_clean`
- `references/all.jsonl` (silver): `appid, name, genres[], len_clean, input_clean, short_description,
  summary{장르, 핵심플레이, 특징}, model, usage`
- `references/gold.jsonl` (★ 파인튜닝용): `appid, name, genres[], input(=정제 본문),
  summary{장르, 핵심플레이, 특징}, source_model, review(auto/fix/keep_reviewed/ea_fix)`
