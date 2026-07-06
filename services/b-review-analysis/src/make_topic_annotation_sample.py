from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# 파일 경로
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "english_reviews_clean.csv"
)

OUTPUT_SAMPLE_PATH = (
    PROJECT_DIR
    / "data"
    / "annotations"
    / "topic_annotation_sample.csv"
)

OUTPUT_SCHEMA_PATH = PROJECT_DIR / "config" / "topic_schema.json"
OUTPUT_TOPIC_GUIDE_PATH = PROJECT_DIR / "config" / "topic_guide.csv"
OUTPUT_SUMMARY_PATH = PROJECT_DIR / "reports" / "topic_sample_summary.txt"


# ============================================================
# 표본 추출 설정
# ============================================================

RANDOM_SEED = 42

# 1차 골든셋: 추천 150건 + 비추천 150건
SAMPLES_PER_LABEL = 150

# 동일한 게임 리뷰가 지나치게 많이 포함되지 않도록 제한
MAX_REVIEWS_PER_APP = 5

# 토픽을 판단하기 어려운 극단적으로 짧거나 긴 리뷰 제외
MIN_WORD_COUNT = 5
MAX_WORD_COUNT = 500


# ============================================================
# 토픽 정의
# ============================================================

TOPIC_SCHEMA = {
    "gameplay": {
        "name_ko": "게임성",
        "description": "전투, 이동, 퍼즐, 핵심 플레이 방식과 전반적인 재미",
    },
    "story": {
        "name_ko": "스토리",
        "description": "줄거리, 서사, 캐릭터, 대사, 결말",
    },
    "graphics": {
        "name_ko": "그래픽",
        "description": "아트 스타일, 모델링, 애니메이션, 시각 효과",
    },
    "audio": {
        "name_ko": "오디오",
        "description": "음악, 효과음, 음성 연기, 사운드트랙",
    },
    "controls": {
        "name_ko": "조작감",
        "description": "키보드, 마우스, 게임패드 조작과 반응성",
    },
    "ui_ux": {
        "name_ko": "UI·UX",
        "description": "메뉴, HUD, 인벤토리, 화면 구성과 사용 편의성",
    },
    "performance": {
        "name_ko": "성능·최적화",
        "description": "프레임 저하, 렉, 끊김, 로딩, 최적화",
    },
    "bugs": {
        "name_ko": "버그",
        "description": "충돌, 튕김, 진행 불가, 저장 오류 등 소프트웨어 문제",
    },
    "difficulty_balance": {
        "name_ko": "난이도·밸런스",
        "description": "난이도, 전투 균형, 캐릭터나 무기의 밸런스",
    },
    "content": {
        "name_ko": "콘텐츠",
        "description": "맵, 퀘스트, 모드, 콘텐츠의 양과 다양성",
    },
    "replayability": {
        "name_ko": "반복 플레이",
        "description": "다회차 가치, 반복 플레이 동기, 랜덤성",
    },
    "value": {
        "name_ko": "가격·가성비",
        "description": "가격 대비 만족도, 구매 가치, 할인",
    },
    "multiplayer": {
        "name_ko": "멀티플레이",
        "description": "협동, 경쟁, 친구와의 플레이 경험",
    },
    "network_server": {
        "name_ko": "네트워크·서버",
        "description": "핑, 연결 끊김, 서버 상태, 매칭 문제",
    },
    "updates_support": {
        "name_ko": "업데이트·지원",
        "description": "패치, 업데이트, 개발자 소통과 사후 지원",
    },
    "monetization": {
        "name_ko": "수익화",
        "description": "DLC, 소액결제, 시즌패스, 페이투윈",
    },
    "accessibility": {
        "name_ko": "접근성",
        "description": "자막, 색약 모드, 글자 크기, 난이도 옵션",
    },
    "localization": {
        "name_ko": "현지화",
        "description": "번역 품질, 언어 지원, 오역",
    },
    "other": {
        "name_ko": "기타",
        "description": "다른 토픽에 명확히 포함되지 않는 내용",
    },
}


def sample_per_label(
    dataframe: pd.DataFrame,
    label: str,
    sample_size: int,
) -> pd.DataFrame:
    """
    하나의 라벨에서 표본을 추출한다.

    같은 appid의 리뷰는 최대 MAX_REVIEWS_PER_APP건까지만 허용한다.
    """

    subset = dataframe[
        dataframe["label"].eq(label)
    ].copy()

    # 먼저 행을 무작위로 섞은 뒤 게임별 최대 5건만 유지
    subset = subset.sample(
        frac=1,
        random_state=RANDOM_SEED,
    )

    subset = (
        subset.groupby(
            "appid",
            group_keys=False,
        )
        .head(MAX_REVIEWS_PER_APP)
    )

    if len(subset) < sample_size:
        raise ValueError(
            f"{label} 표본이 부족합니다. "
            f"필요: {sample_size:,}건, 사용 가능: {len(subset):,}건"
        )

    return subset.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"전처리 파일을 찾을 수 없습니다: {INPUT_PATH.resolve()}"
        )

    print("=" * 70)
    print("토픽 라벨링용 리뷰 표본을 생성합니다.")
    print("=" * 70)

    use_columns = [
        "recommendationid",
        "appid",
        "review_text_clean",
        "voted_up",
        "label",
        "char_count",
        "word_count",
        "votes_up",
        "weighted_vote_score",
    ]

    df = pd.read_csv(
        INPUT_PATH,
        usecols=use_columns,
        low_memory=False,
    )

    original_count = len(df)

    print(f"전처리 영어 리뷰 수: {original_count:,}건")

    # --------------------------------------------------------
    # 1. 토픽 분석에 적합한 길이만 남김
    # --------------------------------------------------------

    df = df[
        df["word_count"].between(
            MIN_WORD_COUNT,
            MAX_WORD_COUNT,
            inclusive="both",
        )
    ].copy()

    length_filtered_count = len(df)

    print(
        f"{MIN_WORD_COUNT}~{MAX_WORD_COUNT}단어 리뷰 수: "
        f"{length_filtered_count:,}건"
    )

    # --------------------------------------------------------
    # 2. 완전히 같은 리뷰 문장 제거
    # 표본 라벨링에서 같은 문장을 여러 번 보는 것을 방지
    # --------------------------------------------------------

    duplicate_text_count = (
        df["review_text_clean"]
        .duplicated()
        .sum()
    )

    df = df.drop_duplicates(
        subset=["review_text_clean"],
        keep="first",
    ).copy()

    print(
        f"동일한 리뷰 문장 제외 수: "
        f"{duplicate_text_count:,}건"
    )

    # --------------------------------------------------------
    # 3. 추천·비추천 각각 표본 추출
    # --------------------------------------------------------

    recommended_sample = sample_per_label(
        dataframe=df,
        label="recommended",
        sample_size=SAMPLES_PER_LABEL,
    )

    not_recommended_sample = sample_per_label(
        dataframe=df,
        label="not_recommended",
        sample_size=SAMPLES_PER_LABEL,
    )

    sample_df = pd.concat(
        [
            recommended_sample,
            not_recommended_sample,
        ],
        ignore_index=True,
    )

    # 최종 순서를 다시 무작위로 섞음
    sample_df = sample_df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. 라벨링용 빈 컬럼 추가
    # --------------------------------------------------------

    sample_df.insert(
        0,
        "annotation_id",
        range(1, len(sample_df) + 1),
    )

    sample_df["positive_topics"] = ""
    sample_df["negative_topics"] = ""
    sample_df["is_valid"] = ""
    sample_df["invalid_reason"] = ""
    sample_df["annotation_note"] = ""
    sample_df["annotation_status"] = "pending"

    # 사람이 직접 라벨링할 때는 쉼표로 구분
    # 예: gameplay,story
    # 예: bugs,performance

    output_columns = [
        "annotation_id",
        "recommendationid",
        "appid",
        "review_text_clean",
        "voted_up",
        "label",
        "positive_topics",
        "negative_topics",
        "is_valid",
        "invalid_reason",
        "annotation_note",
        "annotation_status",
        "word_count",
        "char_count",
        "votes_up",
        "weighted_vote_score",
    ]

    sample_df = sample_df[output_columns]

    sample_df.to_csv(
        OUTPUT_SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 5. 토픽 스키마 저장
    # --------------------------------------------------------

    OUTPUT_SCHEMA_PATH.write_text(
        json.dumps(
            TOPIC_SCHEMA,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    topic_guide_df = pd.DataFrame(
        [
            {
                "topic_code": code,
                "topic_name_ko": information["name_ko"],
                "description": information["description"],
            }
            for code, information in TOPIC_SCHEMA.items()
        ]
    )

    topic_guide_df.to_csv(
        OUTPUT_TOPIC_GUIDE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6. 결과 요약
    # --------------------------------------------------------

    recommended_count = (
        sample_df["label"]
        .eq("recommended")
        .sum()
    )

    not_recommended_count = (
        sample_df["label"]
        .eq("not_recommended")
        .sum()
    )

    unique_app_count = sample_df["appid"].nunique()

    max_reviews_per_app = (
        sample_df.groupby("appid")
        .size()
        .max()
    )

    summary = f"""
토픽 라벨링용 표본 생성 결과
==================================================

입력 리뷰 수: {original_count:,}건
길이 조건 통과 리뷰 수: {length_filtered_count:,}건
동일 문장 제외 수: {duplicate_text_count:,}건

최종 표본 수: {len(sample_df):,}건
추천 리뷰 수: {recommended_count:,}건
비추천 리뷰 수: {not_recommended_count:,}건

포함된 고유 게임 수: {unique_app_count:,}개
게임당 최대 리뷰 수: {max_reviews_per_app:,}건

길이 기준
- 최소 단어 수: {MIN_WORD_COUNT}
- 최대 단어 수: {MAX_WORD_COUNT}

생성 파일
- {OUTPUT_SAMPLE_PATH.name}
- {OUTPUT_SCHEMA_PATH.name}
- {OUTPUT_TOPIC_GUIDE_PATH.name}
- {OUTPUT_SUMMARY_PATH.name}
""".strip()

    OUTPUT_SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)


if __name__ == "__main__":
    main()