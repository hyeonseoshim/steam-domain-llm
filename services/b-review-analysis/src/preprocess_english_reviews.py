from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 파일 경로 설정
# ============================================================

INPUT_PATH = Path("reviews.csv")

OUTPUT_PATH = Path("english_reviews_clean.csv")
SAMPLE_PATH = Path("english_reviews_clean_sample.csv")
SUMMARY_PATH = Path("english_reviews_preprocessing_summary.txt")
LENGTH_SUMMARY_PATH = Path("english_review_length_summary.csv")


# ============================================================
# 2. 사용할 컬럼
# 작성자 Steam ID 등 개인정보성 컬럼은 불러오지 않음
# ============================================================

USE_COLUMNS = [
    "recommendationid",
    "appid",
    "language",
    "review_text",
    "voted_up",
    "votes_up",
    "weighted_vote_score",
    "author_playtime_at_review",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
]


# ============================================================
# 3. 텍스트 정제용 정규표현식
# ============================================================

# [url=https://example.com]표시 문구[/url]
# 주소는 제거하고 표시 문구는 남김
BBCODE_NAMED_URL_PATTERN = re.compile(
    r"\[url=[^\]]+\](.*?)\[/url\]",
    flags=re.IGNORECASE | re.DOTALL,
)

# [url]https://example.com[/url]
# 주소만 들어 있는 링크는 [URL]로 치환
BBCODE_PLAIN_URL_PATTERN = re.compile(
    r"\[url\].*?\[/url\]",
    flags=re.IGNORECASE | re.DOTALL,
)

# [img]이미지 주소[/img]
BBCODE_IMAGE_PATTERN = re.compile(
    r"\[img\].*?\[/img\]",
    flags=re.IGNORECASE | re.DOTALL,
)

# [img=주소] 형태
BBCODE_NAMED_IMAGE_PATTERN = re.compile(
    r"\[img=[^\]]+\].*?\[/img\]",
    flags=re.IGNORECASE | re.DOTALL,
)

# 일반 인터넷 주소
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\[\]\"']+",
    flags=re.IGNORECASE,
)

# 나머지 Steam BBCode 태그
BBCODE_PATTERN = re.compile(
    r"""
    \[
        /?
        (?:
            h[1-6]
            |b
            |i
            |u
            |s
            |strike
            |spoiler
            |quote
            |code
            |list
            |olist
            |table
            |tr
            |th
            |td
            |center
            |left
            |right
            |noparse
            |\*
        )
        (?:=[^\]]*)?
    \]
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# 일반 HTML 태그
HTML_TAG_PATTERN = re.compile(
    r"<[^>]+>",
    flags=re.IGNORECASE,
)

# 보이지 않는 제어문자
CONTROL_CHARACTER_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# 줄바꿈과 연속 공백
WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================
# 4. 정제 함수
# ============================================================

def clean_review_text(value: object) -> str:
    """
    Steam 리뷰 본문을 정제한다.

    처리 내용:
    - HTML 특수문자 변환
    - BBCode 링크의 표시 문구 보존
    - 실제 URL을 [URL]로 치환
    - 이미지 태그 제거
    - HTML 및 BBCode 태그 제거
    - 줄바꿈과 연속 공백 정리
    """

    if pd.isna(value):
        return ""

    text = str(value)

    # &amp; → &, &quot; → " 등으로 변환
    text = html.unescape(text)

    # [url=주소]Diner Dash[/url] → Diner Dash
    text = BBCODE_NAMED_URL_PATTERN.sub(r" \1 ", text)

    # [url]주소[/url] → [URL]
    text = BBCODE_PLAIN_URL_PATTERN.sub(" [URL] ", text)

    # 이미지 태그는 삭제
    text = BBCODE_IMAGE_PATTERN.sub(" ", text)
    text = BBCODE_NAMED_IMAGE_PATTERN.sub(" ", text)

    # 일반 URL → [URL]
    text = URL_PATTERN.sub(" [URL] ", text)

    # 나머지 BBCode 태그 제거
    text = BBCODE_PATTERN.sub(" ", text)

    # HTML 태그 제거
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 제어문자 제거
    text = CONTROL_CHARACTER_PATTERN.sub(" ", text)

    # 줄바꿈 및 연속 공백을 하나의 공백으로 변경
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def normalize_boolean(series: pd.Series) -> pd.Series:
    """문자열 또는 숫자 형태의 값을 불리언으로 통일한다."""

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )


def main() -> None:
    # ========================================================
    # 5. 입력 파일 확인
    # ========================================================

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"reviews.csv 파일을 찾을 수 없습니다.\n"
            f"확인한 위치: {INPUT_PATH.resolve()}"
        )

    print("=" * 70)
    print("Steam 영어 리뷰 1차 전처리를 시작합니다.")
    print("=" * 70)
    print(f"입력 파일: {INPUT_PATH.resolve()}")
    print()

    # ========================================================
    # 6. CSV 불러오기
    # ========================================================

    df = pd.read_csv(
        INPUT_PATH,
        usecols=USE_COLUMNS,
        low_memory=False,
        encoding="utf-8",
    )

    original_count = len(df)

    print(f"원본 전체 리뷰 수: {original_count:,}건")

    # ========================================================
    # 7. 영어 리뷰만 선택
    # ========================================================

    df["language"] = (
        df["language"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df = df[
        df["language"].eq("english")
    ].copy()

    english_count = len(df)

    print(f"영어 리뷰 수: {english_count:,}건")

    # ========================================================
    # 8. 원문 보존 및 정제문 생성
    # ========================================================

    df = df.rename(
        columns={
            "review_text": "review_text_original"
        }
    )

    df["review_text_clean"] = (
        df["review_text_original"]
        .apply(clean_review_text)
    )

    # ========================================================
    # 9. 빈 리뷰 제거
    # ========================================================

    empty_review_count = (
        df["review_text_clean"]
        .eq("")
        .sum()
    )

    df = df[
        df["review_text_clean"].ne("")
    ].copy()

    print(f"빈 리뷰 제거 수: {empty_review_count:,}건")

    # ========================================================
    # 10. 추천 여부 값 정리
    # ========================================================

    df["voted_up"] = normalize_boolean(
        df["voted_up"]
    )

    invalid_label_count = (
        df["voted_up"]
        .isna()
        .sum()
    )

    df = df.dropna(
        subset=["voted_up"]
    ).copy()

    df["voted_up"] = (
        df["voted_up"]
        .astype(bool)
    )

    df["label"] = df["voted_up"].map(
        {
            True: "recommended",
            False: "not_recommended",
        }
    )

    print(
        f"잘못된 추천 라벨 제거 수: "
        f"{invalid_label_count:,}건"
    )

    # ========================================================
    # 11. 중복 리뷰 ID 제거
    # ========================================================

    duplicate_id_count = (
        df["recommendationid"]
        .duplicated()
        .sum()
    )

    df = df.drop_duplicates(
        subset=["recommendationid"],
        keep="first",
    ).copy()

    print(
        f"중복 recommendationid 제거 수: "
        f"{duplicate_id_count:,}건"
    )

    # 같은 리뷰 문장이더라도 서로 다른 사용자가 작성했을 수 있으므로
    # review_text 내용 중복은 현재 단계에서 제거하지 않음

    # ========================================================
    # 12. 리뷰 길이 정보 추가
    # ========================================================

    df["char_count"] = (
        df["review_text_clean"]
        .str.len()
    )

    df["word_count"] = (
        df["review_text_clean"]
        .str.split()
        .str.len()
    )

    # ========================================================
    # 13. 출력 컬럼 순서 정리
    # ========================================================

    output_columns = [
        "recommendationid",
        "appid",
        "language",
        "review_text_original",
        "review_text_clean",
        "voted_up",
        "label",
        "char_count",
        "word_count",
        "votes_up",
        "weighted_vote_score",
        "author_playtime_at_review",
        "steam_purchase",
        "received_for_free",
        "written_during_early_access",
    ]

    df = df[output_columns]

    # ========================================================
    # 14. 전체 전처리 데이터 저장
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # Data Wrangler에서 확인하기 위한 1,000건 표본
    sample_size = min(1_000, len(df))

    sample_df = df.sample(
        n=sample_size,
        random_state=42,
    )

    sample_df.to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 15. 길이 통계 저장
    # ========================================================

    length_summary = (
        df[["char_count", "word_count"]]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    length_summary.to_csv(
        LENGTH_SUMMARY_PATH,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 16. 전처리 통계 계산
    # ========================================================

    final_count = len(df)

    recommended_count = (
        df["voted_up"]
        .eq(True)
        .sum()
    )

    not_recommended_count = (
        df["voted_up"]
        .eq(False)
        .sum()
    )

    unique_game_count = (
        df["appid"]
        .nunique()
    )

    url_token_count = (
        df["review_text_clean"]
        .str.contains(
            r"\[URL\]",
            regex=True,
            na=False,
        )
        .sum()
    )

    # ========================================================
    # 17. 전처리 결과 보고서 저장
    # ========================================================

    summary = f"""
Steam 영어 리뷰 1차 전처리 결과
==================================================

원본 전체 리뷰 수: {original_count:,}건
영어 리뷰 수: {english_count:,}건

빈 리뷰 제거 수: {empty_review_count:,}건
잘못된 추천 라벨 제거 수: {invalid_label_count:,}건
중복 recommendationid 제거 수: {duplicate_id_count:,}건

최종 리뷰 수: {final_count:,}건
고유 게임 수: {unique_game_count:,}개

추천 리뷰 수: {recommended_count:,}건
추천 리뷰 비율: {recommended_count / final_count * 100:.2f}%

비추천 리뷰 수: {not_recommended_count:,}건
비추천 리뷰 비율: {not_recommended_count / final_count * 100:.2f}%

[URL] 토큰이 포함된 리뷰 수: {url_token_count:,}건

생성 파일
- {OUTPUT_PATH.name}
- {SAMPLE_PATH.name}
- {SUMMARY_PATH.name}
- {LENGTH_SUMMARY_PATH.name}

리뷰 길이 통계
--------------------------------------------------
{length_summary.to_string()}
""".strip()

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print("=" * 70)
    print("전처리가 완료되었습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()