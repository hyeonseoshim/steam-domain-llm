from pathlib import Path

import pandas as pd


# ============================================================
# 0. 파일 경로 설정
# ============================================================

# data_audit.py가 reviews.csv와 같은 폴더에 있으면:
CSV_PATH = Path("reviews.csv")

# 상위 프로젝트 폴더에서 실행한다면 위 코드를 주석 처리하고 아래처럼 변경:
# CSV_PATH = Path("steam_dataset_2025_csv/reviews.csv")


# 분석에 필요한 열만 불러와 메모리 사용량을 줄임
USE_COLUMNS = [
    "recommendationid",
    "appid",
    "language",
    "review_text",
    "voted_up",
]


def print_title(title: str) -> None:
    """출력 구역을 구분하기 위한 함수"""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # --------------------------------------------------------
    # 파일 존재 여부 확인
    # --------------------------------------------------------
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {CSV_PATH.resolve()}\n"
            "CSV_PATH를 실제 reviews.csv 위치에 맞게 수정하세요."
        )

    print("CSV 파일을 불러오는 중입니다.")
    print(f"파일 위치: {CSV_PATH.resolve()}")

    # --------------------------------------------------------
    # 필요한 열만 불러오기
    # --------------------------------------------------------
    df = pd.read_csv(
        CSV_PATH,
        usecols=USE_COLUMNS,
        low_memory=False,
    )

    total_rows = len(df)

    print_title("기본 데이터 정보")

    print(f"전체 리뷰 수: {total_rows:,}건")
    print(f"전체 컬럼 수: {len(df.columns)}개")
    print(f"고유 게임 수: {df['appid'].nunique(dropna=True):,}개")
    print(f"불러온 컬럼: {df.columns.tolist()}")

    # ========================================================
    # 1. review_text 빈값 확인
    # ========================================================

    print_title("1. review_text 빈값 확인")

    # 실제 NaN 결측치
    review_nan_count = df["review_text"].isna().sum()

    # NaN, 빈 문자열, 공백만 있는 문자열을 모두 빈 리뷰로 처리
    review_text_clean = (
        df["review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    review_empty_count = review_text_clean.eq("").sum()
    review_valid_count = total_rows - review_empty_count

    print(f"NaN 리뷰 수: {review_nan_count:,}건")
    print(f"NaN·빈 문자열·공백 리뷰 수: {review_empty_count:,}건")
    print(f"정상 리뷰 수: {review_valid_count:,}건")
    print(f"빈 리뷰 비율: {review_empty_count / total_rows * 100:.4f}%")

    # ========================================================
    # 2. language에서 영어 리뷰 비율 확인
    # ========================================================

    print_title("2. 언어 분포 및 영어 리뷰 비율")

    language_clean = (
        df["language"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    english_count = language_clean.eq("english").sum()
    language_missing_count = language_clean.isna().sum()

    print(f"영어 리뷰 수: {english_count:,}건")
    print(f"영어 리뷰 비율: {english_count / total_rows * 100:.2f}%")
    print(f"언어 결측치 수: {language_missing_count:,}건")

    language_table = (
        language_clean
        .fillna("missing")
        .value_counts()
        .rename_axis("language")
        .reset_index(name="count")
    )

    language_table["ratio_percent"] = (
        language_table["count"] / total_rows * 100
    ).round(4)

    print()
    print("언어별 상위 20개 분포:")
    print(language_table.head(20).to_string(index=False))

    language_table.to_csv(
        "language_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 3. voted_up의 True·False 분포 확인
    # ========================================================

    print_title("3. voted_up 추천·비추천 분포")

    # 값이 문자열로 읽힌 경우까지 대응
    voted_up_clean = (
        df["voted_up"]
        .astype("string")
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

    recommended_count = voted_up_clean.eq(True).sum()
    not_recommended_count = voted_up_clean.eq(False).sum()
    voted_missing_count = voted_up_clean.isna().sum()

    print(
        f"True · 추천: {recommended_count:,}건 "
        f"({recommended_count / total_rows * 100:.2f}%)"
    )
    print(
        f"False · 비추천: {not_recommended_count:,}건 "
        f"({not_recommended_count / total_rows * 100:.2f}%)"
    )
    print(
        f"결측 또는 변환 불가: {voted_missing_count:,}건 "
        f"({voted_missing_count / total_rows * 100:.4f}%)"
    )

    voted_table = pd.DataFrame(
        {
            "voted_up": ["True", "False", "Missing"],
            "count": [
                recommended_count,
                not_recommended_count,
                voted_missing_count,
            ],
        }
    )

    voted_table["ratio_percent"] = (
        voted_table["count"] / total_rows * 100
    ).round(4)

    voted_table.to_csv(
        "voted_up_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. appid별 리뷰 개수 확인
    # ========================================================

    print_title("4. appid별 리뷰 개수")

    reviews_per_app = (
        df.dropna(subset=["appid"])
        .groupby("appid")
        .size()
        .sort_values(ascending=False)
        .rename("review_count")
        .reset_index()
    )

    print(f"리뷰가 있는 게임 수: {len(reviews_per_app):,}개")

    print()
    print("게임당 리뷰 수 기초 통계:")
    print(
        reviews_per_app["review_count"]
        .describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        .to_string()
    )

    print()
    print("리뷰 수가 가장 많은 게임 상위 20개:")
    print(reviews_per_app.head(20).to_string(index=False))

    reviews_per_app.to_csv(
        "reviews_per_appid.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 5. 추가 데이터 품질 확인
    # ========================================================

    print_title("추가 데이터 품질 확인")

    duplicated_recommendationid = (
        df["recommendationid"]
        .duplicated()
        .sum()
    )

    duplicated_review_text = (
        review_text_clean[review_text_clean.ne("")]
        .duplicated()
        .sum()
    )

    print(
        f"중복 recommendationid 수: "
        f"{duplicated_recommendationid:,}건"
    )
    print(
        f"완전히 같은 review_text 중복 수: "
        f"{duplicated_review_text:,}건"
    )

    # ========================================================
    # 완료 메시지
    # ========================================================

    print_title("분석 완료")

    print("다음 결과 파일이 생성되었습니다.")
    print("- language_distribution.csv")
    print("- voted_up_distribution.csv")
    print("- reviews_per_appid.csv")


if __name__ == "__main__":
    main()