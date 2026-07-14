from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_INPUT_PATH = "data/processed/english_reviews_clean.csv"
DEFAULT_OUTPUT_DIR = "services/b-review-analysis/data/pseudo_labels"
DEFAULT_GOLDEN_FILE = "services/b-review-analysis/data/annotations/topic_golden_dataset.csv"
DEFAULT_GUIDELINES_FILE = "services/b-review-analysis/data/annotations/topic_guidelines.json"

TEXT_COLUMN_CANDIDATES = [
    "review_text_clean",
    "review_text_original",
    "review_text",
    "cleaned_review",
    "text",
    "review",
    "content",
]

OPTIONAL_META_COLUMNS = [
    "annotation_id",
    "recommendationid",
    "appid",
    "language",
    "voted_up",
    "label",
    "word_count",
    "char_count",
    "votes_up",
    "weighted_vote_score",
    "author_playtime_at_review",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
]


def split_topics(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null", "[]", "-", "없음"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    parts = re.split(r"[,\|;/]+", text)
    return [p.strip() for p in parts if p.strip()]


def load_guidelines(path: str) -> tuple[dict[str, str], list[str]]:
    guideline_path = Path(path)
    if not guideline_path.exists():
        raise FileNotFoundError(f"토픽 가이드라인 파일을 찾을 수 없습니다: {path}")

    with guideline_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    topics = data.get("topics", {})
    rules = data.get("distinction_rules", [])

    if not topics:
        raise ValueError("topic_guidelines.json에 topics가 없습니다.")

    return topics, rules


def load_allowed_topics_from_golden(golden_file: str) -> list[str]:
    path = Path(golden_file)
    if not path.exists():
        raise FileNotFoundError(f"골든셋 파일을 찾을 수 없습니다: {golden_file}")

    df = pd.read_csv(path)

    topics = set()
    for col in ["positive_topics", "negative_topics"]:
        if col not in df.columns:
            raise ValueError(f"{golden_file}에 {col} 컬럼이 없습니다.")

        for value in df[col].dropna():
            topics.update(split_topics(value))

    topics = sorted(t for t in topics if t)

    if not topics:
        raise ValueError("골든셋에서 허용 토픽 목록을 찾지 못했습니다.")

    return topics


def find_text_column(df: pd.DataFrame, text_column: str | None) -> str:
    if text_column:
        if text_column not in df.columns:
            raise ValueError(f"지정한 text column이 없습니다: {text_column}")
        return text_column

    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "리뷰 본문 컬럼을 자동으로 찾지 못했습니다. "
        f"현재 컬럼: {df.columns.tolist()}"
    )


def load_golden_examples(golden_file: str) -> pd.DataFrame:
    df = pd.read_csv(golden_file)

    required_cols = ["review_text_clean", "positive_topics", "negative_topics"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"골든셋에 {col} 컬럼이 없습니다.")

    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(str).str.lower() == "true"].copy()

    if "human_verified" in df.columns:
        df = df[df["human_verified"].astype(str).str.lower() == "true"].copy()

    df = df[df["review_text_clean"].notna()].copy()
    df["review_text_clean"] = df["review_text_clean"].astype(str).str.strip()
    df = df[df["review_text_clean"] != ""].copy()

    df["positive_topic_list"] = df["positive_topics"].apply(split_topics)
    df["negative_topic_list"] = df["negative_topics"].apply(split_topics)

    return df.reset_index(drop=True)


class GoldenRetriever:
    def __init__(self, golden_df: pd.DataFrame):
        self.golden_df = golden_df
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(
            golden_df["review_text_clean"].tolist()
        )

    def retrieve(
        self,
        query: str,
        k: int,
        min_score: float = 0.0,
        exclude_recommendationid=None,
        exclude_text: str | None = None,
    ) -> list[dict]:
        if k <= 0 or len(self.golden_df) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()

        examples = []

        for idx in scores.argsort()[::-1]:
            if len(examples) >= k:
                break

            score = float(scores[idx])

            if score < min_score:
                continue

            row = self.golden_df.iloc[int(idx)]

            if exclude_recommendationid is not None:
                row_id = row.get("recommendationid")

                if (
                    pd.notna(row_id)
                    and str(row_id) == str(exclude_recommendationid)
                ):
                    continue

            if exclude_text:
                candidate_text = str(
                    row["review_text_clean"]
                ).strip()

                if candidate_text == str(exclude_text).strip():
                    continue

            examples.append(
                {
                    "review": row["review_text_clean"],
                    "positive_topics": row["positive_topic_list"],
                    "negative_topics": row["negative_topic_list"],
                    "score": score,
                }
            )

        return examples


def build_topic_context(topic_definitions: dict[str, str], allowed_topics: list[str]) -> str:
    lines = []

    for topic in allowed_topics:
        definition = topic_definitions.get(topic, "")
        lines.append(f"- {topic}: {definition}")

    return "\n".join(lines)


def build_rules_context(rules: list[str]) -> str:
    return "\n".join([f"- {rule}" for rule in rules])


def build_examples_context(examples: list[dict]) -> str:
    if not examples:
        return "No similar examples provided."

    blocks = []

    for i, ex in enumerate(examples, start=1):
        pos = json.dumps(ex["positive_topics"], ensure_ascii=False)
        neg = json.dumps(ex["negative_topics"], ensure_ascii=False)

        blocks.append(
            f"""Example {i}:
Review:
{ex["review"]}

positive_topics: {pos}
negative_topics: {neg}"""
        )

    return "\n\n".join(blocks)


def build_prompt(
    review_text: str,
    allowed_topics: list[str],
    topic_definitions: dict[str, str],
    distinction_rules: list[str],
    examples: list[dict],
) -> str:
    allowed_list = ", ".join(allowed_topics)
    topic_context = build_topic_context(topic_definitions, allowed_topics)
    rules_context = build_rules_context(distinction_rules)
    examples_context = build_examples_context(examples)

    return f"""
You are a Steam game review topic classifier.

Your task is to classify the target review into positive_topics and negative_topics.

Allowed topics:
{allowed_list}

Topic definitions:
{topic_context}

Distinction rules:
{rules_context}

Similar labeled examples from the human-verified golden set:
{examples_context}

Strict rules:
- Use only the allowed topics.
- Do not create new topics.
- If there is no positive topic, output an empty list.
- If there is no negative topic, output an empty list.
- Use a topic only when the review gives clear evidence.
- Output JSON only.
- Do not explain.
- Do not add markdown.

Output format:
{{"positive_topics": ["topic1"], "negative_topics": ["topic2"]}}

Target review:
{review_text}
""".strip()


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float,
    num_predict: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        ollama_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as res:
        body = res.read().decode("utf-8")
        parsed = json.loads(body)

    return str(parsed.get("response", "")).strip()


def extract_json(raw_response: str) -> dict:
    raw = raw_response.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def normalize_topics(values, allowed_topics: list[str]) -> tuple[list[str], list[str]]:
    allowed_map = {t.lower(): t for t in allowed_topics}

    if values is None:
        return [], []

    if isinstance(values, str):
        values = split_topics(values)

    if not isinstance(values, list):
        return [], [str(values)]

    normalized = []
    invalid = []

    for value in values:
        item = str(value).strip().strip('"').strip("'")
        if not item:
            continue

        key = item.lower()
        if key in allowed_map:
            topic = allowed_map[key]
            if topic not in normalized:
                normalized.append(topic)
        else:
            invalid.append(item)

    return normalized, invalid


def parse_response(raw_response: str, allowed_topics: list[str]) -> tuple[list[str], list[str], bool, str]:
    parsed = extract_json(raw_response)

    if not parsed:
        return [], [], False, "json_parse_failed"

    pos, pos_invalid = normalize_topics(parsed.get("positive_topics"), allowed_topics)
    neg, neg_invalid = normalize_topics(parsed.get("negative_topics"), allowed_topics)

    invalid_items = pos_invalid + neg_invalid

    if invalid_items:
        return [], [], False, "invalid_topic"

    return pos, neg, True, ""


def prepare_sample(
    input_path: str,
    text_column: str | None,
    limit: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_path)
    detected_text_column = find_text_column(df, text_column)

    df = df[df[detected_text_column].notna()].copy()
    df[detected_text_column] = df[detected_text_column].astype(str).str.strip()
    df = df[df[detected_text_column] != ""].copy()

    if limit > 0 and len(df) > limit:
        df = df.sample(n=limit, random_state=seed).copy()

    df = df.reset_index(drop=False).rename(columns={"index": "source_index"})

    return df, detected_text_column


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--golden-file", default=DEFAULT_GOLDEN_FILE)
    parser.add_argument("--guidelines-file", default=DEFAULT_GUIDELINES_FILE)
    parser.add_argument("--rag-k", type=int, default=3)
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    topic_definitions, distinction_rules = load_guidelines(args.guidelines_file)
    allowed_topics_from_golden = load_allowed_topics_from_golden(args.golden_file)
    allowed_topics_from_guidelines = sorted(topic_definitions.keys())

    missing_in_guidelines = sorted(set(allowed_topics_from_golden) - set(allowed_topics_from_guidelines))
    if missing_in_guidelines:
        raise ValueError(f"topic_guidelines.json에 없는 토픽이 있습니다: {missing_in_guidelines}")

    allowed_topics = allowed_topics_from_golden

    golden_df = load_golden_examples(args.golden_file)
    retriever = GoldenRetriever(golden_df)

    df, text_col = prepare_sample(
        input_path=args.input,
        text_column=args.text_column,
        limit=args.limit,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = args.model.replace(":", "_").replace("/", "_")
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"pseudo_topics_rag_v2_{safe_model_name}_{len(df)}.csv"

    fieldnames = [
        "source_index",
        "input_text",
        "positive_topics",
        "negative_topics",
        "raw_response",
        "is_valid",
        "invalid_reason",
        "teacher_model",
        "prompt_version",
        "rag_k",
        "retrieved_examples",
        "text_column",
        "created_at",
    ]

    for col in OPTIONAL_META_COLUMNS:
        if col in df.columns and col not in fieldnames:
            fieldnames.insert(1, col)

    print(f"입력 파일: {args.input}")
    print(f"출력 파일: {output_path}")
    print(f"리뷰 컬럼: {text_col}")
    print(f"teacher 모델: {args.model}")
    print(f"샘플 수: {len(df)}")
    print(f"허용 토픽 수: {len(allowed_topics)}")
    print(f"RAG 예시 수: {args.rag_k}")
    print(f"허용 토픽: {allowed_topics}")

    valid_count = 0
    invalid_count = 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in df.iterrows():
            review_text = str(row[text_col]).strip()
            examples = retriever.retrieve(review_text, args.rag_k)

            prompt = build_prompt(
                review_text=review_text,
                allowed_topics=allowed_topics,
                topic_definitions=topic_definitions,
                distinction_rules=distinction_rules,
                examples=examples,
            )

            try:
                raw_response = call_ollama(
                    prompt=prompt,
                    model=args.model,
                    ollama_url=args.ollama_url,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
                positive_topics, negative_topics, is_valid, invalid_reason = parse_response(
                    raw_response,
                    allowed_topics,
                )

            except Exception as e:
                raw_response = f"ERROR: {e}"
                positive_topics = []
                negative_topics = []
                is_valid = False
                invalid_reason = "runtime_error"

            if is_valid:
                valid_count += 1
            else:
                positive_topics = []
                negative_topics = []
                invalid_count += 1

            retrieved_payload = [
                {
                    "review": ex["review"],
                    "positive_topics": ex["positive_topics"],
                    "negative_topics": ex["negative_topics"],
                    "score": round(ex["score"], 4),
                }
                for ex in examples
            ]

            output_row = {
                "source_index": row["source_index"],
                "input_text": review_text,
                "positive_topics": json.dumps(positive_topics, ensure_ascii=False),
                "negative_topics": json.dumps(negative_topics, ensure_ascii=False),
                "raw_response": raw_response if is_valid else "REJECTED_INVALID_OUTPUT",
                "is_valid": is_valid,
                "invalid_reason": invalid_reason,
                "teacher_model": args.model,
                "prompt_version": "pseudo_topics_rag_v2",
                "rag_k": args.rag_k,
                "retrieved_examples": json.dumps(retrieved_payload, ensure_ascii=False),
                "text_column": text_col,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }

            for col in OPTIONAL_META_COLUMNS:
                if col in df.columns:
                    output_row[col] = row[col]

            writer.writerow(output_row)

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"[{i + 1}/{len(df)}] valid={valid_count}, invalid={invalid_count}, "
                    f"pos={positive_topics}, neg={negative_topics}"
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("완료")
    print(f"valid: {valid_count}")
    print(f"invalid: {invalid_count}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
