# query1/generate_universe.py
import os
import sys
import json
import re
from mlx_lm import load, generate

print("⏳ [Build] MLX 파인튜닝 모델 로드 중...")
MODEL_PATH = 'Qwen/Qwen2.5-7B-Instruct'
ADAPTER_PATH = './adapters'

try:
    model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    print("🔥 [Build] MLX 파인튜닝 가중치 로드 성공!")
except Exception as e:
    print(f"⚠️ 모델 로드 경고 (기본 가드레일 모드로 진행): {e}")
    model, tokenizer = None, None

def clean_korean_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    return text.strip()

def parse_json(raw_text: str):
    cleaned = clean_korean_text(raw_text)
    try:
        start_brace = cleaned.find('{')
        end_brace = cleaned.rfind('}') + 1
        if start_brace != -1 and end_brace != -1:
            return json.loads(cleaned[start_brace:end_brace])
    except Exception:
        pass
    return None

def build_universe():
    universe = {}
    dates = ["2024-02-10", "2024-08-15", "2025-05-20", "2026-07-23"]
    print("🚀 100명 가상 유저 및 롱텀 인격 타임라인 시뮬레이션 생성을 시작합니다...")

    for i in range(1, 101):
        uid = f"user_{i:03d}"
        user_name = f"가상게이머_{i}"
        base_persona = "다양한 스팀 게임에 호기심을 지닌 게이머"

        if model is not None and tokenizer is not None:
            prompt = (
                f"<|im_start|>system\n당신은 유저 창작 엔진입니다. 오직 한글 JSON으로만 답변하세요.<|im_end|>\n"
                f"<|im_start|>user\n스팀 게이머 1명의 이름(user_name)과 평소 게임 취향(base_persona)을 한글 JSON으로 생성하세요.\n"
                f'{{"user_name": "한국어이름", "base_persona": "취향 설명 1문장"}}<|im_end|>\n'
                f"<|im_start|>assistant\n"
            )
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=100)
            parsed = parse_json(raw)
            if parsed and isinstance(parsed, dict) and "user_name" in parsed:
                user_name = clean_korean_text(parsed.get("user_name", user_name))
                base_persona = clean_korean_text(parsed.get("base_persona", base_persona))

        user_timeline = {}
        for d in dates:
            life_event = f"{user_name}님이 {d} 시점에 일상 속 새로운 변화와 기분 전환을 경험함."
            evolved_persona = f"{base_persona}를 바탕으로 현재 몰입감과 정서적 만족을 원하는 상태."

            if model is not None and tokenizer is not None:
                sim_prompt = (
                    f"<|im_start|>system\n당신은 인격 진화 시뮬레이터입니다. 오직 한글 JSON으로만 답변하세요.<|im_end|>\n"
                    f"<|im_start|>user\n유저 [{user_name}] ({base_persona})에게 날짜 [{d}]에 일어난 현실 사건(life_event)과 "
                    f"그로 인해 진화한 현재 게이밍 무드(evolved_persona)를 한글 JSON으로 생성하세요.\n"
                    f'{{"life_event": "사건 1문장", "evolved_persona": "진화한 무드 1문장"}}<|im_end|>\n'
                    f"<|im_start|>assistant\n"
                )
                raw_sim = generate(model, tokenizer, prompt=sim_prompt, max_tokens=150)
                parsed_sim = parse_json(raw_sim)
                if parsed_sim and isinstance(parsed_sim, dict) and "life_event" in parsed_sim:
                    life_event = clean_korean_text(parsed_sim.get("life_event", life_event))
                    evolved_persona = clean_korean_text(parsed_sim.get("evolved_persona", evolved_persona))

            user_timeline[d] = {
                "name": user_name,
                "context": life_event,
                "persona": evolved_persona
            }

        universe[uid] = user_timeline
        print(f"  └─ [{i}/100] 유저 시뮬레이션 완료 ({user_name})")

    output_path = "query1/precomputed_universe.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(universe, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 [완료] 총 100명의 타임라인 데이터가 '{output_path}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    build_universe()