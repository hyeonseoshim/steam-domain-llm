# query1/app.py
import os
import sys
import gc
import json
import random
import re
import time
import uvicorn
from typing import Dict, List, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from mlx_lm import load, generate

FT_MODEL = None
FT_TOKENIZER = None

DATA_FILE_PATH = "query1/user_interactions.json"

# 🎯 학습/검증 데이터셋 경로
TRAIN_JSONL_PATH = "data/train.jsonl"
VALID_JSONL_PATH = "data/valid.jsonl"

# 동적으로 로드될 전체 스팀 게임 데이터베이스
GAME_DATABASE: List[Dict[str, Any]] = []

class SearchEvent(BaseModel):
    userId: str
    query: str

class ClickEvent(BaseModel):
    userId: str
    appid: int
    game_name: str = ""

def clean_korean_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    return text.strip()

def parse_json_from_llm(raw_text: str):
    cleaned = clean_korean_text(raw_text)
    try:
        start_brace = cleaned.find('{')
        end_brace = cleaned.rfind('}') + 1
        if start_brace != -1 and end_brace != -1:
            return json.loads(cleaned[start_brace:end_brace])
        
        start_bracket = cleaned.find('[')
        end_bracket = cleaned.rfind(']') + 1
        if start_bracket != -1 and end_bracket != -1:
            return json.loads(cleaned[start_bracket:end_bracket])
    except Exception:
        pass
    return None

# =========================================================================
# 📂 [JSONL Loader] train.jsonl / valid.jsonl 에서 전체 게임 추출
# =========================================================================
def load_games_from_jsonl(file_paths: List[str]) -> List[Dict[str, Any]]:
    """jsonl 파일에서 중복을 제거하고 전체 게임 데이터베이스를 동적 추출합니다."""
    games_map = {}
    
    for path in file_paths:
        if not os.path.exists(path):
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    user_content = ""
                    assistant_content = ""
                    
                    for msg in messages:
                        if msg["role"] == "user":
                            user_content = msg["content"]
                        elif msg["role"] == "assistant":
                            assistant_content = msg["content"]
                    
                    # 1. Game 이름 및 Query 파싱
                    game_match = re.search(r'Game:\s*(.+)', user_content)
                    query_match = re.search(r'Query:\s*(.+)', user_content)
                    
                    game_name = game_match.group(1).strip() if game_match else ""
                    query_text = query_match.group(1).strip() if query_match else ""
                    
                    # 2. Assistant 출력 JSON에서 AppID 및 user_reason 파싱
                    assistant_json = json.loads(assistant_content) if assistant_content else {}
                    dev_reason = assistant_json.get("developer_reason", "")
                    user_reason = assistant_json.get("user_reason", "")
                    
                    appid_match = re.search(r'AppID:\s*(\d+)', dev_reason)
                    appid = int(appid_match.group(1)) if appid_match else random.randint(100000, 999999)
                    
                    if game_name and appid not in games_map:
                        # query_text 키워드를 태그 형태로 분할
                        tags = re.findall(r'\w+', query_text)
                        
                        games_map[appid] = {
                            "appid": appid,
                            "name": game_name,
                            "tags": tags,
                            "query_context": query_text,
                            "default_reason": user_reason if user_reason else f"{game_name}은 고유한 매력을 지닌 추천 작품입니다."
                        }
                except Exception:
                    continue

    return list(games_map.values())

def load_all_interactions() -> Dict[str, Any]:
    if os.path.exists(DATA_FILE_PATH):
        try:
            with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_all_interactions(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)
    with open(DATA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_user_action(user_id: str, action_type: str, payload: Dict[str, Any]):
    all_data = load_all_interactions()
    if user_id not in all_data:
        all_data[user_id] = []
    
    action_record = {
        "type": action_type,
        "payload": payload,
        "timestamp": time.time()
    }
    all_data[user_id].append(action_record)
    save_all_interactions(all_data)

# =========================================================================
# ⏳ [Time-Decay & Random Exploration Engine]
# =========================================================================
def calculate_scores_with_exploration(user_id: str) -> tuple[bool, str, List[Dict[str, Any]]]:
    all_data = load_all_interactions()
    user_history = all_data.get(user_id, [])

    # 클릭/관심 표현한 게임 배제
    interacted_appids = set()
    for act in user_history:
        if act["type"] == "click":
            appid = act["payload"].get("appid")
            if appid:
                interacted_appids.add(appid)

    # 1. 신규 유저 (Cold-Start): 전체 로드된 게임 중 무작위 셔플
    if not user_history:
        candidate_pool = [g for g in GAME_DATABASE if g["appid"] not in interacted_appids]
        if not candidate_pool:
            candidate_pool = list(GAME_DATABASE)
            
        shuffled = list(candidate_pool)
        random.shuffle(shuffled)
        
        results = []
        for g in shuffled:
            random_score = round(random.uniform(0.72, 0.89), 2)
            results.append({
                **g,
                "score": random_score,
                "is_cold": True
            })
        return True, "신규 유저 (게임성 중심 무작위 탐색 모드)", results

    # 2. 기존 유저: 가중치 연산 + 무작위 노이즈 (매번 다른 추천)
    decay_rate = 0.7
    sorted_history = sorted(user_history, key=lambda x: x["timestamp"], reverse=True)
    
    game_scores = {g["appid"]: 0.50 for g in GAME_DATABASE}
    summary_parts = []

    for idx, act in enumerate(sorted_history[:10]):
        weight = (decay_rate ** idx)
        a_type = act["type"]
        payload = act["payload"]

        if a_type == "search":
            query = payload.get("query", "")
            summary_parts.append(f"[{query}] 검색(가중치:{weight:.2f})")
            for g in GAME_DATABASE:
                q_ctx = g.get("query_context", "")
                name = g.get("name", "")
                if any(kw in q_ctx for kw in query.split()) or query.lower() in name.lower():
                    game_scores[g["appid"]] += 0.35 * weight

        elif a_type == "click":
            clicked_appid = payload.get("appid")
            g_name = payload.get("game_name", "")
            summary_parts.append(f"[{g_name}] 관심(가중치:{weight:.2f})")
            
            clicked_game = next((g for g in GAME_DATABASE if g["appid"] == clicked_appid), None)
            if clicked_game:
                clicked_tags = set(clicked_game.get("tags", []))
                for g in GAME_DATABASE:
                    if g["appid"] != clicked_appid:
                        common_tags = set(g.get("tags", [])).intersection(clicked_tags)
                        if common_tags:
                            game_scores[g["appid"]] += (0.15 * len(common_tags)) * weight

    # 🎲 [Random Dynamic Noise]
    scored_games = []
    for g in GAME_DATABASE:
        if g["appid"] in interacted_appids:
            continue  # 이미 클릭해 본 게임은 제외

        base = game_scores[g["appid"]]
        random_multiplier = random.uniform(0.80, 1.20)
        random_noise = random.uniform(0.01, 0.10)
        
        final_score = min(0.96, round((base * random_multiplier) + random_noise, 2))
        
        scored_games.append({
            **g,
            "score": final_score,
            "is_cold": False
        })

    if len(scored_games) < 5:
        for g in GAME_DATABASE:
            scored_games.append({
                **g,
                "score": round(random.uniform(0.70, 0.85), 2),
                "is_cold": False
            })

    context_str = "최근 선호 맥락: " + " / ".join(summary_parts[:3])
    scored_games.sort(key=lambda x: x["score"], reverse=True)
    return False, context_str, scored_games

# =========================================================================
# 🚀 [Lifespan Handler] jsonl 데이터 및 MLX 모델 로드
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global FT_MODEL, FT_TOKENIZER, GAME_DATABASE
    
    print("⏳ [Lifespan] train.jsonl / valid.jsonl 에서 전체 게임 데이터 로딩 중...")
    GAME_DATABASE = load_games_from_jsonl([TRAIN_JSONL_PATH, VALID_JSONL_PATH])
    print(f"✅ [Lifespan] 총 {len(GAME_DATABASE)}개의 스팀 학습 게임 데이터베이스 구축 완료!")

    print("⏳ [Lifespan] MLX 파인튜닝 LoRA 모델 로드 중...")
    MODEL_PATH = 'Qwen/Qwen2.5-7B-Instruct'
    ADAPTER_PATH = './adapters'
    
    try:
        FT_MODEL, FT_TOKENIZER = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
        print("🔥 [Lifespan] MLX 파인튜닝 가중치 로드 성공!")
    except Exception as e:
        print(f"⚠️ 모델 로드 경고 (기본 가드레일 가동): {e}")

    yield
    if FT_MODEL is not None:
        del FT_MODEL, FT_TOKENIZER
        gc.collect()

app = FastAPI(
    title="JSONL Dataset Based Steam Recommender",
    description="학습 데이터셋(JSONL) 자동 파싱 및 동적 무작위 추천 서버",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/events/search")
def log_search_event(event: SearchEvent):
    if event.query.strip():
        append_user_action(event.userId, "search", {"query": event.query.strip()})
    return JSONResponse(content={"status": "success", "message": f"Search event logged for {event.userId}"})

@app.post("/events/click")
def log_click_event(event: ClickEvent):
    game_name = event.game_name
    if not game_name:
        matched = next((g["name"] for g in GAME_DATABASE if g["appid"] == event.appid), "STEAM GAME")
        game_name = matched

    append_user_action(event.userId, "click", {"appid": event.appid, "game_name": game_name})
    return JSONResponse(content={"status": "success", "message": f"Click event logged for {event.userId}"})

def generate_reasons(user_id: str, is_cold: bool, context_str: str, candidate_games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global FT_MODEL, FT_TOKENIZER

    if FT_MODEL is not None and FT_TOKENIZER is not None:
        if is_cold:
            prompt_guide = "유저 행동 맥락을 언급하지 말고 각 게임 본연의 게임성과 재미를 매력적으로 한 문장 설명하세요."
        else:
            prompt_guide = f"유저의 최근 선호 맥락({context_str})에 맞춰 해당 게임을 추천하게 된 사유를 한 문장으로 설명하세요."

        prompt = (
            f"<|im_start|>system\n"
            f"당신은 Steam 추천 AI입니다. {prompt_guide}\n"
            f"응답은 한글 JSON 배열로만 출력하세요: [{{\"appid\": 1222880, \"reason\": \"추천사유\"}}]\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n유저 ID: {user_id}\n추천 게임 목록: {[g['name'] for g in candidate_games]}<|im_end|>\n"
            f"<|im_start|>assistant\n["
        )
        try:
            raw = generate(FT_MODEL, FT_TOKENIZER, prompt=prompt, max_tokens=700)
            if not raw.startswith("["):
                raw = "[" + raw
            parsed = parse_json_from_llm(raw)
            if parsed and isinstance(parsed, list):
                reason_map = {item["appid"]: clean_korean_text(item["reason"]) for item in parsed if "appid" in item and "reason" in item}
                
                results = []
                for g in candidate_games:
                    results.append({
                        "appid": g["appid"],
                        "name": g["name"],
                        "score": g["score"],
                        "reason": reason_map.get(g["appid"], g["default_reason"])
                    })
                return results
        except Exception:
            pass

    results = []
    for g in candidate_games:
        results.append({
            "appid": g["appid"],
            "name": g["name"],
            "score": g["score"],
            "reason": g.get("default_reason", f"{g['name']}은 독창적인 게임성으로 완성도가 높은 추천 작품입니다.")
        })
    return results

@app.get("/search")
def gateway_search(userId: str = "guest_user", k: int = 30):
    is_cold, context_str, scored_games = calculate_scores_with_exploration(userId)
    top_candidates = scored_games[:min(k, len(scored_games))]
    
    final_recommendations = generate_reasons(userId, is_cold, context_str, top_candidates)

    return JSONResponse(content={
        "userId": userId,
        "is_new_user": is_cold,
        "recent_context": context_str,
        "results": final_recommendations
    }, media_type="application/json; charset=utf-8")

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

if __name__ == "__main__":
    uvicorn.run("query1.app:app", host="0.0.0.0", port=8000, reload=True)