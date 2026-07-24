# generate_user_patterns.py
import json
import time
from langchain_ollama import OllamaLLM

# 유저 패턴 생성용으로 Ollama를 단독 활용합니다.
agent_brain = OllamaLLM(model="qwen2.5:7b", temperature=0.8)
DB_FILE = "user_pattern_db.jsonl"

GAME_POOL = [
    "Dark Souls III", "Elden Ring", "Sekiro", "Stardew Valley", 
    "Animal Crossing", "Dave the Diver", "Cyberpunk 2077", "Outer Wilds"
]

class TimedEvolvingHuman:
    def __init__(self, name: str):
        self.name = name
        self.persona = "도전과 성취를 좋아하는 코어 게이머."

    def reflect_and_act(self, step: int) -> dict:
        # 1. 무작위 현실 사건 생성
        event_prompt = f"게이머 {self.name}에게 일어날 수 있는 구체적인 일상 사건이나 건강 상태 변화를 한국어 한 문장으로 무작위 생성하세요."
        life_event = agent_brain.invoke(event_prompt).strip()
        
        # 2. 인격 리플렉션
        persona_prompt = f"이전 성향: {self.persona}\n사건: {life_event}\n위 사건을 겪은 후의 현재 정서 상태를 반영한 인격 설명을 한국어 한 문장으로만 정의하세요."
        self.persona = agent_brain.invoke(persona_prompt).strip()
        
        # 3. 쿼리 및 게임 매칭
        action_prompt = f"""현재 상태: {self.persona}\n후보: {GAME_POOL}\n이 상태의 유저가 선택할 법한 게임 하나와 자연어 쿼리를 JSON으로 출력하세요. 
        {{
            "desire_query": "게임 스타일 요구 문장(한국어)",
            "target_game": "후보 중 선택한 게임 영문 이름"
        }}"""
        raw_action = agent_brain.invoke(action_prompt).strip()
        
        try:
            start_idx = raw_action.find('{')
            end_idx = raw_action.rfind('}') + 1
            decision = json.loads(raw_action[start_idx:end_idx])
            query = decision["desire_query"]
            game = decision["target_game"]
        except:
            query, game = "여유로운 게임", "Stardew Valley"
            
        return {
            "timestamp": int(time.time()) + (step * 3600), # 1시간 간격 시뮬레이션
            "step": step,
            "life_event": life_event,
            "evolved_persona": self.persona,
            "generated_query": query,
            "chosen_game": game
        }

def main():
    print("⏳ [Phase 1] 유저 자율 이용 패턴 및 타임라인 데이터베이스 구축 시작...")
    user = TimedEvolvingHuman(name="재현")
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        for step in range(1, 11): # 총 10개 타임스탬프 누적
            record = user.reflect_and_act(step)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"📦 [DB 적재 완료] Step {step:02d} | 게임: {record['chosen_game']}")
            
    print(f"✅ 유저 패턴 DB 빌드 완료: {DB_FILE}")

if __name__ == "__main__":
    main()