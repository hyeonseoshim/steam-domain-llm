import json
from mlx_lm import load, generate

model_path = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./adapters"

print("⏳ 파인튜닝된 4-bit 모델 및 어댑터 가중치 로드 중...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

# 1. 학습할 때와 100% 일치하는 시스템 및 유저 메시지 구성
messages = [
    {
        "role": "system", 
        "content": "당신은 Steam 게임 추천 엔진입니다. 유저의 자연어 쿼리와 매칭된 게임 이름을 바탕으로, 디버깅용 'developer_reason'과 유저용 'user_reason'을 포함한 정석 JSON 구조체로 답변해야 합니다."
    },
    {
        "role": "user", 
        "content": "Query: 우주를 떠돌며 평화롭게 행성을 탐사하는 몽환적인 게임\nGame: Outer Wilds"
    }
]

# 2. Qwen2.5 고유의 Chat Template 마킹 처리 (토크나이저 바인딩)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n🚀 [C파트 학습 모델] 듀얼 타깃 생성 테스트 시작:\n")
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=300,
    verbose=True # 실시간 토큰 속도 및 메모리 모니터링 출력 활성화
)

print("\n[최종 출력 결과]:")
print(response)