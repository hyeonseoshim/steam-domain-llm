import json
import os
import random

# 1. 원천 데이터 로드
source_file = "synthetic_train_data.json"
if not os.path.exists(source_file):
    raise FileNotFoundError(f"{source_file} 파일이 없습니다. 합성 엔진을 먼저 돌려주세요!")

with open(source_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. MLX Chat Template 규격으로 변환
mlx_dataset = []
for item in raw_data:
    # 소형 모델에게 정체성을 부여하는 시스템 프롬프트
    system_prompt = (
        "당신은 Steam 게임 추천 엔진입니다. 유저의 자연어 쿼리와 매칭된 게임 이름을 바탕으로, "
        "디버깅용 'developer_reason'과 유저용 'user_reason'을 포함한 정석 JSON 구조체로 답변해야 합니다."
    )
    
    # 모델이 입력받을 컨텍스트
    user_input = f"Query: {item['input']['query']}\nGame: {item['input']['matched_game']}"
    
    # 모델이 최종 학습하고 도달해야 하는 정답(Target) JSON 문자열
    # 이 구조를 완벽히 외우도록 유도합니다.
    assistant_output = json.dumps(item['output'], ensure_ascii=False)
    
    chat_format = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output}
        ]
    }
    mlx_dataset.append(chat_format)

# 3. 학습셋(Train)과 검증셋(Valid)을 9:1 비율로 무작위 분할
random.shuffle(mlx_dataset)
split_idx = int(len(mlx_dataset) * 0.9)
train_set = mlx_dataset[:split_idx]
valid_set = mlx_dataset[split_idx:]

# 4. mlx-lm 규격에 맞게 data/ 폴더 하위에 jsonl 파일로 저장
os.makedirs("data", exist_ok=True)

with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for item in train_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("data/valid.jsonl", "w", encoding="utf-8") as f:
    for item in valid_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ MLX 포맷 변환 완료! (Train: {len(train_set)}개, Valid: {len(valid_set)}개)")
print("📂 'data/train.jsonl' 및 'data/valid.jsonl' 파일이 생성되었습니다.")