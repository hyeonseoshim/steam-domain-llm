# query1/fix_causality.py
import json

path = "query1/precomputed_universe.json"

with open(path, "r", encoding="utf-8") as f:
    universe = json.load(f)

# 인과관계가 명확히 일치하는 페르소나 세트
CAUSAL_PAIRS = [
    {
        "event": "격렬한 야외 운동으로 땀을 흘린 후 피지컬적 아드레날린이 최고조로 고조됨.",
        "persona": "묵직한 타격감과 정교한 패링 컨트롤로 극한의 사투를 즐기고 싶은 상태."
    },
    {
        "event": "지속된 업무 과부하와 야근으로 인해 심각한 피로감과 번아웃이 엄습함.",
        "persona": "복잡한 생각 없이 시간에 쫓기지 않는 느긋하고 평화로운 힐링을 원하는 상태."
    },
    {
        "event": "조용한 카페 창가에서 비 오는 소리를 들으며 혼자만의 사색에 잠김.",
        "persona": "방대한 세계관과 깊이 있는 서사적 선택지를 자유롭게 탐험하고자 하는 상태."
    },
    {
        "event": "오랜만에 친구들과 만나 유쾌한 홈파티와 기분 전환을 기획함.",
        "persona": "신선한 탐험 요소와 소소한 경영 타이쿤 루프를 동시에 즐기고 싶은 상태."
    }
]

count = 0
for uid, timeline in universe.items():
    for date_str, item in timeline.items():
        # 인덱스 기반 정밀 인과 매칭 (짝 맞추기)
        pair = CAUSAL_PAIRS[count % len(CAUSAL_PAIRS)]
        name = item.get("name", "게이머")
        item["context"] = f"{name}님이 {pair['event']}"
        item["persona"] = pair["persona"]
        count += 1

with open(path, "w", encoding="utf-8") as f:
    json.dump(universe, f, ensure_ascii=False, indent=2)

print(f"✅ 총 {count}건의 타임라인 인과관계 교정 완료!")