# query1/enrich_universe.py
import json
import random

path = "query1/precomputed_universe.json"

with open(path, "r", encoding="utf-8") as f:
    universe = json.load(f)

events_pool = [
    "마감 프로젝트를 무사히 완수하고 압도적인 해방감과 성취감을 맛봄.",
    "격렬한 야외 운동으로 땀을 흘린 후 피지컬적 아드레날린이 고조됨.",
    "주말에 창밖의 비 소리를 들으며 잔잔한 카페에서 사색을 즐김.",
    "업무 과부하로 인한 야근 지속으로 심각한 번아웃과 휴식이 필요한 상태.",
    "오랜만에 만난 친구들과 홈파티 후 유쾌한 뒷풀이를 기획함."
]

personas_pool = [
    "묵직한 타격감과 타협 없는 난이도의 강렬한 사투를 갈망하는 코어 게이머 성향.",
    "복잡한 생각 없이 시간에 쫓기지 않는 느긋하고 평화로운 힐링을 원하는 상태.",
    "방대한 세계관과 깊이 있는 선택지를 자유롭게 탐험하고자 하는 모험가 성향.",
    "스피디한 템포와 정교한 패링 컨트롤로 스트레스를 시원하게 날리고 싶은 상태.",
    "신선한 탐험 요소와 소소한 경영 타이쿤 루프를 좋아하는 하이브리드 성향."
]

updated_count = 0
for uid, timeline in universe.items():
    for date_str, item in timeline.items():
        # 기본 템플릿 문장이 들어있는 경우 다채로운 무드로 스와핑
        if "호기심을 지닌 게이머" in item.get("persona", ""):
            idx = random.randint(0, len(events_pool) - 1)
            name = item.get("name", "게이머")
            item["context"] = f"{name}님이 {events_pool[idx]}"
            item["persona"] = personas_pool[idx]
            updated_count += 1

with open(path, "w", encoding="utf-8") as f:
    json.dump(universe, f, ensure_ascii=False, indent=2)

print(f"✅ 총 {updated_count}개의 반복 페르소나 데이터 다채화 정제 완료!")