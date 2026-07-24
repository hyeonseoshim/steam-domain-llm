# query1/clean_universe.py
import json
import re

json_path = "query1/precomputed_universe.json"

with open(json_path, "r", encoding="utf-8") as f:
    universe = json.load(f)

# 오염 패턴 검출 (숫자+단위 반복, 한자 등)
bad_pattern = re.compile(r'(\d+kg|\d+m){2,}')

fixed_count = 0
for uid, timeline in universe.items():
    for date_str, item in timeline.items():
        ctx = item.get("context", "")
        persona = item.get("persona", "")
        name = item.get("name", "게이머")

        # 오염 문장 감지 시 정상 문장으로 덮어쓰기
        if bad_pattern.search(ctx) or len(ctx) < 10 or "15kg" in ctx:
            item["context"] = f"{name}님이 {date_str} 시점에 몰입감 높은 게이밍을 통해 기분 전환을 경험함."
            fixed_count += 1

        if bad_pattern.search(persona) or len(persona) < 10 or "15kg" in persona:
            item["persona"] = "깊이 있는 세계관 속에서 새로운 도전과 몰입감을 갈망하는 상태."
            fixed_count += 1

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(universe, f, ensure_ascii=False, indent=2)

print(f"✅ 오염된 타임라인 문장 총 {fixed_count}건 정제 완료!")