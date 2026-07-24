# evaluate_recommendations.py
import json
import requests
import time  # 💡 시간 지연 모듈 추가

DB_FILE = "user_pattern_db.jsonl"
PART_C_URL = "http://localhost:8000/infer"
OUTPUT_ANALYSIS_FILE = "final_recommendation_analysis.md"

def main():
    print("🚀 [Phase 2] 고정 패턴 DB 기반 파인튜닝 추천 사유 직렬 분석 개시...")
    analysis_report = "# 📊 유저 인격 진화 궤적에 따른 파인튜닝 추천 사유 최종 분석\n\n"
    
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"❌ {DB_FILE} 파일이 없습니다. Phase 1을 먼저 실행하세요.")
        return

    for record in records:
        print(f"🔄 시뮬레이션 Step {record['step']} 분석 요청 중...")
        
        payload = {
            "query": record["generated_query"],
            "matched_game": record["chosen_game"],
            "similarity_score": 0.78
        }
        
        user_reason = "추론 실패"
        try:
            response = requests.post(PART_C_URL, json=payload, timeout=90)
            if response.status_code == 200:
                res = response.json()
                if res["status"] == "success":
                    user_reason = res["data"]["user_reason"]
                    print(f"   ✓ Step {record['step']} 성공 수신 완료.")
                else:
                    user_reason = f"🛑 가드레일 작동 차단: {res['data']['user_reason']}"
        except Exception as e:
            user_reason = f"❌ 통신 실패 ({str(e)})"

        analysis_report += f"## ⏱️ 타임스탬프 Step {record['step']}\n"
        analysis_report += f"* **주입된 현실 사건:** {record['life_event']}\n"
        analysis_report += f"* **진화한 자아 인격:** {record['evolved_persona']}\n"
        analysis_report += f"* **에이전트 직조 쿼리:** `{record['generated_query']}`\n"
        analysis_report += f"* **선택한 스팀 게임:** **{record['chosen_game']}**\n"
        analysis_report += f"* **🔥 파인튜닝 LoRA 최종 추천사유:**\n  > {user_reason}\n"
        analysis_report += "---\n\n"

        # 💡 [VRAM 완충지대] 서버 가드레일이 작동하여 메모리를 비울 수 있도록 3초간 숨을 고릅니다.
        print("⏳ 다음 요청 전 GPU VRAM 리프레시 대기 중 (3초)...")
        time.sleep(3)

    with open(OUTPUT_ANALYSIS_FILE, "w", encoding="utf-8") as f:
        f.write(analysis_report)
        
    print(f"✅ 최종 데이터 분석 리포트 생성 완료: {OUTPUT_ANALYSIS_FILE}")

if __name__ == "__main__":
    main()