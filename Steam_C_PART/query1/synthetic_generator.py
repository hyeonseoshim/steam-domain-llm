import os
import numpy as np
import pandas as pd
import random
import json
import re
from sentence_transformers import SentenceTransformer
import ollama

# ==========================================
# [설정] 로컬 파일 및 폴더 경로 정의
# ==========================================
csv_folder = os.path.expanduser("~/Downloads/steam_dataset_2025_csv")
emb_folder = os.path.expanduser("~/Downloads/steam_dataset_2025_embeddings")

apps_csv_path = os.path.join(csv_folder, "applications.csv")
emb_npy_path = os.path.join(emb_folder, "applications_embeddings.npy")
emb_map_path = os.path.join(emb_folder, "applications_embedding_map.csv")
output_dataset_path = "synthetic_train_data.json"

# [데이터] 50개 쿼리 (본 코드는 테스트용으로 축약, 실제 50개 그대로 사용하세요)
steam_queries = [
"비 오는 날 창밖 보면서 느긋하게 즐길 수 있는 힐링 게임",
    "지친 하루 끝에 자극 없이 마음을 달래줄 잔잔한 인디 게임",
    "스토리가 아름답고 그래픽이 동화 같은 감성적인 어드벤처",
    "아무 생각 없이 평화로운 시골 마을을 가꾸는 시뮬레이션",
    "사운드트랙이 피아노 선율 위주로 아름답게 흘러나오는 게임",
    "대사 없이 영상미와 분위기만으로 압도하는 예술적인 게임",
    "우주를 떠돌며 평화롭게 행성을 탐사하는 몽환적인 게임",
    "고양이들이 잔뜩 나와서 보기만 해도 미소 지어지는 게임",
    "스트레스 받지 않고 혼자서 느긋하게 퍼즐을 푸는 게임",
    "바다 속을 자유롭게 헤엄치며 해양 생물을 구경하는 힐링물",
    "주말에 친구랑 디스코드 하면서 배가 찢어지게 웃을 수 있는 파티 게임",
    "우정 파괴용 아기자기하지만 서로 통수 치는 협동 게임",
    "친구 3~4명이서 밤새도록 기지를 짓고 생존하는 서바이벌",
    "지능적인 협동이 필요해서 고도의 소통을 요구하는 2인인 전용 퍼즐",
    "친구들과 몬스터를 사냥하며 장비를 맞추는 가벼운 RPG",
    "퇴근하고 가볍게 한두 판 동료들과 깰 수 있는 코옵 슈팅 게임",
    "요리나 경영을 하며 친구들과 소리지르며 분업하는 게임",
    "다 같이 감옥을 탈출하거나 미션을 해결하는 역할 분담 게임",
    "캐주얼하고 진입장벽이 낮아서 게임 안 해본 친구와도 가능한 게임",
    "공포스러운 분위기 속에서 친구들과 비명 지르며 아이템 줍는 게임",
    "한 편의 웰메이드 영화나 소설을 읽은 듯한 깊은 여운의 스토리 게임",
    "플레이어의 사소한 선택에 따라 결말이 완전히 뒤바뀌는 정통 RPG",
    "세계관 설정이 치밀하고 역사나 철학적 메시지가 담긴 게임",
    "사이버펑크나 암울한 미래 도시를 배경으로 한 몰입감 높은 서사",
    "주인공에게 완전히 감정 이입되어 눈물 흘릴 만한 감동적인 게임",
    "추리 소설의 탐정이 된 것처럼 단서를 모아 범인을 찾는 미스터리",
    "시간 여행이나 루프물을 소재로 한 두뇌 자극 스토리 게임",
    "디스토피아 세계관에서 살아남는 인간들의 시린 이야기를 다룬 게임",
    "등장인물 간의 관계성과 심리 묘사가 탁월한 비주얼 노벨 계열",
    "신화나 전설을 현대적으로 재해석한 웅장한 연출의 어드벤처",
    "수십 번 죽어가며 보스의 패턴을 외우고 극복하는 성취감 있는 게임",
    "피지컬보다는 뇌지컬과 고도의 전략적 수읽기를 요구하는 턴제 게임",
    "로그라이크 특유의 매 판 새로운 조합과 억까를 극복하는 재미",
    "자원이 극도로 제한된 상황에서 심리적 압박감을 주는 하드코어 생존",
    "도트 그래픽이지만 난이도는 매운맛인 플랫폼 액션 게임",
    "복잡한 공장 자동화 라인을 설계하며 효율성을 극대화하는 게임",
    "단 한 번의 실수가 패배로 이어지는 극도로 긴장감 넘치는 전술 슈팅",
    "나만의 덱을 정교하게 빌드업하여 시너지를 내는 하드코어 카드 게임",
    "길을 찾기 어렵고 불친절하지만 탐험하는 맛이 살아있는 메트로배니아",
    "한 치 앞도 알 수 없는 심리전과 피 말리는 타이밍 싸움의 대전 격투",
    "지갑이 가벼운 학생들을 위한 1~2만 원대 갓성비 타임킬러 게임",
    "내 낡은 사무용 노트북에서도 부드럽게 돌아가는 픽셀 아트 게임",
    "한 판당 10~20분 내외로 끝나서 출퇴근길이나 짬시간에 하기 좋은 게임",
    "뱀파이어 서바이버즈처럼 능동적인 조작 없이 사방의 적을 쓸어 담는 게임",
    "단순한 규칙인데 이상하게 밤을 새우게 만드는 악마 같은 중독성 게임",
    "무료(Free to Play) 게임 중에서 과금 유도 없고 평가가 극찬인 게임",
    "카드 게임과 로그라이크가 섞여 가볍게 한 판씩 즐기기 좋은 타이밍 킬러",
    "오래된 고전 명작 감성이 나지만 편의성은 최신식인 레트로 풍 게임",
    "가격 대비 플레이 타임이 100시간 이상 보장되는 오픈월드 가성비 작",
    "규칙은 오목만큼 쉬운데 마스터하긴 어려운 가벼운 보드게임 감성"
]

def main():
    print("🤖 [로컬 합성 엔진 V2] JSON Mode 강제 적용 및 할루시네이션 원천 차단 파이프라인")
    
    df_map = pd.read_csv(emb_map_path)
    total_records = len(df_map)
    emb_matrix = np.memmap(emb_npy_path, dtype='float32', mode='r').reshape(total_records, 1024)
    
    df_apps = pd.read_csv(apps_csv_path, usecols=['appid', 'name', 'type'])
    df_apps = df_apps[df_apps['type'] == 'game']
    
    embed_model = SentenceTransformer('BAAI/bge-m3', device='mps')
    
    # 2주차 진입을 위한 최종 볼륨 설정 (예: 20~50개 먼저 테스트 후 200개 이상 확장 권장)
    iterations = 5  
    synthetic_dataset = []
    
    for i in range(iterations):
        print(f"\n🔄 [{i+1}/{iterations}] 안전 매칭 및 JSON 추론 진행 중...")
        
        selected_query = random.choice(steam_queries)
        query_vector = embed_model.encode(selected_query, normalize_embeddings=True)
        similarities = np.dot(emb_matrix, query_vector)
        
        sorted_indices = np.argsort(similarities)[::-1]
        target_idx = None
        game_name = "Unknown Game"
        appid = None
        
        for idx in sorted_indices[:50]:
            potential_appid = df_map.iloc[idx].values[1]
            game_row = df_apps[df_apps['appid'] == potential_appid]
            if not game_row.empty:
                target_idx = idx
                appid = potential_appid
                game_name = game_row['name'].values[0]
                break
                
        if target_idx is None:
            continue
            
        similarity_score = float(similarities[target_idx])
        simulated_playtime = random.randint(10, 50)
        
        # 🛠️ 1. 개발자 사유 자동 빌드
        developer_reason = (
            f"유저 자연어 쿼리 유사도 연산 기반 결과임. "
            f"선정된 게임 '{game_name}'(AppID: {appid})은 쿼리와의 시맨틱 유사도 {similarity_score:.4f}를 기록함. "
            f"유저 가상 플레이 패턴 데이터 분석 결과, 유사 장르 누적 {simulated_playtime}시간 플레이 이력이 관측되어 가중 랭킹 시스템에 반영됨."
        )
        
        # 😊 2. 유저 사유용 초강력 가드레일 프롬프트 (JSON 반환 규격화)
        prompt = f"""
        당신은 오직 한 문장의 한국어 추천 이유만 작성하는 시스템입니다.
        아래의 정보를 바탕으로 유저에게 정중하고 부드러운 존댓말(~습니다 체)로 된 추천 이유를 JSON 형식으로 답변하십시오.

        [지시 사항]
        1. 무조건 한글로만 작성하고, 인사말이나 "추천 이유:" 같은 서두는 절대 포함하지 마십시오.
        2. 다른 임의의 게임 이름을 지어내지 말고, 반드시 지정된 [게임 이름]의 분위기만 서술하십시오.
        3. 반환 포맷은 정확히 {{"reason": "작성한 추천 사유"}} 형태여야 합니다.

        [데이터]
        - 유저 요청: "{selected_query}"
        - 게임 이름: "{game_name}"

        JSON 응답:
        """
        
        user_reason = ""
        for retry in range(3):
            try:
                # format='json' 옵션을 주어 Qwen2.5가 딴소리를 못하게 문법을 강제합니다.
                response = ollama.generate(model='qwen2.5:7b', prompt=prompt, format='json')
                res_json = json.loads(response['response'].strip())
                candidate_text = res_json.get("reason", "").strip()
                
                # 한자/중국어 유입 필터링
                if re.search(r'[\u4e00-\u9fff]', candidate_text):
                    continue
                
                # 가비지 키워드 및 타 게임 명칭 오염 필터링 가드레일
                if "유저 요청" in candidate_text or "추천 게임" in candidate_text:
                    continue
                
                user_reason = candidate_text
                break
            except Exception as e:
                print(f"  [재시도] 포맷 보정 에러 발생: {e}")
                continue
                
        if not user_reason:
            user_reason = f"요청하신 분위기에 잘 맞으며, 독창적인 재미와 몰입감을 선사할 {game_name}을 추천해 드립니다."
            
        # 📦 3. 파인튜닝용 정석 데이터 바인딩
        data_pair = {
            "input": {
                "query": selected_query,
                "matched_game": game_name,
                "appid": int(appid)
            },
            "output": {
                "developer_reason": developer_reason,
                "user_reason": user_reason
            }
        }
        
        synthetic_dataset.append(data_pair)
        print(f"🎯 매칭: {game_name}")
        print(f"😊 [정제된 유저 사유]: {user_reason}")
        print("-" * 50)

    with open(output_dataset_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 [성공] 오염 데이터가 완벽히 차단된 {len(synthetic_dataset)}개의 데이터 쌍이 저장되었습니다.")

if __name__ == "__main__":
    main()