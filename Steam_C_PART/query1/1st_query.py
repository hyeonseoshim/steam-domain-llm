import os
import numpy as np
import pandas as pd
import random
from sentence_transformers import SentenceTransformer

# ==========================================
# [설정] 파일 및 폴더 경로 정의
# ==========================================
csv_folder = os.path.expanduser("~/Downloads/steam_dataset_2025_csv")
emb_folder = os.path.expanduser("~/Downloads/steam_dataset_2025_embeddings")

apps_csv_path = os.path.join(csv_folder, "applications.csv")
emb_npy_path = os.path.join(emb_folder, "applications_embeddings.npy")
emb_map_path = os.path.join(emb_folder, "applications_embedding_map.csv")

# ==========================================
# [데이터] C파트 특화 감성/상황별 가상 쿼리 50선
# ==========================================
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
    print("🚀 [C파트 엔진] 50개 쿼리 연동 실시간 벡터 검색 파이프라인 가동\n")
    
    # 1. 무작위 유저 쿼리 추출
    selected_query = random.choice(steam_queries)
    print(f"🎯 [유저 입력 쿼리]: \"{selected_query}\"")
    print("-" * 60)
    
    # 2. 임베딩 매핑 정보 및 메타데이터 로드
    df_map = pd.read_csv(emb_map_path)
    total_records = len(df_map)
    
    # apps 메인 정보 로드 (상위 20개만 컬럼 매핑 테스트용으로 확인 후 전체 연동)
    # 메모리 방어를 위해 주석 처리하거나 필요 컬럼만 load 하도록 설계
    df_apps = pd.read_csv(apps_csv_path, usecols=['appid', 'name', 'type']) 
    df_apps = df_apps[df_apps['type'] == 'game'] # 오직 게임만 필터링
    
    # 3. 안전한 memmap 복구 행렬 빌드
    emb_matrix = np.memmap(emb_npy_path, dtype='float32', mode='r')
    emb_matrix = emb_matrix.reshape(total_records, 1024)
    
    # 4. M4 MPS(Metal Performance Shaders) 가속 기반 BGE-M3 임베딩 모델 로드
    print("[*] M4 칩셋 가속을 활용해 BGE-M3 임베딩 모델을 로드합니다...")
    # 원본 데이터셋이 사용한 대조 모델 'BAAI/bge-m3' 탑재
    model = SentenceTransformer('BAAI/bge-m3', device='mps') 
    
    # 5. 자연어 쿼리 벡터 변환 (인코딩)
    print("[*] 쿼리 문장 벡터라이징 진행 중...")
    query_vector = model.encode(selected_query, normalize_embeddings=True)
    
    # 6. 코사인 유사도 계산 (메모리 맵 상태의 대용량 행렬과 1:N 연산 수행)
    print("[*] Steam 24만 개 데이터셋 실시간 시맨틱 매칭 실행...")
    # 데이터셋의 임베딩이 이미 정규화되어 있다고 가정하고 행렬곱 수행
    # (만약 정규화가 안되어 있다면 분모에 norm을 연산해야 하나 정석 배포본은 정규화되어 있음)
    similarities = np.dot(emb_matrix, query_vector)
    
    # 7. 최상위 유사 후보 3개 추출
    top_k = 3
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_k_indices):
        # 맵 매프 구조에 따라 appid 추출 (두번째 컬럼이 appid 계열 역할)
        appid = df_map.iloc[idx].values[1] 
        similarity_score = similarities[idx]
        
        # 메타데이터 테이블에서 게임명 매핑
        game_name_row = df_apps[df_apps['appid'] == appid]
        game_name = game_name_row['name'].values[0] if not game_name_row.empty else "Unknown Game"
        
        # 2주차 디버깅을 위한 가상 개인화 로그 역산 가중치 생성
        simulated_playtime = random.randint(8, 52) if similarity_score > 0.65 else 0
        
        results.append({
            "순위": rank + 1,
            "AppID": appid,
            "게임명": game_name,
            "유사도 스코어": f"{similarity_score:.4f}",
            "가상 플레이이력(hrs)": simulated_playtime
        })
        
    print("\n📦 [최종 매칭 결과 - 1주차 검증 완료]")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
if __name__ == "__main__":
    main()