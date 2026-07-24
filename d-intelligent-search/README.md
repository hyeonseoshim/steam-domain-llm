# Steam Game Recommendation System (Part D)

스팀 게임 데이터셋을 활용한 대규모 언어 모델(LLM) 기반 지능형 추천 시스템입니다. 자연어 쿼리를 분석하여 사용자 의도에 맞는 게임을 정밀하게 추출하고 랭킹화합니다.

## 핵심 기능
- 자연어 쿼리 분석: 사용자의 복잡한 요구사항(가격, 장르, 멀티플레이 여부 등)을 LLM으로 추출
- 하이브리드 검색: SQL Hard Filter와 Vector Search(BGE-M3)를 결합한 고성능 검색
- 정밀 랭킹: Cross-Encoder 및 리뷰 긍부정 신호를 활용한 다단계 재정렬(Reranking)
- LoRA 최적화: Exaone-2.4B 모델을 제약 조건 추출에 특화되도록 미세 조정(Fine-tuning)

## 시스템 아키텍처
1. Query Parsing: LLM이 자연어에서 검색 조건을 추출
2. SQL Filtering: DB 레벨에서 명시적 제약 조건(무료, 플랫폼 등) 필터링
3. Vector Retrieval: 의미론적 유사도 기반 후보군 추출
4. Reranking: Cross-Encoder 및 리뷰 피드백 점수 반영
5. Diverse Final Ranking: 결과의 다양성 확보

## 설치 및 실행
1. 의존성 설치:
   pip install -r requirements.txt

2. 환경 설정:
   .env 파일에 LLM API 키 및 DB 경로 설정

3. 서버 실행:
   python main.py --port 8004

## API 명세
- GET /search: 통합 검색 엔드포인트
- POST /recommend: 상세 추천 요청

## 기술 스택
- Backend: FastAPI, SQLAlchemy
- LLM: Exaone (LoRA Fine-tuned)
- Vector DB/Search: BGE-M3, Cosine Similarity
- Database: SQLite (Steam Dataset 2025)
