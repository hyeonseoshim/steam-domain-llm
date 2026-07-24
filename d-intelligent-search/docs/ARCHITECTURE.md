# System Architecture

이 문서는 Steam Part D 추천 시스템의 내부 파이프라인과 기술적 의사결정을 설명합니다.

## 1. 추천 파이프라인 (Recommendation Pipeline)

시스템은 7단계의 순차적/병렬적 과정을 거쳐 최종 결과를 생성합니다.

1) LLM Query Parsing (Stage 1)
   - 입력: 사용자의 자연어 쿼리
   - 출력: 구조화된 제약 조건 (Structured Constraints)
   - 기술: LoRA로 미세 조정된 Exaone 모델

2) SQL Hard Filtering (Stage 2)
   - 역할: DB 인덱스를 활용하여 명시적 조건(가격, 플랫폼 등)을 만족하는 후보군 1차 필터링

3) Vector Retrieval (Stage 3)
   - 역할: BGE-M3 임베딩 모델을 사용하여 쿼리와 게임 설명 간의 의미론적 유사도 계산

4) Cross-Encoder Reranking (Stage 4)
   - 역할: 1차 후보군에 대해 쿼리와의 관련성을 정밀 재계산 (다단계 랭킹의 핵심)

5) Review Signal Fusion (Stage 5)
   - 역할: 스팀 리뷰의 긍정 비율 및 평점을 가중치로 반영하여 추천 품질 향상

6) Semantic Diversity (Stage 6)
   - 역할: 유사한 게임만 추천되는 현상을 방지하기 위한 MMR(Maximal Marginal Relevance) 적용

7) Final Recommendation (Stage 7)
   - 출력: 최종 추천 리스트 및 추천 사유(Reason) 생성

## 2. LLM & LoRA 연동

- 모델: Exaone-2.4B-Instruct
- 최적화: 유저의 검색 쿼리에서 '가격', '장르', '카테고리' 등 8가지 속성을 정확히 추출하기 위해 LoRA(Low-Rank Adaptation) 적용
- 성과: 기본 모델 대비 제약 조건 준수율(Constraint Accuracy) 약 15% 향상

## 3. 데이터 아키텍처 (No DB Changes)

- 원본 데이터셋의 무결성을 유지하기 위해 DB 스키마 변경 없이 In-memory Feature 추출 및 검색 기술을 활용합니다.
