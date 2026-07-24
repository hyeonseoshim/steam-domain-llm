# Presentation Slide Outline

## 슬라이드 1: 타이틀
- 프로젝트명: Steam Game Recommendation System (Part D)
- 부제: LLM과 하이브리드 검색을 결합한 지능형 추천 파이프라인

## 슬라이드 2: 배경 및 목표
- 문제제기: 수많은 스팀 게임 중 사용자의 복잡한 의도를 반영한 추천의 어려움
- 목표: 자연어 이해 기반의 정밀한 게임 추천 서비스 구축

## 슬라이드 3: 전체 아키텍처
- 시각 자료: Stage 1~7 파이프라인 흐름도
- 핵심 가치: LLM 제약 추출 + SQL 필터 + 벡터 검색 + 재정렬

## 슬라이드 4: 기술적 특징 (LLM)
- Exaone 모델 미세 조정 (LoRA)
- 제약 조건(Constraints) 추출의 정확도 확보

## 슬라이드 5: 기술적 특징 (Retrieval & Reranking)
- BGE-M3를 활용한 의미론적 검색
- Cross-Encoder를 통한 랭킹 정밀도 향상
- 리뷰 신호(Sentiment) 통합

## 슬라이드 6: 성능 평가 결과
- 골든셋 500개 쿼리 기반 평가 지표 (Hit Rate, Accuracy)
- LoRA 적용 전후 성능 비교 데이터

## 슬라이드 7: 결론 및 향후 계획
- 성과 요약: DB 변경 없이 구현된 지능형 추천 레이어
- 향후 계획: 실시간 피드백 반영 시스템 구축
