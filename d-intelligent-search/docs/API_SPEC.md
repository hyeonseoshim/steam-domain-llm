# API Specification

## 1. GET /search
통합 추천 서비스의 표준 엔드포인트입니다.

### Request
- q (string, 필수): 자연어 검색 쿼리
- k (int, 기본값 30): 최대 결과 수

### Response (JSON)
{
  "results": [
    {
      "appid": 10,
      "name": "Counter-Strike",
      "score": 0.952,
      "reason": "정통 FPS의 정수를 경험할 수 있는 팀 기반 전술 게임입니다."
    }
  ],
  "note": "추출 조건: 무료 / Multi-player"
}

## 2. GET /health
시스템 상태 및 모델/DB 연결 상태를 확인합니다.
