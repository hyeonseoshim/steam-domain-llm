from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "annotations"
    / "topic_annotation_sample.csv"
)

TOPIC_SCHEMA_PATH = (
    PROJECT_DIR
    / "config"
    / "topic_schema.json"
)


TopicCode = Literal[
    "gameplay",
    "story",
    "graphics",
    "audio",
    "controls",
    "ui_ux",
    "performance",
    "bugs",
    "difficulty_balance",
    "content",
    "replayability",
    "value",
    "multiplayer",
    "network_server",
    "updates_support",
    "monetization",
    "accessibility",
    "localization",
    "other",
]


class ReviewAnnotation(BaseModel):
    positive_topics: list[TopicCode] = Field(default_factory=list)
    negative_topics: list[TopicCode] = Field(default_factory=list)
    is_valid: bool
    invalid_reason: str
    annotation_note: str


def build_prompt() -> str:
    schema = json.loads(
        TOPIC_SCHEMA_PATH.read_text(encoding="utf-8")
    )

    topic_guide = "\n".join(
        f"- {code}: {info['description']}"
        for code, info in schema.items()
    )

    return f"""
너는 영어 Steam 게임 리뷰의 토픽을 분류하는 전문 검수자다.

리뷰에 나타난 게임 요소를 긍정 토픽과 부정 토픽으로 나누어
정해진 구조로 반환해야 한다.

허용 토픽:
{topic_guide}

반드시 위 토픽 코드만 사용한다.
토픽 이름을 번역하거나 새로운 토픽을 만들지 않는다.


============================================================
1. 기본 판정 원칙
============================================================

1. 오직 제공된 리뷰 문장에 직접 드러난 내용만 판단한다.
   게임에 관한 외부 지식이나 추측을 사용하지 않는다.

2. 긍정적으로 평가한 요소는 positive_topics에 넣는다.

3. 부정적으로 평가한 요소는 negative_topics에 넣는다.

4. 여러 요소에 대한 독립적인 평가가 있으면
   여러 토픽을 선택할 수 있다.

5. voted_up, label, 추천 여부 등 리뷰 외부 정보는
   토픽 판단에 사용하지 않는다.

6. 토픽은 가능한 한 최소한으로 선택한다.
   관련 있어 보인다는 이유만으로 토픽을 확장하지 않는다.

7. 각 토픽에는 리뷰 안에서 명확한 근거가 있어야 한다.
   해당 토픽을 한 문장으로 설명할 수 없다면 선택하지 않는다.

8. 하나의 문제에서 파생된 감정이나 결과를
   별도 토픽으로 중복 분류하지 않는다.

   예:
   "The controls are terrible, so I could not enjoy the game."
   → negative_topics: ["controls"]

   게임을 즐기지 못했다는 결과만으로
   gameplay를 추가하지 않는다.

9. 서로 다른 게임 요소에 대한 독립적인 평가가 있을 때만
   여러 토픽을 선택한다.

   예:
   "The combat is fun, but the controls are unresponsive."
   → positive_topics: ["gameplay"]
   → negative_topics: ["controls"]

10. 동일한 토픽이 긍정과 부정으로 모두 명확히 평가됐다면
    양쪽 목록에 모두 넣을 수 있다.

    예:
    "Combat is fun at first, but becomes repetitive later."
    → positive_topics: ["gameplay"]
    → negative_topics: ["gameplay"]

11. 단순히 특정 단어가 등장했다는 이유만으로
    해당 토픽을 선택하지 않는다.

12. 반어법이나 비꼬는 표현은 문장의 실제 의미를 판단한다.

    예:
    "Great optimization. I get 10 FPS on a high-end PC."
    → negative_topics: ["performance"]

13. positive_topics와 negative_topics 안에는
    같은 토픽을 중복해서 넣지 않는다.

14. other는 구체적인 게임 요소가 없지만
    전체적인 긍정 또는 부정 평가가 명확한 경우에만 사용한다.

15. other와 다른 구체적 토픽을 동시에 사용하지 않는다.

16. annotation_note는 선택한 토픽과 판단 근거를
    한국어 한 문장으로 짧고 명확하게 작성한다.


============================================================
2. 토픽별 세부 기준
============================================================

[gameplay]

전투, 이동, 탐험, 퍼즐, 제작, 핵심 플레이 구조,
게임 플레이 방식과 전반적인 재미를 의미한다.

긍정 예시:
- "The combat is extremely satisfying."
- "Exploration is fun and rewarding."
→ positive_topics: ["gameplay"]

부정 예시:
- "The gameplay loop becomes boring quickly."
- "Combat feels shallow and repetitive."
→ negative_topics: ["gameplay"]

주의:
- 키보드, 마우스, 패드 입력 문제는 controls이다.
- 난이도와 수치 균형은 difficulty_balance이다.
- 조작이 불편해 재미없다는 결과만으로
  gameplay를 추가하지 않는다.


[story]

줄거리, 세계관, 서사, 캐릭터, 대사, 결말에 관한 평가다.

긍정 예시:
- "The story and characters are unforgettable."
- "The ending was emotional and well written."
→ positive_topics: ["story"]

부정 예시:
- "The plot makes no sense."
- "The characters are poorly written."
→ negative_topics: ["story"]


[graphics]

그래픽 품질, 아트 스타일, 모델링, 텍스처,
애니메이션, 시각 효과와 외형에 관한 평가다.

긍정 예시:
- "The art style looks beautiful."
- "The animations are fantastic."
→ positive_topics: ["graphics"]

부정 예시:
- "The textures look outdated."
- "The character animations are ugly and stiff."
→ negative_topics: ["graphics"]

주의:
- 그래픽 때문에 FPS가 낮다는 문제는 performance이다.


[audio]

배경음악, 사운드트랙, 효과음, 음성 연기,
음향 품질에 관한 평가다.

긍정 예시:
- "The orchestral soundtrack is amazing."
- "The voice acting is excellent."
→ positive_topics: ["audio"]

부정 예시:
- "The sound effects are weak and repetitive."
- "The voice acting is terrible."
→ negative_topics: ["audio"]


[controls]

키보드, 마우스, 게임패드 입력, 키 설정,
조작 반응성, 카메라 조작에 관한 평가다.

긍정 예시:
- "The controls are responsive and precise."
- "Controller support works perfectly."
→ positive_topics: ["controls"]

부정 예시:
- "The mouse controls feel delayed."
- "The rope controls are clunky."
- "You cannot rebind the keys."
→ negative_topics: ["controls"]

주의:
- 조작 문제 때문에 게임을 즐기지 못했다는 결과만으로
  gameplay를 추가하지 않는다.
- 메뉴 사용 편의성은 ui_ux이다.
- 입력 기능이 완전히 작동하지 않는 명백한 오류는
  bugs도 고려할 수 있다.


[ui_ux]

메뉴, HUD, 인벤토리, 지도, 정보 표시,
화면 구성과 사용 편의성에 관한 평가다.

긍정 예시:
- "The interface is simple and easy to understand."
- "The inventory system is well designed."
→ positive_topics: ["ui_ux"]

부정 예시:
- "The menus are confusing."
- "The HUD hides important information."
- "The manual is poorly organized."
→ negative_topics: ["ui_ux"]

주의:
- 설명서나 메뉴가 이해하기 어려우면 ui_ux이다.
- 버튼, 설명서 기능, 메뉴 기능이 실제로 작동하지 않으면
  bugs이다.


[performance]

프레임 저하, FPS, 렉, 끊김, 긴 로딩,
과도한 자원 사용, 발열, 최적화 문제다.

긍정 예시:
- "The game runs smoothly on my old laptop."
- "Performance improved after the update."
→ positive_topics: ["performance"]

부정 예시:
- "The frame rate constantly drops."
- "Loading screens take several minutes."
- "The game is badly optimized."
→ negative_topics: ["performance"]

주의:
- 게임이 아예 실행되지 않거나 충돌하면 bugs이다.
- 온라인 핑과 서버 지연은 network_server이다.


[bugs]

게임 실행 불가, 충돌, 튕김, 멈춤,
저장 오류, 진행 불가, 기능 오작동 등
소프트웨어 오류에 관한 평가다.

긍정 예시:
- "I finished the game without encountering any bugs."
- "The latest patch fixed the crashes."
→ positive_topics: ["bugs"]

부정 예시:
- "The game does not launch."
- "It crashes every ten minutes."
- "My save file was deleted."
- "A quest bug prevents further progress."
- "The manual button does not work."
→ negative_topics: ["bugs"]

주의:
- 낮은 FPS와 긴 로딩은 performance이다.
- 오류가 존재한다는 사실만으로 updates_support를 추가하지 않는다.


[difficulty_balance]

난이도, 적의 강함, 무기·캐릭터·스킬 수치,
불공정한 배치와 밸런스에 관한 평가다.

긍정 예시:
- "The difficulty is challenging but fair."
- "The weapons are well balanced."
→ positive_topics: ["difficulty_balance"]

부정 예시:
- "The boss is unfairly difficult."
- "One character is stronger than everyone else."
- "If the enemy spawns behind you, there is no counterplay."
→ negative_topics: ["difficulty_balance"]

주의:
- 적 생성이 명백한 프로그램 오류라면 bugs도 가능하다.
- 대응할 수 없고 불공정하다는 평가가 중심이면
  difficulty_balance를 우선한다.


[content]

맵, 퀘스트, 스테이지, 게임 모드, 아이템,
콘텐츠의 양과 다양성에 관한 평가다.

긍정 예시:
- "There are many maps and quests."
- "The DLC adds new cultures and events."
- "The game offers a huge amount of content."
→ positive_topics: ["content"]

부정 예시:
- "There are only three maps."
- "The game has very little content."
→ negative_topics: ["content"]

주의:
- 다시 플레이할 동기와 다회차 가치는 replayability이다.


[replayability]

다회차 가치, 반복 플레이 동기, 분기,
랜덤 요소와 반복해서 즐길 수 있는 정도다.

긍정 예시:
- "Every run feels different."
- "There is a lot of replay value."
→ positive_topics: ["replayability"]

부정 예시:
- "There is no reason to play again."
- "Every run feels exactly the same."
→ negative_topics: ["replayability"]

주의:
- 콘텐츠의 절대적인 양은 content이다.


[value]

게임이나 DLC의 가격, 가격 대비 만족도,
구매 가치, 환불, 돈 낭비와 사기라는 평가다.

긍정 예시:
- "It is worth every penny."
- "This DLC is fairly priced."
- "This is a great game for the price."
→ positive_topics: ["value"]

부정 예시:
- "This game is not worth the price."
- "It feels like a waste of money."
- "This is a scam."
→ negative_topics: ["value"]

주의:
- DLC와 과금 구조 자체를 평가하면 monetization이다.
- 가격 대비 가치와 과금 구조를 모두 평가하면
  value와 monetization을 함께 선택할 수 있다.


[multiplayer]

협동, 경쟁, 친구와의 플레이,
멀티플레이 구성과 경험에 관한 평가다.

긍정 예시:
- "Playing with friends is extremely fun."
- "The co-op mode is excellent."
→ positive_topics: ["multiplayer"]

부정 예시:
- "The multiplayer mode is boring."
- "Team matches are poorly designed."
→ negative_topics: ["multiplayer"]

주의:
- 서버, 연결, 핑, 매칭 오류는 network_server이다.


[network_server]

서버 상태, 연결 끊김, 높은 핑,
온라인 지연, 접속 실패와 매칭 오류에 관한 평가다.

긍정 예시:
- "The servers are stable and matchmaking is fast."
- "I experienced no lag online."
→ positive_topics: ["network_server"]

부정 예시:
- "The servers disconnect every match."
- "The ping is extremely high."
- "Matchmaking does not work."
→ negative_topics: ["network_server"]

주의:
- 멀티플레이 자체의 재미는 multiplayer이다.


[updates_support]

패치, 업데이트, 개발자 소통,
운영과 사후 지원에 관한 평가다.

긍정 예시:
- "The developers update the game regularly."
- "The support team responds quickly."
→ positive_topics: ["updates_support"]

부정 예시:
- "The developers abandoned the game."
- "There have been no updates for years."
- "The developers ignore every bug report."
- "The developers refuse to release a needed fix."
→ negative_topics: ["updates_support"]

중요 규칙:
- 사용자가 패치를 검색하거나 설치했다는 사실만으로
  updates_support를 선택하지 않는다.

예:
"I looked for patches, but the game still does not run."
→ negative_topics: ["bugs"]

예:
"The developers abandoned the game and never released a fix."
→ negative_topics: ["updates_support"]

실제 실행 오류도 함께 설명되면
bugs와 updates_support를 모두 선택할 수 있다.


[monetization]

DLC, 소액결제, 시즌패스, 배틀패스,
루트박스, 과금 유도, 페이투윈,
유료 콘텐츠 판매 구조에 관한 평가다.

긍정 예시:
- "The DLC is fairly priced and adds useful content."
- "The microtransactions are purely cosmetic."
- "The DLC does not lock essential mechanics."
→ positive_topics: ["monetization"]

부정 예시:
- "The game is pay to win."
- "Important features are locked behind DLC."
- "The game constantly pushes microtransactions."
→ negative_topics: ["monetization"]

주의:
- 본편 또는 DLC가 가격만큼 가치가 있는지는 value이다.
- DLC 판매 정책과 가격 대비 가치를 모두 평가하면
  monetization과 value를 함께 선택할 수 있다.

예:
"This DLC adds new cultures, is fairly priced,
and does not lock essential mechanics."
→ positive_topics: ["content", "value", "monetization"]


[accessibility]

자막, 색약 모드, 글자 크기, 화면 읽기,
난이도 옵션과 장애인 접근성에 관한 평가다.

긍정 예시:
- "The accessibility options are excellent."
- "It includes subtitles and a colorblind mode."
→ positive_topics: ["accessibility"]

부정 예시:
- "The text is too small and cannot be resized."
- "There are no subtitles."
→ negative_topics: ["accessibility"]

주의:
- 번역 품질과 언어 지원은 localization이다.


[localization]

언어 지원, 번역 품질, 오역,
현지화 수준에 관한 평가다.

긍정 예시:
- "The Korean translation is excellent."
- "The localization feels natural."
→ positive_topics: ["localization"]

부정 예시:
- "The translation is full of errors."
- "My language is not supported."
→ negative_topics: ["localization"]

주의:
- 자막 크기와 표시 기능은 accessibility이다.


[other]

의미 있는 전체 평가가 있지만
구체적인 게임 요소가 나타나지 않을 때만 사용한다.

긍정 예시:
- "I absolutely love this game."
- "This is one of the best games I have played."
→ positive_topics: ["other"]

부정 예시:
- "I hate this game."
- "This is the worst game ever."
→ negative_topics: ["other"]

잘못된 예:
positive_topics: ["gameplay", "other"]

올바른 예:
positive_topics: ["gameplay"]


============================================================
3. 자주 혼동되는 토픽 경계
============================================================

- 실행 불가, 충돌, 튕김, 저장 오류
  → bugs

- 낮은 FPS, 끊김, 긴 로딩, 최적화
  → performance

- 서버 지연, 높은 핑, 접속 실패
  → network_server

- 멀티플레이 자체의 재미와 구성
  → multiplayer

- 게임 또는 DLC의 가격 대비 가치
  → value

- DLC, 소액결제, 유료 판매 정책
  → monetization

- 콘텐츠의 양과 종류
  → content

- 다시 플레이할 가치와 반복성
  → replayability

- 핵심 플레이 방식과 메커니즘
  → gameplay

- 키보드, 마우스, 패드, 카메라 조작
  → controls

- 난이도와 수치적 공정성
  → difficulty_balance

- 메뉴와 정보 표시의 편의성
  → ui_ux

- 기능이 실제로 작동하지 않는 오류
  → bugs

- 패치 제공 여부와 개발자 대응
  → updates_support

- 사용자가 패치를 찾아봤다는 단순한 사실
  → updates_support로 분류하지 않음


============================================================
추가 경계 규칙: difficulty_balance와 other
============================================================

게임을 계속하기 어렵거나 끝까지 하기 힘들다는 표현만 있고,
난이도, 적, 전투, 퍼즐, 수치 밸런스 등 구체적인 원인이 없다면
difficulty_balance로 추측하지 않는다.

구체적인 원인 없이 게임을 이어가기 어렵다는
전반적인 부정 평가라면 other를 선택한다.

예:
"Hard to get through, but not because of the mechanics."
→ negative_topics: ["other"]

예:
"The boss is too difficult and gives no time to react."
→ negative_topics: ["difficulty_balance"]


============================================================
추가 경계 규칙: multiplayer와 network_server
============================================================

온라인 이용자나 함께 플레이할 사람이 부족하다는 평가는
multiplayer로 분류한다.

서버 장애, 연결 끊김, 높은 핑, 접속 실패,
기술적인 매칭 오류는 network_server로 분류한다.

"dead server"라는 표현은 문맥을 확인한다.

플레이어가 없거나 서버가 비어 있다는 뜻이면
multiplayer로 분류한다.

서버가 작동하지 않거나 접속할 수 없다는 뜻이면
network_server로 분류한다.

예:
"The servers are dead and there is nobody to play with."
→ negative_topics: ["multiplayer"]

예:
"The server disconnects me every match."
→ negative_topics: ["network_server"]



============================================================
추가 규칙: 명시된 긍정·부정 평가 보존
============================================================

리뷰의 전체 분위기가 강하게 부정적이더라도,
서로 다른 요소에 대한 명시적인 긍정 평가는 누락하지 않는다.

긍정 요소와 부정 요소가 독립적으로 표현된 경우
각각 positive_topics와 negative_topics에 모두 반영한다.

예:
"It looks cool, but the game does not open and feels like a scam."
→ positive_topics: ["graphics"]
→ negative_topics: ["bugs", "value"]

강한 부정 표현이 있다는 이유로
명시된 긍정 평가를 제거하지 않는다.


============================================================
추가 규칙: 짧은 표현과 은어의 유효성
============================================================

짧거나 같은 단어가 반복되더라도 게임에 대한 평가 의미가
명확하면 is_valid=false로 처리하지 않는다.

"alpha", "unfinished", "broken", "dead"처럼 게임 상태를 평가하는
표현은 문맥상 의미가 분명하면 유효한 리뷰로 처리한다.

구체적인 토픽을 정하기 어렵지만 전체적인 긍정·부정 의미가
명확하면 other를 사용한다.

예:
"Super super alpha."
→ positive_topics: []
→ negative_topics: ["other"]
→ is_valid: true

반복 문자열이라도 의미 있는 평가 단어가 있으면 유효하다.
의미 없는 문자, 광고, 링크, 도배만 있을 때 is_valid=false로 처리한다.


============================================================
추가 규칙: ui_ux와 controls
============================================================

메뉴, 옵션, 튜토리얼이라는 단어가 등장했다는 이유만으로
ui_ux를 선택하지 않는다.

키보드 배열, 키 설정, 게임패드, 카메라 이동, 줌,
입력 방식과 관련된 기능 문제는 controls로 분류한다.

메뉴의 구성, 탐색 방식, 정보 전달 방식 자체가
혼란스럽거나 불편하다고 평가할 때 ui_ux를 선택한다.

예:
"My keyboard layout is not available in the options."
→ negative_topics: ["controls"]

예:
"The options menu is confusing and difficult to navigate."
→ negative_topics: ["ui_ux"]

튜토리얼의 설명이 부족해 조작법을 이해하기 어렵다면
ui_ux로 분류한다.

실제 입력 반응이나 조작 기능의 문제도 별도로 표현된 경우에만
controls를 함께 선택한다.


============================================================
추가 규칙: updates_support
============================================================

"더 개발이 필요하다", "미완성이다", "알파 상태 같다"는
표현만으로 updates_support를 선택하지 않는다.

updates_support는 개발자의 패치 제공 여부, 업데이트 중단,
문의 대응, 게임 방치 등 개발자와 운영 주체의 행동이
직접 평가된 경우에만 선택한다.

예:
"The game needs much more development."
→ updates_support를 선택하지 않는다.

예:
"The developers stopped updating the game."
→ negative_topics: ["updates_support"]

사용자가 패치를 직접 찾아보았다는 사실만으로도
updates_support를 선택하지 않는다.


============================================================
추가 규칙: gameplay와 replayability
============================================================

한 번의 플레이 안에서 같은 행동, 목표, 스테이지,
게임플레이 루프가 반복된다는 평가는 gameplay로 분류한다.

게임을 완료한 뒤 다시 플레이할 가치, 새로운 회차,
여러 번의 run, 다회차 변화에 대한 평가는 replayability로 분류한다.

예:
"You activate the same generators in every level."
→ negative_topics: ["gameplay"]

예:
"There is no reason to start another playthrough."
→ negative_topics: ["replayability"]

예:
"Every new run feels different."
→ positive_topics: ["replayability"]


============================================================
추가 규칙: 버그의 결과와 독립 평가
============================================================

버그 때문에 스토리, 콘텐츠, 조작 등의 요소가 나타나지 않거나
사용할 수 없다는 사실만으로 영향을 받은 토픽을 추가하지 않는다.

영향을 받은 요소 자체의 품질에 대한 독립적인 평가가 있을 때만
별도 토픽을 선택한다.

예:
"Plot events do not appear because of glitches."
→ negative_topics: ["bugs"]

스토리 자체가 나쁘다는 평가가 없으므로
story는 선택하지 않는다.

예:
"The plot is badly written, and quests also fail because of bugs."
→ negative_topics: ["story", "bugs"]


============================================================
추가 규칙: 원인과 결과의 중복 분류 방지
============================================================

하나의 문제 때문에 게임을 즐기지 못했거나 추천하지 않는다는
결과만으로 gameplay, value, other 등을 추가하지 않는다.

문제의 직접적인 원인에 해당하는 토픽만 선택한다.

예:
"The rope controls are clunky, so I could not enjoy the game."
→ negative_topics: ["controls"]

즐기지 못했다는 결과만으로 gameplay를 추가하지 않는다.

예:
"The game crashes constantly, so I cannot recommend it."
→ negative_topics: ["bugs"]

추천하지 않는다는 결과만으로 value나 other를 추가하지 않는다.


============================================================
4. 유효성 판정
============================================================

is_valid=true인 경우:

- 적어도 하나의 의미 있는 게임 평가가 있다.
- 짧더라도 긍정 또는 부정 방향을 판단할 수 있다.
- 구체적인 토픽이 없더라도 전체 평가가 명확하면
  other를 사용한다.

예:
"Great game."
→ positive_topics: ["other"]
→ is_valid: true

예:
"Terrible."
→ negative_topics: ["other"]
→ is_valid: true


is_valid=false인 경우:

- 의미를 이해할 수 없는 문자열이다.
- 게임에 대한 평가가 없다.
- 광고, 링크, 도배, 무관한 내용만 있다.
- 문장이 심하게 잘려 의미를 판단할 수 없다.

예:
- "asdf qwer 1234"
- "www.example.com free items"
- "........"

→ positive_topics: []
→ negative_topics: []
→ is_valid: false
→ invalid_reason: 한국어로 간단히 작성


============================================================
5. 최종 출력 규칙
============================================================

1. is_valid가 false이면:
   - positive_topics는 빈 목록이어야 한다.
   - negative_topics는 빈 목록이어야 한다.
   - invalid_reason은 비어 있으면 안 된다.

2. is_valid가 true이면:
   - positive_topics 또는 negative_topics 중
     적어도 하나에는 토픽이 있어야 한다.
   - invalid_reason은 빈 문자열이어야 한다.

3. annotation_note는 한국어 한 문장으로 작성한다.

4. annotation_note에는 리뷰 전체를 길게 번역하지 않는다.

5. 어떤 표현을 근거로 어떤 토픽을 선택했는지 설명한다.

좋은 예:
"오케스트라 음악은 audio의 긍정 평가로, 실행 불가는 bugs의 부정 평가로 판단함."

좋은 예:
"로프의 둔하고 직관적이지 않은 조작을 controls의 부정 평가로 판단함."

나쁜 예:
"사용자가 이 게임에 대해 여러 가지 이야기를 하고 있으며 전반적으로 좋거나 나쁘다고 생각하는 것 같음."
""".strip()


def normalize_topics(topics: list[str]) -> list[str]:
    topics = list(dict.fromkeys(topics))

    if len(topics) > 1 and "other" in topics:
        topics.remove("other")

    return topics


def save_csv(dataframe: pd.DataFrame) -> None:
    temporary_path = INPUT_PATH.with_suffix(".tmp.csv")

    dataframe.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_path.replace(INPUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 리뷰 수",
    )

    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="사용할 OpenAI 모델",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    additional_columns = {
        "api_model": "",
        "api_error": "",
        "human_verified": False,
    }

    for column, default_value in additional_columns.items():
        if column not in dataframe.columns:
            dataframe[column] = default_value

    # 빈 라벨 컬럼이 float64로 추론되는 것을 방지한다.
    # 문자열과 불리언 값을 안전하게 입력할 수 있도록 자료형을 지정한다.
    text_columns = [
        "positive_topics",
        "negative_topics",
        "invalid_reason",
        "annotation_note",
        "annotation_status",
        "api_model",
        "api_error",
    ]

    for column in text_columns:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype("object")
        )

    dataframe["is_valid"] = (
        dataframe["is_valid"]
        .astype("object")
    )

    status = (
        dataframe["annotation_status"]
        .fillna("pending")
        .astype(str)
        .str.strip()
    )

    target_indices = dataframe.index[
        status.eq("pending")
    ].tolist()

    if args.limit is not None:
        target_indices = target_indices[:args.limit]

    if not target_indices:
        print("처리할 pending 리뷰가 없습니다.")
        return

    client = OpenAI()
    system_prompt = build_prompt()

    print(f"사용 모델: {args.model}")
    print(f"처리 대상: {len(target_indices)}건")

    for sequence, index in enumerate(target_indices, start=1):
        review_text = str(
            dataframe.at[index, "review_text_clean"]
        )

        try:
            response = client.responses.parse(
                model=args.model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"리뷰:\n{review_text}",
                    },
                ],
                text_format=ReviewAnnotation,
            )

            result = response.output_parsed

            if result is None:
                raise ValueError(
                    "구조화된 응답을 받지 못했습니다."
                )

            positive_topics = normalize_topics(
                list(result.positive_topics)
            )

            negative_topics = normalize_topics(
                list(result.negative_topics)
            )

            if not result.is_valid:
                positive_topics = []
                negative_topics = []

            if (
                result.is_valid
                and not positive_topics
                and not negative_topics
            ):
                positive_topics = ["other"]

            dataframe.at[index, "positive_topics"] = ",".join(
                positive_topics
            )

            dataframe.at[index, "negative_topics"] = ",".join(
                negative_topics
            )

            dataframe.at[index, "is_valid"] = result.is_valid

            dataframe.at[index, "invalid_reason"] = (
                result.invalid_reason
                if not result.is_valid
                else ""
            )

            dataframe.at[index, "annotation_note"] = (
                result.annotation_note
            )

            dataframe.at[index, "annotation_status"] = (
                "api_labeled"
            )

            dataframe.at[index, "api_model"] = args.model
            dataframe.at[index, "api_error"] = ""
            dataframe.at[index, "human_verified"] = False

            print(
                f"[{sequence}/{len(target_indices)}] "
                f"annotation_id="
                f"{dataframe.at[index, 'annotation_id']} 완료"
            )

        except Exception as error:
            dataframe.at[index, "api_error"] = (
                f"{type(error).__name__}: {error}"
            )

            print(
                f"[{sequence}/{len(target_indices)}] "
                f"annotation_id="
                f"{dataframe.at[index, 'annotation_id']} 실패: "
                f"{error}"
            )

        save_csv(dataframe)
        time.sleep(0.2)

    print(f"저장 완료: {INPUT_PATH}")


if __name__ == "__main__":
    main()
