"""LangGraph State 정의. docs/agentic-rag-design.md 11장 State 표를 그대로 구현함.

필드명·타입·갱신 방식(덮어쓰기 vs 누적)은 설계서 표와 1:1로 대응시킴.
evidence/references만 여러 노드가 동시에 쓰므로 operator.add reducer를 둠(12장,
INVALID_CONCURRENT_GRAPH_UPDATE 회피).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Camp = Literal["SW", "HW"]
Role = Literal["target", "comparison"]
Stance = Literal["지지", "반대"]
SourceType = Literal["논문", "웹"]
Perspective = Literal["tech_research", "trl", "market", "stakeholder", "domain"]


class TechSpec(BaseModel):
    """3장: 조가 설정 파일에서 읽어 오는 대상 기술 정의."""

    name: str
    camp: Camp
    role: Role
    search_anchor: str  # market_eval/stakeholder_eval 웹 검색용 앵커 키워드 (7.1, 7.5)


class Evidence(BaseModel):
    """11장: 번호, 기술, 관점, 입장, 출처 유형, 출처, 인용문."""

    id: int
    tech: str
    perspective: Perspective
    stance: Stance
    source_type: SourceType
    source: str  # 논문명/절/쪽 또는 URL
    quote: str


class Reference(BaseModel):
    """13장 REFERENCE 표기 형식(특허/논문/웹페이지)에 대응."""

    id: int
    type: Literal["patent", "paper", "web"]
    author_or_org: str
    year: str
    title: str
    venue: str | None = None  # 학술지, 사이트명 등
    url: str | None = None


class TechProfile(BaseModel):
    """7.2 tech_research 출력. 원문 청크가 아닌 구조화 결과만 저장함."""

    tech: str
    overview: str
    scope: str
    limitations: str
    differentiation: str  # 같은 진영 다른 방식과의 차이
    evidence_ids: list[int] = Field(default_factory=list)


class Claim(BaseModel):
    """ViewResult 내 문장 단위 주장. 반드시 근거 번호를 참조함(10장 중립성 확보)."""

    statement: str
    evidence_ids: list[int] = Field(default_factory=list)


class TechViewResult(BaseModel):
    """기술 하나에 대한 관점 평가 결과(확인된 사실/반대 사실/미확인 항목)."""

    confirmed_facts: list[Claim] = Field(default_factory=list)
    counter_facts: list[Claim] = Field(default_factory=list)
    unconfirmed_items: list[str] = Field(default_factory=list)


class ViewResult(BaseModel):
    """trl_result/market_result/stakeholder_result/domain_result 공통 형태.
    기술명 -> TechViewResult로 두 기술을 나란히 담음."""

    by_tech: dict[str, TechViewResult] = Field(default_factory=dict)


class Conflict(BaseModel):
    topic: str
    explanation: str  # 왜 갈리는지(단순 나열 금지, 7.8)


class Synthesis(BaseModel):
    """7.8 synthesize 출력. 일치점, 상충점, SUMMARY."""

    agreements: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    summary: str = ""


class JudgeFeedback(BaseModel):
    """7.9 judge 출력. 점수화하지 않고 예/아니오 + 비고만 둠."""

    has_biased_expression: bool
    sentences_without_evidence: list[str] = Field(default_factory=list)
    is_balanced: bool
    notes: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph 그래프 전체가 공유하는 State (11장 표)."""

    techs: list[TechSpec]
    domain: str
    tech_profiles: dict[str, TechProfile]

    trl_result: ViewResult
    market_result: ViewResult
    stakeholder_result: ViewResult
    domain_result: ViewResult

    evidence: Annotated[list[Evidence], operator.add]
    references: Annotated[list[Reference], operator.add]

    retry_targets: list[str]
    retry_count: int

    synthesis: Synthesis
    # 12장 "반복 2"(judge -> synthesize 재작성, 1회 한정)의 횟수. 11장 표에는 없지만
    # retry_count와 같은 역할이라 추가함. synthesize가 쓰고 그래프 조건 분기가 읽음.
    rewrite_count: int
    judge_feedback: JudgeFeedback

    report_md: str
    report_path: str
    report_json_path: str
