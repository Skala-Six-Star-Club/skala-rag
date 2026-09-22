"""LangGraph State 정의.

필드명·타입·갱신 방식(덮어쓰기 vs 누적)은 설계서 표와 1:1로 대응시킴.

병렬 관점 노드는 먼저 ``raw_evidence``/``raw_references``에 임시 결과를 누적한다.
최종화 노드가 재시도까지 끝난 뒤 번호를 확정해 ``evidence``/``references``에
기록한다. 이렇게 분리해야 LangGraph의 ``operator.add`` reducer가 최종 결과를
다시 붙여서 중복시키지 않는다.
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
    """수집 중에는 임시 ``key``를, 최종화 후에는 연속 정수 ``id``를 사용한다."""

    id: int | None = None
    key: str | None = None
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
    evidence_keys: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    """ViewResult 내 문장 단위 주장. 반드시 근거 번호를 참조함(10장 중립성 확보)."""

    statement: str
    evidence_ids: list[int] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)


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
    """7.8 synthesize의 서술 결과와 관점별 근거량의 집계값."""

    agreements: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    summary: str = ""
    perspective_confidence: dict[str, dict[str, float]] = Field(default_factory=dict)
    overall_confidence: float = 0.0
    weakest_perspective: str | None = None


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

    # 여러 병렬 노드가 쓰는 누적 영역. evidence_finalize 이후에도 감사/디버깅용으로
    # 남겨 두지만, 보고서와 인용 검증은 아래의 확정된 evidence를 사용한다.
    raw_evidence: Annotated[list[Evidence], operator.add]
    raw_references: Annotated[list[Reference], operator.add]

    # evidence_finalize가 한 번에 기록하는 확정 결과(덮어쓰기 필드).
    evidence: list[Evidence]
    references: list[Reference]
    evidence_finalized: bool

    retry_targets: list[str]
    retry_count: int
    rewrite_count: int

    # evidence_check가 계산한 관점별·기술별 근거량 점수(0~1).
    perspective_confidence: dict[str, dict[str, float]]

    synthesis: Synthesis
    judge_feedback: JudgeFeedback

    report_md: str
    report_path: str
