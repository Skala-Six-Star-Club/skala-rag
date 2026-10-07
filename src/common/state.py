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
    # 이 근거가 속한 참고문헌의 URL(논문은 arXiv abs, 웹은 페이지 URL). report(7.10)가
    # 본문에 인용된 근거만 골라 REFERENCE를 만들 때 Reference.url과 이 값을 맞춰 봄.
    reference_url: str | None = None


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
    # True면 reducer가 by_tech의 해당 기술을 통째로 교체함(evidence_check 그라운딩 필터,
    # evidence_finalize 번호 확정처럼 이미 병합된 결과를 다시 쓰는 경우). False(worker
    # 출력)면 같은 기술의 기존 결과 뒤에 이어 붙임.
    replace: bool = False


def _append_tech_view(left: TechViewResult, right: TechViewResult) -> TechViewResult:
    def _claims(a: list[Claim], b: list[Claim]) -> list[Claim]:
        seen = {(c.statement, tuple(c.evidence_keys), tuple(c.evidence_ids)) for c in a}
        return [*a, *(c for c in b if (c.statement, tuple(c.evidence_keys), tuple(c.evidence_ids)) not in seen)]

    return TechViewResult(
        confirmed_facts=_claims(left.confirmed_facts, right.confirmed_facts),
        counter_facts=_claims(left.counter_facts, right.counter_facts),
        unconfirmed_items=list(dict.fromkeys([*left.unconfirmed_items, *right.unconfirmed_items])),
    )


def merge_view_results(left: ViewResult | None, right: ViewResult | None) -> ViewResult:
    """관점 결과 reducer.

    orchestrator가 같은 (관점, 기술)에 초점이 다른 서브 태스크를 여러 개 만들면 같은
    superstep에서 같은 기술 키로 결과가 여러 번 들어오므로, 기술 키가 겹치면 덮어쓰지
    않고 이어 붙임. ``replace=True``인 갱신(그라운딩 필터, 번호 확정)만 교체함.
    """
    if right is None:
        return left if left is not None else ViewResult()
    right = right if isinstance(right, ViewResult) else ViewResult.model_validate(right)
    if left is None:
        return ViewResult(by_tech=dict(right.by_tech))
    left = left if isinstance(left, ViewResult) else ViewResult.model_validate(left)
    merged = dict(left.by_tech)
    for tech, view in right.by_tech.items():
        if right.replace or tech not in merged:
            merged[tech] = view
        else:
            merged[tech] = _append_tech_view(merged[tech], view)
    return ViewResult(by_tech=merged)


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


ViewPerspective = Literal["trl", "market", "stakeholder", "domain"]


class SubTask(BaseModel):
    """orchestrator가 만드는 worker 실행 단위. (관점, 기술, 초점) 하나가 Send 하나임.
    focus는 src/common/focus.py 카탈로그의 초점 id(관점 필수 항목 하나에 대응)."""

    subtask_id: str
    perspective: ViewPerspective
    tech: str
    focus: str
    round: int = 0  # 0: 최초 계획, 1 이상: re-plan
    reason: str = ""

    @property
    def node(self) -> str:
        return f"{self.perspective}_eval"


class Gap(BaseModel):
    """re-plan 요청 한 건. evidence_check(근거 부족)나 quality_eval(커버리지, 편향 미달)이 씀."""

    perspective: ViewPerspective
    tech: str
    focus: str | None = None  # None이면 orchestrator가 그 칸에서 아직 조사하지 않은 초점을 고름
    reason: str = ""
    source: Literal["evidence_check", "quality_eval"] = "evidence_check"


class Plan(BaseModel):
    """orchestrator 출력. 이번 round에 실행할 서브 태스크 목록과 계획 사유."""

    round: int = 0
    subtasks: list[SubTask] = Field(default_factory=list)
    rationale: str = ""
    # 계획 검증이 LLM 계획을 고친 내역(빈 칸 보정, 대칭 보정, 상한 조정 등)
    corrections: list[str] = Field(default_factory=list)


Criterion = Literal["groundedness", "neutrality", "bias_control", "coverage"]


class CriterionResult(BaseModel):
    name: Criterion
    passed: bool
    method: Literal["rule", "llm", "hybrid"]
    issues: list[str] = Field(default_factory=list)


class EvalVerdict(BaseModel):
    """quality_eval 출력. 항목별 판정과 미달 시 되돌아갈 노드."""

    passed: bool
    criteria: list[CriterionResult] = Field(default_factory=list)
    # 커버리지, 편향 미달은 orchestrator에 넘길 부족 칸
    gaps: list[Gap] = Field(default_factory=list)
    route: Literal["synthesize", "orchestrator", "end"] = "end"
    attempt: int = 0

    def issues_for(self, *names: str) -> list[str]:
        return [i for c in self.criteria if c.name in names and not c.passed for i in c.issues]


class AgentState(TypedDict, total=False):
    """LangGraph 그래프 전체가 공유하는 State (11장 표)."""

    techs: list[TechSpec]
    domain: str
    tech_profiles: dict[str, TechProfile]

    # (관점, 기술) 단위 병렬 실행이 같은 관점 필드에 동시에 쓰므로 기술 키로 합침
    trl_result: Annotated[ViewResult, merge_view_results]
    market_result: Annotated[ViewResult, merge_view_results]
    stakeholder_result: Annotated[ViewResult, merge_view_results]
    domain_result: Annotated[ViewResult, merge_view_results]

    # Send fan-out으로 관점 노드를 호출할 때 그래프가 넣어 주는 실행 범위(기술명 하나).
    # 없으면 노드는 techs 전체를 처리함(독립 실행 스크립트, 레거시 호환).
    tech_scope: str

    # 여러 병렬 노드가 쓰는 누적 영역. evidence_finalize 이후에도 감사/디버깅용으로
    # 남겨 두지만, 보고서와 인용 검증은 아래의 확정된 evidence를 사용한다.
    raw_evidence: Annotated[list[Evidence], operator.add]
    raw_references: Annotated[list[Reference], operator.add]

    # evidence_finalize가 한 번에 기록하는 확정 결과(덮어쓰기 필드).
    evidence: list[Evidence]
    references: list[Reference]
    evidence_finalized: bool

    retry_targets: list[str]
    # 재검색을 기술 단위로 좁히기 위한 범위: 노드 이름 -> 부족한 기술명 목록.
    # evidence_check가 채우고 graph.py가 (노드, 기술)별 Send로 씀. 비어 있으면 전체 기술.
    retry_scopes: dict[str, list[str]]
    retry_count: int
    rewrite_count: int

    # evidence_check가 계산한 관점별·기술별 근거량 점수(0~1).
    perspective_confidence: dict[str, dict[str, float]]

    synthesis: Synthesis
    # 12장 "반복 2"(judge -> synthesize 재작성, 1회 한정)의 횟수. 11장 표에는 없지만
    # retry_count와 같은 역할이라 추가함. synthesize가 쓰고 그래프 조건 분기가 읽음.
    rewrite_count: int
    judge_feedback: JudgeFeedback

    report_md: str
    report_path: str
    report_json_path: str
