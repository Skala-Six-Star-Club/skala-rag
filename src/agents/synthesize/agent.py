"""평가 종합 에이전트 (7.8절 + 확장).

관점 결과 4종을 종합하고 evidence_check가 계산한 관점별 근거 신뢰도를
결정론적으로 집계한다. 신뢰도 수치는 기술 간 우열이 아니라 관점별 근거량을
설명하는 값으로만 사용한다.

입력: state의 관점 결과 4종, perspective_confidence(+ 재작성 시 judge_feedback)
출력: {"synthesis": ...}

judge(7.9)에서 반려되면 judge_feedback을 프롬프트에 추가해 동일 입력으로 1회
재실행함(12장 "반복 2"). 재작성 시에도 이전 synthesis 초안은 프롬프트에 남기지
않고 매번 관점 결과 4종에서 새로 작성함(부분 수정이 아닌 답습 위험 회피).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.common.models import get_generation_llm
from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Conflict, Synthesis

_PROMPT_TEMPLATE = """\
아래는 TurboQuant(SW)와 ITME(HW) 두 기술에 대한 4개 관점(기술 성숙도, 시장성,
이해관계자, 도메인 적용) 평가 결과임. 이를 종합해:
- 일치점 1건 이상
- 상충점 1건 이상(단순 나열이 아니라 왜 갈리는지 설명 포함)
- 1/2쪽 이내 SUMMARY (관점 4종을 한 줄씩 언급, 가장 큰 상충 지점 명시)
를 작성해줘. 우열 판정 표현은 쓰지 말고, 모든 문장에 근거 번호를 붙여줘.

[기술 성숙도] {trl_result}
[시장성] {market_result}
[이해관계자] {stakeholder_result}
[도메인 적용] {domain_result}

[관점별 근거 신뢰도(0~1, evidence_check가 근거량 기준으로 계산함)]
{confidence}
위 수치는 "이 관점에서 수집된 근거가 얼마나 충분한가"를 나타낼 뿐, 기술 우열과
무관함. 특정 관점의 신뢰도가 낮으면 그 관점의 결론은 잠정적임을 서술에 반영하되,
두 기술을 비교해 우열을 매기는 데는 쓰지 말 것.
"""


class _SynthesisNarrative(BaseModel):
    """LLM 구조화 출력용 서술 스키마.

    동적 dictionary인 perspective_confidence는 LLM Structured Outputs에 직접
    요청하지 않고, 코드가 Synthesis에 별도로 채운다.
    """

    agreements: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    summary: str = ""


def _aggregate_confidence(
    confidence: dict[str, dict[str, float]],
) -> tuple[float, str | None]:
    """관점별 신뢰도를 평균 내어 전체 평균과 최약 관점을 반환한다."""

    if not confidence:
        return 0.0, None

    per_perspective_avg: dict[str, float] = {}
    all_scores: list[float] = []
    for perspective, by_tech in confidence.items():
        scores = list(by_tech.values())
        if not scores:
            continue
        per_perspective_avg[perspective] = sum(scores) / len(scores)
        all_scores.extend(scores)

    if not all_scores:
        return 0.0, None

    overall = round(sum(all_scores) / len(all_scores), 2)
    weakest = min(per_perspective_avg, key=per_perspective_avg.get)
    return overall, weakest


class SynthesizeAgent(BaseAgent):
    name = "synthesize"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        confidence = state.get("perspective_confidence", {})
        overall_confidence, weakest_perspective = _aggregate_confidence(confidence)

        prompt = _PROMPT_TEMPLATE.format(
            trl_result=state.get("trl_result"),
            market_result=state.get("market_result"),
            stakeholder_result=state.get("stakeholder_result"),
            domain_result=state.get("domain_result"),
            confidence=confidence,
        )

        feedback = state.get("judge_feedback")
        if feedback is not None:
            prompt += f"\n[이전 검수에서 지적된 사항, 반드시 반영할 것]\n{feedback}"

        # 동적 confidence dictionary는 코드에서 조립하고, LLM에는 고정 스키마인
        # 서술 부분만 요청한다.
        llm = get_generation_llm().with_structured_output(_SynthesisNarrative)
        narrative: _SynthesisNarrative = llm.invoke(prompt)  # type: ignore[assignment]
        synthesis = Synthesis(
            agreements=narrative.agreements,
            conflicts=narrative.conflicts,
            summary=narrative.summary,
            perspective_confidence=confidence,
            overall_confidence=overall_confidence,
            weakest_perspective=weakest_perspective,
        )

        rewrite_count = state.get("rewrite_count", 0)
        if feedback is not None and rewrite_count < 1:
            rewrite_count += 1
        return {"synthesis": synthesis, "rewrite_count": rewrite_count}
