"""평가 종합 에이전트 (7.8절 + 확장). 관점 결과 4종을 종합해 일치점/상충점/SUMMARY를
생성하고, evidence_check(7.7)가 계산한 근거 신뢰도를 하나의 통합 구조(Synthesis)로
가중 집계함. 추가 검색 없는 순수 생성 작업 + 코드 집계.

입력: state의 관점 결과 4종 + perspective_confidence(+ 재작성 시 judge_feedback)
출력: {"synthesis": ...}

judge(7.9)에서 반려되면 judge_feedback을 프롬프트에 추가해 동일 입력으로 1회
재실행함(12장 "반복 2"). 재작성 시에도 이전 synthesis 초안은 프롬프트에 남기지
않고 매번 관점 결과 4종에서 새로 작성함(부분 수정이 아닌 답습 위험 회피).

확장 — 근거 검증 & 데이터 종합 역할(팀원 4, 신소영):
evidence_check가 관점별·기술별로 계산한 perspective_confidence(0~1, 근거량 기반)를
그대로 받아 아래 두 값을 코드로 결정론적으로 집계함(LLM 판단 아님, 재현 가능):
- overall_confidence: 전체 평균 — 이번 조사가 전반적으로 근거가 탄탄한지
- weakest_perspective: 평균이 가장 낮은 관점 — 한계점 절에서 "근거가 상대적으로
  부족한 관점"을 구체적으로 짚을 때 씀
이 값들은 기술 간 우열이 아니라 "관점(TRL/시장성/이해관계자/도메인)" 단위의 근거
탄탄함을 나타내므로 10장 중립성 원칙과 충돌하지 않음. LLM에도 이 수치를 함께
넘기되, 프롬프트에서 "기술 비교가 아니라 관점별 근거량 설명에만 쓸 것"을 명시해
judge(7.9)의 우열 판정 표현 검사를 통과하도록 함.
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
    """LLM 구조화 출력 전용 스키마. Synthesis에서 서술 부분(agreements/conflicts/
    summary)만 떼어 씀 — perspective_confidence(dict[str, dict[str, float]])처럼
    키가 동적으로 정해지는 필드는 OpenAI Structured Outputs가 지원하지 않아
    (모든 필드가 고정된 properties/required를 가져야 함), 그 필드들은 LLM에게
    아예 요청하지 않고 코드가 별도로 계산해 Synthesis에 채워 넣음(아래 run()).
    """

    agreements: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    summary: str = ""


def _aggregate_confidence(
    confidence: dict[str, dict[str, float]],
) -> tuple[float, str | None]:
    """관점별 평균 신뢰도를 계산해 (전체 평균, 최약 관점)을 반환함. 결정론적 집계."""
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
    weakest = min(per_perspective_avg, key=per_perspective_avg.get) if per_perspective_avg else None
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
        is_rewrite = feedback is not None
        if is_rewrite:
            prompt += f"\n[이전 검수에서 지적된 사항, 반드시 반영할 것]\n{feedback}"

        # TODO(담당자): 프롬프트 문구를 다듬을 것(예: 관점 결과를 dict 그대로
        # 넣지 말고 사람이 읽기 좋은 형태로 직렬화). 구조화 출력 호출 자체는 동작함.
        llm = get_generation_llm().with_structured_output(_SynthesisNarrative)
        narrative: _SynthesisNarrative = llm.invoke(prompt)  # type: ignore[assignment]

        # 신뢰도 집계는 LLM에게 요청하지 않고 코드가 결정론적으로 채움(재현성
        # 확보 + perspective_confidence의 동적 dict 스키마는 애초에 OpenAI
        # Structured Outputs로 못 보냄, 위 _SynthesisNarrative 독스트링 참고).
        synthesis = Synthesis(
            agreements=narrative.agreements,
            conflicts=narrative.conflicts,
            summary=narrative.summary,
            perspective_confidence=confidence,
            overall_confidence=overall_confidence,
            weakest_perspective=weakest_perspective,
        )

        # 12장 "반복 2" 예산(1회)을 그래프 조건 분기가 확인할 수 있게 재작성 횟수를 기록함
        rewrite_count = state.get("rewrite_count", 0) + (1 if is_rewrite else 0)
        return {"synthesis": synthesis, "rewrite_count": rewrite_count}
