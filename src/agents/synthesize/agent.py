"""평가 종합 에이전트 (7.8절). 관점 결과 4종을 종합해 일치점/상충점/SUMMARY를 생성함.
추가 검색 없는 순수 생성 작업.

입력: state의 관점 결과 4종(+ 재작성 시 judge_feedback)
출력: {"synthesis": ...}

judge(7.9)에서 반려되면 judge_feedback을 프롬프트에 추가해 동일 입력으로 1회
재실행함(12장 "반복 2"). 재작성 시에도 이전 synthesis 초안은 프롬프트에 남기지
않고 매번 관점 결과 4종에서 새로 작성함(부분 수정이 아닌 답습 위험 회피).
"""

from __future__ import annotations

from typing import Any

from src.common.models import get_generation_llm
from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Synthesis

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
"""


class SynthesizeAgent(BaseAgent):
    name = "synthesize"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        prompt = _PROMPT_TEMPLATE.format(
            trl_result=state.get("trl_result"),
            market_result=state.get("market_result"),
            stakeholder_result=state.get("stakeholder_result"),
            domain_result=state.get("domain_result"),
        )

        feedback = state.get("judge_feedback")
        is_rewrite = feedback is not None
        if is_rewrite:
            prompt += f"\n[이전 검수에서 지적된 사항, 반드시 반영할 것]\n{feedback}"

        # TODO(담당자): 프롬프트 문구를 다듬을 것(예: 관점 결과를 dict 그대로
        # 넣지 말고 사람이 읽기 좋은 형태로 직렬화). 구조화 출력 호출 자체는 동작함.
        llm = get_generation_llm().with_structured_output(Synthesis)
        synthesis: Synthesis = llm.invoke(prompt)  # type: ignore[assignment]
        # 12장 "반복 2" 예산(1회)을 그래프 조건 분기가 확인할 수 있게 재작성 횟수를 기록함
        rewrite_count = state.get("rewrite_count", 0) + (1 if is_rewrite else 0)
        return {"synthesis": synthesis, "rewrite_count": rewrite_count}
