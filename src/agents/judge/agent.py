"""중립성 검수 에이전트 (7.9절). synthesis를 대상으로 세 가지를 예/아니오로만
판정하는 저추론 검사. 점수화하지 않음(6.3절: 판단을 단순화해야 8B급 모델에서도
결과가 안정됨).

입력: state["synthesis"], state["evidence"]
출력: {"judge_feedback": ...}

생성(GPT-5 mini)과 계열이 다른 Qwen3-8B(Ollama 로컬)를 씀 — 자기 선호 편향을
구조적으로 피하기 위함(Zheng et al., 2023; Panickssery et al., 2024).
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.models import get_judge_llm
from src.common.state import AgentState, JudgeFeedback

_PROMPT_TEMPLATE = """\
아래 종합 평가문을 검사해줘. 점수를 매기지 말고 예/아니오로만 답해줘.
1. 우열 판정 표현(어느 기술이 더 낫다는 식의 서술)이 있는가
2. 근거 번호가 없는 문장이 있는가 (있다면 어떤 문장인지 나열)
3. 두 기술의 서술 분량과 어조가 불균형한가

[synthesis] {synthesis}
[evidence 개수] {evidence_count}건
"""


class JudgeAgent(BaseAgent):
    name = "judge"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        synthesis = state.get("synthesis")
        evidence = state.get("evidence", [])

        prompt = _PROMPT_TEMPLATE.format(
            synthesis=synthesis, evidence_count=len(evidence)
        )
        llm = get_judge_llm().with_structured_output(JudgeFeedback)
        feedback: JudgeFeedback = llm.invoke(prompt)  # type: ignore[assignment]
        return {"judge_feedback": feedback}
