"""중립성 검수 에이전트 (7.9절). synthesis를 대상으로 세 가지를 예/아니오로만
판정하는 저추론 검사. 점수화하지 않음(6.3절: 판단을 단순화해야 8B급 모델에서도
결과가 안정됨).

입력: state["synthesis"], state["evidence"]
출력: {"judge_feedback": ...}

생성(GPT-5 mini)과 계열이 다른 Qwen3-8B(Ollama 로컬)를 씀 — 자기 선호 편향을
구조적으로 피하기 위함(Zheng et al., 2023; Panickssery et al., 2024).

재작성 루프(12장 "반복 2")는 graph.py의 조건부 엣지가 judge_passed()로 판정함.
이 노드는 judge_feedback만 쓰고 라우팅은 하지 않음.
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.models import get_judge_llm
from src.common.state import AgentState, Evidence, JudgeFeedback, Synthesis

# 8B 모델이 기준을 흔들리지 않게 항목마다 판정 경계와 위반/통과 예시를 1개씩 명시함.
# 특히 "관점별 병렬 서술"을 우열 판정으로 오인하는 것과, is_balanced를 근거 분포가
# 아닌 문장 인상으로 판단하는 것을 막는 것이 목적임.
_PROMPT_TEMPLATE = """\
당신은 두 경쟁 기술 평가 보고서의 중립성만 검사하는 검수자입니다.
내용의 사실 여부는 검사 대상이 아닙니다. 오직 표현과 형식만 봅니다.
점수를 매기지 말고 아래 세 항목을 판정해 구조화된 필드로만 답하세요. 추론 과정은 적지 마세요.

[항목 1] has_biased_expression — 우열 판정 표현이 있는가
- 한쪽 기술이 전반적으로 더 낫다고 결론짓는 표현("더 우수하다", "훨씬 낫다", "명백히 앞선다", "갈 길이 멀다", "도입을 추천한다")이 있으면 true.
- 관점별로 각 기술의 특징을 나란히 서술하는 것은 우열 판정이 아니므로 false.
- 위반 예: "A가 B보다 훨씬 우수하고 실용적인 기술임." -> true
- 통과 예: "A는 SW 관점에서, B는 HW 관점에서 각각 접근함." -> false

[항목 2] sentences_without_evidence — [근거#N] 표기가 없는 사실 주장 문장
- 사실을 주장하는 문장인데 [근거#N] 표기가 하나도 없으면 그 문장 전체를 리스트에 넣으세요.
- 접속 문장이나 단순 안내 문장은 제외합니다. 개수가 아니라 문장 자체를 나열하세요.
- 모든 문장에 근거 번호가 있으면 빈 리스트 []로 답하세요.

[항목 3] is_balanced — 두 기술의 서술이 균형 잡혀 있는가
- 아래 [근거 분포]에서 두 기술 모두 반대 근거가 1건 이상이면 true.
- 어느 한 기술이라도 반대 근거가 0건이면 false.
- 근거 분포가 균형이어도 한 기술만 분량이 압도적으로 많거나 어조가 한쪽에만 부정적이면 false.

[항목 4] notes — 위 판정의 근거를 한두 문장으로 요약

===== 검사 대상 =====

[일치점]
{agreements}

[상충점]
{conflicts}

[SUMMARY]
{summary}

[근거 분포] (기술별 지지/반대 근거 건수)
{evidence_breakdown}
"""


def _format_evidence_breakdown(evidence: list[Evidence]) -> str:
    """기술별 지지/반대 건수를 집계함. is_balanced 판정의 유일한 정량 신호라
    개수 하나가 아니라 기술·입장별로 나눠 넘김."""
    by_tech: dict[str, dict[str, int]] = {}
    for e in evidence:
        counts = by_tech.setdefault(e.tech, {"지지": 0, "반대": 0})
        counts[e.stance] += 1
    if not by_tech:
        return "(근거 없음)"
    return "\n".join(f"- {tech}: 지지 {c['지지']}건, 반대 {c['반대']}건" for tech, c in by_tech.items())


def _format_synthesis(synthesis: Synthesis | None) -> dict[str, str]:
    if synthesis is None:
        return {"agreements": "(없음)", "conflicts": "(없음)", "summary": "(없음)"}
    return {
        "agreements": "\n".join(f"- {a}" for a in synthesis.agreements) or "(없음)",
        "conflicts": "\n".join(f"- {c.topic}: {c.explanation}" for c in synthesis.conflicts) or "(없음)",
        "summary": synthesis.summary or "(없음)",
    }


def build_prompt(synthesis: Synthesis | None, evidence: list[Evidence]) -> str:
    return _PROMPT_TEMPLATE.format(
        **_format_synthesis(synthesis),
        evidence_breakdown=_format_evidence_breakdown(evidence),
    )


def judge_passed(feedback: JudgeFeedback) -> bool:
    """graph.py 조건부 엣지(12장 "반복 2")용 순수 판정. 세 조건을 모두 만족해야
    synthesize로 되돌리지 않고 report로 진행함."""
    return (
        not feedback.has_biased_expression
        and not feedback.sentences_without_evidence
        and feedback.is_balanced
    )


class JudgeAgent(BaseAgent):
    name = "judge"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        prompt = build_prompt(state.get("synthesis"), state.get("evidence", []))
        llm = get_judge_llm().with_structured_output(JudgeFeedback)
        feedback: JudgeFeedback = llm.invoke(prompt)  # type: ignore[assignment]
        return {"judge_feedback": feedback}
