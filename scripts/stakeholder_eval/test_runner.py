"""stakeholder_eval 독립 테스트 러너. RAG를 쓰지 않아 Hit Rate/MRR 대신 8.2절
LLM-as-a-Judge 루브릭(정확성/완전성/중립성/근거연결성, 1~5)으로 달성도를
확인함(schedule.md 2절).

주의: src/agents/stakeholder_eval/agent.py의 TODO(구조화 추출)를 담당자가 채우기
전까지는 by_tech가 빈 TechViewResult()라 낮은 점수가 나오는 게 정상임. 지금
당장 검증하려는 게 아니라, TODO를 채운 뒤 "만든다 -> 돌린다 -> 채점한다 ->
프롬프트를 고쳐 재실행한다"(schedule.md 1절) 루프에서 쓰라고 만든 스크립트임.

실행: python -m scripts.stakeholder_eval.test_runner
전제: .env에 OPENAI_API_KEY, TAVILY_API_KEY(에이전트 실행용) 설정,
      Ollama에 검수 모델(OLLAMA_JUDGE_MODEL, 기본 qwen3:8b) 로드.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.stakeholder_eval.agent import StakeholderEvalAgent
from src.common.eval_utils import plot_rubric_scores, score_with_rubric
from src.common.models import get_judge_llm
from src.common.state import TechSpec

AGENT_NAME = "stakeholder_eval"
HERE = Path(__file__).parent
THRESHOLD = 3  # 8.2절 1~5 척도, 담당자가 프롬프트를 완성하기 전에는 미달이 정상임

FIXTURE_STATE = {
    "techs": [
        TechSpec(name="TurboQuant", camp="SW", role="target", search_anchor="Google"),
        TechSpec(name="ITME", camp="HW", role="target", search_anchor="SK hynix"),
    ],
    "tech_profiles": {},
    "evidence": [],
}


def build_report(scores, output_text: str) -> str:
    rows = "\n".join(
        f"| {k} | {v} |"
        for k, v in [
            ("정확성", scores.accuracy),
            ("완전성", scores.completeness),
            ("중립성", scores.neutrality),
            ("근거 연결성", scores.evidence_linkage),
        ]
    )
    passed = all(
        v >= THRESHOLD
        for v in (scores.accuracy, scores.completeness, scores.neutrality, scores.evidence_linkage)
    )
    verdict = (
        f"모든 항목이 임계값({THRESHOLD}점) 이상임."
        if passed
        else f"임계값({THRESHOLD}점) 미달 항목 있음 — "
        "src/agents/stakeholder_eval/agent.py의 TODO(구조화 추출 로직)를 아직 채우지 않았다면 "
        "이는 예상된 결과임. 담당자가 프롬프트를 완성한 뒤 재실행할 것."
    )
    return f"""# stakeholder_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.6절 / docs/schedule.md 2절
(RAG 미사용이라 8.2절 LLM-as-a-Judge 루브릭으로 달성도를 확인함)

## 채점 결과

| 항목 | 점수(1~5) |
|---|---|
{rows}

![루브릭 점수](report_assets/rubric_scores.png)

## 검사 대상 산출물

```
{output_text}
```

## 결론

{verdict}
"""


def main() -> None:
    agent = StakeholderEvalAgent()
    result = agent.run(FIXTURE_STATE)
    output_text = str(result["stakeholder_result"])

    judge_llm = get_judge_llm()
    scores = score_with_rubric(
        judge_llm,
        target_text=output_text,
        context="stakeholder_eval(7.6절): 경쟁 진영/도입 기업·개발자/투자 업계 반응을 두 기술에 대해 대칭적으로 조사한 결과",
    )
    plot_rubric_scores(scores, f"{AGENT_NAME} 루브릭 점수", HERE / "report_assets" / "rubric_scores.png")

    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(scores, output_text), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
