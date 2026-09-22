"""market_eval 독립 테스트 러너. RAG를 쓰지 않아 Hit Rate/MRR 대신 8.2절
LLM-as-a-Judge 루브릭(정확성/완전성/중립성/근거연결성, 1~5)으로 달성도를
확인함(schedule.md 2절).

채점 컨텍스트에 실제 수집된 근거 원문(quote)을 같이 넣어줌 — 안 그러면 "정확성"
(인용한 근거와 실제로 부합하는가)을 판정 모델이 검증할 방법이 없어 사실상 찍게 됨.

실행: python -m scripts.market_eval.test_runner
전제: .env에 OPENAI_API_KEY, TAVILY_API_KEY(에이전트 실행용) 설정,
      Ollama에 검수 모델(OLLAMA_JUDGE_MODEL, 기본 qwen3:8b) 로드.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.market_eval.agent import MarketEvalAgent
from src.common.eval_utils import plot_rubric_scores, render_view_result_md, score_with_rubric
from src.common.models import get_judge_llm
from src.common.state import TechSpec

AGENT_NAME = "market_eval"
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
        "src/agents/market_eval/agent.py의 프롬프트(_EXTRACTION_PROMPT)를 다듬고 재실행할 것."
    )
    return f"""# market_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.5절 / docs/schedule.md 2절
(RAG 미사용이라 8.2절 LLM-as-a-Judge 루브릭으로 달성도를 확인함)

## 채점 결과

| 항목 | 점수(1~5) |
|---|---|
{rows}

![루브릭 점수](report_assets/rubric_scores.png)

## 검사 대상 산출물

{output_text}

## 결론

{verdict}
"""


def main() -> None:
    agent = MarketEvalAgent()
    result = agent.run(FIXTURE_STATE)
    output_text = render_view_result_md(result["market_result"])
    evidence_text = "\n".join(
        f"[근거#{e.id}] ({e.tech}, {e.stance}) {e.quote}" for e in result["evidence"]
    )

    judge_llm = get_judge_llm()
    scores = score_with_rubric(
        judge_llm,
        target_text=output_text,
        context=(
            "market_eval(7.5절): 시장 규모/성장성/채택 현황/생태계 지지를 두 기술에 "
            "대해 대칭적으로 조사한 결과. 아래는 실제 수집된 근거 원문임 — 산출물이 "
            "인용한 근거 번호가 이 원문과 실제로 부합하는지 확인할 것.\n\n" + evidence_text
        ),
    )
    plot_rubric_scores(scores, f"{AGENT_NAME} 루브릭 점수", HERE / "report_assets" / "rubric_scores.png")

    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(scores, output_text), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
