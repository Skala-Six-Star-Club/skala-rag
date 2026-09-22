"""synthesize 독립 테스트 러너. 순수 생성 작업이라 8.2절 LLM-as-a-Judge
루브릭(정확성/완전성/중립성/근거연결성, 1~5)으로 달성도를 확인함(schedule.md 2절).

synthesize/agent.py는 자리표시자가 아니라 실제 구조화 출력 호출까지 돼 있어 지금
실행해도 됨. 다만 프롬프트(_PROMPT_TEMPLATE)는 다듬어지지 않은 초안이라, 낮은
점수가 나오면 "만든다 -> 돌린다 -> 채점한다 -> 프롬프트를 고쳐 재실행한다"
(schedule.md 1절) 루프를 그대로 밟으면 됨.

실행: python -m scripts.synthesize.test_runner
전제: .env에 OPENAI_API_KEY(생성) 설정, Ollama에 검수 모델 로드(채점용).
"""

from __future__ import annotations

from pathlib import Path

from src.agents.synthesize.agent import SynthesizeAgent
from src.common.eval_utils import plot_rubric_scores, score_with_rubric
from src.common.models import get_judge_llm
from src.common.state import Claim, TechViewResult, ViewResult

AGENT_NAME = "synthesize"
HERE = Path(__file__).parent
THRESHOLD = 3

# 4개 관점이 이미 채점을 마쳤다고 가정한 손으로 만든 픽스처(실제로는 관점 노드 4종의 출력).
_TURBOQUANT = TechViewResult(
    confirmed_facts=[Claim(statement="채널당 3.5비트에서 품질 저하가 없다고 보고됨", evidence_ids=[1])],
    counter_facts=[Claim(statement="독립 검증에서 3비트 설정 시 최대 20점 저하가 보고됨", evidence_ids=[2])],
)
_ITME = TechViewResult(
    confirmed_facts=[Claim(statement="양산급 CMM 제품으로 성능을 평가함", evidence_ids=[3])],
    counter_facts=[Claim(statement="FPGA 시제품 단계로 실제 제품 검증은 아직 부족함", evidence_ids=[4])],
)

FIXTURE_STATE = {
    "trl_result": ViewResult(by_tech={"TurboQuant": _TURBOQUANT, "ITME": _ITME}),
    "market_result": ViewResult(by_tech={"TurboQuant": _TURBOQUANT, "ITME": _ITME}),
    "stakeholder_result": ViewResult(by_tech={"TurboQuant": _TURBOQUANT, "ITME": _ITME}),
    "domain_result": ViewResult(by_tech={"TurboQuant": _TURBOQUANT, "ITME": _ITME}),
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
        else f"임계값({THRESHOLD}점) 미달 항목 있음 — src/agents/synthesize/agent.py 프롬프트를 다듬을 것."
    )
    return f"""# synthesize 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.8절 / docs/schedule.md 2절

## 채점 결과

| 항목 | 점수(1~5) |
|---|---|
{rows}

![루브릭 점수](report_assets/rubric_scores.png)

## 검사 대상 산출물 (synthesis)

```
{output_text}
```

## 결론

{verdict}
"""


def main() -> None:
    agent = SynthesizeAgent()
    result = agent.run(FIXTURE_STATE)
    output_text = str(result["synthesis"])

    judge_llm = get_judge_llm()
    scores = score_with_rubric(
        judge_llm,
        target_text=output_text,
        context="synthesize(7.8절): 4개 관점 평가 결과를 종합한 일치점/상충점/SUMMARY",
    )
    plot_rubric_scores(scores, f"{AGENT_NAME} 루브릭 점수", HERE / "report_assets" / "rubric_scores.png")

    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(scores, output_text), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
