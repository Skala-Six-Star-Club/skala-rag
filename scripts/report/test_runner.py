"""report 독립 테스트 러너. schedule.md 2절: "8.2 루브릭 + 인용 번호 안전장치
단위 테스트"를 함께 함.

1부(citation safety net, API 키 없이 바로 실행 가능): 환각 근거 번호가 섞인
   입력을 일부러 넣어 원문 문장으로 되돌아가는지 확인함(7.10절).
2부(8.2 루브릭, API 필요): 조립된 보고서 초안을 LLM-as-a-Judge로 채점함.

실행: python -m scripts.report.test_runner
전제(2부만): .env에 OPENAI_API_KEY 설정, Ollama에 검수 모델 로드.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.report.agent import ReportAgent, verify_citations
from src.common.eval_utils import UnitCheck, plot_rubric_scores, plot_unit_checks, score_with_rubric
from src.common.models import get_judge_llm
from src.common.state import Evidence, Reference, Synthesis, TechSpec

AGENT_NAME = "report"
HERE = Path(__file__).parent
THRESHOLD = 3


def run_citation_safety_checks() -> list[UnitCheck]:
    """7.10절 인용 번호 안전장치: 환각 번호 섞인 문장은 원문으로 되돌아가야 함."""
    original = "TurboQuant는 채널당 3.5비트에서 품질 저하가 없다고 보고됨[근거#1]."
    valid_ids = {1, 2, 3}

    hallucinated = "TurboQuant는 채널당 3.5비트에서 품질 저하가 없다고 보고됨[근거#99]."
    reverted = verify_citations(original, hallucinated, valid_ids)

    valid_edit = "TurboQuant는 채널당 3.5비트 수준에서 품질 저하가 거의 없다고 보고됨[근거#1]."
    kept = verify_citations(original, valid_edit, valid_ids)

    return [
        UnitCheck("환각 근거 번호 -> 원문으로 되돌림", reverted == original, reverted),
        UnitCheck("유효 근거 번호 -> 다듬은 문장 유지", kept == valid_edit, kept),
    ]


def run_rubric_check():
    fixture_state = {
        "techs": [
            TechSpec(name="TurboQuant", camp="SW", role="target", search_anchor="Google"),
            TechSpec(name="ITME", camp="HW", role="target", search_anchor="SK hynix"),
        ],
        "tech_profiles": {},
        "trl_result": None,
        "market_result": None,
        "stakeholder_result": None,
        "domain_result": None,
        "synthesis": Synthesis(summary="두 기술은 SW/HW 접근이 다름[근거#1][근거#2]."),
        "judge_feedback": None,
        "evidence": [
            Evidence(id=1, tech="TurboQuant", perspective="tech_research", stance="지지", source_type="논문", source="paper-a", quote="q"),
            Evidence(id=2, tech="ITME", perspective="tech_research", stance="지지", source_type="논문", source="paper-b", quote="q"),
        ],
        "references": [
            Reference(id=1, type="paper", author_or_org="Zandieh, A.", year="2025", title="TurboQuant", url="paper-a"),
            Reference(id=2, type="paper", author_or_org="Jang, H.", year="2026", title="ITME", url="paper-b"),
        ],
    }
    result = ReportAgent().run(fixture_state)
    report_md = result["report_md"]
    scores = score_with_rubric(
        get_judge_llm(), target_text=report_md, context="report(7.10절): 13장 목차에 맞춰 조립된 보고서 초안"
    )
    return scores, report_md


def build_report(citation_checks: list[UnitCheck], scores=None, report_md: str = "") -> str:
    citation_rows = "\n".join(
        f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail[:60]} |" for c in citation_checks
    )
    citation_passed = sum(c.passed for c in citation_checks) == len(citation_checks)

    rubric_section = "(2부 미실행 — API 키/Ollama 설정 후 재실행할 것)"
    if scores is not None:
        rubric_rows = "\n".join(
            f"| {k} | {v} |"
            for k, v in [
                ("정확성", scores.accuracy),
                ("완전성", scores.completeness),
                ("중립성", scores.neutrality),
                ("근거 연결성", scores.evidence_linkage),
            ]
        )
        rubric_section = f"| 항목 | 점수(1~5) |\n|---|---|\n{rubric_rows}\n\n![루브릭 점수](report_assets/rubric_scores.png)"

    return f"""# report 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.10절 / docs/schedule.md 2절

## 1부: 인용 번호 안전장치 단위 테스트

| 테스트 | 결과 | 비고 |
|---|---|---|
{citation_rows}

![안전장치 테스트](report_assets/citation_checks.png)

{"안전장치가 의도대로 동작함." if citation_passed else "안전장치 실패 — src/agents/report/agent.py의 verify_citations 확인 필요."}

## 2부: LLM-as-a-Judge 루브릭 (8.2절)

{rubric_section}

## 검사 대상 보고서 초안

```
{report_md}
```
"""


def main() -> None:
    citation_checks = run_citation_safety_checks()
    plot_unit_checks(
        citation_checks, f"{AGENT_NAME} 인용 번호 안전장치", HERE / "report_assets" / "citation_checks.png"
    )

    scores, report_md = None, ""
    try:
        scores, report_md = run_rubric_check()
        plot_rubric_scores(scores, f"{AGENT_NAME} 루브릭 점수", HERE / "report_assets" / "rubric_scores.png")
    except Exception as exc:  # noqa: BLE001 - 2부는 API 키가 없으면 실패할 수 있음, 1부 결과는 보존함
        print(f"2부(루브릭) 실행 실패, 1부 결과만 저장함: {exc}")

    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(citation_checks, scores, report_md), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
