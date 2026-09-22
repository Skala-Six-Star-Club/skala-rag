"""stakeholder_eval 독립 테스트 러너. RAG를 쓰지 않아 Hit Rate/MRR 대신 설계서 8장의
생성·에이전트 지표로 검증함(schedule.md 2절).

1. 8.2절 LLM-as-a-Judge 루브릭(정확성/완전성/중립성/근거연결성, 1~5) — 기술별로 채점
2. 9.3절 필수 항목 커버리지 — 8.2 "완전성"을 항목 단위로 분해(기술별, 항목별 판정)
3. 규칙 기반 구조 검사 — 8.2 "근거 연결성"을 코드로 확정(주장 대비 유효 근거 참조 비율),
   7.7 반대 근거 유무, 두 기술 간 주장·근거 수 대칭(10장)
4. 8.3절 Tool Calling Accuracy — 실제 던진 web_search 질의가 코드에 고정한 템플릿과 일치하는가

실행: python -m scripts.stakeholder_eval.test_runner
전제: .env에 OPENAI_API_KEY(없으면 OLLAMA_GENERATION_MODEL 폴백), TAVILY_API_KEY(없으면
      DuckDuckGo 폴백), Ollama에 검수 모델(OLLAMA_JUDGE_MODEL, 기본 qwen3:8b) 로드.
"""

from __future__ import annotations

from pathlib import Path


from src.agents.stakeholder_eval.agent import _QUERY_TEMPLATES, REQUIRED_ITEMS, StakeholderEvalAgent
from src.common.eval_utils import (
    RubricScore,
    format_evidence,
    format_view_result,
    plot_bar_comparison,
    plot_status_matrix,
    score_required_items,
    score_with_rubric,
    tool_calling_accuracy,
    view_structure_stats,
)
from src.common.models import get_judge_llm
from src.common.state import TechSpec

AGENT_NAME = "stakeholder_eval"
HERE = Path(__file__).parent
THRESHOLD = 3  # 8.2절 1~5 척도 임계값
THRESHOLD_COVERAGE = 2 / 3  # 필수 항목 3개 중 2개 이상
THRESHOLD_LINKAGE = 1.0  # 모든 주장이 유효 근거를 참조해야 함(10장)

FIXTURE_STATE = {
    "techs": [
        TechSpec(name="TurboQuant", camp="SW", role="target", search_anchor="Google"),
        TechSpec(name="ITME", camp="HW", role="target", search_anchor="SK hynix"),
    ],
    "tech_profiles": {},
    "evidence": [],
}

_RUBRIC_KEYS = ["accuracy", "completeness", "neutrality", "evidence_linkage"]
_RUBRIC_LABELS = ["정확성", "완전성", "중립성", "근거연결성"]


def _plot_rubric_by_tech(scores: dict[str, RubricScore], save_path: Path) -> None:
    """루브릭 4항목(명목형) x 기술 2개. 항목 사이에 순서가 없어 선 대신 막대로 둠. 1~5 척도 고정."""
    plot_bar_comparison(
        _RUBRIC_LABELS,
        {tech: [getattr(r, k) for k in _RUBRIC_KEYS] for tech, r in scores.items()},
        f"{AGENT_NAME} 8.2 루브릭 (기술별, 임계값 {THRESHOLD})", "score (1-5)", save_path, ylim=(0, 5.6),
    )


def _plot_coverage(coverage: dict[str, dict[str, bool]], save_path: Path) -> None:
    """포함/미포함 이진 판정이라 막대 대신 항목 x 기술 상태 행렬로 그림."""
    plot_status_matrix(
        REQUIRED_ITEMS, list(coverage), coverage,
        f"{AGENT_NAME} 9.3절 필수 항목 커버리지", save_path,
    )


def _finalized(result: dict) -> dict:
    """provisional key를 최종 번호로 바꿔(evidence_finalize와 동일 로직) 채점에 씀."""
    from src.common.evidence import finalize_evidence

    state = {"techs": FIXTURE_STATE["techs"], "raw_evidence": result["raw_evidence"], "stakeholder_result": result["stakeholder_result"]}
    return finalize_evidence(state)


def evaluate(agent: StakeholderEvalAgent, result: dict) -> dict:
    final = _finalized(result)
    view_result = final["stakeholder_result"]
    evidence = final["evidence"]
    techs = FIXTURE_STATE["techs"]
    judge = get_judge_llm()

    rubric: dict[str, RubricScore] = {}
    coverage: dict[str, dict[str, bool]] = {}
    coverage_reason: dict[str, dict[str, str]] = {}
    stats = {}
    view_texts = {}
    for tech in techs:
        view = view_result.by_tech[tech.name]
        text = format_view_result(view)
        view_texts[tech.name] = text
        rubric[tech.name] = score_with_rubric(
            judge,
            target_text=text,
            context=(
                f"stakeholder_eval(7.3절): {tech.name}에 대한 이해관계자 관점 평가 결과. "
                "완전성은 아래 9.3절 필수 항목이 빠짐없이 다뤄졌는지로 판단할 것: "
                + ", ".join(REQUIRED_ITEMS)
                + "\n\n[근거 목록]\n" + format_evidence(evidence, tech.name)
            ),
        )
        cov = score_required_items(judge, text, tech.name, REQUIRED_ITEMS)
        coverage[tech.name] = {it.item: it.covered for it in cov.items}
        coverage_reason[tech.name] = {it.item: it.reason for it in cov.items}
        for item in REQUIRED_ITEMS:  # 채점자가 항목을 빠뜨리면 미포함으로 처리
            coverage[tech.name].setdefault(item, False)
            coverage_reason[tech.name].setdefault(item, "채점자 응답 누락")
        stats[tech.name] = view_structure_stats(view, evidence, tech.name)

    tca, tca_issues = tool_calling_accuracy(agent.last_queries, techs, _QUERY_TEMPLATES)
    return dict(
        rubric=rubric, coverage=coverage, coverage_reason=coverage_reason,
        stats=stats, view_texts=view_texts, tca=tca, tca_issues=tca_issues,
        queries=agent.last_queries,
    )


def build_report(ev: dict) -> str:
    techs = list(ev["rubric"])
    rubric_rows = "\n".join(
        f"| {label} | " + " | ".join(str(getattr(ev["rubric"][t], k)) for t in techs) + " |"
        for label, k in zip(_RUBRIC_LABELS, _RUBRIC_KEYS)
    )
    cov_rows = "\n".join(
        f"| {item} | " + " | ".join(
            ("포함" if ev["coverage"][t][item] else "미포함") + f" ({ev['coverage_reason'][t][item][:40]})"
            for t in techs
        ) + " |"
        for item in REQUIRED_ITEMS
    )
    cov_ratio = {t: sum(ev["coverage"][t].values()) / len(REQUIRED_ITEMS) for t in techs}
    st = ev["stats"]
    struct_rows = "\n".join([
        "| 확인된 사실 수 | " + " | ".join(str(st[t].n_confirmed) for t in techs) + " |",
        "| 반대 사실 수 | " + " | ".join(str(st[t].n_counter) for t in techs) + " |",
        "| 미확인 항목 수 | " + " | ".join(str(st[t].n_unconfirmed) for t in techs) + " |",
        "| 수집 근거 수(반대 근거 수) | " + " | ".join(f"{st[t].n_evidence}({st[t].n_evidence_counter})" for t in techs) + " |",
        "| 근거 연결 비율(유효 근거 참조 주장/전체 주장) | " + " | ".join(f"{st[t].evidence_linkage_ratio:.2f}" for t in techs) + " |",
    ])
    n_claims = [st[t].n_claims for t in techs]
    symmetry = f"두 기술 주장 수 차이 {abs(n_claims[0] - n_claims[1])}건, 근거 수 차이 {abs(st[techs[0]].n_evidence - st[techs[1]].n_evidence)}건"
    query_rows = "\n".join(f"| {t} | {q} |" for t, qs in ev["queries"].items() for q in qs)

    failures = []
    for t in techs:
        r = ev["rubric"][t]
        low = [l for l, k in zip(_RUBRIC_LABELS, _RUBRIC_KEYS) if getattr(r, k) < THRESHOLD]
        if low:
            failures.append(f"{t} 루브릭 미달: {', '.join(low)}")
        if cov_ratio[t] < THRESHOLD_COVERAGE:
            failures.append(f"{t} 필수 항목 커버리지 {cov_ratio[t]:.2f} < {THRESHOLD_COVERAGE:.2f}")
        if st[t].evidence_linkage_ratio < THRESHOLD_LINKAGE:
            failures.append(f"{t} 근거 연결 비율 {st[t].evidence_linkage_ratio:.2f} < {THRESHOLD_LINKAGE:.2f}")
        if st[t].n_counter == 0:
            failures.append(f"{t} 반대 근거 없음(7.7 재검색 대상)")
    if ev["tca"] < 1.0:
        failures.append(f"Tool Calling Accuracy {ev['tca']:.2f} < 1.00: " + "; ".join(ev["tca_issues"]))
    verdict = "모든 지표가 임계값 이상임." if not failures else "임계값 미달 항목:\n" + "\n".join(f"- {f}" for f in failures)

    views = "\n\n".join(f"### {t}\n\n```\n{ev['view_texts'][t]}\n```" for t in techs)

    return f"""# stakeholder_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.3절·9.3절 / 8.2절·8.3절 / docs/schedule.md 2절
(RAG 미사용이라 Hit Rate/MRR 대신 생성·에이전트 지표로 확인함)

## 1. 8.2절 LLM-as-a-Judge 루브릭 (기술별, 1~5, 임계값 {THRESHOLD})

| 항목 | {" | ".join(techs)} |
|---|{"---|" * len(techs)}
{rubric_rows}

![루브릭 점수](report_assets/rubric_scores.png)

## 2. 9.3절 필수 항목 커버리지 (완전성 세부, 임계값 {THRESHOLD_COVERAGE:.2f})

| 필수 항목 | {" | ".join(techs)} |
|---|{"---|" * len(techs)}
{cov_rows}

커버리지: {", ".join(f"{t} {cov_ratio[t]:.2f}" for t in techs)}

![필수 항목 커버리지](report_assets/coverage.png)

## 3. 규칙 기반 구조 검사 (근거 연결성·반대 근거·대칭)

| 항목 | {" | ".join(techs)} |
|---|{"---|" * len(techs)}
{struct_rows}

대칭(10장): {symmetry}

## 4. 8.3절 Tool Calling Accuracy

일치 비율: {ev["tca"]:.2f} (질의 {sum(len(q) for q in ev["queries"].values())}건, 템플릿 {len(_QUERY_TEMPLATES)}종 × 기술 {len(techs)}건)

| 기술 | 실제 질의 |
|---|---|
{query_rows}

## 5. 검사 대상 산출물

{views}

## 6. 결론

{verdict}
"""


def main() -> None:
    agent = StakeholderEvalAgent()
    result = agent.run(FIXTURE_STATE)
    ev = evaluate(agent, result)
    _plot_rubric_by_tech(ev["rubric"], HERE / "report_assets" / "rubric_scores.png")
    _plot_coverage(ev["coverage"], HERE / "report_assets" / "coverage.png")
    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(ev), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
