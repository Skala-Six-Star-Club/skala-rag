"""evidence_check 독립 테스트 러너. 규칙 기반 노드라 입력→기대 출력 단위
테스트로 검증함(schedule.md 2절). API 키 없이 바로 실행 가능함.

실행: python -m scripts.evidence_check.test_runner
"""

from __future__ import annotations

from pathlib import Path

from src.agents.evidence_check.agent import EvidenceCheckAgent
from src.common.eval_utils import UnitCheck, plot_unit_checks
from src.common.state import Evidence, TechSpec

AGENT_NAME = "evidence_check"
HERE = Path(__file__).parent

TECHS = [
    TechSpec(name="A", camp="SW", role="target", search_anchor="a"),
    TechSpec(name="B", camp="HW", role="target", search_anchor="b"),
]


def _evidence(tech: str, perspective: str, stance: str, n: int, start_id: int) -> list[Evidence]:
    return [
        Evidence(
            id=start_id + i, tech=tech, perspective=perspective, stance=stance,
            source_type="논문", source="s", quote="q",
        )
        for i in range(n)
    ]


def run_checks() -> list[UnitCheck]:
    agent = EvidenceCheckAgent()
    checks: list[UnitCheck] = []

    # 1. 충분한 근거(3건 이상, 지지/반대 모두 존재, 비율 2배 이내) -> 재검색 대상 아님
    evidence = (
        _evidence("A", "trl", "지지", 3, 1) + _evidence("A", "trl", "반대", 1, 100)
        + _evidence("B", "trl", "지지", 3, 200) + _evidence("B", "trl", "반대", 1, 300)
    )
    result = agent.run({"techs": TECHS, "evidence": evidence, "retry_count": 0})
    checks.append(UnitCheck("충분한 근거 -> 재검색 없음", "trl_eval" not in result["retry_targets"]))

    # 2. 근거 수 3건 미만(규칙 1) -> 재검색 대상
    evidence = (
        _evidence("A", "trl", "지지", 1, 1) + _evidence("A", "trl", "반대", 1, 100)
        + _evidence("B", "trl", "지지", 3, 200) + _evidence("B", "trl", "반대", 1, 300)
    )
    result = agent.run({"techs": TECHS, "evidence": evidence, "retry_count": 0})
    checks.append(UnitCheck("근거 부족(<3건, 규칙 1) -> 재검색 대상", "trl_eval" in result["retry_targets"]))

    # 3. 반대 근거 0건(규칙 2) -> 재검색 대상
    evidence = (
        _evidence("A", "trl", "지지", 3, 1) + _evidence("A", "trl", "반대", 1, 50)
        + _evidence("B", "trl", "지지", 3, 200)
    )
    result = agent.run({"techs": TECHS, "evidence": evidence, "retry_count": 0})
    checks.append(UnitCheck("반대 근거 0건(규칙 2) -> 재검색 대상", "trl_eval" in result["retry_targets"]))

    # 4. 두 기술 근거 수 비율 2배 초과(규칙 3) -> 재검색 대상
    evidence = (
        _evidence("A", "trl", "지지", 3, 1) + _evidence("A", "trl", "반대", 1, 50)
        + _evidence("B", "trl", "지지", 8, 200) + _evidence("B", "trl", "반대", 2, 300)
    )
    result = agent.run({"techs": TECHS, "evidence": evidence, "retry_count": 0})
    checks.append(UnitCheck("근거 수 비율 2배 초과(규칙 3) -> 재검색 대상", "trl_eval" in result["retry_targets"]))

    # 5. 재시도 이미 1회 소진 -> 규칙을 만족해도 더 이상 재검색 대상에 넣지 않음
    evidence = _evidence("A", "trl", "지지", 1, 1)  # 명백히 부족
    result = agent.run({"techs": TECHS, "evidence": evidence, "retry_count": 1})
    checks.append(UnitCheck("재시도 소진 -> 더 이상 재검색 안 함", result["retry_targets"] == []))

    return checks


def build_report(checks: list[UnitCheck]) -> str:
    rows = "\n".join(
        f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |" for c in checks
    )
    passed = sum(c.passed for c in checks)
    verdict = (
        "전부 통과함."
        if passed == len(checks)
        else f"{len(checks) - passed}건 실패 — src/agents/evidence_check/agent.py 규칙 로직 확인 필요."
    )
    return f"""# evidence_check 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.7절 / docs/schedule.md 2절
(LLM 미사용, 규칙 기반 노드라 단위 테스트로 검증함)

| 테스트 | 결과 | 비고 |
|---|---|---|
{rows}

![단위 테스트 결과](report_assets/unit_checks.png)

## 결론

{verdict}
"""


def main() -> None:
    checks = run_checks()
    plot_unit_checks(
        checks, f"{AGENT_NAME} 단위 테스트", HERE / "report_assets" / "unit_checks.png"
    )
    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(checks), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
