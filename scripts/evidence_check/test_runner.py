"""evidence_check 독립 테스트 러너. 규칙 기반 노드라 입력→기대 출력 단위
테스트로 검증함(schedule.md 2절). API 키 없이 바로 실행 가능함(테스트 1~5).

테스트 6~10은 확장된 규칙 4(그라운딩 검증/환각 필터링)와 perspective_confidence
계산을 검증함. bge-m3 로컬 임베딩만 쓰고 API 키는 필요 없지만, 최초 1회 모델
다운로드(로컬 캐시)가 필요함.

테스트 11~12는 evidence_ids가 여러 개인 Claim의 유사도 집계 방식(개별 quote와
비교한 값 중 max)을 검증함. 테스트 13~14는 재검색 카운터(retry_count)가
규칙별이 아니라 evidence_check 호출 전체에 걸친 전역 값임을 검증함.

실행: python -m scripts.evidence_check.test_runner
"""

from __future__ import annotations

from pathlib import Path

from src.agents.evidence_check.agent import EvidenceCheckAgent
from src.common.eval_utils import UnitCheck, plot_unit_checks
from src.common.state import Claim, Evidence, TechSpec, TechViewResult, ViewResult

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

    # --- 규칙 4(신규): 그라운딩 검증(환각 필터링) ---
    grounded_evidence = Evidence(
        id=1, tech="A", perspective="trl", stance="지지", source_type="논문",
        source="s", quote="TurboQuant achieves 3.5-bit quantization with no quality loss.",
    )

    # 6. evidence_ids가 없는 주장(출처 없음) -> 제거됨
    view = ViewResult(by_tech={"A": TechViewResult(
        confirmed_facts=[Claim(statement="근거 없이 쓴 주장", evidence_ids=[])],
    )})
    result = agent.run({
        "techs": TECHS, "evidence": [grounded_evidence], "retry_count": 0, "trl_result": view,
    })
    checks.append(UnitCheck(
        "출처 없는 주장(규칙 4) -> confirmed_facts에서 제거",
        result["trl_result"].by_tech["A"].confirmed_facts == [],
    ))

    # 7. 존재하지 않는 근거 번호를 참조하는 주장 -> 제거됨
    view = ViewResult(by_tech={"A": TechViewResult(
        confirmed_facts=[Claim(statement="없는 근거를 인용한 주장", evidence_ids=[999])],
    )})
    result = agent.run({
        "techs": TECHS, "evidence": [grounded_evidence], "retry_count": 0, "trl_result": view,
    })
    checks.append(UnitCheck(
        "존재하지 않는 근거 번호(규칙 4) -> 제거",
        result["trl_result"].by_tech["A"].confirmed_facts == [],
    ))

    # 8. 인용 근거와 무관한 주장(임베딩 유사도 낮음) -> 제거됨
    view = ViewResult(by_tech={"A": TechViewResult(
        confirmed_facts=[Claim(statement="오늘 점심 메뉴는 김치찌개였다", evidence_ids=[1])],
    )})
    result = agent.run({
        "techs": TECHS, "evidence": [grounded_evidence], "retry_count": 0, "trl_result": view,
    })
    checks.append(UnitCheck(
        "근거와 무관한 주장(규칙 4) -> 제거",
        result["trl_result"].by_tech["A"].confirmed_facts == [],
    ))

    # 9. 인용 근거와 부합하는 주장 -> 유지됨
    view = ViewResult(by_tech={"A": TechViewResult(
        confirmed_facts=[Claim(statement="TurboQuant는 3.5비트에서 품질 저하가 없다고 보고됨", evidence_ids=[1])],
    )})
    result = agent.run({
        "techs": TECHS, "evidence": [grounded_evidence], "retry_count": 0, "trl_result": view,
    })
    checks.append(UnitCheck(
        "근거와 부합하는 주장(규칙 4) -> 유지됨",
        len(result["trl_result"].by_tech["A"].confirmed_facts) == 1,
    ))

    # 10. perspective_confidence: 목표치(5건) 이상이면 1.0, 절반이면 0.5 근사
    evidence = _evidence("A", "trl", "지지", 5, 1) + _evidence("B", "trl", "지지", 2, 100)
    result = agent.run({"techs": TECHS, "evidence": evidence, "retry_count": 0})
    conf = result["perspective_confidence"]["trl"]
    checks.append(UnitCheck(
        "perspective_confidence: 목표치 이상 -> 1.0, 목표치 미만 -> 비례",
        conf["A"] == 1.0 and conf["B"] == 0.4,
        f"A={conf['A']}, B={conf['B']}",
    ))

    # --- 유사도 집계 방식(max) 검증: evidence_ids가 2개 이상인 Claim ---
    irrelevant_a = Evidence(
        id=2, tech="A", perspective="trl", stance="지지", source_type="논문",
        source="s", quote="오늘 점심 메뉴는 김치찌개였다.",
    )
    irrelevant_b = Evidence(
        id=3, tech="A", perspective="trl", stance="지지", source_type="논문",
        source="s", quote="제주도 여행 첫날에는 성산일출봉에 갔다.",
    )

    # 11. 인용 근거 2개 중 하나만 유사도 높음 -> max 집계라 통과(유지)해야 함
    view = ViewResult(by_tech={"A": TechViewResult(
        confirmed_facts=[Claim(
            statement="TurboQuant는 3.5비트에서 품질 저하가 없다고 보고됨",
            evidence_ids=[1, 2],  # 1=grounded_evidence(관련 높음), 2=무관
        )],
    )})
    result = agent.run({
        "techs": TECHS, "evidence": [grounded_evidence, irrelevant_a],
        "retry_count": 0, "trl_result": view,
    })
    checks.append(UnitCheck(
        "다중 근거 중 1개만 고유사도(max 집계) -> 유지됨",
        len(result["trl_result"].by_tech["A"].confirmed_facts) == 1,
    ))

    # 12. 인용 근거 2개 다 유사도 낮음 -> 제거돼야 함
    view = ViewResult(by_tech={"A": TechViewResult(
        confirmed_facts=[Claim(
            statement="TurboQuant는 3.5비트에서 품질 저하가 없다고 보고됨",
            evidence_ids=[2, 3],  # 둘 다 무관
        )],
    )})
    result = agent.run({
        "techs": TECHS, "evidence": [irrelevant_a, irrelevant_b],
        "retry_count": 0, "trl_result": view,
    })
    checks.append(UnitCheck(
        "다중 근거 모두 저유사도(max 집계) -> 제거됨",
        result["trl_result"].by_tech["A"].confirmed_facts == [],
    ))

    # --- 재검색 카운터가 규칙별이 아니라 evidence_check 호출 전역임을 검증 ---

    # 13. 예산 소진(retry_count=1) 상태에서 규칙1과 규칙4가 동시에 걸려도
    #     재검색 대상에 추가되지 않음 -> 규칙4가 별도 예산을 갖지 않음(전역 공유) 증명
    evidence = _evidence("A", "trl", "지지", 1, 1)  # 규칙1 위반(<3건)
    view = ViewResult(by_tech={"A": TechViewResult(
        confirmed_facts=[Claim(statement="근거 없이 쓴 주장", evidence_ids=[])],  # 규칙4 위반
    )})
    result = agent.run({
        "techs": TECHS, "evidence": evidence, "retry_count": 1, "trl_result": view,
    })
    checks.append(UnitCheck(
        "예산 소진 상태에서 규칙1+규칙4 동시 위반 -> 재검색 안 함(전역 카운터)",
        result["retry_targets"] == [] and result["retry_count"] == 1,
    ))

    # 14. 예산이 남은 상태(retry_count=0)에서 규칙1(A)과 규칙4(A)가 동시에 걸려도
    #     같은 관점(trl_eval)은 중복 없이 1건만 추가되고, retry_count는 1로만 증가함
    grounded_b = Evidence(
        id=200, tech="B", perspective="trl", stance="지지", source_type="논문",
        source="s", quote="B tech has abundant support evidence.",
    )
    evidence = (
        _evidence("A", "trl", "지지", 1, 1)  # 규칙1 위반(A: <3건)
        + [grounded_b] + _evidence("B", "trl", "지지", 2, 201) + _evidence("B", "trl", "반대", 1, 210)
    )
    view = ViewResult(by_tech={
        "A": TechViewResult(confirmed_facts=[Claim(statement="근거 없이 쓴 주장", evidence_ids=[])]),  # 규칙4 위반
        "B": TechViewResult(confirmed_facts=[Claim(
            statement="B tech has abundant support evidence.", evidence_ids=[200],
        )]),  # 그라운딩 통과
    })
    result = agent.run({"techs": TECHS, "evidence": evidence, "retry_count": 0, "trl_result": view})
    checks.append(UnitCheck(
        "동시 다중 규칙 위반 -> trl_eval 중복 없이 1건, retry_count=1(전역 카운터)",
        result["retry_targets"].count("trl_eval") == 1 and result["retry_count"] == 1,
        f"retry_targets={result['retry_targets']}, retry_count={result['retry_count']}",
    ))

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
