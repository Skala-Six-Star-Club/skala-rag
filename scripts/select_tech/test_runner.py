"""select_tech 독립 테스트 러너. LLM을 쓰지 않는 규칙 기반 노드라 8.2 루브릭
대신 입력→기대 출력 단위 테스트로 검증함(schedule.md 2절). API 키 없이 바로
실행 가능함.

실행: python -m scripts.select_tech.test_runner
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.agents.select_tech.agent import SelectTechAgent
from src.common.eval_utils import UnitCheck, plot_unit_checks

AGENT_NAME = "select_tech"
HERE = Path(__file__).parent

VALID_CONFIG = {
    "domain": "에이전트형 AI 코딩 서비스의 멀티턴 장문맥 서빙",
    "techs": [
        {"name": "TurboQuant", "camp": "SW", "role": "target", "search_anchor": "Google"},
        {"name": "ITME", "camp": "HW", "role": "target", "search_anchor": "SK hynix"},
    ],
}


def _write_config(data: dict, tmp_dir: Path) -> Path:
    path = tmp_dir / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_checks() -> list[UnitCheck]:
    checks: list[UnitCheck] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        path = _write_config(VALID_CONFIG, tmp_dir)
        result = SelectTechAgent(config_path=path).run({})
        checks.append(
            UnitCheck(
                "정상 설정 로드",
                len(result["techs"]) == 2 and result["domain"] == VALID_CONFIG["domain"],
            )
        )

        bad = json.loads(json.dumps(VALID_CONFIG))
        bad["techs"][0]["camp"] = "INVALID"
        path = _write_config(bad, tmp_dir)
        try:
            SelectTechAgent(config_path=path).run({})
            checks.append(UnitCheck("잘못된 camp 값 거부", False, "예외가 발생하지 않음"))
        except ValueError:
            checks.append(UnitCheck("잘못된 camp 값 거부", True))

        bad = json.loads(json.dumps(VALID_CONFIG))
        bad["techs"][0]["role"] = "INVALID"
        path = _write_config(bad, tmp_dir)
        try:
            SelectTechAgent(config_path=path).run({})
            checks.append(UnitCheck("잘못된 role 값 거부", False, "예외가 발생하지 않음"))
        except ValueError:
            checks.append(UnitCheck("잘못된 role 값 거부", True))

        bad = {"techs": VALID_CONFIG["techs"]}  # domain 누락
        path = _write_config(bad, tmp_dir)
        try:
            SelectTechAgent(config_path=path).run({})
            checks.append(UnitCheck("domain 누락 거부", False, "예외가 발생하지 않음"))
        except ValueError:
            checks.append(UnitCheck("domain 누락 거부", True))

    return checks


def build_report(checks: list[UnitCheck]) -> str:
    rows = "\n".join(
        f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |" for c in checks
    )
    passed = sum(c.passed for c in checks)
    verdict = (
        "전부 통과함."
        if passed == len(checks)
        else f"{len(checks) - passed}건 실패 — src/agents/select_tech/agent.py 검증 로직 확인 필요."
    )
    return f"""# select_tech 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.1절 / docs/schedule.md 2절
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
