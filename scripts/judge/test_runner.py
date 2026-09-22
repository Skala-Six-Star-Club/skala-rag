"""judge 독립 테스트 러너. 이진 규칙 판정이라 8.2 루브릭 대신 판정 정확도
스팟체크로 검증함(schedule.md 2절): 위반이 뚜렷한 입력과 깨끗한 입력을 각각
넣어 JudgeFeedback이 기대한 대로 나오는지 확인함. LLM 호출 결과라 완벽한
재현성을 보장하진 않는 스팟체크임.

judge/agent.py는 자리표시자가 아니라 실제 구조화 출력 호출까지 돼 있어 지금
실행해도 됨. 다만 프롬프트(_PROMPT_TEMPLATE)는 다듬어지지 않은 초안이므로,
스팟체크가 실패하면 "만든다 -> 돌린다 -> 채점한다 -> 프롬프트를 고쳐 재실행한다"
(schedule.md 1절) 루프를 그대로 밟으면 됨.

실행: python -m scripts.judge.test_runner
전제: Ollama에 검수 모델(OLLAMA_JUDGE_MODEL, 기본 qwen3:8b) 로드.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.judge.agent import JudgeAgent
from src.common.eval_utils import UnitCheck, plot_unit_checks
from src.common.state import Evidence, Synthesis

AGENT_NAME = "judge"
HERE = Path(__file__).parent

_BIASED_SYNTHESIS = Synthesis(
    agreements=["두 기술 모두 공개된 실험만으로는 실사용 검증이 부족함"],
    conflicts=[],
    summary=(
        "TurboQuant가 ITME보다 훨씬 우수하고 실용적인 기술임. "
        "ITME는 아직 갈 길이 멀다."  # 우열 판정 + 근거 번호 없음(의도적으로 위반 유발)
    ),
)

_CLEAN_SYNTHESIS = Synthesis(
    agreements=["두 기술 모두 공개 정보 기반 추정의 한계를 가짐[근거#1][근거#2]"],
    conflicts=[],
    summary=(
        "TurboQuant는 SW 관점에서[근거#1], ITME는 HW 관점에서[근거#2] 각각 KV cache "
        "병목에 접근함. 두 기술 모두 실사용 규모의 독립 검증은 아직 부족함[근거#3][근거#4]."
    ),
)

_EVIDENCE = [
    Evidence(id=i, tech=t, perspective="trl", stance="지지", source_type="논문", source="s", quote="q")
    for i, t in enumerate(["TurboQuant", "ITME", "TurboQuant", "ITME"], start=1)
]


def run_checks() -> list[UnitCheck]:
    agent = JudgeAgent()
    checks: list[UnitCheck] = []

    biased_feedback = agent.run({"synthesis": _BIASED_SYNTHESIS, "evidence": _EVIDENCE})["judge_feedback"]
    checks.append(
        UnitCheck(
            "우열 판정 표현 탐지",
            biased_feedback.has_biased_expression is True,
            str(biased_feedback),
        )
    )
    checks.append(
        UnitCheck(
            "근거 없는 문장 탐지",
            len(biased_feedback.sentences_without_evidence) > 0,
            str(biased_feedback),
        )
    )

    clean_feedback = agent.run({"synthesis": _CLEAN_SYNTHESIS, "evidence": _EVIDENCE})["judge_feedback"]
    checks.append(
        UnitCheck(
            "깨끗한 입력 -> 위반 없음",
            clean_feedback.has_biased_expression is False,
            str(clean_feedback),
        )
    )

    return checks


def build_report(checks: list[UnitCheck]) -> str:
    rows = "\n".join(
        f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |" for c in checks
    )
    passed = sum(c.passed for c in checks)
    verdict = (
        "스팟체크 전부 통과함."
        if passed == len(checks)
        else f"{len(checks) - passed}건 실패 — Qwen3-8B 프롬프트(7.9절)를 다듬을 것. "
        "LLM 판정이므로 1회 실패가 곧 설계 결함을 의미하진 않음, 여러 번 재실행해 볼 것."
    )
    return f"""# judge 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.9절 / docs/schedule.md 2절
(이진 판정 노드라 판정 정확도 스팟체크로 검증함)

| 테스트 | 결과 | 비고 |
|---|---|---|
{rows}

![스팟체크 결과](report_assets/unit_checks.png)

## 결론

{verdict}
"""


def main() -> None:
    checks = run_checks()
    plot_unit_checks(
        checks, f"{AGENT_NAME} 판정 스팟체크", HERE / "report_assets" / "unit_checks.png"
    )
    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(checks), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
