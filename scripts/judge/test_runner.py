"""judge 독립 테스트 러너. 이진 규칙 판정이라 8.2 루브릭 대신 판정 정확도
스팟체크로 검증함(schedule.md 2절): 위반이 뚜렷한 입력과 깨끗한 입력을 각각
넣어 JudgeFeedback이 기대한 대로 나오는지 확인함. LLM 호출 결과라 완벽한
재현성을 보장하진 않는 스팟체크임.

1부(LLM 미사용): judge_passed() — graph.py 조건부 엣지가 쓸 순수 판정 함수.
2부(Ollama 필요): 우열 표현 / 근거 없는 문장 / 균형 여부 스팟체크.
   is_balanced는 설계서 7.9절 정의("두 기술 모두 반대 근거 1건 이상")대로 근거
   분포가 균형인 픽스처와 한쪽 반대 근거가 0건인 픽스처를 각각 넣어 검증함.

실행: python -m scripts.judge.test_runner
전제(2부만): Ollama에 검수 모델(OLLAMA_JUDGE_MODEL, 기본 qwen3:8b) 로드.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.judge.agent import JudgeAgent, judge_passed
from src.common.eval_utils import UnitCheck, plot_unit_checks
from src.common.state import Evidence, JudgeFeedback, Synthesis

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


def _evidence(stances: list[tuple[str, str]]) -> list[Evidence]:
    return [
        Evidence(id=i, tech=t, perspective="trl", stance=s, source_type="논문", source="s", quote="q")
        for i, (t, s) in enumerate(stances, start=1)
    ]


# 두 기술 모두 지지 1건 + 반대 1건 -> is_balanced 기대값 True
_EVIDENCE = _evidence([("TurboQuant", "지지"), ("TurboQuant", "반대"), ("ITME", "지지"), ("ITME", "반대")])
# ITME 반대 근거 0건 -> is_balanced 기대값 False (문장은 _CLEAN_SYNTHESIS 그대로)
_EVIDENCE_IMBALANCED = _evidence([("TurboQuant", "지지"), ("TurboQuant", "반대"), ("ITME", "지지"), ("ITME", "지지")])


def run_pure_checks() -> list[UnitCheck]:
    """judge_passed()는 LLM과 무관한 순수 함수라 항상 통과해야 함."""
    ok = JudgeFeedback(has_biased_expression=False, sentences_without_evidence=[], is_balanced=True)
    biased = ok.model_copy(update={"has_biased_expression": True})
    uncited = ok.model_copy(update={"sentences_without_evidence": ["근거 없는 문장"]})
    unbalanced = ok.model_copy(update={"is_balanced": False})
    return [
        UnitCheck("judge_passed: 모두 통과 -> True", judge_passed(ok) is True, str(ok)),
        UnitCheck("judge_passed: 우열 표현 -> False", judge_passed(biased) is False, str(biased)),
        UnitCheck("judge_passed: 근거 없는 문장 -> False", judge_passed(uncited) is False, str(uncited)),
        UnitCheck("judge_passed: 불균형 -> False", judge_passed(unbalanced) is False, str(unbalanced)),
    ]


def run_llm_checks() -> list[UnitCheck]:
    agent = JudgeAgent()
    checks: list[UnitCheck] = []

    biased_feedback = agent.run({"synthesis": _BIASED_SYNTHESIS, "evidence": _EVIDENCE})["judge_feedback"]
    checks.append(UnitCheck("우열 판정 표현 탐지", biased_feedback.has_biased_expression is True, str(biased_feedback)))
    checks.append(
        UnitCheck("근거 없는 문장 탐지", len(biased_feedback.sentences_without_evidence) > 0, str(biased_feedback))
    )
    checks.append(UnitCheck("위반 입력 -> judge_passed False", judge_passed(biased_feedback) is False, str(biased_feedback)))

    clean_feedback = agent.run({"synthesis": _CLEAN_SYNTHESIS, "evidence": _EVIDENCE})["judge_feedback"]
    checks.append(UnitCheck("깨끗한 입력 -> 우열 표현 없음", clean_feedback.has_biased_expression is False, str(clean_feedback)))
    checks.append(
        UnitCheck("깨끗한 입력 -> 근거 없는 문장 없음", len(clean_feedback.sentences_without_evidence) == 0, str(clean_feedback))
    )
    checks.append(UnitCheck("근거 분포 균형 -> is_balanced True", clean_feedback.is_balanced is True, str(clean_feedback)))
    checks.append(UnitCheck("깨끗한 입력 -> judge_passed True", judge_passed(clean_feedback) is True, str(clean_feedback)))

    imbalanced_feedback = agent.run({"synthesis": _CLEAN_SYNTHESIS, "evidence": _EVIDENCE_IMBALANCED})["judge_feedback"]
    checks.append(
        UnitCheck("한쪽 반대 근거 0건 -> is_balanced False", imbalanced_feedback.is_balanced is False, str(imbalanced_feedback))
    )
    return checks


def build_report(pure_checks: list[UnitCheck], llm_checks: list[UnitCheck], llm_error: str | None) -> str:
    def rows(checks: list[UnitCheck]) -> str:
        return "\n".join(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |" for c in checks)

    pure_verdict = "순수 함수 검사 전부 통과함." if all(c.passed for c in pure_checks) else "judge_passed() 실패 — 조건식 확인 필요."

    if llm_error:
        llm_section = f"(2부 미실행 — Ollama 설정 후 재실행할 것: {llm_error})"
    else:
        passed = sum(c.passed for c in llm_checks)
        verdict = (
            "스팟체크 전부 통과함."
            if passed == len(llm_checks)
            else f"{len(llm_checks) - passed}건 실패 — Qwen3-8B 프롬프트(7.9절)를 다듬을 것. "
            "LLM 판정이므로 1회 실패가 곧 설계 결함을 의미하진 않음, 여러 번 재실행해 볼 것."
        )
        llm_section = f"| 테스트 | 결과 | 비고 |\n|---|---|---|\n{rows(llm_checks)}\n\n{verdict}"

    return f"""# judge 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.9절, 12장 "반복 2" / docs/schedule.md 2절
(이진 판정 노드라 판정 정확도 스팟체크로 검증함)

## 1부: judge_passed() 순수 함수 (graph.py 조건부 엣지용)

| 테스트 | 결과 | 비고 |
|---|---|---|
{rows(pure_checks)}

{pure_verdict}

## 2부: Qwen3-8B 판정 스팟체크

{llm_section}

![스팟체크 결과](report_assets/unit_checks.png)
"""


def main() -> None:
    pure_checks = run_pure_checks()
    llm_checks: list[UnitCheck] = []
    llm_error: str | None = None
    try:
        llm_checks = run_llm_checks()
    except Exception as exc:  # noqa: BLE001 - Ollama 미기동 시 1부 결과는 보존함
        llm_error = str(exc)
        print(f"2부(LLM 스팟체크) 실행 실패, 1부 결과만 저장함: {exc}")

    for c in pure_checks + llm_checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}")

    plot_unit_checks(pure_checks + llm_checks, f"{AGENT_NAME} 판정 스팟체크", HERE / "report_assets" / "unit_checks.png")
    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(pure_checks, llm_checks, llm_error), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
