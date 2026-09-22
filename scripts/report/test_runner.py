"""report 독립 테스트 러너. schedule.md 2절: "8.2 루브릭 + 인용 번호 안전장치
단위 테스트"를 함께 함.

1부(API 키 없이 바로 실행 가능):
   - 인용 번호 안전장치: 환각 근거 번호가 섞인 입력이 원문으로 되돌아가는지(7.10절)
   - 챕터 직렬화(render_*): 9.5절 기준 문구가 포함되는지
   - REFERENCE 유형별 표기(13장), JSON 구조, PDF 변환(한글 폰트 임베딩)
2부(8.2 루브릭, API 필요): 조립된 보고서 초안을 LLM-as-a-Judge로 채점하고
   report_path/report_json_path가 실제 파일을 가리키는지 확인함.

실행: python -m scripts.report.test_runner
전제(2부만): .env에 OPENAI_API_KEY 설정, Ollama에 검수 모델 로드.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.agents.report.agent import (
    ReportAgent,
    build_report_json,
    convert_to_pdf,
    format_reference,
    render_implications,
    render_limitations,
    render_tech_selection,
    render_view_evaluation,
    verify_citations,
)
from src.common.eval_utils import UnitCheck, plot_rubric_scores, plot_unit_checks, score_with_rubric
from src.common.models import get_judge_llm
from src.common.state import (
    Claim,
    Conflict,
    Evidence,
    JudgeFeedback,
    Reference,
    Synthesis,
    TechProfile,
    TechSpec,
    TechViewResult,
    ViewResult,
)

AGENT_NAME = "report"
HERE = Path(__file__).parent
THRESHOLD = 3

_TECHS = [
    TechSpec(name="TurboQuant", camp="SW", role="target", search_anchor="Google"),
    TechSpec(name="ITME", camp="HW", role="target", search_anchor="SK hynix"),
]


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


def run_section_checks() -> list[UnitCheck]:
    """챕터 직렬화가 9.5절 기준 문구를 담는지(LLM 미사용)."""
    checks: list[UnitCheck] = []

    selection = render_tech_selection(_TECHS)
    checks.append(
        UnitCheck(
            "기술 선정: 두 기술명 + Human 기반 명시",
            all(t.name in selection for t in _TECHS) and "Human" in selection,
            selection[:80],
        )
    )

    view = ViewResult(
        by_tech={
            "TurboQuant": TechViewResult(
                confirmed_facts=[Claim(statement="공개 구현이 vLLM에 통합됨", evidence_ids=[1])],
                counter_facts=[Claim(statement="3비트에서 코딩 벤치마크 저하 보고", evidence_ids=[2])],
                unconfirmed_items=["양산 서비스 적용 사례"],
            ),
            "ITME": TechViewResult(
                confirmed_facts=[Claim(statement="FPGA 시제품으로 동작 확인", evidence_ids=[3])],
            ),
        }
    )
    evaluation = render_view_evaluation(view, view, view, view, _TECHS)
    perspective_blocks = [b for b in evaluation.split("### ") if b.strip()]
    checks.append(
        UnitCheck(
            "관점별 평가: 4관점 + TRL 추정 라벨 + 관점마다 두 기술",
            len(perspective_blocks) == 4
            and "공개 정보 기반 추정" in evaluation
            and all("TurboQuant" in b and "ITME" in b for b in perspective_blocks),
            f"블록 {len(perspective_blocks)}개",
        )
    )
    checks.append(
        UnitCheck(
            "관점별 평가: 모든 주장에 근거 번호",
            "[근거#1]" in evaluation and "[근거#2]" in evaluation and "[근거#3]" in evaluation,
            evaluation[:80],
        )
    )

    implications = render_implications(
        Synthesis(
            agreements=["두 기술 모두 실사용 검증 부족[근거#1][근거#3]"],
            conflicts=[Conflict(topic="비용", explanation="SW는 정확도, HW는 인프라 비용을 치름")],
        )
    )
    checks.append(
        UnitCheck(
            "시사점: 일치점·상충점 각 1건 + 상충 이유",
            "일치점" in implications and "상충점" in implications and "치름" in implications,
            implications[:80],
        )
    )

    limitations = render_limitations(
        JudgeFeedback(has_biased_expression=False, sentences_without_evidence=["문장 X"], is_balanced=True, notes="ok"),
        ["양산 일정"],
    )
    checks.append(
        UnitCheck(
            "한계점: 공개 정보 한계 + 미확인 항목 + 편향 조치 + 검수 반영",
            all(k in limitations for k in ("공개 정보", "양산 일정", "확증 편향", "문장 X")),
            limitations[:80],
        )
    )
    return checks


def run_reference_checks() -> list[UnitCheck]:
    """13장 특허/논문/웹페이지 표기 형식."""
    paper = format_reference(
        Reference(id=1, type="paper", author_or_org="Zandieh, A. et al.", year="2025", title="TurboQuant", venue="arXiv, 2504.19874")
    )
    patent = format_reference(
        Reference(id=2, type="patent", author_or_org="SK hynix", year="2026-01", title="계층 메모리", venue="KR10-2026-0000001", url="https://x")
    )
    web = format_reference(
        Reference(id=3, type="web", author_or_org="vLLM Team", year="2026-05-11", title="TurboQuant Study", venue="vLLM Blog", url="https://vllm.ai")
    )
    return [
        UnitCheck("논문: 저자(YYYY). 제목. *학술지*.", paper == "Zandieh, A. et al.(2025). TurboQuant. *arXiv, 2504.19874*.", paper),
        UnitCheck("특허: 출원인(YYYY-MM). *특허명*, 번호, URL", patent == "SK hynix(2026-01). *계층 메모리*, KR10-2026-0000001, https://x", patent),
        UnitCheck("웹: 작성자(YYYY-MM-DD). *제목*. 사이트, URL", web == "vLLM Team(2026-05-11). *TurboQuant Study*. vLLM Blog, https://vllm.ai", web),
    ]


def run_output_format_checks() -> list[UnitCheck]:
    """JSON 구조 왕복 + PDF 변환(한글 임베딩) 확인."""
    checks: list[UnitCheck] = []

    data = build_report_json(
        domain="d",
        techs=_TECHS,
        sections={"SUMMARY": "요약[근거#1]"},
        references=[Reference(id=1, type="paper", author_or_org="A", year="2025", title="T")],
        cited_ids={1},
        judge_feedback=JudgeFeedback(has_biased_expression=False, sentences_without_evidence=[], is_balanced=True),
        generated_at="2026-01-01T00:00:00+00:00",
    )
    roundtrip = json.loads(json.dumps(data, ensure_ascii=False))
    expected_keys = {"generated_at", "domain", "techs", "sections", "references", "cited_evidence_ids", "judge_feedback", "judge_passed"}
    checks.append(
        UnitCheck(
            "JSON: 왕복 + 필수 키 + judge_passed",
            expected_keys <= set(roundtrip) and roundtrip["judge_passed"] is True and roundtrip["cited_evidence_ids"] == [1],
            ", ".join(sorted(roundtrip)),
        )
    )

    try:
        import xhtml2pdf  # noqa: F401
    except ImportError:
        checks.append(UnitCheck("PDF: 변환 (건너뜀)", True, "xhtml2pdf 미설치 — pip install -r requirements.txt 후 재실행"))
        return checks

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "t.pdf"
        ok = convert_to_pdf("# 제목\n\n한글 본문[근거#1]\n\n| a | b |\n|---|---|\n| 가 | 나 |\n", pdf_path)
        detail = f"ok={ok}, size={pdf_path.stat().st_size if pdf_path.exists() else 0}"
        korean_extracted = False
        if ok:
            try:
                import fitz

                korean_extracted = "한글 본문" in fitz.open(str(pdf_path))[0].get_text()
                detail += f", 한글 추출={korean_extracted}"
            except ImportError:
                korean_extracted = True
                detail += ", (pymupdf 없음: 텍스트 추출 생략)"
        checks.append(UnitCheck("PDF: 변환 성공 + 한글 임베딩", ok and korean_extracted, detail))
    return checks


def run_rubric_check():
    fixture_state = {
        "domain": "에이전트형 AI 코딩 서비스의 멀티턴 장문맥 서빙",
        "techs": _TECHS,
        "tech_profiles": {
            "TurboQuant": TechProfile(
                tech="TurboQuant",
                overview="입력 벡터를 무작위 회전시킨 뒤 좌표별 스칼라 양자화를 적용함",
                scope="보정 데이터 없이 토큰 도착 즉시 양자화 가능",
                limitations="3비트 설정에서 코딩 벤치마크 저하가 보고됨",
                differentiation="KIVI와 달리 사전 보정 데이터가 필요 없음",
                evidence_ids=[1],
            ),
            "ITME": TechProfile(
                tech="ITME",
                overview="CXL-Hybrid 메모리를 TB 규모 원격 메모리로 제공하는 계층 구조",
                scope="prefix cache의 예측 가능한 접근 패턴을 이용한 선제 이동",
                limitations="FPGA 시제품 단계로 양산 적용 사례가 없음",
                differentiation="PIM/CXL과 달리 연산을 메모리로 옮기지 않음",
                evidence_ids=[2],
            ),
        },
        "trl_result": ViewResult(
            by_tech={
                "TurboQuant": TechViewResult(confirmed_facts=[Claim(statement="vLLM 통합 구현 공개", evidence_ids=[1])]),
                "ITME": TechViewResult(confirmed_facts=[Claim(statement="FPGA 시제품 동작 확인", evidence_ids=[2])]),
            }
        ),
        "market_result": None,
        "stakeholder_result": None,
        "domain_result": None,
        "synthesis": Synthesis(
            agreements=["두 기술 모두 실사용 규모의 독립 검증은 부족함[근거#1][근거#2]"],
            conflicts=[Conflict(topic="도입 비용", explanation="SW는 정확도 저하를, HW는 새 인프라 비용을 치름")],
            summary="두 기술은 SW/HW 접근이 다름[근거#1][근거#2].",
        ),
        "judge_feedback": JudgeFeedback(has_biased_expression=False, sentences_without_evidence=[], is_balanced=True),
        "evidence": [
            Evidence(id=1, tech="TurboQuant", perspective="tech_research", stance="지지", source_type="논문", source="paper-a", quote="q"),
            Evidence(id=2, tech="ITME", perspective="tech_research", stance="지지", source_type="논문", source="paper-b", quote="q"),
        ],
        "references": [
            Reference(id=1, type="paper", author_or_org="Zandieh, A.", year="2025", title="TurboQuant", venue="arXiv", url="paper-a"),
            Reference(id=2, type="paper", author_or_org="Jang, H.", year="2026", title="ITME", venue="arXiv", url="paper-b"),
        ],
    }
    result = ReportAgent().run(fixture_state)
    report_md = result["report_md"]

    output_checks = [
        UnitCheck("report_path 파일 존재", Path(result["report_path"]).exists(), result["report_path"]),
        UnitCheck("report_path가 PDF", result["report_path"].endswith(".pdf"), result["report_path"]),
    ]
    json_path = Path(result["report_json_path"])
    try:
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        output_checks.append(UnitCheck("report_json 파싱 + sections 포함", "sections" in parsed, str(json_path)))
    except Exception as exc:  # noqa: BLE001
        output_checks.append(UnitCheck("report_json 파싱 + sections 포함", False, str(exc)))

    scores = score_with_rubric(
        get_judge_llm(), target_text=report_md, context="report(7.10절): 13장 목차에 맞춰 조립된 보고서 초안"
    )
    return scores, report_md, output_checks


def _rows(checks: list[UnitCheck]) -> str:
    return "\n".join(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail[:70]} |" for c in checks)


def build_report(unit_checks: list[UnitCheck], scores=None, report_md: str = "", output_checks=None) -> str:
    unit_passed = sum(c.passed for c in unit_checks) == len(unit_checks)

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
        if output_checks:
            rubric_section += f"\n\n산출물 확인:\n\n| 테스트 | 결과 | 비고 |\n|---|---|---|\n{_rows(output_checks)}"

    return f"""# report 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.10절, 9.5절, 13장 / docs/schedule.md 2절

## 1부: 단위 테스트 (인용 안전장치 · 챕터 직렬화 · REFERENCE 표기 · JSON/PDF)

| 테스트 | 결과 | 비고 |
|---|---|---|
{_rows(unit_checks)}

![단위 테스트](report_assets/citation_checks.png)

{"단위 테스트 전부 통과함." if unit_passed else "단위 테스트 실패 — src/agents/report/agent.py 확인 필요."}

## 2부: LLM-as-a-Judge 루브릭 (8.2절)

{rubric_section}

## 검사 대상 보고서 초안

```
{report_md}
```
"""


def main() -> None:
    unit_checks = (
        run_citation_safety_checks() + run_section_checks() + run_reference_checks() + run_output_format_checks()
    )
    plot_unit_checks(unit_checks, f"{AGENT_NAME} 단위 테스트", HERE / "report_assets" / "citation_checks.png")
    for c in unit_checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}")

    scores, report_md, output_checks = None, "", None
    try:
        scores, report_md, output_checks = run_rubric_check()
        plot_rubric_scores(scores, f"{AGENT_NAME} 루브릭 점수", HERE / "report_assets" / "rubric_scores.png")
    except Exception as exc:  # noqa: BLE001 - 2부는 API 키가 없으면 실패할 수 있음, 1부 결과는 보존함
        print(f"2부(루브릭) 실행 실패, 1부 결과만 저장함: {exc}")

    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(build_report(unit_checks, scores, report_md, output_checks), encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
