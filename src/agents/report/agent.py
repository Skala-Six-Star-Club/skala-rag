"""보고서 생성 에이전트 (7.10절). 13장 목차에 맞춰 State를 조립하고, 다듬기
과정에서 존재하지 않는 근거 번호가 섞여 들어가지 않도록 사후 검증함.

입력: State 전체
출력: {"report_md": ..., "report_path": ...}

다듬기(polish_and_verify)는 [근거#N] 표기를 문장 표현과 분리된 고정 토큰으로
다루도록 지시하고, 결과에 유효 근거 번호 집합 밖의 번호가 남아 있으면 원문으로
되돌림(7.10절 "인용 번호 안전장치"). PDF 변환은 아직 구현하지 않음(TODO).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.common.base_agent import BaseAgent
from src.common.models import get_generation_llm
from src.common.state import AgentState, Reference

_CITATION_RE = re.compile(r"\[근거#(\d+)\]")
_REPORT_OUTPUT_PATH = Path("output/report.md")


def collect_valid_evidence_ids(state: AgentState) -> set[int]:
    """synthesis + 관점 결과 4종에 실제로 등록된 근거 번호를 모음(7.10절)."""
    return {e.id for e in state.get("evidence", []) if e.id is not None}


def verify_citations(original_text: str, candidate_text: str, valid_ids: set[int]) -> str:
    """다듬어진 문장에 유효 집합 밖 근거 번호가 있으면 원문으로 되돌림(7.10절 안전장치).

    LLM 호출과 분리된 순수 함수로 둬서, 환각 번호가 섞인 입력을 넣었을 때
    원문으로 되돌아가는지를 LLM 없이도 단위 테스트할 수 있게 함(schedule.md 2절).
    """
    cited = {int(n) for n in _CITATION_RE.findall(candidate_text)}
    if not cited.issubset(valid_ids):
        return original_text
    return candidate_text


def polish_and_verify(raw_text: str, valid_ids: set[int]) -> str:
    """문장을 다듬되, 결과에 유효 집합 밖 근거 번호가 있으면 원문으로 되돌림."""
    if not raw_text.strip():
        return raw_text
    llm = get_generation_llm()
    polished = llm.invoke(
        "다음 문단을 자연스럽게 다듬어줘. [근거#N] 표기는 절대 바꾸거나 새로 "
        "만들지 말고 있는 그대로 유지해.\n\n" + raw_text
    ).content
    return verify_citations(raw_text, polished, valid_ids)


def filter_references(references: list[Reference], cited_evidence_sources: set[str]) -> list[Reference]:
    """본문에서 실제로 인용된 근거의 출처만 REFERENCE에 남김(13장)."""
    return [r for r in references if r.url in cited_evidence_sources or r.title in cited_evidence_sources]


class ReportAgent(BaseAgent):
    name = "report"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        valid_ids = collect_valid_evidence_ids(state)
        synthesis = state.get("synthesis")
        judge_feedback = state.get("judge_feedback")

        # TODO(담당자): State 필드를 사람이 읽기 좋은 문장으로 직렬화하는 로직을
        # 채울 것(지금은 각 절이 원본 객체를 그대로 문자열로 넣은 자리표시자임).
        sections = {
            "SUMMARY": getattr(synthesis, "summary", ""),
            "1. 분석 배경": str(state.get("tech_profiles", {})),
            "2. 기술 선정": str(state.get("techs", [])),
            "3. 기술 개요": str(state.get("tech_profiles", {})),
            "4. 관점별 평가": "\n".join(
                str(state.get(k))
                for k in ("trl_result", "market_result", "stakeholder_result", "domain_result")
            ),
            "5. 시사점": str(synthesis),
            "6. 한계점": str(judge_feedback),
        }

        polished_sections = {
            title: polish_and_verify(body, valid_ids) for title, body in sections.items()
        }

        cited_ids = {
            int(n) for body in polished_sections.values() for n in _CITATION_RE.findall(body)
        }
        cited_sources = {e.source for e in state.get("evidence", []) if e.id in cited_ids}
        references = filter_references(state.get("references", []), cited_sources)

        body = "\n\n".join(f"## {title}\n\n{text}" for title, text in polished_sections.items())
        ref_lines = "\n".join(f"- [{r.id}] {r.author_or_org}({r.year}). {r.title}" for r in references)
        report_md = f"{body}\n\n## REFERENCE\n\n{ref_lines}\n"

        _REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_OUTPUT_PATH.write_text(report_md, encoding="utf-8")
        # TODO(담당자): PDF 변환(예: weasyprint/pandoc)을 추가해 report_path가
        # 실제 PDF를 가리키게 할 것. 지금은 마크다운 경로를 그대로 둠.

        return {"report_md": report_md, "report_path": str(_REPORT_OUTPUT_PATH)}
