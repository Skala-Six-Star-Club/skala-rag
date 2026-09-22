"""보고서 생성 에이전트 (7.10절). 13장 목차에 맞춰 State를 조립하고, 다듬기
과정에서 존재하지 않는 근거 번호가 섞여 들어가지 않도록 사후 검증함.

입력: State 전체
출력: {"report_md": ..., "report_path": ..., "report_json_path": ...}

구조는 코드가 결정하고 문장 표현만 모델이 담당함(7.10절): 챕터별 render_* 함수가
State 필드를 마크다운으로 직렬화하고(LLM 미사용), polish_and_verify가 문장만
다듬음. [근거#N] 표기는 고정 토큰으로 두고 결과에 유효 집합 밖 번호가 남으면
원문으로 되돌림("인용 번호 안전장치"). 10장 관계사 고지문은 다듬기 이후에
결정론적으로 덧붙여 모델이 바꾸지 못하게 함.

산출물: output/report.md(본문), output/report.pdf(xhtml2pdf, 실패 시 report_path는
.md로 폴백), output/report.json(챕터·참고문헌·메타데이터 구조화).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.judge.agent import judge_passed
from src.common.base_agent import BaseAgent
from src.common.models import get_generation_llm
from src.common.state import (
    AgentState,
    Evidence,
    JudgeFeedback,
    Reference,
    Synthesis,
    TechProfile,
    TechSpec,
    TechViewResult,
    ViewResult,
)

_CITATION_RE = re.compile(r"\[근거#(\d+)\]")
# synthesize 등 상위 노드가 "[근거#25,#26, 27]"처럼 한 괄호에 여러 번호를 넣는 경우가 있음.
# 인용 안전장치와 REFERENCE 집계는 [근거#N] 단일 토큰만 세므로, 조립 직후 이를 풀어 씀.
_MULTI_CITATION_RE = re.compile(r"\[근거#\s*\d+(?:\s*,\s*#?\s*\d+)+\s*\]")


def normalize_citations(text: str) -> str:
    """[근거#25,#26, 27] -> [근거#25][근거#26][근거#27]. 단일 토큰은 그대로 둠."""

    def _expand(m: re.Match) -> str:
        nums = re.findall(r"\d+", m.group(0))
        return "".join(f"[근거#{n}]" for n in nums)

    return _MULTI_CITATION_RE.sub(_expand, text)
_OUTPUT_DIR = Path("output")
_REPORT_OUTPUT_PATH = _OUTPUT_DIR / "report.md"
_REPORT_PDF_PATH = _OUTPUT_DIR / "report.pdf"
_REPORT_JSON_PATH = _OUTPUT_DIR / "report.json"

_FONT_PATH = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "NanumGothic-Regular.ttf"
_FONT_ALIAS = "NanumGothic-Regular.ttf"

_PERSPECTIVE_ORDER = ("trl", "market", "stakeholder", "domain")
_PERSPECTIVE_LABELS = {
    "trl": "기술 성숙도 (TRL, 공개 정보 기반 추정)",
    "market": "시장성",
    "stakeholder": "이해관계자",
    "domain": "도메인 적용 (에이전트 코딩 멀티턴 서빙)",
}
_CAMP_LABELS = {"SW": "SW 진영 (데이터를 줄이는 접근)", "HW": "HW 진영 (담을 공간을 넓히는 접근)"}

# 설계서 1장 개요. tech_profiles만으로는 병목의 기술적 배경이 드러나지 않을 수 있어
# 고정 도입문을 앞에 둠(9.5절 "KV cache가 병목이 되는 이유가 설명되는가").
_BACKGROUND_INTRO = (
    "KV cache는 LLM이 앞서 계산한 Key와 Value를 재사용하게 해 주는 장치이지만, "
    "문맥이 길어질수록 가속기 메모리를 빠르게 소진함. 에이전트형 코딩 서비스처럼 "
    "긴 공통 문맥 위에서 수십 턴을 이어 가는 워크로드에서는 세션이 끝날 때까지 "
    "캐시를 보관해야 하므로, 캐시를 얼마나 작게 만들 수 있는지와 어디에 싸게 둘 수 "
    "있는지가 서비스 원가로 직결됨. 이 병목을 두고 데이터를 줄이려는 SW 진영과 담을 "
    "공간을 넓히려는 HW 진영이 서로 다른 해법을 내놓고 있음."
)

# 설계서 3장 "선정 이유" 요약. TechSpec에 사유 필드가 없어 State가 아닌 코드 상수로 둠.
_SELECTION_RATIONALE = (
    "기술 선정은 조가 직접 수행한 Human 기반 결정이며, 설정 파일(configs/tech_selection.json)에 "
    "기록된 결과를 그래프의 첫 노드가 읽어 옴. 선정 이유는 세 가지임.\n\n"
    "1. 진영의 발상을 가장 선명하게 대표함 — 한쪽은 기존 하드웨어를 그대로 두고 데이터만 "
    "줄이며, 다른 쪽은 데이터를 그대로 두고 담을 곳을 넓힘.\n"
    "2. 성숙 단계가 뚜렷하게 다름 — 하나는 소프트웨어만으로 적용할 수 있고 다른 하나는 새 "
    "인프라가 필요해 기술 성숙도 관점에서 대비가 분명함.\n"
    "3. 네 관점 모두에서 확인할 공개 자료가 있음 — 시장 반응, 독립 검증, 메모리 업계의 "
    "대응이 공개돼 있어 이해관계자 관점에서 두 기술이 서로 맞물림."
)

_PUBLIC_INFO_LIMITATION = (
    "본 보고서의 모든 평가는 논문, 기사, 제조사 발표 자료 등 공개 정보에 한정한 추정임. "
    "기술을 직접 구현하거나 성능을 재현하지 않았으며, 기술 성숙도(TRL) 구간 역시 공개된 "
    "실험 수준과 제품 발표 보도를 근거로 한 추정이지 실측이 아님. 따라서 공개되지 않은 "
    "내부 실험 결과, 양산 일정, 계약 조건 등은 반영되지 않았음."
)

_BIAS_MITIGATION = (
    "확증 편향을 줄이기 위해 다음 조치를 취함: 두 기술에 같은 질의 구조를 같은 횟수로 "
    "실행하고, 관점마다 한계·비판·실패 사례를 묻는 질의를 따로 실행해 반대 근거로 표시함. "
    "두 기술 사이의 근거 수와 지지 대 반대 비율이 기울면 다시 검색했으며, 근거 번호가 없는 "
    "문장과 근거 목록에 없는 번호를 참조하는 문장은 출력 검증에서 걸러 냄. 종합 평가문의 "
    "중립성 검수는 생성 모델(GPT-5 mini)과 제공사·학습 데이터가 다른 Qwen3-8B가 맡음."
)

# 10장 마지막 항목. 회사명이 바뀌거나 문장이 통째로 사라지면 안 되므로 다듬기 LLM을
# 거치지 않고 run()에서 다듬기 이후에 덧붙임.
_ITME_DISCLOSURE = (
    "평가 대상 기술 중 ITME는 본 교육 과정 관계사인 SK hynix가 제안한 기술임을 밝힘. "
    "이 점을 고려해 위의 중립성 확보 조치를 두 기술에 동일하게 적용했으며, 보고서는 어느 "
    "기술이 나은지를 가리지 않고 관점에 따라 어떻게 다르게 평가되는지를 보여 주는 데 목적을 둠."
)


# ---------------------------------------------------------------------------
# 인용 번호 안전장치 (7.10절)
# ---------------------------------------------------------------------------


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
        "다음 마크다운 문단의 문장 표현만 자연스럽게 다듬어줘. 제목(#), 목록(-), 굵게(**) 같은 "
        "마크다운 구조와 내용의 순서는 그대로 유지하고, 새로운 사실이나 우열 판정을 추가하지 마. "
        "[근거#N] 표기는 절대 바꾸거나 새로 만들지 말고 있는 그대로 유지해.\n\n" + raw_text
    ).content
    return verify_citations(raw_text, polished, valid_ids)


def order_references_by_citation(
    references: list[Reference], cited_order: list[int], evidence: list[Evidence]
) -> list[Reference]:
    """REFERENCE를 본문 인용 순서로 정렬하고 1부터 다시 번호를 매김(13장).

    evidence_finalize는 참고문헌을 URL 순으로 두는데, 보고서에서는 처음 인용된 자료가
    먼저 나오는 편이 관례임. 근거 번호가 본문에 처음 나온 순서를 따라 그 근거의
    reference_url(없으면 source)에 해당하는 참고문헌을 차례로 세우고, 인용 위치를 찾지
    못한 항목은 뒤에 URL 순으로 붙임. 본문은 [근거#N]만 쓰므로 번호 재부여는 안전함.
    """
    by_id = {e.id: e for e in evidence if e.id is not None}
    first_pos: dict[str, int] = {}
    for pos, eid in enumerate(cited_order):
        e = by_id.get(eid)
        if e is None:
            continue
        for key in (getattr(e, "reference_url", None), e.source):
            if key and key not in first_pos:
                first_pos[key] = pos

    def _rank(r: Reference) -> tuple[int, str]:
        candidates = [first_pos[k] for k in (r.url, r.title) if k and k in first_pos]
        return (min(candidates) if candidates else len(cited_order), r.url or r.title)

    ordered = sorted(references, key=_rank)
    return [r.model_copy(update={"id": i}) for i, r in enumerate(ordered, 1)]


def filter_references(references: list[Reference], cited_evidence_sources: set[str]) -> list[Reference]:
    """본문에서 실제로 인용된 근거의 출처만 REFERENCE에 남김(13장).

    cited_evidence_sources에는 인용된 Evidence의 reference_url(논문 arXiv URL, 웹 URL)과
    source를 함께 넣음. 논문 근거의 source는 "기술 p.쪽 절" 형식이라 URL로만 맞음.
    """
    return [r for r in references if r.url in cited_evidence_sources or r.title in cited_evidence_sources]


# ---------------------------------------------------------------------------
# 챕터별 직렬화 (13장 목차, 9.5절 기준). LLM 미사용 순수 함수.
# ---------------------------------------------------------------------------


def _cite(evidence_ids: list[int]) -> str:
    return "".join(f"[근거#{i}]" for i in evidence_ids)


def _tech_names(techs: list[TechSpec]) -> list[str]:
    return [t.name for t in techs]


def render_summary(synthesis: Synthesis | None) -> str:
    """SUMMARY는 synthesize(7.8)가 이미 결론형·관점 4종·최대 상충점 기준으로 생성함.
    report는 재생성하지 않고 그대로 직렬화만 함."""
    if synthesis is None or not synthesis.summary.strip():
        return "(종합 요약 없음)"
    return synthesis.summary.strip()


def render_background(techs: list[TechSpec], tech_profiles: dict[str, TechProfile]) -> str:
    """KV cache 병목 도입문 + 캠프별로 묶은 기술 개요로 SW·HW 대립 구도를 드러냄."""
    parts = [_BACKGROUND_INTRO]
    for camp in ("SW", "HW"):
        camp_techs = [t for t in techs if t.camp == camp]
        if not camp_techs:
            continue
        lines = [f"### {_CAMP_LABELS[camp]}"]
        for tech in camp_techs:
            profile = tech_profiles.get(tech.name)
            if profile is None:
                lines.append(f"- **{tech.name}**: (기술 조사 결과 없음)")
            else:
                lines.append(f"- **{tech.name}**: {profile.overview.strip()}{_cite(profile.evidence_ids)}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def render_tech_selection(techs: list[TechSpec]) -> str:
    """선정 기술 2건 + 선정 사유 + Human 기반 선정 명시(9.5절)."""
    if not techs:
        return _SELECTION_RATIONALE
    rows = ["| 기술 | 진영 | 역할 | 검색 앵커 |", "|---|---|---|---|"]
    rows += [f"| {t.name} | {t.camp} | {t.role} | {t.search_anchor} |" for t in techs]
    return "\n".join(rows) + "\n\n" + _SELECTION_RATIONALE


def render_tech_overview(tech_profiles: dict[str, TechProfile]) -> str:
    """기술별 핵심 접근·적용 범위·한계·같은 진영 다른 방식과의 차이(9.5절).

    TechProfile.evidence_ids는 문장 단위가 아니라 프로필 단위라, 근거 번호는 기술마다
    한 번 붙임(state.py 스키마 한계)."""
    if not tech_profiles:
        return "(기술 조사 결과 없음)"
    parts = []
    for name, profile in tech_profiles.items():
        parts.append(
            f"### {name}\n\n"
            f"- **핵심 접근**: {profile.overview.strip()}\n"
            f"- **적용 범위**: {profile.scope.strip()}\n"
            f"- **한계**: {profile.limitations.strip()}\n"
            f"- **같은 진영 다른 방식과의 차이**: {profile.differentiation.strip()}\n\n"
            f"근거: {_cite(profile.evidence_ids) or '(없음)'}"
        )
    return "\n\n".join(parts)


def _render_tech_view(tech_name: str, view: TechViewResult | None) -> str:
    """기술 하나의 관점 결과를 확인/반대/미확인 항목으로 렌더링. 모든 Claim에 근거 번호를 붙임."""
    lines = [f"**{tech_name}**"]
    if view is None:
        lines.append("- (평가 결과 없음)")
        return "\n".join(lines)
    for claim in view.confirmed_facts:
        lines.append(f"- (확인) {claim.statement.strip()}{_cite(claim.evidence_ids)}")
    for claim in view.counter_facts:
        lines.append(f"- (반대) {claim.statement.strip()}{_cite(claim.evidence_ids)}")
    if view.unconfirmed_items:
        lines.append("- 미확인 항목: " + "; ".join(i.strip() for i in view.unconfirmed_items))
    if len(lines) == 1:
        lines.append("- (확인된 사실 없음)")
    return "\n".join(lines)


def render_view_evaluation(
    trl_result: ViewResult | None,
    market_result: ViewResult | None,
    stakeholder_result: ViewResult | None,
    domain_result: ViewResult | None,
    techs: list[TechSpec],
) -> str:
    """4개 관점을 순서대로, 관점마다 두 기술을 나란히 서술(9.5절, 13장)."""
    results = {
        "trl": trl_result,
        "market": market_result,
        "stakeholder": stakeholder_result,
        "domain": domain_result,
    }
    parts = []
    for idx, key in enumerate(_PERSPECTIVE_ORDER, start=1):
        result = results[key]
        block = [f"### 4.{idx} {_PERSPECTIVE_LABELS[key]}"]
        for name in _tech_names(techs):
            view = result.by_tech.get(name) if result is not None else None
            block.append(_render_tech_view(name, view))
        parts.append("\n\n".join(block))
    return "\n\n".join(parts)


def render_implications(synthesis: Synthesis | None) -> str:
    """일치점·상충점(왜 갈리는지 포함)만 서술. summary는 SUMMARY 챕터가 이미 씀."""
    if synthesis is None:
        return "(종합 결과 없음)"
    agreements = "\n".join(f"- {a.strip()}" for a in synthesis.agreements) or "- (일치점 없음)"
    conflicts = (
        "\n".join(f"- **{c.topic.strip()}**: {c.explanation.strip()}" for c in synthesis.conflicts)
        or "- (상충점 없음)"
    )
    return f"### 관점 간 일치점\n\n{agreements}\n\n### 관점 간 상충점\n\n{conflicts}"


def collect_unconfirmed_items(*results: ViewResult | None) -> list[str]:
    """관점 결과 4종의 미확인 항목을 순서 유지·중복 제거로 모음(9.5절 한계점 기준)."""
    seen: dict[str, None] = {}
    for result in results:
        if result is None:
            continue
        for view in result.by_tech.values():
            for item in view.unconfirmed_items:
                seen.setdefault(item.strip(), None)
    return list(seen)


def render_limitations(judge_feedback: JudgeFeedback | None, unconfirmed_items: list[str] | None = None) -> str:
    """공개 정보 한계 + 미확인 항목 + 편향 완화 조치 + judge_feedback 반영(9.5절).
    10장 관계사 고지는 여기 넣지 않고 run()이 다듬기 이후에 덧붙임."""
    parts = [f"### 공개 정보 기반 추정의 한계\n\n{_PUBLIC_INFO_LIMITATION}"]

    if unconfirmed_items:
        items = "\n".join(f"- {i}" for i in unconfirmed_items)
        parts.append(f"### 확인하지 못한 항목\n\n{items}")

    parts.append(f"### 확증 편향을 줄이기 위한 조치\n\n{_BIAS_MITIGATION}")

    if judge_feedback is not None:
        lines = []
        lines.append(
            "- 우열 판정 표현: " + ("검수에서 지적돼 재작성함" if judge_feedback.has_biased_expression else "없음")
        )
        lines.append("- 두 기술 서술 균형: " + ("균형" if judge_feedback.is_balanced else "불균형 — 한쪽 기술의 반대 근거가 부족함"))
        if judge_feedback.sentences_without_evidence:
            listed = "\n".join(f"  - {s.strip()}" for s in judge_feedback.sentences_without_evidence)
            lines.append(f"- 근거 번호 없이 서술돼 지적된 문장:\n{listed}")
        if judge_feedback.notes.strip():
            lines.append(f"- 검수 비고: {judge_feedback.notes.strip()}")
        parts.append("### 중립성 검수 결과\n\n" + "\n".join(lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# REFERENCE (13장 표기 형식)
# ---------------------------------------------------------------------------


def format_reference(ref: Reference) -> str:
    """13장 특허/논문/웹페이지 표기 형식.

    Reference 모델에 특허번호·권(호)·페이지 전용 필드가 없어 venue를 유형별로 다른
    의미로 씀(특허: 특허번호/공개번호, 논문: 학술지/학회명, 웹: 사이트명). state.py는
    형제 노드들과 공유하는 스키마라 여기서 필드를 늘리지 않고, 값이 없으면 자리표시
    문구로 대신함(설계서 REFERENCE 절도 arXiv 논문을 권(호)/페이지 없이 표기함).
    """
    if ref.type == "patent":
        identifier = ref.venue or "특허번호 미기재"
        tail = f", {ref.url}" if ref.url else ""
        return f"{ref.author_or_org}({ref.year}). *{ref.title}*, {identifier}{tail}"
    if ref.type == "paper":
        venue = ref.venue or "출처 미기재"
        return f"{ref.author_or_org}({ref.year}). {ref.title}. *{venue}*."
    if ref.type == "web":
        venue = ref.venue or "사이트명 미기재"
        tail = f", {ref.url}" if ref.url else ""
        return f"{ref.author_or_org}({ref.year}). *{ref.title}*. {venue}{tail}"
    raise ValueError(f"unknown reference type: {ref.type}")


def render_references(references: list[Reference]) -> str:
    if not references:
        return "(본문에서 인용된 자료 없음)"
    return "\n".join(f"- [{r.id}] {format_reference(r)}" for r in references)


# ---------------------------------------------------------------------------
# 출력 형식: JSON / PDF
# ---------------------------------------------------------------------------


def build_report_json(
    *,
    domain: str,
    techs: list[TechSpec],
    sections: dict[str, str],
    references: list[Reference],
    cited_ids: set[int],
    judge_feedback: JudgeFeedback | None,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "domain": domain,
        "techs": [t.model_dump() for t in techs],
        "sections": sections,
        "references": [{**r.model_dump(), "formatted": format_reference(r)} for r in references],
        "cited_evidence_ids": sorted(cited_ids),
        "judge_feedback": judge_feedback.model_dump() if judge_feedback is not None else None,
        "judge_passed": judge_passed(judge_feedback) if judge_feedback is not None else None,
    }


def write_report_json(data: dict[str, Any], path: Path = _REPORT_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def render_html(report_md: str) -> str:
    """마크다운 -> HTML. 폰트는 CSS에 ASCII 별칭만 두고 convert_to_pdf의 link_callback이
    실제 경로로 바꿔 줌 — xhtml2pdf CSS 파서가 url() 안의 한글 경로를 못 읽기 때문."""
    import markdown

    body = markdown.markdown(report_md, extensions=["tables", "fenced_code"])
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@font-face {{ font-family: NanumGothic; src: url({_FONT_ALIAS}); }}
* {{ font-family: NanumGothic; }}
@page {{ size: A4; margin: 2cm; }}
body {{ font-size: 10.5pt; line-height: 1.5; }}
h1 {{ font-size: 18pt; margin-top: 18pt; }}
h2 {{ font-size: 14pt; margin-top: 16pt; border-bottom: 1px solid #999; }}
h3 {{ font-size: 12pt; margin-top: 12pt; }}
table {{ width: 100%; margin: 6pt 0; }}
th, td {{ border: 1px solid #999; padding: 4pt; font-size: 9.5pt; }}
th {{ background: #eee; }}
</style></head><body>{body}</body></html>"""


def convert_to_pdf(report_md: str, output_path: Path = _REPORT_PDF_PATH) -> bool:
    """report_md를 PDF로 변환. 실패(미설치·폰트 없음·변환 오류)하면 False만 반환하고
    파일을 남기지 않음 — report_path가 항상 존재하는 파일을 가리키게 하기 위함."""
    try:
        from xhtml2pdf import pisa
        from xhtml2pdf.config.resources import default_policy
    except ImportError:
        return False
    if not _FONT_PATH.exists():
        return False

    def _resolve_font(uri: str, _rel: str | None) -> str | None:
        return str(_FONT_PATH) if uri == _FONT_ALIAS else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("wb") as f:
            result = pisa.CreatePDF(
                render_html(report_md),
                dest=f,
                encoding="utf-8",
                link_callback=_resolve_font,
                resource_policy=default_policy(_FONT_PATH.parent),
            )
        ok = not result.err and output_path.stat().st_size > 0
    except Exception:  # noqa: BLE001 - 변환 실패는 .md 폴백으로 흡수함
        ok = False
    if not ok:
        output_path.unlink(missing_ok=True)
    return ok


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------


class ReportAgent(BaseAgent):
    name = "report"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        valid_ids = collect_valid_evidence_ids(state)
        techs = state.get("techs", [])
        tech_profiles = state.get("tech_profiles", {}) or {}
        synthesis = state.get("synthesis")
        judge_feedback = state.get("judge_feedback")
        view_results = tuple(
            state.get(k) for k in ("trl_result", "market_result", "stakeholder_result", "domain_result")
        )

        sections = {
            "SUMMARY": render_summary(synthesis),
            "1. 분석 배경": render_background(techs, tech_profiles),
            "2. 기술 선정": render_tech_selection(techs),
            "3. 기술 개요": render_tech_overview(tech_profiles),
            "4. 관점별 평가": render_view_evaluation(*view_results, techs),
            "5. 시사점": render_implications(synthesis),
            "6. 한계점": render_limitations(judge_feedback, collect_unconfirmed_items(*view_results)),
        }

        sections = {title: normalize_citations(body) for title, body in sections.items()}
        polished_sections = {
            title: polish_and_verify(body, valid_ids) for title, body in sections.items()
        }
        polished_sections["6. 한계점"] = polished_sections["6. 한계점"].rstrip() + "\n\n" + _ITME_DISCLOSURE

        # 본문(SUMMARY -> 6. 한계점 순)에 [근거#N]이 처음 등장하는 순서
        cited_order: list[int] = []
        for body in polished_sections.values():
            for n in _CITATION_RE.findall(body):
                if int(n) not in cited_order:
                    cited_order.append(int(n))
        cited_ids = set(cited_order)
        cited_sources = {
            key
            for e in state.get("evidence", [])
            if e.id in cited_ids
            for key in (e.source, getattr(e, "reference_url", None))
            if key
        }
        references = filter_references(state.get("references", []), cited_sources)
        references = order_references_by_citation(references, cited_order, state.get("evidence", []))

        title = f"# KV cache 최적화 기술 다관점 평가 보고서: {' vs '.join(_tech_names(techs)) or '(기술 미지정)'}"
        body = "\n\n".join(f"## {t}\n\n{text}" for t, text in polished_sections.items())
        report_md = f"{title}\n\n{body}\n\n## REFERENCE\n\n{render_references(references)}\n"

        _REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_OUTPUT_PATH.write_text(report_md, encoding="utf-8")

        report_path = _REPORT_PDF_PATH if convert_to_pdf(report_md, _REPORT_PDF_PATH) else _REPORT_OUTPUT_PATH

        write_report_json(
            build_report_json(
                domain=state.get("domain", ""),
                techs=techs,
                sections=polished_sections,
                references=references,
                cited_ids=cited_ids,
                judge_feedback=judge_feedback,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
            _REPORT_JSON_PATH,
        )

        return {
            "report_md": report_md,
            "report_path": str(report_path),
            "report_json_path": str(_REPORT_JSON_PATH),
        }
