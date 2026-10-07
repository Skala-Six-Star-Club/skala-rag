"""평가 종합 에이전트 (7.8절 + 확장).

관점 결과 4종을 종합하고 evidence_check가 계산한 관점별 근거 신뢰도를
결정론적으로 집계한다. 신뢰도 수치는 기술 간 우열이 아니라 관점별 근거량을
설명하는 값으로만 사용한다.

입력: state의 관점 결과 4종, perspective_confidence(+ 재작성 시 judge_feedback)
출력: {"synthesis": ...}

judge(7.9)에서 반려되면 judge_feedback을 프롬프트에 추가해 동일 입력으로 1회
재실행함(12장 "반복 2"). 재작성 시에도 이전 synthesis 초안은 프롬프트에 남기지
않고 매번 관점 결과 4종에서 새로 작성함(부분 수정이 아닌 답습 위험 회피).

확장 — 근거 검증 & 데이터 종합 역할(팀원 4, 신소영):
evidence_check가 관점별·기술별로 계산한 perspective_confidence(0~1, 근거량 기반)를
그대로 받아 아래 두 값을 코드로 결정론적으로 집계함(LLM 판단 아님, 재현 가능):
- overall_confidence: 전체 평균 — 이번 조사가 전반적으로 근거가 탄탄한지
- weakest_perspective: 평균이 가장 낮은 관점 — 한계점 절에서 "근거가 상대적으로
  부족한 관점"을 구체적으로 짚을 때 씀
이 값들은 기술 간 우열이 아니라 "관점(TRL/시장성/이해관계자/도메인)" 단위의 근거
탄탄함을 나타내므로 10장 중립성 원칙과 충돌하지 않음. LLM에도 이 수치를 함께
넘기되, 프롬프트에서 "기술 비교가 아니라 관점별 근거량 설명에만 쓸 것"을 명시해
judge(7.9)의 우열 판정 표현 검사를 통과하도록 함.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.common.models import get_generation_llm
from src.common.base_agent import BaseAgent
from src.common.observability import log_decision
from src.common.state import AgentState, Claim, Conflict, Synthesis, ViewResult
from src.common.tools import strip_citation_tokens

_PROMPT_TEMPLATE = """\
아래는 {tech_pair} 두 기술에 대한 4개 관점(기술 성숙도, 시장성,
이해관계자, 도메인 적용) 평가 결과임. 각 줄 끝의 (근거 3, 7)이 그 사실의 근거 번호임. 이를 종합해:
- 일치점 1건 이상
- 상충점 1건 이상(단순 나열이 아니라 왜 갈리는지 설명 포함)
- 1/2쪽 이내 SUMMARY (관점 4종을 한 문장씩 언급, 가장 큰 상충 지점 명시)
를 작성해줘. 우열 판정 표현은 쓰지 않음.

출력 규칙:
1. 모든 문장은 text(근거 번호 표기 없이 한 문장)와 evidence_ids(그 문장의 근거가 된 번호 목록)로 나눠 씀
2. evidence_ids에는 아래 평가 결과에 실제로 적힌 근거 번호만 넣음. 근거 번호를 댈 수 없는 문장은 쓰지 않음
3. 여러 관점의 사실을 묶는 문장이면 각 관점의 대표 근거 번호를 넣되, 한 문장의 evidence_ids는 5개 이하
4. 문장 끝은 "~함", "~임", "~됨"처럼 명사형으로 맺고 "~다", "~습니다"는 쓰지 않음

{views}

[관점별 근거 신뢰도(0~1, evidence_check가 근거량 기준으로 계산함)]
{confidence}
위 수치는 "이 관점에서 수집된 근거가 얼마나 충분한가"를 나타낼 뿐, 기술 우열과
무관함. 특정 관점의 신뢰도가 낮으면 그 관점의 결론은 잠정적임을 서술에 반영하되,
두 기술을 비교해 우열을 매기는 데는 쓰지 말 것.
"""

_VIEW_LABELS = {
    "trl_result": "기술 성숙도",
    "market_result": "시장성",
    "stakeholder_result": "이해관계자",
    "domain_result": "도메인 적용",
}


class _CitedSentence(BaseModel):
    text: str
    evidence_ids: list[int] = Field(default_factory=list)


class _ConflictDraft(BaseModel):
    topic: str
    sentences: list[_CitedSentence] = Field(default_factory=list)


class _SynthesisDraft(BaseModel):
    """LLM 구조화 출력용 스키마. 문장마다 근거 번호를 따로 받아 코드가 검증하고 [근거#N]을 붙임.

    동적 dictionary인 perspective_confidence는 LLM Structured Outputs에 직접
    요청하지 않고, 코드가 Synthesis에 별도로 채운다.
    """

    agreements: list[_CitedSentence] = Field(default_factory=list)
    conflicts: list[_ConflictDraft] = Field(default_factory=list)
    summary: list[_CitedSentence] = Field(default_factory=list)


def format_views(state: AgentState) -> str:
    """관점 결과 4종을 "(확인) 문장 (근거 3, 7)" 줄로 직렬화. LLM이 근거 번호를 그대로 옮길 수 있게 함."""
    blocks = []
    for key, label in _VIEW_LABELS.items():
        result = state.get(key)
        lines = [f"[{label}]"]
        if result is not None:
            result = result if isinstance(result, ViewResult) else ViewResult.model_validate(result)
            for tech, view in result.by_tech.items():
                for kind, claims in (("확인", view.confirmed_facts), ("반대", view.counter_facts)):
                    for c in claims:
                        c = c if isinstance(c, Claim) else Claim.model_validate(c)
                        ids = ", ".join(map(str, c.evidence_ids)) or "없음"
                        lines.append(f"- {tech} ({kind}) {c.statement.strip()} (근거 {ids})")
                if view.unconfirmed_items:
                    lines.append(f"- {tech} (미확인) " + "; ".join(view.unconfirmed_items[:5]))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# 한 문장에 근거 번호가 많으면 어느 근거가 무엇을 뒷받침하는지 읽히지 않으므로 앞 번호만 남김
_MAX_IDS_PER_SENTENCE = 5


def render_sentence(sentence: _CitedSentence, valid_ids: set[int]) -> str | None:
    """실제 근거 번호가 하나 이상 있는 문장만 "문장 [근거#3][근거#7]." 형태로 돌려줌. 없으면 None."""
    ids = [i for i in dict.fromkeys(sentence.evidence_ids) if i in valid_ids][:_MAX_IDS_PER_SENTENCE]
    text = strip_citation_tokens(sentence.text).strip().rstrip(".")
    if not ids or not text:
        return None
    return f"{text} " + "".join(f"[근거#{i}]" for i in ids) + "."


def assemble_synthesis(draft: _SynthesisDraft, valid_ids: set[int]) -> tuple[list[str], list[Conflict], str, int]:
    """구조화 초안을 Synthesis 서술 필드로 조립. (일치점, 상충점, SUMMARY, 버린 문장 수)."""
    dropped = 0

    def keep(sentences: list[_CitedSentence]) -> list[str]:
        nonlocal dropped
        rendered = [render_sentence(s, valid_ids) for s in sentences]
        dropped += sum(r is None for r in rendered)
        return [r for r in rendered if r]

    agreements = keep(draft.agreements)
    conflicts = []
    for c in draft.conflicts:
        sentences = keep(c.sentences)
        if sentences:
            conflicts.append(Conflict(topic=c.topic.strip(), explanation=" ".join(sentences)))
    summary = " ".join(keep(draft.summary))
    return agreements, conflicts, summary, dropped


def _aggregate_confidence(
    confidence: dict[str, dict[str, float]],
) -> tuple[float, str | None]:
    """관점별 신뢰도를 평균 내어 전체 평균과 최약 관점을 반환한다."""

    if not confidence:
        return 0.0, None

    per_perspective_avg: dict[str, float] = {}
    all_scores: list[float] = []
    for perspective, by_tech in confidence.items():
        scores = list(by_tech.values())
        if not scores:
            continue
        per_perspective_avg[perspective] = sum(scores) / len(scores)
        all_scores.extend(scores)

    if not all_scores:
        return 0.0, None

    overall = round(sum(all_scores) / len(all_scores), 2)
    weakest = min(per_perspective_avg, key=per_perspective_avg.get)
    return overall, weakest


class SynthesizeAgent(BaseAgent):
    name = "synthesize"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        confidence = state.get("perspective_confidence", {})
        overall_confidence, weakest_perspective = _aggregate_confidence(confidence)

        tech_pair = "와 ".join(f"{t.name}({t.camp})" for t in state.get("techs", []) or []) or "두 기술"
        prompt = _PROMPT_TEMPLATE.format(
            tech_pair=tech_pair,
            views=format_views(state),
            confidence=confidence,
        )

        feedback = state.get("judge_feedback")
        if feedback is not None:
            prompt += f"\n[이전 검수에서 지적된 사항, 반드시 반영할 것]\n{feedback}"
        # 보고서 품질 평가(quality_eval)가 서술 문제로 되돌려 보낸 경우의 지적 사항
        verdict = state.get("eval_result")
        eval_issues = verdict.issues_for("groundedness", "neutrality") if verdict is not None and not verdict.passed else []
        if eval_issues:
            listed = "\n".join(f"- {i}" for i in eval_issues)
            prompt += f"\n[보고서 품질 평가에서 지적된 사항, 반드시 반영할 것]\n{listed}"
        is_rewrite = feedback is not None or bool(eval_issues)

        # 동적 confidence dictionary는 코드에서 조립하고, LLM에는 고정 스키마인
        # 서술 부분만 요청한다.
        llm = get_generation_llm().with_structured_output(_SynthesisDraft)
        draft: _SynthesisDraft = llm.invoke(prompt)  # type: ignore[assignment]
        valid_ids = {e.id for e in state.get("evidence", []) or [] if e.id is not None}
        agreements, conflicts, summary, dropped = assemble_synthesis(draft, valid_ids)
        if dropped:
            log_decision(
                state.get("trace_id"), self.name, "drop_uncited",
                f"근거 번호가 없거나 실제 근거가 아닌 문장 {dropped}개를 종합에서 제외",
            )
        synthesis = Synthesis(
            agreements=agreements,
            conflicts=conflicts,
            summary=summary,
            perspective_confidence=confidence,
            overall_confidence=overall_confidence,
            weakest_perspective=weakest_perspective,
        )

        # 12장 "반복 2" 예산(1회)을 그래프 조건 분기가 확인할 수 있게 재작성 횟수를 기록함
        rewrite_count = state.get("rewrite_count", 0) + (1 if is_rewrite else 0)
        return {"synthesis": synthesis, "rewrite_count": rewrite_count}
