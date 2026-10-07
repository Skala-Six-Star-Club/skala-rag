"""보고서 품질 평가 에이전트. 보고서 생성 뒤에 실행하고, 미달이면 Loop를 요청함.

평가 방식은 Hybrid(규칙 + LLM Judge). 판정 대상은 report가 남긴 ``report.json``의
``evidence_token_sections``(인용이 [근거#N] 토큰으로 남아 있는 본문)임.

| 항목 | 방식 | 판정 |
|---|---|---|
| groundedness | 규칙 + 임베딩 | 인용 번호가 모두 실제 근거인지, 인용이 붙은 문장과 인용문의 코사인 유사도 |
| neutrality | 규칙 + LLM | 금지 표현 목록, 검수 LLM(Qwen3-8B)의 우열 판정 표현과 근거 없는 사실 문장 |
| bias_control | 규칙 | (관점, 기술)별 인용 출처 수, 기술별 반대 근거 인용, 기술 간 인용 수 비율 |
| coverage | 규칙 | 관점별 평가 절에 4개 관점 x 기술 모든 칸의 인용이 있는지 |

미달 시 되돌아갈 곳
- coverage, bias_control 미달: orchestrator로 부족한 (관점, 기술)만 re-plan (MAX_QUALITY_REPLANS회)
- groundedness, neutrality 미달: synthesize로 재작성 (MAX_QUALITY_REWRITES회)
- 두 예산은 따로 셈. 둘 다 미달이면 재조사를 먼저 하고, 재조사 뒤 다시 들어온 synthesize가 서술 지적도 반영함
- 예산을 소진했으면 종료하고 미달 내용을 report.json과 보고서 한계점에 남김
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.models import get_embedding_model, get_judge_llm
from src.common.observability import log_decision
from src.common.state import AgentState, CriterionResult, EvalVerdict, Evidence, Gap

_CITATION_RE = re.compile(r"\[근거#(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# "문장임. [근거#3]"처럼 마침표 뒤에 붙은 인용을 마침표 앞으로 옮겨 다음 문장에 붙지 않게 함
_TRAILING_CITATION_RE = re.compile(r"([.!?])\s*((?:\[근거#\d+\]\s*)+)")
_VIEW_SECTION = "4. 관점별 평가"
_NARRATIVE_SECTIONS = ("SUMMARY", "5. 시사점")
_PERSPECTIVES = ("trl", "market", "stakeholder", "domain")

# 우열 판정, 추천 표현. LLM Judge와 별개로 결정론적으로 걸러냄
_BIASED_PATTERNS = [
    r"더\s*(우수|낫|뛰어나|유리)",
    r"(우월|열등)",
    r"(압도적|명백히)\s*앞",
    r"도입을\s*(추천|권장)",
    r"(추천|권장)한다",
    r"뒤처",
    r"갈 길이 멀",
]

_JUDGE_PROMPT = """\
당신은 두 기술의 다관점 평가 보고서에서 중립성만 검사하는 검수자임.
사실 여부는 검사하지 않고 표현과 형식만 봄. 점수를 매기지 말고 구조화된 필드로만 답함.

[biased_sentences] 한 기술이 전반적으로 더 낫다고 결론짓거나 도입을 추천하는 문장을 그대로 옮기고 kind를 붙임.
kind는 "우열 판정"(두 기술 중 하나가 전반적으로 낫다고 결론), "도입 추천"(특정 기술 도입을 권함),
"해당 없음"(의심했지만 위 두 경우가 아님) 중 하나.
두 기술의 특징, 반응, 제약을 "A는 ~한 반면 B는 ~"처럼 대비해 서술하는 문장은 우열 판정이 아님.
아래 예시의 "기술 A", "기술 B"는 설명용 이름이며 검사 대상 문장이 아님. 예시를 답에 옮기지 말 것.
- 해당 예: "기술 A가 기술 B보다 실용적이므로 도입을 권장함."
- 해당 예: "기술 B는 기술 A에 비해 훨씬 앞선 기술임."
- 해당 아님: "기술 A는 투자자 불확실성을 유발한 반면, 기술 B는 설비투자 흐름과 연결되어 평가됨."
[sentences_without_evidence] [근거#N] 표기가 하나도 없는 사실 주장 문장을 그대로 나열.
접속 문장, 안내 문장, 관점 이름만 적은 문장, 앞 문장의 근거를 요약만 하는 결론 문장은 제외.
[notes] 판정 근거 한두 문장.

===== 검사 대상 =====
{text}
"""


class _BiasFlag(BaseModel):
    sentence: str
    kind: Literal["우열 판정", "도입 추천", "해당 없음"]


class _NeutralityJudgment(BaseModel):
    biased_sentences: list[_BiasFlag] = Field(default_factory=list)
    sentences_without_evidence: list[str] = Field(default_factory=list)
    notes: str = ""


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)


def _sentences(text: str) -> list[str]:
    text = _TRAILING_CITATION_RE.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", text)
    return [s.strip(" -*") for s in _SENTENCE_SPLIT_RE.split(text) if s and len(s.strip(" -*")) > 5]


def _cited_ids(text: str) -> list[int]:
    return [int(n) for n in _CITATION_RE.findall(text)]


def load_report_json(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 규칙 판정. LLM과 임베딩 없이 단위 테스트할 수 있게 순수 함수로 둠
# ---------------------------------------------------------------------------


def check_coverage(sections: dict[str, str], evidence_by_id: dict[int, Evidence], techs: list[str]) -> tuple[CriterionResult, list[Gap]]:
    cited = [evidence_by_id[i] for i in _cited_ids(sections.get(_VIEW_SECTION, "")) if i in evidence_by_id]
    present = {(e.perspective, e.tech) for e in cited}
    missing = [(p, t) for p in _PERSPECTIVES for t in techs if (p, t) not in present]
    gaps = [
        Gap(perspective=p, tech=t, reason="보고서 관점별 평가에 인용 근거 없음", source="quality_eval")
        for p, t in missing
    ]
    return (
        CriterionResult(
            name="coverage",
            passed=not missing,
            method="rule",
            issues=[f"{p}/{t}: 관점별 평가에 인용 근거 없음" for p, t in missing],
        ),
        gaps,
    )


def check_bias_control(sections: dict[str, str], evidence_by_id: dict[int, Evidence], techs: list[str]) -> tuple[CriterionResult, list[Gap]]:
    ids = {i for body in sections.values() for i in _cited_ids(body)}
    cited = [evidence_by_id[i] for i in ids if i in evidence_by_id and evidence_by_id[i].perspective in _PERSPECTIVES]
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    per_tech: dict[str, int] = {t: 0 for t in techs}
    counter_by_tech: dict[str, int] = {t: 0 for t in techs}
    for e in cited:
        sources[(e.perspective, e.tech)].add(e.reference_url or e.source)
        per_tech[e.tech] = per_tech.get(e.tech, 0) + 1
        if e.stance == "반대":
            counter_by_tech[e.tech] = counter_by_tech.get(e.tech, 0) + 1

    issues: list[str] = []
    gaps: list[Gap] = []
    for (perspective, tech), srcs in sorted(sources.items()):
        if len(srcs) < config.QUALITY_MIN_SOURCES:
            issues.append(f"{perspective}/{tech}: 인용 출처 {len(srcs)}곳(단일 출처 편중)")
            gaps.append(Gap(perspective=perspective, tech=tech, reason="단일 출처 편중", source="quality_eval"))
    for tech in techs:
        if counter_by_tech.get(tech, 0) == 0:
            issues.append(f"{tech}: 인용된 반대 근거 없음")
            gaps.extend(
                Gap(perspective=p, tech=tech, focus="counter", reason="반대 근거 인용 없음", source="quality_eval")
                for p in _PERSPECTIVES
                if (p, tech) in sources
            )
    counts = [c for c in per_tech.values() if c > 0]
    if len(counts) == len(techs) >= 2 and max(counts) / min(counts) > config.QUALITY_MAX_RATIO:
        low = min(per_tech, key=per_tech.get)
        issues.append(f"기술 간 인용 수 비율 {max(counts)}:{min(counts)} > {config.QUALITY_MAX_RATIO:g}")
        gaps.extend(
            Gap(perspective=p, tech=low, reason="기술 간 인용 불균형", source="quality_eval")
            for p in _PERSPECTIVES
        )
    unique: dict[tuple[str, str, str], Gap] = {}
    for g in gaps:
        unique.setdefault((g.perspective, g.tech, g.focus), g)
    return CriterionResult(name="bias_control", passed=not issues, method="rule", issues=issues), list(unique.values())


def check_biased_phrases(sections: dict[str, str]) -> list[str]:
    hits = []
    for title in (*_NARRATIVE_SECTIONS, _VIEW_SECTION):
        for sentence in _sentences(sections.get(title, "")):
            if any(re.search(p, sentence) for p in _BIASED_PATTERNS):
                hits.append(sentence)
    return hits


DATA_CRITERIA = {"coverage", "bias_control"}
TEXT_CRITERIA = {"groundedness", "neutrality"}


def _norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", _CITATION_RE.sub("", text))


def match_sentence(flag: str, sentences: list[str]) -> str | None:
    """LLM이 지적한 문장을 실제 평가 대상 문장과 대조함. 찾지 못하면 None(LLM이 지어낸 지적).

    검수 모델이 프롬프트 예시나 다른 문장을 베껴 오는 경우가 있어, 앞 20자(공백, 기호 제외)가
    실제 문장에 들어 있거나 실제 문장의 앞 20자가 지적 문장에 들어 있을 때만 같은 문장으로 봄.
    """
    key = _norm(flag)[:20]
    if len(key) < 8:
        return None
    for sentence in sentences:
        norm = _norm(sentence)
        if key in norm or (len(norm) >= 8 and norm[:20] in _norm(flag)):
            return sentence
    return None


def verify_judgment(judgment: "_NeutralityJudgment", sections: dict[str, str]) -> tuple["_NeutralityJudgment", int]:
    """실제 문장과 대조되지 않는 지적, 근거 번호가 있는데 근거 없다고 한 지적을 버림. (검증된 판정, 버린 수)."""
    sentences = [s for t in _NARRATIVE_SECTIONS for s in _sentences(sections.get(t, ""))]
    dropped = 0
    biased = []
    for flag in judgment.biased_sentences:
        if match_sentence(flag.sentence, sentences) is None:
            dropped += 1
        else:
            biased.append(flag)
    uncited = []
    for flag in judgment.sentences_without_evidence:
        matched = match_sentence(flag, sentences)
        if matched is None or _CITATION_RE.search(matched):
            dropped += 1
        else:
            uncited.append(flag)
    return judgment.model_copy(update={"biased_sentences": biased, "sentences_without_evidence": uncited}), dropped


def decide_route(criteria: list[CriterionResult], gaps: list[Gap], replans_left: bool, rewrites_left: bool) -> str:
    failed = {c.name for c in criteria if not c.passed}
    if failed & DATA_CRITERIA and gaps and replans_left:
        return "orchestrator"
    if failed & TEXT_CRITERIA and rewrites_left:
        return "synthesize"
    return "end"


class QualityEvalAgent(BaseAgent):
    name = "quality_eval"
    uses_rag = False

    def __init__(self, embed: Any | None = None, judge_llm: Any | None = None, use_llm: bool | None = None):
        self._embed = embed
        self._judge_llm = judge_llm
        self.use_llm = config.QUALITY_USE_LLM_JUDGE if use_llm is None else use_llm
        self._trace_id: str | None = None

    def run(self, state: AgentState) -> dict[str, Any]:
        trace_id = state.get("trace_id")
        self._trace_id = trace_id
        attempt = state.get("eval_count", 0)
        report = load_report_json(state.get("report_json_path"))
        sections: dict[str, str] = report.get("evidence_token_sections") or {}
        evidence = list(state.get("evidence") or [])
        evidence_by_id = {e.id: e for e in evidence if e.id is not None}
        techs = [t.name for t in state.get("techs", []) or []]

        coverage, coverage_gaps = check_coverage(sections, evidence_by_id, techs)
        bias, bias_gaps = check_bias_control(sections, evidence_by_id, techs)
        judgment = self._judge(sections, trace_id)
        criteria = [
            self._groundedness(sections, evidence_by_id, judgment),
            self._neutrality(sections, judgment),
            bias,
            coverage,
        ]
        if not sections:
            criteria = [
                CriterionResult(name=c.name, passed=False, method=c.method, issues=["report.json의 평가 본문 없음"])
                for c in criteria
            ]

        gaps = [*coverage_gaps, *bias_gaps]
        replans = state.get("quality_replan_count", 0)
        rewrites = state.get("quality_rewrite_count", 0)
        route = decide_route(
            criteria, gaps, replans < config.MAX_QUALITY_REPLANS, rewrites < config.MAX_QUALITY_REWRITES
        )
        passed = all(c.passed for c in criteria)
        verdict = EvalVerdict(
            passed=passed,
            criteria=criteria,
            gaps=gaps if route == "orchestrator" else [],
            route=route,  # type: ignore[arg-type]
            attempt=attempt,
        )

        reason = "모든 항목 통과" if passed else "; ".join(
            f"{c.name}: {', '.join(c.issues[:2])}" for c in criteria if not c.passed
        )
        if not passed and route == "end":
            reason = f"Loop 예산 소진(재계획 {replans}/{config.MAX_QUALITY_REPLANS}, 재작성 {rewrites}/{config.MAX_QUALITY_REWRITES}). {reason}"
        log_decision(
            trace_id, self.name, f"route={route}", reason,
            attempt=attempt, criteria={c.name: c.passed for c in criteria},
        )
        self._append_to_report(state.get("report_json_path"), verdict)
        if route == "end":
            from src.agents.report.agent import append_quality_result

            append_quality_result(verdict, state.get("report_json_path"))

        updates: dict[str, Any] = {
            "eval_result": verdict,
            "eval_count": attempt + 1,
            "pending_gaps": verdict.gaps,
            "quality_replan_count": replans + (route == "orchestrator"),
            "quality_rewrite_count": rewrites + (route == "synthesize"),
        }
        return updates

    # -- LLM, 임베딩 판정 ---------------------------------------------------------
