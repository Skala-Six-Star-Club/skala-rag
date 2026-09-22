"""근거 점검 에이전트 (7.7절 + 확장). LLM을 쓰지 않는 규칙 기반 노드.

입력: state의 관점 결과 4종 + evidence
출력: {"retry_targets": [...], "retry_count": int, "perspective_confidence": {...},
       trl_result/market_result/stakeholder_result/domain_result(그라운딩 필터링 후)}

기존 규칙(임계값은 통계적으로 도출한 값이 아니라 경험적 휴리스틱, 7.7절):
1. 관점별·기술별 근거 수가 3건 미만이면 재검색 대상
2. 해당 조합에 반대 근거가 0건이면 재검색 대상
3. 같은 관점에서 두 기술 간 근거 수 비율이 2배를 넘으면 재검색 대상

확장 — 근거 검증 & 데이터 종합 역할(팀원 4, 신소영):
기존 3규칙은 "검색된 근거(Evidence)가 충분히 모였는가"만 보고, 관점 에이전트가
그 근거로 LLM 합성한 주장(TechViewResult.confirmed_facts/counter_facts)이 실제로
그 근거와 부합하는지는 점검하지 않았음. 이를 규칙 4(그라운딩 검증)로 추가함.

주장 하나가 아래 중 하나라도 해당하면 환각 후보로 보고 제거함:
- evidence_ids가 비어 있음 (출처 없는 문장, 10장 중립성 확보 방안과 동일 기준)
- evidence_ids가 실제 evidence 목록에 없는 번호를 참조함 (존재하지 않는 근거 인용)
- 인용된 evidence.quote와 statement 간 bge-m3 임베딩 코사인 유사도가 임계값 미만
  (근거와 무관해 보이는 문장)

bge-m3를 재사용하는 이유: 6.1절에서 한국어 질의/영어 논문 교차언어 검색을 위해
이미 선택된 모델이라, 한국어 주장(statement)과 영어 원문 인용문(quote) 사이의
정합성을 비교하는 이 용도에도 별도 모델 도입 없이 그대로 들어맞음. LLM 판단 대신
임베딩 유사도로 검증하는 이유는 evidence_check를 LLM-프리 상태로 유지해 재현성을
지키기 위함(원 설계 의도와 동일 논리, 7.7절 "이유" 참고).

필터링 후 남은 근거량으로 관점별·기술별 신뢰도(perspective_confidence)를 계산해
State에 얹음. synthesize(7.8)가 이를 읽어 종합 시 가중치로 참고함. 필터링 후
한 기술의 confirmed_facts/counter_facts가 모두 비면(=합성된 주장이 근거를 하나도
못 건졌으면) 반대 근거 부재(규칙 2)와 같은 성격의 결함으로 보고 재검색 대상에
추가함.

재시도가 이미 1회 소진되었으면 규칙을 만족해도 더 이상 재검색 대상에 넣지 않음
(12장 "반복 1", 최대 1회). 그라운딩 필터링 자체는 재시도 예산과 무관하게 매번
수행함 — 예산이 소진됐어도 환각 후보를 보고서에 그대로 흘려보내지 않기 위함.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.models import get_embedding_model
from src.common.state import AgentState, Claim, Evidence, TechViewResult, ViewResult

_MIN_EVIDENCE = 3
_MAX_RATIO = 2.0
_PERSPECTIVE_TO_AGENT = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
_PERSPECTIVE_TO_RESULT_KEY = {
    "trl": "trl_result",
    "market": "market_result",
    "stakeholder": "stakeholder_result",
    "domain": "domain_result",
}


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)


def _claim_is_grounded(
    claim: Claim, evidence_by_id: dict[int, Evidence], embed: Any
) -> tuple[bool, str]:
    """주장 하나가 인용한 근거로 실제 뒷받침되는지 판정함. (통과 여부, 사유)."""
    if not claim.evidence_ids:
        return False, "출처 없음"

    quotes: list[str] = []
    for eid in claim.evidence_ids:
        ev = evidence_by_id.get(eid)
        if ev is None:
            return False, f"존재하지 않는 근거 번호 참조(#{eid})"
        quotes.append(ev.quote)

    statement_vec = embed.embed_query(claim.statement)
    quote_vecs = embed.embed_documents(quotes)
    best_sim = max(_cosine(statement_vec, qv) for qv in quote_vecs)
    if best_sim < config.GROUNDING_MIN_SIMILARITY:
        return False, f"근거와 무관해 보임(유사도 {best_sim:.2f})"
    return True, ""


def _filter_view_result(
    view_result: ViewResult, evidence_by_id: dict[int, Evidence], embed: Any
) -> ViewResult:
    """환각 후보 주장을 제거한 새 ViewResult를 반환함(원본은 건드리지 않음)."""
    new_by_tech: dict[str, TechViewResult] = {}
    for tech, tvr in view_result.by_tech.items():
        kept_confirmed = [
            c for c in tvr.confirmed_facts if _claim_is_grounded(c, evidence_by_id, embed)[0]
        ]
        kept_counter = [
            c for c in tvr.counter_facts if _claim_is_grounded(c, evidence_by_id, embed)[0]
        ]
        new_by_tech[tech] = TechViewResult(
            confirmed_facts=kept_confirmed,
            counter_facts=kept_counter,
            unconfirmed_items=tvr.unconfirmed_items,
        )
    return ViewResult(by_tech=new_by_tech)


class EvidenceCheckAgent(BaseAgent):
    name = "evidence_check"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        retry_count = state.get("retry_count", 0)
        budget_left = retry_count < 1

        evidence = state.get("evidence", [])
        evidence_by_id = {e.id: e for e in evidence}
        techs = [t.name for t in state["techs"]]

        updates: dict[str, Any] = {}
        retry_targets: list[str] = []
        confidence: dict[str, dict[str, float]] = {}
        embed = None  # 지연 로딩: 그라운딩 검증이 실제로 필요할 때만 bge-m3를 올림

        for perspective, agent_name in _PERSPECTIVE_TO_AGENT.items():
            counts: dict[str, dict[str, int]] = defaultdict(lambda: {"지지": 0, "반대": 0})
            for e in evidence:
                if e.perspective == perspective:
                    counts[e.tech][e.stance] += 1

            totals = {tech: counts[tech]["지지"] + counts[tech]["반대"] for tech in techs}
            needs_retry = False
            if any(total < _MIN_EVIDENCE for total in totals.values()):  # 규칙 1
                needs_retry = True
            if any(counts[tech]["반대"] == 0 for tech in techs):  # 규칙 2
                needs_retry = True
            nonzero = [t for t in totals.values() if t > 0]
            if len(totals) == 2 and len(nonzero) == 2 and max(nonzero) / min(nonzero) > _MAX_RATIO:  # 규칙 3
                needs_retry = True

            # 규칙 4(신규): 그라운딩 검증 — 관점 결과가 이미 도착했으면 환각 후보 제거
            result_key = _PERSPECTIVE_TO_RESULT_KEY[perspective]
            view_result: ViewResult | None = state.get(result_key)
            if view_result is not None and view_result.by_tech:
                if embed is None:
                    embed = get_embedding_model()
                filtered = _filter_view_result(view_result, evidence_by_id, embed)
                updates[result_key] = filtered
                for tech in techs:
                    tvr = filtered.by_tech.get(tech)
                    if tvr is not None and not tvr.confirmed_facts and not tvr.counter_facts:
                        needs_retry = True  # 필터링 후 남은 주장이 없음 = 지지할 게 없음

            if needs_retry and budget_left:
                retry_targets.append(agent_name)

            confidence[perspective] = {
                tech: round(min(1.0, totals[tech] / config.TARGET_EVIDENCE_COUNT), 2)
                for tech in techs
            }

        updates["retry_targets"] = retry_targets
        updates["retry_count"] = retry_count + (1 if retry_targets else 0)
        updates["perspective_confidence"] = confidence
        return updates
