"""근거 점검 에이전트 (7.7절 + 확장). LLM을 쓰지 않는 규칙 기반 노드.

Evidence가 충분한지 확인하는 세 가지 규칙과, 관점 에이전트가 만든 Claim을
인용 근거와 임베딩으로 확인하는 그라운딩 규칙을 수행한다. 병렬 수집 단계에서는
``raw_evidence``와 provisional key를 사용하고, 레거시 단위 호출에서는 최종
``evidence``와 정수 ID도 계속 지원한다.

출력:
    ``retry_targets``, ``retry_count``, ``perspective_confidence``와
    그라운딩 검증 후의 관점 결과(검증이 필요한 경우)를 반환한다.
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
    claim: Claim,
    evidence_by_id: dict[int, Evidence],
    evidence_by_key: dict[str, Evidence],
    embed: Any,
) -> tuple[bool, str]:
    """Claim이 인용한 Evidence와 의미적으로 연결되는지 판정한다."""

    cited: list[Evidence] = []
    for key in claim.evidence_keys:
        evidence = evidence_by_key.get(key)
        if evidence is None:
            return False, f"존재하지 않는 provisional key 참조({key})"
        cited.append(evidence)

    for evidence_id in claim.evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            return False, f"존재하지 않는 근거 번호 참조(#{evidence_id})"
        cited.append(evidence)

    if not cited:
        return False, "출처 없음"

    # 동일 Evidence가 key와 id 양쪽으로 들어온 경우에는 한 번만 비교한다.
    unique_quotes = list(dict.fromkeys(evidence.quote for evidence in cited if evidence.quote))
    if not unique_quotes:
        return False, "인용문 없음"

    statement_vec = embed.embed_query(claim.statement)
    quote_vecs = embed.embed_documents(unique_quotes)
    best_sim = max(_cosine(statement_vec, quote_vec) for quote_vec in quote_vecs)
    if best_sim < config.GROUNDING_MIN_SIMILARITY:
        return False, f"근거와 무관해 보임(유사도 {best_sim:.2f})"
    return True, ""


def _filter_view_result(
    view_result: ViewResult,
    evidence_by_id: dict[int, Evidence],
    evidence_by_key: dict[str, Evidence],
    embed: Any,
) -> ViewResult:
    """그라운딩에 실패한 주장을 제거한 새 ViewResult를 반환한다."""

    if not isinstance(view_result, ViewResult):
        view_result = ViewResult.model_validate(view_result)

    new_by_tech: dict[str, TechViewResult] = {}
    for tech, raw_result in view_result.by_tech.items():
        tech_result = (
            raw_result
            if isinstance(raw_result, TechViewResult)
            else TechViewResult.model_validate(raw_result)
        )
        kept_confirmed = [
            claim
            for claim in tech_result.confirmed_facts
            if _claim_is_grounded(
                claim, evidence_by_id, evidence_by_key, embed
            )[0]
        ]
        kept_counter = [
            claim
            for claim in tech_result.counter_facts
            if _claim_is_grounded(
                claim, evidence_by_id, evidence_by_key, embed
            )[0]
        ]
        new_by_tech[tech] = TechViewResult(
            confirmed_facts=kept_confirmed,
            counter_facts=kept_counter,
            unconfirmed_items=tech_result.unconfirmed_items,
        )
    return ViewResult(by_tech=new_by_tech)


def _has_claims(view_result: ViewResult) -> bool:
    """자리표시자 결과처럼 Claim이 하나도 없으면 임베딩을 불러오지 않는다."""

    if not isinstance(view_result, ViewResult):
        view_result = ViewResult.model_validate(view_result)
    return any(
        tech_result.confirmed_facts or tech_result.counter_facts
        for tech_result in view_result.by_tech.values()
    )


class EvidenceCheckAgent(BaseAgent):
    name = "evidence_check"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        retry_count = state.get("retry_count", 0)
        budget_left = retry_count < 1

        # 통합 Graph에서는 병렬 reducer가 누적한 raw_evidence를 읽는다. raw가
        # 비어 있는 레거시 호출에서는 최종 evidence를 fallback으로 사용한다.
        evidence = state.get("raw_evidence")
        if evidence is None or (not evidence and state.get("evidence")):
            evidence = state.get("evidence", [])
        evidence = list(evidence or [])
        evidence_by_id = {
            evidence_item.id: evidence_item
            for evidence_item in evidence
            if evidence_item.id is not None
        }
        evidence_by_key = {
            evidence_item.key: evidence_item
            for evidence_item in evidence
            if evidence_item.key
        }
        techs = [tech.name for tech in state["techs"]]

        updates: dict[str, Any] = {}
        retry_targets: list[str] = []
        confidence: dict[str, dict[str, float]] = {}
        embed = None  # Claim이 실제로 있을 때만 bge-m3를 지연 로딩한다.

        for perspective, agent_name in _PERSPECTIVE_TO_AGENT.items():
            counts: dict[str, dict[str, int]] = defaultdict(
                lambda: {"지지": 0, "반대": 0}
            )
            for evidence_item in evidence:
                if evidence_item.perspective == perspective:
                    counts[evidence_item.tech][evidence_item.stance] += 1

            totals = {
                tech: counts[tech]["지지"] + counts[tech]["반대"]
                for tech in techs
            }
            needs_retry = any(total < _MIN_EVIDENCE for total in totals.values())
            if any(counts[tech]["반대"] == 0 for tech in techs):
                needs_retry = True

            nonzero = [total for total in totals.values() if total > 0]
            if (
                len(totals) == 2
                and len(nonzero) == 2
                and max(nonzero) / min(nonzero) > _MAX_RATIO
            ):
                needs_retry = True

            # 규칙 4: 결과에 Claim이 있을 때만 그라운딩을 수행한다. 현재 담당
            # 에이전트의 빈 자리표시자 결과 때문에 불필요하게 임베딩을 로드하지 않는다.
            result_key = _PERSPECTIVE_TO_RESULT_KEY[perspective]
            view_result = state.get(result_key)
            if view_result is not None and _has_claims(view_result):
                if embed is None:
                    embed = get_embedding_model()
                filtered = _filter_view_result(
                    view_result, evidence_by_id, evidence_by_key, embed
                )
                updates[result_key] = filtered
                for tech in techs:
                    tech_result = filtered.by_tech.get(tech)
                    if tech_result is not None and not (
                        tech_result.confirmed_facts or tech_result.counter_facts
                    ):
                        needs_retry = True

            if needs_retry and budget_left:
                retry_targets.append(agent_name)

            target_count = max(config.TARGET_EVIDENCE_COUNT, 1)
            confidence[perspective] = {
                tech: round(min(1.0, totals[tech] / target_count), 2)
                for tech in techs
            }

        updates["retry_targets"] = retry_targets
        updates["retry_count"] = retry_count + (1 if retry_targets else 0)
        updates["perspective_confidence"] = confidence
        return updates
