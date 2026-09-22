"""근거 점검 에이전트 (7.7절). LLM을 쓰지 않는 규칙 기반 노드.

입력: state의 관점 결과 4종 + evidence
출력: {"retry_targets": [...], "retry_count": int}

규칙(임계값은 통계적으로 도출한 값이 아니라 경험적 휴리스틱, 7.7절):
1. 관점별·기술별 근거 수가 3건 미만이면 재검색 대상
2. 해당 조합에 반대 근거가 0건이면 재검색 대상
3. 같은 관점에서 두 기술 간 근거 수 비율이 2배를 넘으면 재검색 대상
재시도가 이미 1회 소진되었으면 규칙을 만족해도 더 이상 재검색 대상에 넣지 않음
(12장 "반복 1", 최대 1회).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.common.base_agent import BaseAgent
from src.common.state import AgentState

_MIN_EVIDENCE = 3
_MAX_RATIO = 2.0
_PERSPECTIVE_TO_AGENT = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}


class EvidenceCheckAgent(BaseAgent):
    name = "evidence_check"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        retry_count = state.get("retry_count", 0)
        if retry_count >= 1:
            # 재시도 예산 소진: 더 재검색시키지 않고 미확인으로 넘어가게 신호만 비움
            return {"retry_targets": [], "retry_count": retry_count}

        evidence = state.get("evidence", [])
        techs = [t.name for t in state["techs"]]
        retry_targets: list[str] = []

        for perspective, agent_name in _PERSPECTIVE_TO_AGENT.items():
            counts: dict[str, dict[str, int]] = defaultdict(lambda: {"지지": 0, "반대": 0})
            for e in evidence:
                if e.perspective == perspective:
                    counts[e.tech][e.stance] += 1

            needs_retry = False
            totals = [counts[tech]["지지"] + counts[tech]["반대"] for tech in techs]
            for total in totals:
                if total < _MIN_EVIDENCE:  # 규칙 1
                    needs_retry = True
            if any(counts[tech]["반대"] == 0 for tech in techs):  # 규칙 2
                needs_retry = True
            if len(totals) == 2 and min(totals) > 0 and max(totals) / min(totals) > _MAX_RATIO:  # 규칙 3
                needs_retry = True

            if needs_retry:
                retry_targets.append(agent_name)

        return {
            "retry_targets": retry_targets,
            "retry_count": retry_count + (1 if retry_targets else 0),
        }
