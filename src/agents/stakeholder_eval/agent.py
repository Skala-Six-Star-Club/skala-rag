"""이해관계자 평가 에이전트 (7.6절). 경쟁 진영, 도입 기업과 개발자, 투자
업계의 반응을 웹 검색만으로 조사함. RAG 미사용(5장 원칙).

입력: state["tech_profiles"], state["techs"]
출력: {"stakeholder_result": ..., "raw_evidence": [...]}

market_eval과 동일한 검색 앵커 키워드를 재사용해 질의 경로를 코드로 고정함
(4장 도구 선택 원칙, 10장 대칭 질의).
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, TechViewResult, ViewResult
from src.common.tools import web_search

_QUERY_TEMPLATES = [
    "{tech} {anchor} 경쟁 기술 반응",
    "{tech} {anchor} 개발자 커뮤니티 반응",
    "{tech} {anchor} 투자 업계 평가",
]


class StakeholderEvalAgent(BaseAgent):
    name = "stakeholder_eval"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        is_retry = self.name in (state.get("retry_targets") or [])
        new_evidence: list[Evidence] = []
        ordinal = 0
        by_tech: dict[str, TechViewResult] = {}

        for tech in techs:
            passages: list[str] = []
            for template in _QUERY_TEMPLATES:
                query = template.format(tech=tech.name, anchor=tech.search_anchor)
                for r in web_search(query, max_results=3):
                    evidence_key = self.provisional_evidence_key(state, tech.name, ordinal)
                    new_evidence.append(
                        Evidence(
                            key=evidence_key,
                            tech=tech.name,
                            perspective="stakeholder",
                            stance="반대" if is_retry else "지지",
                            source_type="웹",
                            source=r.url,
                            quote=r.content[:200],
                        )
                    )
                    passages.append(f"[임시근거#{evidence_key}] {r.content[:300]}")
                    ordinal += 1

            # TODO(담당자, 박기연): passages를 GPT-5 mini structured output으로
            # 넘겨 TechViewResult(confirmed_facts/counter_facts/unconfirmed_items)를
            # 채울 것(7.6절).
            by_tech[tech.name] = TechViewResult()

        return {
            "stakeholder_result": ViewResult(by_tech=by_tech),
            "raw_evidence": new_evidence,
            "evidence": new_evidence,
        }
