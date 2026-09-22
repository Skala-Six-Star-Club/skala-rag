"""시장 평가 에이전트 (7.5절). 시장 규모/성장성/채택 현황/생태계 지지를 웹
검색만으로 조사함. RAG 미사용(5장: 답이 논문이 아니라 시장 리포트·산업 기사에 있음).

입력: state["tech_profiles"], state["techs"]
출력: {"market_result": ..., "evidence": [...]}

두 기술 모두 같은 질의 템플릿·같은 횟수로 실행해 10장 중립성(대칭 질의) 원칙을
지킴. 앵커 키워드(TechSpec.search_anchor)만 값이 다름.
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, TechViewResult, ViewResult
from src.common.tools import web_search

_QUERY_TEMPLATES = [
    "{tech} {anchor} 시장 규모",
    "{tech} {anchor} 채택 사례",
    "{tech} {anchor} 프레임워크 지원",
]


class MarketEvalAgent(BaseAgent):
    name = "market_eval"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        new_evidence: list[Evidence] = []
        next_id = self.next_evidence_id(state)
        by_tech: dict[str, TechViewResult] = {}

        for tech in techs:
            passages: list[str] = []
            for template in _QUERY_TEMPLATES:
                query = template.format(tech=tech.name, anchor=tech.search_anchor)
                for r in web_search(query, max_results=3):
                    new_evidence.append(
                        Evidence(
                            id=next_id,
                            tech=tech.name,
                            perspective="market",
                            stance="지지",
                            source_type="웹",
                            source=r.url,
                            quote=r.content[:200],
                        )
                    )
                    passages.append(f"[근거#{next_id}] {r.content[:300]}")
                    next_id += 1

            # TODO(담당자, 문관록): passages를 GPT-5 mini structured output(get_generation_llm)
            # 으로 넘겨 TechViewResult(confirmed_facts/counter_facts/unconfirmed_items)를
            # 채울 것(7.5절). 지금은 자리표시자만 있음.
            by_tech[tech.name] = TechViewResult()

        return {"market_result": ViewResult(by_tech=by_tech), "evidence": new_evidence}
