"""이해관계자 평가 에이전트 (7.6절). 경쟁 진영, 도입 기업과 개발자, 투자
업계의 반응을 웹 검색만으로 조사함. RAG 미사용(5장 원칙).

입력: state["tech_profiles"], state["techs"]
출력: {"stakeholder_result": ..., "evidence": [...]}

market_eval과 동일한 검색 앵커 키워드를 재사용해 질의 경로를 코드로 고정함
(4장 도구 선택 원칙, 10장 대칭 질의). 검색 결과는 9.3절 필수 항목(경쟁 진영의
반응, 도입 기업과 개발자의 의견과 도입 장벽, 투자 업계의 평가)에 맞춰 구조화
출력으로 ViewResult에 담김.
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, TechViewResult, ViewResult
from src.common.tools import extract_view_result, web_search

_QUERY_TEMPLATES = [
    "{tech} {anchor} 경쟁 기술 반응",
    "{tech} {anchor} 개발자 커뮤니티 반응",
    "{tech} {anchor} 투자 업계 평가",
]

# 9.3절 평가 기준: 완전성(8.2) 채점 시 빠짐없이 다뤄야 하는 필수 항목
REQUIRED_ITEMS = [
    "경쟁 진영의 반응과 대응 기술",
    "도입 기업과 개발자의 의견과 도입 장벽",
    "투자 업계의 평가",
]
PERSPECTIVE_LABEL = "이해관계자"
MAX_RESULTS_PER_QUERY = 3


class StakeholderEvalAgent(BaseAgent):
    name = "stakeholder_eval"
    uses_rag = False

    def __init__(self) -> None:
        # 8.3절 Tool Calling Accuracy 측정용: 실제로 던진 질의 문자열을 남김
        self.last_queries: dict[str, list[str]] = {}

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        new_evidence: list[Evidence] = []
        next_id = self.next_evidence_id(state)
        by_tech: dict[str, TechViewResult] = {}
        self.last_queries = {}

        for tech in techs:
            passages: list[str] = []
            tech_ids: set[int] = set()
            self.last_queries[tech.name] = []
            for template in _QUERY_TEMPLATES:
                query = template.format(tech=tech.name, anchor=tech.search_anchor)
                self.last_queries[tech.name].append(query)
                for r in web_search(query, max_results=MAX_RESULTS_PER_QUERY):
                    new_evidence.append(
                        Evidence(
                            id=next_id,
                            tech=tech.name,
                            perspective="stakeholder",
                            stance="지지",
                            source_type="웹",
                            source=r.url,
                            quote=r.content[:200],
                        )
                    )
                    passages.append(f"[근거#{next_id}] ({r.title}) {r.content[:300]}")
                    tech_ids.add(next_id)
                    next_id += 1

            # 7.3절: 발췌를 구조화 출력으로 넘겨 ViewResult 형태로 종합함
            view = extract_view_result(
                passages, tech.name, PERSPECTIVE_LABEL, REQUIRED_ITEMS, tech_ids
            )
            by_tech[tech.name] = view

            # counter_facts가 참조한 근거는 stance를 "반대"로 바꿔 evidence_check(7.7)의
            # 반대 근거 유무 규칙이 실제 값을 보게 함
            counter_ids = {i for c in view.counter_facts for i in c.evidence_ids}
            for ev in new_evidence:
                if ev.id in counter_ids:
                    ev.stance = "반대"

        return {"stakeholder_result": ViewResult(by_tech=by_tech), "evidence": new_evidence}
