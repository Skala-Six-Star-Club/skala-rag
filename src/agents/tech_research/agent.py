"""기술 조사 에이전트 (7.2절). 논문에서 기술 개요, 적용 범위, 한계, 같은 진영
다른 방식과의 차이를 추출함.

입력: state["techs"]
출력: {"tech_profiles": ..., "evidence": [...], "references": [...]}

내부 로직(TODO 표시)만 담당자가 채우면 됨 — paper_search 호출 지점과 State
입출력 키는 이미 고정돼 있으므로 인터페이스를 바꿀 필요가 없음.
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, TechProfile
from src.common.tools import paper_search


class TechResearchAgent(BaseAgent):
    name = "tech_research"
    uses_rag = True

    def __init__(self, index: FAISS):
        self.index = index

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        tech_profiles: dict[str, TechProfile] = {}
        new_evidence: list[Evidence] = []
        next_id = self.next_evidence_id(state)

        for tech in techs:
            # role=target: 기술 개요는 자기 논문에서만 근거를 가져옴 (7.2 RAG 및 재검색 전략)
            overview_query = self.rewrite_query(
                f"{tech.name} 기술 개요와 핵심 접근 방식", tech.name
            )
            overview_docs = paper_search(
                self.index, overview_query, k=config.DEFAULT_TOP_K, role="target"
            )

            # camp 전체: 같은 진영 다른 방식과의 차이를 물을 때는 role 조건을 풂
            diff_query = self.rewrite_query(
                f"{tech.name}과 같은 진영의 다른 방식이 가지는 차이", tech.name
            )
            diff_docs = paper_search(
                self.index, diff_query, k=config.DEFAULT_TOP_K, camp=tech.camp
            )

            # TODO(담당자): overview_docs + diff_docs를 GPT-5 mini structured output으로
            # 넘겨 TechProfile(overview, scope, limitations, differentiation)을 채움(7.2절).
            evidence_ids: list[int] = []
            for doc in overview_docs + diff_docs:
                new_evidence.append(
                    Evidence(
                        id=next_id,
                        tech=tech.name,
                        perspective="tech_research",
                        stance="지지",
                        source_type="논문",
                        source=f"{tech.name} p.{doc.metadata.get('page')} {doc.metadata.get('section')}",
                        quote=doc.page_content[:200],
                    )
                )
                evidence_ids.append(next_id)
                next_id += 1

            tech_profiles[tech.name] = TechProfile(
                tech=tech.name,
                overview="TODO: overview_docs 기반 요약 추출 필요",
                scope="TODO",
                limitations="TODO",
                differentiation="TODO: diff_docs 기반 비교 서술 필요",
                evidence_ids=evidence_ids,
            )

        return {"tech_profiles": tech_profiles, "evidence": new_evidence}
