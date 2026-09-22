"""기술 성숙도 에이전트 (7.3절). TRL 구간을 추정하고 근거와 정보 공백을 정리함.

입력: state["tech_profiles"]
출력: {"trl_result": ..., "evidence": [...]}

evidence_check(7.7)가 이 관점을 retry_targets에 넣으면, 다음 실행에서 질의 초점을
"구현/공식 발표" -> "한계·실패 사례·후속 검증"으로 바꿔 반대 근거를 보강함(재검색 최대 1회).
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, TechViewResult, ViewResult
from src.common.tools import paper_search, web_search


class TrlEvalAgent(BaseAgent):
    name = "trl_eval"
    uses_rag = True

    def __init__(self, index: FAISS):
        self.index = index

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        is_retry = self.name in (state.get("retry_targets") or [])
        new_evidence: list[Evidence] = []
        next_id = self.next_evidence_id(state)
        by_tech: dict[str, TechViewResult] = {}

        for tech in techs:
            focus = (
                "한계와 실패 사례, 후속 검증" if is_retry else "구현과 공식 발표 현황"
            )
            query = self.rewrite_query(f"{tech.name} {focus}", tech.name)
            paper_docs = paper_search(
                self.index, query, k=config.DEFAULT_TOP_K, role="target"
            )
            web_results = web_search(
                f"{tech.name} {tech.search_anchor} TRL 제품 발표 오픈소스 구현",
                max_results=5,
            )

            # TODO(담당자): paper_docs + web_results를 GPT-5 mini structured output으로
            # 넘겨 TRL 구간 추정 + confirmed_facts/counter_facts/unconfirmed_items를
            # 채움(7.3절). 아래는 인터페이스만 맞춘 자리표시자.
            for doc in paper_docs:
                new_evidence.append(
                    Evidence(
                        id=next_id,
                        tech=tech.name,
                        perspective="trl",
                        stance="반대" if is_retry else "지지",
                        source_type="논문",
                        source=f"{tech.name} p.{doc.metadata.get('page')}",
                        quote=doc.page_content[:200],
                    )
                )
                next_id += 1
            for r in web_results:
                new_evidence.append(
                    Evidence(
                        id=next_id,
                        tech=tech.name,
                        perspective="trl",
                        stance="반대" if is_retry else "지지",
                        source_type="웹",
                        source=r.url,
                        quote=r.content[:200],
                    )
                )
                next_id += 1

            by_tech[tech.name] = TechViewResult()  # TODO: confirmed/counter/unconfirmed 채움

        return {"trl_result": ViewResult(by_tech=by_tech), "evidence": new_evidence}
