"""도메인 평가 에이전트 (7.4절). 에이전트 코딩 서비스의 멀티턴 장문맥 서빙에
도입할 때의 조건과 장벽을 정리함.

입력: state["tech_profiles"], state["domain"]
출력: {"domain_result": ..., "raw_evidence": [...]}

이 에이전트에 한해, 유사도가 동률일 때 "실험 환경"/"평가" 절 청크를 우선하는
경량 규칙 기반 재랭킹을 둠(7.4절 2번 항목). 정식 reranker는 두지 않음.
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, TechViewResult, ViewResult
from src.common.tools import web_search

_EXPERIMENT_SECTIONS = ("experiment", "evaluation", "실험", "평가")


def _rerank_experiment_first(index: FAISS, query: str, k: int, role: str):
    """동일 유사도일 때 실험 환경/평가 절을 우선하는 경량 재랭킹(7.4절)."""
    candidates = index.similarity_search_with_score(query, k=k * 2, filter={"role": role})
    candidates.sort(
        key=lambda pair: (
            round(pair[1], 4),
            0 if str(pair[0].metadata.get("section", "")).lower().startswith(_EXPERIMENT_SECTIONS) else 1,
        )
    )
    return [doc for doc, _ in candidates[:k]]


class DomainEvalAgent(BaseAgent):
    name = "domain_eval"
    uses_rag = True

    def __init__(self, index: FAISS):
        self.index = index

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        domain = state["domain"]
        is_retry = self.name in (state.get("retry_targets") or [])
        new_evidence: list[Evidence] = []
        ordinal = 0
        by_tech: dict[str, TechViewResult] = {}

        for tech in techs:
            focus = (
                "한계와 실패 사례, 후속 검증" if is_retry else "실험 환경과 요구 하드웨어"
            )
            query = self.rewrite_query(f"{tech.name} {domain} {focus}", tech.name)
            paper_docs = _rerank_experiment_first(
                self.index, query, k=config.DEFAULT_TOP_K, role="target"
            )
            web_results = web_search(
                f"{tech.name} {tech.search_anchor} {domain} 서빙 프레임워크 지연시간",
                max_results=5,
            )

            # TODO(담당자): paper_docs + web_results를 GPT-5 mini structured output으로
            # 넘겨 도메인 적용 조건/장벽을 confirmed_facts/counter_facts/unconfirmed_items
            # 형태로 채움(7.4절).
            for doc in paper_docs:
                evidence_key = self.provisional_evidence_key(state, tech.name, ordinal)
                new_evidence.append(
                    Evidence(
                        key=evidence_key,
                        tech=tech.name,
                        perspective="domain",
                        stance="반대" if is_retry else "지지",
                        source_type="논문",
                        source=f"{tech.name} p.{doc.metadata.get('page')}",
                        quote=doc.page_content[:200],
                    )
                )
                ordinal += 1
            for r in web_results:
                evidence_key = self.provisional_evidence_key(state, tech.name, ordinal)
                new_evidence.append(
                    Evidence(
                        key=evidence_key,
                        tech=tech.name,
                        perspective="domain",
                        stance="반대" if is_retry else "지지",
                        source_type="웹",
                        source=r.url,
                        quote=r.content[:200],
                    )
                )
                ordinal += 1

            by_tech[tech.name] = TechViewResult()  # TODO: confirmed/counter/unconfirmed 채움

        return {
            "domain_result": ViewResult(by_tech=by_tech),
            "raw_evidence": new_evidence,
            "evidence": new_evidence,
        }
