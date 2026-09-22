"""도메인 평가 에이전트 (7.4절). 에이전트 코딩 서비스의 멀티턴 장문맥 서빙에
도입할 때의 조건과 장벽을 정리함.

입력: state["techs"], state["tech_profiles"], state["domain"]
출력: {"domain_result": ..., "raw_evidence": [...]}
(근거는 provisional key로 발급되고 evidence_finalize가 최종 번호를 부여함)

검색 설정(3차 비교실험 채택안): 절 인식 청킹 + Qwen3-Embedding-0.6B + Query
Rewriting 끔. 이 에이전트에 한해, 유사도가 동률일 때 "실험 환경"/"평가" 절 청크를
우선하는 경량 규칙 기반 재랭킹을 둠(7.4절). 정식 reranker는 두지 않음.
재검색 시 초점을 한계·실패 사례로 바꾸고 1차의 미확인 항목만 이어받음(최대 1회).
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, TechViewResult, ViewResult
from src.common.tools import (
    extract_view_result,
    format_paper_source,
    get_shared_index,
    paper_search,
    web_search,
)

_EXPERIMENT_SECTIONS = ("experiment", "evaluation", "실험", "평가")


def _rerank_experiment_first(index: FAISS, query: str, k: int, role: str, tech: str):
    """동일 유사도일 때 실험 환경/평가 절을 우선하는 경량 재랭킹(7.4절). 자기 논문(tech)만 봄."""
    candidates = index.similarity_search_with_score(
        query, k=k * 2, filter={"role": role, "tech": tech}, fetch_k=index.index.ntotal
    )
    candidates.sort(
        key=lambda pair: (
            round(pair[1], 4),
            0 if str(pair[0].metadata.get("section", "")).lower().startswith(_EXPERIMENT_SECTIONS) else 1,
        )
    )
    return [doc for doc, _ in candidates[:k]]


# 9.4절 평가 기준: 완전성(8.2) 채점 시 빠짐없이 다뤄야 하는 필수 항목
REQUIRED_ITEMS = [
    "세션 간 캐시 재사용에 미치는 영향",
    "턴당 응답 지연",
    "캐시 보관 비용(메모리, 저장 장치)",
    "누적 턴에서의 정확도 유지",
    "도입에 필요한 하드웨어와 소프트웨어 변경 범위",
]
PERSPECTIVE_LABEL = "도메인 적용"
EXTRA_INSTRUCTIONS = (
    "6. 도메인은 '{domain}'임. 각 항목을 이 도메인 조건에 비추어 서술하고, "
    "논문 실험 환경이 도메인과 다르면 그 차이를 unconfirmed_items나 counter_facts에 적음"
)
WEB_MAX_RESULTS = 5


class DomainEvalAgent(BaseAgent):
    name = "domain_eval"
    uses_rag = True

    def __init__(self, index: FAISS | None = None):
        self.index = index or get_shared_index()
        # 8.3절 Tool Calling Accuracy 측정용: 실제로 던진 질의를 남김
        self.last_queries: dict[str, dict[str, str]] = {}

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        domain = state["domain"]
        is_retry = self.name in (state.get("retry_targets") or [])
        prior = state.get("domain_result")
        new_evidence: list[Evidence] = []
        ordinal = 0
        by_tech: dict[str, TechViewResult] = {}
        self.last_queries = {}

        for tech in techs:
            # 12장 "반복 1": 재검색 시 질의 초점을 한계·실패 사례·후속 검증으로 바꿈
            focus = "한계와 실패 사례, 후속 검증" if is_retry else "실험 환경과 요구 하드웨어, 지연 시간, 메모리 비용"
            paper_query = self.rewrite_query(f"{tech.name} {domain} {focus}", tech.name)
            web_query = f"{tech.name} {tech.search_anchor} {domain} " + ("한계 실패 사례" if is_retry else "서빙 프레임워크 지연시간 메모리")
            self.last_queries[tech.name] = {"paper": paper_query, "web": web_query}

            paper_docs = _rerank_experiment_first(self.index, paper_query, k=config.DEFAULT_TOP_K, role="target", tech=tech.name)
            web_results = web_search(web_query, max_results=WEB_MAX_RESULTS)

            passages: list[str] = []
            key_by_num: dict[int, str] = {}

            def _add(ev: Evidence, label: str, body: str) -> None:
                new_evidence.append(ev)
                num = len(key_by_num) + 1  # 프롬프트용 로컬 번호(기술마다 1부터)
                key_by_num[num] = ev.key
                passages.append(f"[근거#{num}] ({label}) {body}")

            for doc in paper_docs:
                _add(
                    self.new_evidence(
                        state, tech.name, ordinal, perspective="domain", source_type="논문",
                        source=format_paper_source(tech.name, doc), quote=doc.page_content[:200],
                    ),
                    f"논문 p.{doc.metadata.get('page')} {doc.metadata.get('section', '')}",
                    doc.page_content,
                )
                ordinal += 1
            for r in web_results:
                _add(
                    self.new_evidence(
                        state, tech.name, ordinal, perspective="domain", source_type="웹",
                        source=r.url, quote=r.content[:200],
                    ),
                    f"웹: {r.title} {r.published_date or ''}",
                    r.content[:600],
                )
                ordinal += 1

            # 재검색 시 1차 결과의 미확인 항목만 이어받음(7.4절 Context 및 Memory)
            prior_unconfirmed = (
                list(prior.by_tech[tech.name].unconfirmed_items)
                if is_retry and prior is not None and tech.name in prior.by_tech
                else None
            )
            view = extract_view_result(
                passages,
                tech.name,
                PERSPECTIVE_LABEL,
                REQUIRED_ITEMS,
                key_by_num,
                extra_instructions=EXTRA_INSTRUCTIONS.format(domain=domain),
                prior_unconfirmed=prior_unconfirmed,
            )
            by_tech[tech.name] = view

            # counter_facts가 참조한 근거는 stance를 "반대"로 바꿔 evidence_check(7.7)가 실제 값을 보게 함
            counter_keys = {k for c in view.counter_facts for k in c.evidence_keys}
            for ev in new_evidence:
                if ev.key in counter_keys:
                    ev.stance = "반대"

        return {
            "domain_result": ViewResult(by_tech=by_tech),
            "raw_evidence": new_evidence,
            # 독립 실행 스크립트 호환. 통합 Graph는 raw 영역으로만 병합함.
            "evidence": new_evidence,
        }
