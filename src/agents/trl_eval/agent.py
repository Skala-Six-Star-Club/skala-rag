"""기술 성숙도 에이전트 (7.3절). TRL 구간을 추정하고 근거와 정보 공백을 정리함.

입력: state["techs"], state["tech_profiles"]
출력: {"trl_result": ..., "raw_evidence": [...]}
(근거는 provisional key로 발급되고 evidence_finalize가 최종 번호를 부여함)

검색 설정(3차 비교실험 채택안): 절 인식 청킹 + Qwen3-Embedding-0.6B + Query
Rewriting 끔. 논문 Top-5(role=target)와 웹 Top-5를 합쳐 구조화 출력 1회로
TRL 구간(1~3, 4~6, 7~9)을 추정함. TRL은 추정임을 문장에 명시함(9.5절 필수 항목).
evidence_check(7.7)가 이 관점을 retry_targets에 넣으면 질의 초점을
"구현/공식 발표" -> "한계·실패 사례·후속 검증"으로 바꾸고, 1차의 미확인 항목만
이어받아 재실행함(재검색 최대 1회).
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

# 9.1절 평가 기준: 완전성(8.2) 채점 시 빠짐없이 다뤄야 하는 필수 항목
REQUIRED_ITEMS = [
    "TRL 구간 추정(1~3, 4~6, 7~9)과 추정임을 명시한 판단 사유",
    "논문의 실험 수준(시뮬레이션, 프로토타입, 실제 시스템)",
    "공개 구현 유무(오픈소스, 코드 공개)",
    "시제품과 제품 발표, 샘플 공급과 양산 보도",
    "공개 정보로 확인되지 않는 부분",
]
PERSPECTIVE_LABEL = "기술 성숙도(TRL)"
EXTRA_INSTRUCTIONS = (
    "5. confirmed_facts의 첫 문장은 반드시 'TRL 추정 구간: N~M (추정)' 형식으로 시작하고, "
    "그 구간을 고른 사유를 근거 번호와 함께 이어 씀. TRL은 공개 정보 기반 추정임을 명시함"
)
WEB_MAX_RESULTS = 5


class TrlEvalAgent(BaseAgent):
    name = "trl_eval"
    uses_rag = True

    def __init__(self, index: FAISS | None = None):
        self.index = index or get_shared_index()
        # 8.3절 Tool Calling Accuracy 측정용: 실제로 던진 질의를 남김
        self.last_queries: dict[str, dict[str, str]] = {}

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        is_retry = self.name in (state.get("retry_targets") or [])
        prior = state.get("trl_result")
        new_evidence: list[Evidence] = []
        ordinal = 0
        by_tech: dict[str, TechViewResult] = {}
        self.last_queries = {}

        for tech in techs:
            # 12장 "반복 1": 재검색 시 질의 초점을 한계·실패 사례·후속 검증으로 바꿈
            focus = "한계와 실패 사례, 후속 검증" if is_retry else "구현과 공식 발표 현황, 실험 수준, 공개 구현"
            paper_query = self.rewrite_query(f"{tech.name} {focus}", tech.name)
            web_query = f"{tech.name} {tech.search_anchor} " + ("한계 실패 사례 후속 검증" if is_retry else "제품 발표 오픈소스 구현 TRL")
            self.last_queries[tech.name] = {"paper": paper_query, "web": web_query}

            paper_docs = paper_search(self.index, paper_query, k=config.DEFAULT_TOP_K, role="target", tech=tech.name)
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
                        state, tech.name, ordinal, perspective="trl", source_type="논문",
                        source=format_paper_source(tech.name, doc), quote=doc.page_content[:200],
                    ),
                    f"논문 p.{doc.metadata.get('page')} {doc.metadata.get('section', '')}",
                    doc.page_content,
                )
                ordinal += 1
            for r in web_results:
                _add(
                    self.new_evidence(
                        state, tech.name, ordinal, perspective="trl", source_type="웹",
                        source=r.url, quote=r.content[:200],
                    ),
                    f"웹: {r.title} {r.published_date or ''}",
                    r.content[:600],
                )
                ordinal += 1

            # 재검색 시 1차 결과의 미확인 항목만 이어받음(7.3절 Context 및 Memory)
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
                extra_instructions=EXTRA_INSTRUCTIONS,
                prior_unconfirmed=prior_unconfirmed,
            )
            by_tech[tech.name] = view

            # counter_facts가 참조한 근거는 stance를 "반대"로 바꿔 evidence_check(7.7)가 실제 값을 보게 함
            counter_keys = {k for c in view.counter_facts for k in c.evidence_keys}
            for ev in new_evidence:
                if ev.key in counter_keys:
                    ev.stance = "반대"

        return {
            "trl_result": ViewResult(by_tech=by_tech),
            "raw_evidence": new_evidence,
            # 독립 실행 스크립트 호환. 통합 Graph는 raw 영역으로만 병합함.
            "evidence": new_evidence,
        }
