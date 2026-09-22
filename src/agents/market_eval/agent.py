"""시장 평가 에이전트 (7.5절). 시장 규모/성장성/채택 현황/생태계 지지를 웹
검색만으로 조사함. RAG 미사용(5장: 답이 논문이 아니라 시장 리포트·산업 기사에 있음).

입력: state["tech_profiles"], state["techs"]
출력: {"market_result": ..., "raw_evidence": [...]}
(근거는 provisional key로 발급되고 evidence_finalize가 최종 번호를 부여함)

두 기술 모두 같은 질의 템플릿·같은 횟수로 실행해 10장 중립성(대칭 질의) 원칙을
지킴. 앵커 키워드(TechSpec.search_anchor)만 값이 다름. 검색 결과는 9.2절 필수
항목(시장 규모와 성장성, 상용화와 채택 현황, 생태계 지지)에 맞춰 구조화 출력으로
ViewResult에 담김.
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.state import AgentState, Evidence, Reference, TechViewResult, ViewResult
from src.common.tools import extract_view_result, web_reference, web_search

_QUERY_TEMPLATES = [
    "{tech} {anchor} 시장 규모",
    "{tech} {anchor} 채택 사례",
    "{tech} {anchor} 프레임워크 지원",
]

# 9.2절 평가 기준: 완전성(8.2) 채점 시 빠짐없이 다뤄야 하는 필수 항목
REQUIRED_ITEMS = [
    "시장 규모와 성장성",
    "상용화와 채택 현황",
    "프레임워크 지원과 표준화 같은 생태계 지지",
]
PERSPECTIVE_LABEL = "시장성"
MAX_RESULTS_PER_QUERY = 3


class MarketEvalAgent(BaseAgent):
    name = "market_eval"
    uses_rag = False

    def __init__(self) -> None:
        # 8.3절 Tool Calling Accuracy 측정용: 실제로 던진 질의 문자열을 남김
        self.last_queries: dict[str, list[str]] = {}

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = self.scoped_techs(state)  # Send fan-out이면 기술 하나, 아니면 전체
        new_evidence: list[Evidence] = []
        new_references: list[Reference] = []
        ordinal = 0
        by_tech: dict[str, TechViewResult] = {}
        if not state.get("tech_scope"):
            self.last_queries = {}

        for tech in techs:
            passages: list[str] = []
            key_by_num: dict[int, str] = {}
            self.last_queries[tech.name] = []
            for template in _QUERY_TEMPLATES:
                query = template.format(tech=tech.name, anchor=tech.search_anchor)
                self.last_queries[tech.name].append(query)
                for r in web_search(query, max_results=MAX_RESULTS_PER_QUERY):
                    ev = self.new_evidence(
                        state, tech.name, ordinal, perspective="market", source_type="웹",
                        source=r.url, quote=r.content[:200], reference_url=r.url,
                    )
                    new_evidence.append(ev)
                    new_references.append(web_reference(r))
                    num = len(key_by_num) + 1
                    key_by_num[num] = ev.key
                    passages.append(f"[근거#{num}] ({r.title}) {r.content[:300]}")
                    ordinal += 1

            # 7.2절: 발췌를 구조화 출력으로 넘겨 ViewResult 형태로 종합함
            view = extract_view_result(
                passages, tech.name, PERSPECTIVE_LABEL, REQUIRED_ITEMS, key_by_num
            )
            by_tech[tech.name] = view

            # counter_facts가 참조한 근거는 stance를 "반대"로 바꿔 evidence_check(7.7)의
            # 반대 근거 유무 규칙이 실제 값을 보게 함
            counter_keys = {k for c in view.counter_facts for k in c.evidence_keys}
            for ev in new_evidence:
                if ev.key in counter_keys:
                    ev.stance = "반대"

        return {
            "market_result": ViewResult(by_tech=by_tech),
            "raw_evidence": new_evidence,
            "raw_references": new_references,
            # 독립 실행 스크립트 호환. 통합 Graph는 raw 영역으로만 병합함.
            "evidence": new_evidence,
            "references": new_references,
        }
