"""이해관계자 평가 에이전트 (7.6절). 경쟁 진영, 도입 기업과 개발자, 투자
업계의 반응을 웹 검색만으로 조사함. RAG 미사용(5장 원칙).

입력: state["tech_profiles"], state["techs"]
출력: {"stakeholder_result": ..., "raw_evidence": [...]}
(근거는 provisional key로 발급되고 evidence_finalize가 최종 번호를 부여함)

market_eval과 동일한 검색 앵커 키워드를 재사용해 질의 경로를 코드로 고정함
(4장 도구 선택 원칙, 10장 대칭 질의). 검색 결과는 9.3절 필수 항목(경쟁 진영의
반응, 도입 기업과 개발자의 의견과 도입 장벽, 투자 업계의 평가)에 맞춰 구조화
출력으로 ViewResult에 담김.
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.focus import COUNTER, FOCI, non_counter_foci
from src.common.state import AgentState, Evidence, Reference, TechViewResult, ViewResult
from src.common.tools import extract_view_result, web_reference, web_search_ladder

# 질의와 필수 항목은 src/common/focus.py 카탈로그가 정본. 초점 하나가 필수 항목 하나에 대응함.
# 단독 실행(서브 태스크 없음)은 기존처럼 필수 항목 전체의 질의를 한 번에 돌림.
_BASE_FOCI = non_counter_foci("stakeholder")
_QUERY_TEMPLATES = [FOCI["stakeholder"][f].web[0][0] for f in _BASE_FOCI]
# 0건 분기용 질의 사다리. 한국어 기본 질의가 0건이면 같은 뜻의 영어 질의, 그다음 앵커를 뺀 질의 순으로
# 넓힘. 두 기술에 같은 사다리를 적용하므로 10장 대칭 질의 원칙은 유지됨.
_QUERY_TEMPLATES_EN = [FOCI["stakeholder"][f].web[0][1] for f in _BASE_FOCI]
# 9장 평가 기준: 완전성(8.2) 채점 시 빠짐없이 다뤄야 하는 필수 항목
REQUIRED_ITEMS = [FOCI["stakeholder"][f].items[0] for f in _BASE_FOCI]
PERSPECTIVE_LABEL = "이해관계자"
MAX_RESULTS_PER_QUERY = 3
# 초점의 질의가 하나뿐이면 그 질의에서 결과를 더 받아 칸당 근거 수를 맞춤
SINGLE_QUERY_MAX_RESULTS = 5


class StakeholderEvalAgent(BaseAgent):
    name = "stakeholder_eval"
    uses_rag = False

    def __init__(self) -> None:
        # 8.3절 Tool Calling Accuracy 측정용: 코드가 고정한 기본 질의(템플릿 기준)와
        # 실제로 결과를 낸 질의(사다리 단계, 0건이면 None)를 따로 남김
        self.last_queries: dict[str, list[str]] = {}
        self.last_queries_used: dict[str, list[str | None]] = {}

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = self.scoped_techs(state)  # Send fan-out이면 기술 하나, 아니면 전체
        new_evidence: list[Evidence] = []
        new_references: list[Reference] = []
        ordinal = 0
        by_tech: dict[str, TechViewResult] = {}
        if not state.get("tech_scope"):
            self.last_queries = {}
            self.last_queries_used = {}
        spec = self.focus_spec(state, "stakeholder")
        if spec is None:
            pairs = FOCI["stakeholder"][COUNTER].web if self.focus(state) == COUNTER else tuple(zip(_QUERY_TEMPLATES, _QUERY_TEMPLATES_EN))
            required = REQUIRED_ITEMS
        else:
            pairs, required = spec.web, list(spec.items)
        max_results = MAX_RESULTS_PER_QUERY if len(pairs) > 1 else SINGLE_QUERY_MAX_RESULTS

        for tech in techs:
            passages: list[str] = []
            key_by_num: dict[int, str] = {}
            self.last_queries[tech.name] = []
            self.last_queries_used[tech.name] = []
            for template, template_en in pairs:
                query = template.format(tech=tech.name, anchor=tech.search_anchor)
                self.last_queries[tech.name].append(query)
                ladder = [
                    query,
                    template_en.format(tech=tech.name, anchor=tech.search_anchor),
                    template.format(tech=tech.name, anchor="").replace("  ", " ").strip(),
                ]
                results, used = web_search_ladder(ladder, max_results=max_results)
                self.last_queries_used[tech.name].append(used)
                for r in results:
                    ev = self.new_evidence(
                        state, tech.name, ordinal, perspective="stakeholder", source_type="웹",
                        source=r.url, quote=r.content[:200], reference_url=r.url,
                    )
                    new_evidence.append(ev)
                    new_references.append(web_reference(r))
                    num = len(key_by_num) + 1
                    key_by_num[num] = ev.key
                    passages.append(f"[근거#{num}] ({r.title}) {r.content[:300]}")
                    ordinal += 1

            # 7.3절: 발췌를 구조화 출력으로 넘겨 ViewResult 형태로 종합함
            if not passages:
                # 사다리 전부 0건: 근거 없이 판단하지 않고 미확인으로 명시함. 근거 0건은
                # evidence_check(7.7)가 재검색 대상으로 잡고, 재검색도 0건이면 보고서
                # 한계점에 그대로 드러남(12장 "예산 소진 시 미확인 상태로 진행").
                print(f"[{self.name}] {tech.name}: 웹 검색 결과 0건 (질의 {len(ladder) * len(pairs)}종 시도)")
                by_tech[tech.name] = TechViewResult(
                    unconfirmed_items=[*required, f"웹 검색 결과 없음: {', '.join(self.last_queries[tech.name])}"]
                )
                continue
            view = extract_view_result(
                passages, tech.name, PERSPECTIVE_LABEL, required, key_by_num
            )
            by_tech[tech.name] = view

            # counter_facts가 참조한 근거는 stance를 "반대"로 바꿔 evidence_check(7.7)의
            # 반대 근거 유무 규칙이 실제 값을 보게 함
            counter_keys = {k for c in view.counter_facts for k in c.evidence_keys}
            for ev in new_evidence:
                if ev.key in counter_keys:
                    ev.stance = "반대"

        return {
            "stakeholder_result": ViewResult(by_tech=by_tech),
            "raw_evidence": new_evidence,
            "raw_references": new_references,
            # 독립 실행 스크립트 호환. 통합 Graph는 raw 영역으로만 병합함.
            "evidence": new_evidence,
            "references": new_references,
        }
