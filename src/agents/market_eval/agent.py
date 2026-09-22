"""시장 평가 에이전트 (7.5절). 시장 규모/성장성/채택 현황/생태계 지지를 웹
검색만으로 조사함. RAG 미사용(5장: 답이 논문이 아니라 시장 리포트·산업 기사에 있음).

입력: state["tech_profiles"], state["techs"]
출력: {"market_result": ..., "raw_evidence": [...]}

두 기술 모두 같은 질의 템플릿·같은 횟수로 실행해 10장 중립성(대칭 질의) 원칙을
지킴. 앵커 키워드(TechSpec.search_anchor)만 값이 다름.
"""

from __future__ import annotations

from typing import Any

from src.common.base_agent import BaseAgent
from src.common.models import get_generation_llm
from src.common.state import AgentState, Evidence, TechViewResult, ViewResult
from src.common.tools import web_search

_QUERY_TEMPLATES = [
    "{tech} {anchor} 시장 규모",
    "{tech} {anchor} market size billion CAGR forecast",
    "{tech} {anchor} 채택 사례",
    "{tech} {anchor} 프레임워크 지원",
]

# 9.2절 시장성 평가 기준
_EXTRACTION_PROMPT = """\
다음은 '{tech}' 기술의 시장성에 대한 웹 검색 근거임. 아래 세 기준(9.2절)으로 정리해줘.
- 시장 규모와 성장성
- 상용화, 채택 현황
- 생태계 지지(프레임워크 지원, 표준화 동향)

세 기준 각각이 confirmed_facts·counter_facts·unconfirmed_items 어디에든 최소 1번은
언급되게 해줘 — 근거가 있으면 confirmed/counter_facts에 넣고, 그 경우 unconfirmed_items에
같은 기준을 "근거 없음"이라고 다시 쓰지 마(모순임). confirmed_facts와 counter_facts
어디에도 전혀 등장하지 않는 기준만 unconfirmed_items에 "OO 기준: 근거 없음"처럼 넣어줘.

각 사실을 confirmed_facts(시장에서 긍정적으로 받아들여지고 있다는 근거) 또는
counter_facts(시장 반응이 부정적이거나 채택에 걸림돌이 있다는 근거)로 나누고,
판단하기엔 근거가 부족한 부분은 unconfirmed_items에 넣어줘. 우열 판정 표현은 쓰지 마.
각 사실에는 반드시 아래 [근거#N] 번호를 evidence_ids에 넣고, 목록에 없는 번호를
만들어내지 마.

{passages}
"""


class MarketEvalAgent(BaseAgent):
    name = "market_eval"
    uses_rag = False

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        is_retry = self.name in (state.get("retry_targets") or [])
        new_evidence: list[Evidence] = []
        ordinal = 0
        by_tech: dict[str, TechViewResult] = {}
        llm = get_generation_llm().with_structured_output(TechViewResult)

        for tech in techs:
            passages: list[str] = []
            tech_evidence: list[Evidence] = []
            local_id_to_key: dict[int, str] = {}
            for template in _QUERY_TEMPLATES:
                query = template.format(tech=tech.name, anchor=tech.search_anchor)
                for r in web_search(query, max_results=3):
                    evidence_key = self.provisional_evidence_key(state, tech.name, ordinal)
                    evidence = Evidence(
                        key=evidence_key,
                        tech=tech.name,
                        perspective="market",
                        stance="반대" if is_retry else "지지",  # 기본값. 아래 분류 결과로 조정함
                        source_type="웹",
                        source=r.url,
                        quote=r.content[:200],
                    )
                    tech_evidence.append(evidence)
                    local_id = ordinal + 1
                    local_id_to_key[local_id] = evidence_key
                    # LLM에는 짧은 로컬 정수로 보여줌 — 긴 provisional key를 그대로
                    # 되읊게 하면 오탈자·환각 위험이 커서, 응답은 로컬 정수로 받고
                    # 아래에서 evidence_keys로 되돌림.
                    passages.append(f"[근거#{local_id}] {r.content[:300]}")
                    ordinal += 1

            valid_local_ids = set(local_id_to_key)
            view_result: TechViewResult = llm.invoke(
                _EXTRACTION_PROMPT.format(tech=tech.name, passages="\n\n".join(passages))
            )  # type: ignore[assignment]

            # 환각 근거 번호 방지 + 로컬 정수 -> provisional key 역매핑
            # (7.10절 인용 안전장치와 동일 원칙)
            for claim in view_result.confirmed_facts + view_result.counter_facts:
                kept = [i for i in claim.evidence_ids if i in valid_local_ids]
                claim.evidence_ids = []
                claim.evidence_keys = [local_id_to_key[i] for i in kept]

            confirmed_keys = {
                k for claim in view_result.confirmed_facts for k in claim.evidence_keys
            }
            counter_keys = {
                k for claim in view_result.counter_facts for k in claim.evidence_keys
            }
            for e in tech_evidence:
                if e.key in counter_keys:
                    e.stance = "반대"
                elif e.key in confirmed_keys:
                    e.stance = "지지"

            new_evidence.extend(tech_evidence)
            by_tech[tech.name] = view_result

        return {
            "market_result": ViewResult(by_tech=by_tech),
            "raw_evidence": new_evidence,
            "evidence": new_evidence,
        }
