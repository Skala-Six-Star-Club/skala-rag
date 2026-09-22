"""그래프 흐름 검증(API 키·PDF 불필요). stub 노드로 12장 흐름과 (관점, 기술) Send fan-out을 확인함.

확인 항목:
1. tech_research 뒤 관점 4개 x 기술 2개 = 8회 호출이 같은 superstep에서 실행됨
2. 같은 관점 필드에 두 호출이 쓴 ViewResult가 기술 키로 합쳐짐(merge_view_results)
3. evidence_check가 특정 (관점, 기술)만 부족하다고 하면 그 조합만 재검색됨(retry_scopes)
4. 재검색 결과의 provisional key가 1차와 충돌하지 않고 finalize가 연속 번호를 부여함
5. judge 위반 -> synthesize 재작성 1회 -> report

실행: python -m scripts.graph_flow_check
"""

from __future__ import annotations

from collections import Counter

from src.common.base_agent import BaseAgent
from src.common.evidence import finalize_evidence
from src.common.state import Claim, JudgeFeedback, Synthesis, TechSpec, TechViewResult, ViewResult
from src.graph import ALL_NODES, NODE_EVIDENCE_FINALIZE, VIEW_NODES, build_graph

TECHS = [
    TechSpec(name="TurboQuant", camp="SW", role="target", search_anchor="Google"),
    TechSpec(name="ITME", camp="HW", role="target", search_anchor="SK hynix"),
]
CALLS: Counter = Counter()  # (node, tech, attempt) -> 호출 수
RESULT_KEY = {"trl_eval": "trl_result", "market_eval": "market_result", "stakeholder_eval": "stakeholder_result", "domain_eval": "domain_result"}
PERSPECTIVE = {"trl_eval": "trl", "market_eval": "market", "stakeholder_eval": "stakeholder", "domain_eval": "domain"}


class StubView(BaseAgent):
    """관점 노드 stub: 기술당 근거 2건(재검색이면 반대 1건 포함)과 Claim 1건을 만듦."""

    def __init__(self, name: str):
        self.name = name

    def run(self, state):
        techs = self.scoped_techs(state)
        assert len(techs) == 1, f"Send fan-out이면 기술 하나만 받아야 함: {[t.name for t in techs]}"
        is_retry = self.name in (state.get("retry_targets") or [])
        evidence, by_tech = [], {}
        for tech in techs:
            CALLS[(self.name, tech.name, int(is_retry))] += 1
            e1 = self.new_evidence(state, tech.name, 0, perspective=PERSPECTIVE[self.name], source_type="논문", source="p.1", quote="q1")
            e2 = self.new_evidence(state, tech.name, 1, perspective=PERSPECTIVE[self.name], source_type="웹", source="u", quote="q2", stance="반대" if is_retry else "지지")
            evidence += [e1, e2]
            by_tech[tech.name] = TechViewResult(confirmed_facts=[Claim(statement=f"{tech.name} fact", evidence_keys=[e1.key])])
        return {RESULT_KEY[self.name]: ViewResult(by_tech=by_tech), "raw_evidence": evidence}


class StubEvidenceCheck(BaseAgent):
    """규칙 stub: 1차에서는 trl_eval/ITME, domain_eval/TurboQuant만 부족하다고 판정, 2차는 통과."""

    name = "evidence_check"

    def run(self, state):
        retry_count = state.get("retry_count", 0)
        # 병합 검증: 네 관점 모두 두 기술이 by_tech에 있어야 함
        for node, key in RESULT_KEY.items():
            assert set(state[key].by_tech) == {"TurboQuant", "ITME"}, f"{key} by_tech 병합 실패: {list(state[key].by_tech)}"
        if retry_count == 0:
            return {"retry_targets": ["trl_eval", "domain_eval"], "retry_scopes": {"trl_eval": ["ITME"], "domain_eval": ["TurboQuant"]}, "retry_count": 1}
        return {"retry_targets": [], "retry_scopes": {}, "retry_count": retry_count}


def stub_synthesize(state):
    return {"synthesis": Synthesis(summary="s"), "rewrite_count": state.get("rewrite_count", 0) + (1 if state.get("judge_feedback") else 0)}


def stub_judge(state):
    CALLS[("judge", "-", 0)] += 1
    violated = CALLS[("judge", "-", 0)] == 1
    return {"judge_feedback": JudgeFeedback(has_biased_expression=violated, is_balanced=True)}


def stub_report(state):
    ids = [e.id for e in state["evidence"]]
    assert ids == sorted(ids) and len(ids) == len(set(ids)), f"최종 번호 오류: {ids}"
    return {"report_md": "ok"}


def main() -> None:
    nodes = {
        "select_tech": lambda s: {"techs": TECHS, "domain": "d"},
        "tech_research": lambda s: {"tech_profiles": {}, "raw_evidence": []},
        **{v: StubView(v) for v in VIEW_NODES},
        "evidence_check": StubEvidenceCheck(),
        NODE_EVIDENCE_FINALIZE: finalize_evidence,
        "synthesize": stub_synthesize,
        "judge": stub_judge,
        "report": stub_report,
    }
    assert set(nodes) == set(ALL_NODES)
    final = build_graph(nodes).invoke({}, config={"recursion_limit": 50})

    first = {k: v for k, v in CALLS.items() if k[0] in VIEW_NODES and k[2] == 0}
    retry = {k: v for k, v in CALLS.items() if k[0] in VIEW_NODES and k[2] == 1}
    assert len(first) == 8 and all(v == 1 for v in first.values()), first
    assert set(retry) == {("trl_eval", "ITME", 1), ("domain_eval", "TurboQuant", 1)}, retry
    keys = [e.key for e in final["raw_evidence"]]
    assert len(keys) == len(set(keys)) == 20, (len(keys), len(set(keys)))
    assert final["retry_count"] == 1 and final["rewrite_count"] == 1
    assert len(final["evidence"]) == 20 and final["evidence"][-1].id == 20
    # 재검색된 기술은 반대 근거가 생기고, 재검색 안 된 기술은 그대로여야 함
    counter = Counter((e.perspective, e.tech) for e in final["evidence"] if e.stance == "반대")
    assert counter == {("trl", "ITME"): 1, ("domain", "TurboQuant"): 1}, counter
    print("1차 fan-out:", len(first), "회 (관점 4 x 기술 2)")
    print("재검색:", sorted(k[:2] for k in retry))
    print("근거:", len(final["evidence"]), "건, 최종 번호 1~", final["evidence"][-1].id, ", key 중복 0")
    print("retry_count", final["retry_count"], "rewrite_count", final["rewrite_count"], "report", final["report_md"])
    print("OK")


if __name__ == "__main__":
    main()
