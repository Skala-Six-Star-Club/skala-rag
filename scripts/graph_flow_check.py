"""그래프 흐름 검증(API 키, PDF, 임베딩 모델 불필요).

orchestrator, evidence_check, evidence_finalize, quality_eval은 실제 코드를 쓰고, LLM과 검색이
필요한 노드와 임베딩만 stub으로 바꿔 Orchestrator-Workers 흐름을 확인함.

확인 항목
1. Dynamic Fan-out: orchestrator가 계획한 (관점, 기술, 초점) 서브 태스크 수만큼 worker가 실행됨.
   LLM 계획의 빈 칸, 비대칭, 잘못된 초점을 계획 검증이 보정하고 사유를 남김
2. 같은 (관점, 기술)의 서브 태스크 결과가 덮어쓰이지 않고 이어 붙음, 임시 key 충돌 없음
3. Fall-back: 한 번 실패한 worker는 재시도로 성공, 계속 실패한 worker는 제외되고 기록됨
4. evidence_check의 부족 칸만 re-plan(round 1)으로 재조사
5. quality_eval 커버리지 미달 -> orchestrator Loop(round 2) -> 통과 후 종료. 재계획, 재작성 예산은 따로 셈
6. finalize 재실행 시 앞서 확정한 근거 번호가 바뀌지 않음
7. 체크포인트 재개: synthesize에서 중단된 실행을 같은 thread_id로 이어서 완료
8. 결정 로그(JSONL)에 orchestrator, evidence_check, quality_eval 결정과 사유가 남음

실행: python -m scripts.graph_flow_check
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="graph_flow_check_"))
os.environ["DECISION_LOG_DIR"] = str(_TMP / "logs")
os.environ["MAX_QUALITY_REPLANS"] = "1"
os.environ["MAX_QUALITY_REWRITES"] = "1"

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import src.agents.evidence_check.agent as evidence_check_module  # noqa: E402
from src.agents.evidence_check.agent import EvidenceCheckAgent  # noqa: E402
from src.agents.quality_eval.agent import QualityEvalAgent, check_biased_phrases, decide_route  # noqa: E402
from src.common.base_agent import BaseAgent  # noqa: E402
from src.common.evidence import finalize_evidence  # noqa: E402
from src.common.observability import read_decisions, run_config  # noqa: E402
from src.common.state import Claim, CriterionResult, Gap, Synthesis, TechProfile, TechSpec, TechViewResult, ViewResult  # noqa: E402
from src.graph import ALL_NODES, NODE_EVIDENCE_FINALIZE, VIEW_NODES, build_graph  # noqa: E402
from src.orchestration.orchestrator import OrchestratorAgent, _Pick, _PlanDraft, validate_plan  # noqa: E402

TECHS = [
    TechSpec(name="TurboQuant", camp="SW", role="target", search_anchor="Google"),
    TechSpec(name="ITME", camp="HW", role="target", search_anchor="SK hynix"),
]
RESULT_KEY = {"trl_eval": "trl_result", "market_eval": "market_result", "stakeholder_eval": "stakeholder_result", "domain_eval": "domain_result"}
CALLS: Counter = Counter()  # (node, tech, focus, round) -> 실행 수
FAILURES: Counter = Counter()
FLAKY = ("market_eval", "ITME", "size", 0)  # 1회 실패 후 성공
BROKEN = ("stakeholder_eval", "TurboQuant", "competitor", 0)  # 계속 실패 -> 제외
CRASH = {"synthesize": True}  # 첫 synthesize에서 중단해 체크포인트 재개를 확인


class FakeEmbed:
    """모든 문장을 같은 벡터로 보내 그라운딩 규칙을 통과시키는 stub."""

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _pick(perspective, tech, focus, reason="stub"):
    return _Pick(perspective=perspective, tech=tech, focus=focus, reason=reason)


class FakeOrchestratorLLM:
    """일부러 비대칭, 빈 칸, 잘못된 초점이 섞인 계획을 내는 stub."""

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return _PlanDraft(
            picks=[
                _pick("trl", "TurboQuant", "estimate"), _pick("trl", "ITME", "estimate"),
                _pick("trl", "ITME", "productization", "ITME 양산 근거 공백"),
                _pick("market", "TurboQuant", "adoption"), _pick("market", "ITME", "adoption"),
                _pick("market", "TurboQuant", "size", "시장 규모 근거 공백"),
                _pick("market", "ITME", "foo", "카탈로그에 없는 초점"),
                _pick("domain", "TurboQuant", "latency"), _pick("domain", "ITME", "latency"),
                _pick("domain", "TurboQuant", "counter"), _pick("domain", "ITME", "counter"),
            ],
            rationale="stub 계획",
        )


class StubView(BaseAgent):
    """관점 worker stub: 서브 태스크 하나당 근거 3건과 Claim 1건.
    market 이외 관점과 모든 counter 초점은 반대 근거 1건을 포함함. 따라서 round 0 뒤
    evidence_check는 반대 근거가 없는 market 두 칸과, 제외되어 근거가 0건인 stakeholder/TurboQuant만
    부족하다고 판정해야 함."""

    def __init__(self, name: str):
        self.name = name

    def run(self, state):
        subtask = state["subtask"]
        techs = self.scoped_techs(state)
        assert len(techs) == 1 and techs[0].name == subtask.tech
        key = (self.name, subtask.tech, subtask.focus, subtask.round)
        CALLS[key] += 1
        if key == BROKEN or (key == FLAKY and FAILURES[key] == 0):
            FAILURES[key] += 1
            raise RuntimeError(f"stub 실패 {key}")
        perspective = self.name.removesuffix("_eval")
        stance = "반대" if subtask.focus == "counter" or perspective != "market" else "지지"
        e1 = self.new_evidence(state, subtask.tech, 0, perspective=perspective, source_type="논문", source="p.1", quote="q", reference_url=f"https://paper/{perspective}")
        e2 = self.new_evidence(state, subtask.tech, 1, perspective=perspective, source_type="웹", source="u", quote="q", stance=stance, reference_url=f"https://web/{perspective}/{subtask.focus}/{subtask.round}")
        e3 = self.new_evidence(state, subtask.tech, 2, perspective=perspective, source_type="웹", source="u2", quote="q", reference_url=f"https://web2/{perspective}")
        claim = Claim(statement=f"{subtask.tech} {perspective} {subtask.focus} r{subtask.round}", evidence_keys=[e1.key, e2.key])
        view = TechViewResult(**({"counter_facts": [claim]} if stance == "반대" else {"confirmed_facts": [claim]}))
        return {RESULT_KEY[self.name]: ViewResult(by_tech={subtask.tech: view}), "raw_evidence": [e1, e2, e3]}


def stub_tech_research(state):
    return {
        "tech_profiles": {t.name: TechProfile(tech=t.name, overview="o", scope="s", limitations="l", differentiation="d") for t in TECHS},
        "raw_evidence": [],
    }


def stub_synthesize(state):
    if CRASH["synthesize"]:
        CRASH["synthesize"] = False
        raise RuntimeError("stub 중단: 체크포인트 재개 확인용")
    CALLS[("synthesize", "-", "-", 0)] += 1
    return {"synthesis": Synthesis(summary="두 기술은 관점별로 다르게 평가됨 [근거#1]."), "rewrite_count": state.get("rewrite_count", 0)}


FIRST_FINAL_IDS: dict[str, int] = {}


def stub_report(state):
    """4장 관점별 평가에 (관점, 기술)마다 근거 하나를 인용. 첫 보고서는 domain/ITME를 빠뜨려 coverage 미달을 만듦."""
    n = CALLS[("report", "-", "-", 0)]
    CALLS[("report", "-", "-", 0)] += 1
    if n == 0:
        FIRST_FINAL_IDS.update({e.key: e.id for e in state["evidence"]})
    cited: dict[tuple[str, str], list[int]] = {}
    for e in state["evidence"]:
        cited.setdefault((e.perspective, e.tech), []).append(e.id)
    lines = []
    for (perspective, tech), ids in cited.items():
        if n == 0 and (perspective, tech) == ("domain", "ITME"):
            continue
        lines.append(f"- {tech} {perspective} 서술 " + "".join(f"[근거#{i}]" for i in ids))
    sections = {
        "SUMMARY": "두 기술은 관점별로 다르게 평가됨 [근거#1].",
        "4. 관점별 평가": "\n".join(lines),
        "5. 시사점": "관점에 따라 평가가 갈림 [근거#2].",
    }
    path = _TMP / "report.json"
    path.write_text(json.dumps({"evidence_token_sections": sections}, ensure_ascii=False), encoding="utf-8")
    return {"report_md": "ok", "report_path": str(_TMP / "report.md"), "report_json_path": str(path)}


def unit_checks() -> None:
    techs = [t.name for t in TECHS]
    # rule 모드(빈 계획): 칸마다 기본 초점 하나 = 8개, 보정 사유 8건
    subtasks, corrections = validate_plan([], techs, 16)
    assert len(subtasks) == 8 and len(corrections) == 8, (len(subtasks), corrections)
    # 상한: 칸이 비지 않고 필수 초점(trl/estimate)은 남긴 채 줄임
    many = [_pick(p, t, f) for p in ("trl", "market") for f in ("estimate", "experiment", "open_impl", "size", "adoption", "ecosystem", "counter") for t in techs]
    subtasks, corrections = validate_plan(many, techs, 10)
    cells = {(s.perspective, s.tech) for s in subtasks}
    assert len(subtasks) <= 10 and len(cells) == 8, (len(subtasks), cells)
    assert {("trl", "estimate", t) for t in techs} <= {(s.perspective, s.focus, s.tech) for s in subtasks}
    assert any(c.startswith("상한 조정") for c in corrections)
    assert check_biased_phrases({"SUMMARY": "A가 B보다 더 우수함 [근거#1]."}), "금지 표현 규칙"
    assert not check_biased_phrases({"SUMMARY": "A는 SW, B는 HW 관점에서 접근함 [근거#1]."})
    ok = CriterionResult(name="coverage", passed=True, method="rule")
    bad_text = CriterionResult(name="neutrality", passed=False, method="rule", issues=["x"])
    bad_data = CriterionResult(name="coverage", passed=False, method="rule", issues=["x"])
    gap = [Gap(perspective="trl", tech="ITME", source="quality_eval")]
    assert decide_route([ok, bad_text], [], True, True) == "synthesize"
    assert decide_route([ok, bad_text], [], True, False) == "end"
    assert decide_route([ok], [], True, True) == "end"
    # 조사와 서술이 모두 미달이면 재조사 먼저. 재계획 예산을 쓴 뒤에도 재작성 기회는 남음
    assert decide_route([bad_data, bad_text], gap, True, True) == "orchestrator"
    assert decide_route([bad_data, bad_text], gap, False, True) == "synthesize"
    assert decide_route([bad_data], gap, False, True) == "end"
    # counter 초점만 있는 칸은 기본 초점을 더해 균형 보정
    subtasks, corrections = validate_plan([_pick("domain", t, "counter") for t in techs], techs, 16)
    domain = {(s.focus, s.tech) for s in subtasks if s.perspective == "domain"}
    assert {("cache_reuse", t) for t in techs} <= domain, domain
    assert sum(c.startswith("균형 보정") for c in corrections) == 2, corrections


def main() -> None:
    unit_checks()
    evidence_check_module.get_embedding_model = lambda: FakeEmbed()
    nodes = {
        "select_tech": lambda s: {"techs": TECHS, "domain": "d"},
        "tech_research": stub_tech_research,
        "orchestrator": OrchestratorAgent(mode="llm", llm=FakeOrchestratorLLM()),
        **{v: StubView(v) for v in VIEW_NODES},
        "evidence_check": EvidenceCheckAgent(),
        NODE_EVIDENCE_FINALIZE: finalize_evidence,
        "synthesize": stub_synthesize,
        "report": stub_report,
        "quality_eval": QualityEvalAgent(embed=FakeEmbed(), use_llm=False),
    }
    assert set(nodes) == set(ALL_NODES)
    graph = build_graph(nodes, checkpointer=InMemorySaver())
    trace_id = "flowcheck"
    cfg = run_config(trace_id)

    try:
        graph.invoke({"trace_id": trace_id}, config=cfg)
        raise AssertionError("synthesize stub이 중단되지 않음")
    except RuntimeError as exc:
        assert "체크포인트 재개" in str(exc), exc
    calls_before_resume = sum(v for k, v in CALLS.items() if k[0] in VIEW_NODES)
    final = graph.invoke(None, config=cfg)  # 같은 thread_id로 재개
    # 재개 직후 worker를 다시 돌리지 않았는지: round 0, 1의 worker 실행 수는 중단 전과 같아야 함
    round01 = sum(v for k, v in CALLS.items() if k[0] in VIEW_NODES and k[3] in (0, 1))
    assert round01 == calls_before_resume, (round01, calls_before_resume)

    decisions = read_decisions(trace_id)
    plans = [d for d in decisions if d["node"] == "orchestrator"]
    round0 = plans[0]["subtasks"]
    corrections = plans[0]["corrections"]
    # 1. LLM 계획 10개(잘못된 초점 1개 제외) + 대칭 보정 2 + 빈 칸 보정 2(stakeholder) = 14
    assert len(round0) == 14, round0
    assert "r0:trl:productization:TurboQuant" in round0 and "r0:market:size:ITME" in round0
    assert {"r0:stakeholder:competitor:TurboQuant", "r0:stakeholder:competitor:ITME"} <= set(round0)
    assert sum(c.startswith("대칭 보정") for c in corrections) == 2, corrections
    assert sum(c.startswith("빈 칸 보정") for c in corrections) == 2, corrections
    assert any(c.startswith("제거") for c in corrections), corrections
    first = {k: v for k, v in CALLS.items() if k[0] in VIEW_NODES and k[3] == 0}
    assert len(first) == 14, first

    # 3. Fall-back
    assert CALLS[FLAKY] == 2 and CALLS[BROKEN] == 2, (CALLS[FLAKY], CALLS[BROKEN])
    excluded = [s.subtask_id for s in final["excluded_subtasks"]]
    assert excluded == ["r0:stakeholder:competitor:TurboQuant"], excluded
    assert final["node_status"]["r0:stakeholder:competitor:TurboQuant"] == "excluded"
    # 작업 추적: 모든 round의 계획이 누적되고, 끝난 뒤 pending으로 남은 작업이 없음
    planned = [s.subtask_id for s in final["planned_subtasks"]]
    assert len(planned) == len(round0) + len(plans[1]["subtasks"]) + len(plans[2]["subtasks"]), planned
    assert "pending" not in {final["node_status"][sid] for sid in planned}
    assert set(final["task_errors"]) == {"r0:stakeholder:competitor:TurboQuant", "r0:market:size:ITME"}, final["task_errors"]
    assert any(d["decision"] == "retry_succeeded" for d in decisions)

    # 4. evidence_check re-plan은 부족 칸만(round 1)
    round1 = plans[1]["subtasks"]
    assert set(round1) == {"r1:market:counter:TurboQuant", "r1:market:counter:ITME", "r1:stakeholder:counter:TurboQuant"}, round1

    # 5. quality_eval coverage 미달 -> orchestrator round 2(그 칸에서 아직 안 한 초점) -> 상한 후 종료
    round2 = plans[2]["subtasks"]
    assert "r2:domain:cache_reuse:ITME" in round2, round2
    qe = [d for d in decisions if d["node"] == "quality_eval"]
    assert qe[0]["decision"] == "route=orchestrator" and qe[-1]["decision"] == "route=end", [d["decision"] for d in qe]
    assert final["eval_count"] == 2
    assert final["quality_replan_count"] == 1 and final["quality_rewrite_count"] == 0

    # 2. 같은 (관점, 기술) 결과 병합과 key 고유성
    trl_itme = final["trl_result"].by_tech["ITME"]
    statements = [c.statement for c in [*trl_itme.confirmed_facts, *trl_itme.counter_facts]]
    assert any("estimate" in s for s in statements) and any("productization" in s for s in statements), statements
    keys = [e.key for e in final["evidence"]]
    assert len(keys) == len(set(keys)), "임시 key 중복"
    ids = [e.id for e in final["evidence"]]
    assert ids == list(range(1, len(ids) + 1)), ids

    # 6. finalize 재실행 후에도 앞서 확정한 번호 유지
    second = {e.key: e.id for e in final["evidence"]}
    assert all(second[k] == v for k, v in FIRST_FINAL_IDS.items()), "번호가 바뀜"

    # 8. 결정 로그
    nodes_logged = {d["node"] for d in decisions}
    assert {"orchestrator", "evidence_check", "quality_eval"} <= nodes_logged, nodes_logged
    assert final["step_count"] > 0 and "report_md" not in final

    print("round 0 서브 태스크:", len(round0), "개, 계획 보정:", corrections)
    print("round 1 re-plan:", round1)
    print("round 2 re-plan:", round2)
    print("제외:", excluded, ", 재시도 성공:", "/".join(map(str, FLAKY)))
    print("근거:", len(ids), "건, 번호 유지 확인", len(FIRST_FINAL_IDS), "건")
    print("quality_eval:", [d["decision"] for d in qe], "eval_count", final["eval_count"], "step_count", final["step_count"])
    print("결정 로그:", len(decisions), "줄 ->", _TMP / "logs" / f"{trace_id}.jsonl")
    print("OK")


if __name__ == "__main__":
    main()
