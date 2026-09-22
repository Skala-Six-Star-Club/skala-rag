"""LangGraph 그래프 조립. docs/agentic-rag-design.md 12장 Graph 설계를 그대로 구현함.

이 파일은 10개 비즈니스 에이전트 **공통**으로 하나만 존재함. 각 `src/agents/{agent}/agent.py`는
`BaseAgent.run(state) -> dict` 계약만 지키는 노드이고, 그 노드들을 어떤 순서·조건으로
잇는지는 전부 여기서 정함. 에이전트별 `scripts/{agent}/test_runner.py`는 그래프 없이
노드 하나만 따로 돌리므로 이 파일과 무관함(schedule.md 2절). 이 파일은 전 에이전트가
붙은 뒤의 "통합 실행"(schedule.md 1절, 8.3 Agent System 지표)에 쓰임.

    select_tech -> tech_research -> Send x (관점 4 x 기술 N): [trl_eval | market_eval | stakeholder_eval | domain_eval]
        -> evidence_check --(근거 부족, 재시도 0회)--> 부족한 (관점, 기술)만 Send로 다시 -> evidence_check
        -> evidence_finalize -> synthesize -> judge --(위반 발견, 재작성 0회)--> synthesize
        -> report

구현 시 지킨 12장 규칙:
- 병렬 분기: tech_research 뒤에 관점 노드 4개를 기술별로 나눠(Send, `fan_out_views`) 같은
  superstep에서 동시 실행함. 각 호출은 전체 state에 `tech_scope`(기술명 하나)를 얹어 받고,
  `BaseAgent.scoped_techs`가 그 기술만 처리함. 같은 관점 필드(`trl_result` 등)에 두 호출이
  동시에 쓰므로 state.py의 `merge_view_results` reducer가 by_tech를 기술 키로 합침.
- 합류: 관점 노드 4개 각각에서 evidence_check로 일반 edge를 둠. 같은 superstep에 실행된
  노드들의 완료 신호는 다음 superstep에 한 번에 모이므로 evidence_check는 1회만 실행됨.
  `add_edge([4개], "evidence_check")` 형태의 barrier 합류를 쓰지 않은 이유는, 반복 1에서
  부족한 관점 노드만 재실행할 때 4개가 전부 도착하지 않아 evidence_check가 영영 깨어나지
  않기 때문임.
- 반복 1: evidence_check가 채운 `retry_targets`(노드 이름)와 `retry_scopes`(노드 -> 부족한
  기술)로 부족한 (관점, 기술)만 Send로 재실행함. 각 관점 노드는 `self.name in
  state["retry_targets"]`로 재검색 초점을 바꿈. 횟수 예산(1회)은 evidence_check가
  `retry_count`로 관리하고, 그래프는 안전장치로 한 번 더 확인함.
- Evidence ID 확정: evidence_check가 끝난 뒤 내부 finalizer가 임시 key를 정렬하고
  연속 정수 ID를 부여한다. 이 노드는 비즈니스 에이전트 수에 포함하지 않는다.
- 반복 2: judge_feedback에 위반이 있고 재작성 예산(1회)이 남아 있으면 synthesize로 돌아감.
  재작성 횟수는 synthesize가 `rewrite_count`에 기록함.
- 조건 분기: 예산 소진 시 부족한 항목을 미확인 상태로 둔 채 다음 단계로 진행함.

실행: python -m src.graph
전제: .env(OPENAI_API_KEY, TAVILY_API_KEY), Ollama(qwen3:8b), data/doc_pool/*.pdf
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.common import config
from src.common.doc_pool import DOC_POOL_SPECS
from src.common.state import AgentState, JudgeFeedback

# 12장 그림의 노드 이름. 각 agent.py의 `name` 속성과 반드시 일치해야 함
# (evidence_check가 retry_targets에 이 이름을 그대로 넣고, 그래프가 그 이름으로 라우팅함).
NODE_SELECT_TECH = "select_tech"
NODE_TECH_RESEARCH = "tech_research"
NODE_EVIDENCE_CHECK = "evidence_check"
NODE_EVIDENCE_FINALIZE = "evidence_finalize"
NODE_SYNTHESIZE = "synthesize"
NODE_JUDGE = "judge"
NODE_REPORT = "report"
VIEW_NODES: tuple[str, ...] = ("trl_eval", "market_eval", "stakeholder_eval", "domain_eval")
ALL_NODES: tuple[str, ...] = (
    NODE_SELECT_TECH,
    NODE_TECH_RESEARCH,
    *VIEW_NODES,
    NODE_EVIDENCE_CHECK,
    NODE_EVIDENCE_FINALIZE,
    NODE_SYNTHESIZE,
    NODE_JUDGE,
    NODE_REPORT,
)

# 12장 "반복 1·2는 1회 한정". 그래프 쪽 안전장치용 상한(실제 예산 관리는 노드가 함).
MAX_RETRY = 1
MAX_REWRITE = 1

NodeFn = Callable[[AgentState], dict[str, Any]]


def _normalize_parallel_update(node_name: str, node: NodeFn) -> NodeFn:
    """구 에이전트의 ``evidence`` 반환을 raw 누적 영역으로 호환 변환한다.

    새 노드는 ``raw_evidence``를 반환하지만, 독립 실행 스크립트나 외부 팀 코드가
    예전 계약인 ``evidence``를 반환할 수 있다. Graph에서는 최종화 노드만 확정
    ``evidence``를 쓰므로, 업무 노드의 레거시 반환은 reducer 영역으로 옮긴다.
    """

    if node_name == NODE_EVIDENCE_FINALIZE:
        return node

    def normalized(state: AgentState) -> dict[str, Any]:
        update = dict(node(state))
        if "evidence" in update:
            update.setdefault("raw_evidence", update["evidence"])
            update.pop("evidence", None)
        if "references" in update:
            update.setdefault("raw_references", update["references"])
            update.pop("references", None)
        return update

    return normalized


# ---------------------------------------------------------------------------
# 조건 분기 (12장 표 "반복 1", "반복 2", "조건 분기")
# ---------------------------------------------------------------------------


def _send_view(node: str, state: AgentState, tech: str) -> Send:
    """관점 노드를 (관점, 기술) 단위로 호출하는 Send. 노드는 payload를 state로 받으므로
    전체 state에 tech_scope만 얹어 보냄(BaseAgent.scoped_techs가 이 값으로 기술을 고름)."""
    return Send(node, {**state, "tech_scope": tech})


def fan_out_views(state: AgentState) -> list[Send]:
    """tech_research 뒤: 관점 노드 4개 x 기술 N개를 같은 superstep에서 병렬 실행함.

    노드 단위(4개)가 아니라 (관점, 기술) 단위(4 x N)로 나누는 이유는 (1) 기술별 검색과
    LLM 호출이 서로 독립이라 실행 시간이 기술 수만큼 줄고, (2) 반복 1에서 부족한
    기술만 골라 재검색할 수 있기 때문(evidence_check의 retry_scopes). 관점 결과는
    state.merge_view_results reducer가 기술 키로 합침.
    """
    techs = [t.name for t in state.get("techs", []) or []]
    return [_send_view(view, state, tech) for view in VIEW_NODES for tech in techs]


def route_after_evidence_check(state: AgentState) -> list[Send] | str:
    """evidence_check 뒤: 부족한 (관점, 기술)만 재검색하거나 evidence_finalize로.

    evidence_check(7.7)가 retry_targets(노드)와 retry_scopes(노드 -> 부족한 기술)를
    채움. scopes에 없는 노드는 전체 기술을 다시 검색함(구 evidence_check 호환).
    예산을 소진했으면 retry_targets가 비어 오고 ID 최종화로 진행함. retry_count가
    상한을 넘었는데도 targets가 남아 있는 비정상 상황은 무한 루프 대신 최종화로
    진행시킴(12장 "조건 분기").
    """
    targets = [t for t in (state.get("retry_targets") or []) if t in VIEW_NODES]
    if targets and state.get("retry_count", 0) <= MAX_RETRY:
        all_techs = [t.name for t in state.get("techs", []) or []]
        scopes = state.get("retry_scopes") or {}
        return [
            _send_view(node, state, tech)
            for node in targets
            for tech in (scopes.get(node) or all_techs)
        ]
    return NODE_EVIDENCE_FINALIZE


def judge_found_violation(feedback: JudgeFeedback | None) -> bool:
    """7.9의 세 검사 항목 중 하나라도 걸리면 위반(12장 그림 "위반 발견")."""
    if feedback is None:
        return False
    return (
        feedback.has_biased_expression
        or bool(feedback.sentences_without_evidence)
        or not feedback.is_balanced
    )


def route_after_judge(state: AgentState) -> str:
    """judge 뒤: 위반이 있고 재작성 예산이 남았으면 synthesize, 아니면 report."""
    if judge_found_violation(state.get("judge_feedback")) and state.get("rewrite_count", 0) < MAX_REWRITE:
        return NODE_SYNTHESIZE
    return NODE_REPORT


# ---------------------------------------------------------------------------
# 그래프 조립
# ---------------------------------------------------------------------------


def build_graph(nodes: dict[str, NodeFn]) -> CompiledStateGraph:
    """노드 이름 -> 노드 함수 dict를 받아 12장 그래프를 조립함.

    노드 함수는 `BaseAgent` 인스턴스(호출 가능) 또는 같은 계약의 아무 callable이면 됨.
    실제 에이전트 대신 stub을 넣으면 API 키·Doc Pool 없이 그래프 흐름만 검증할 수 있음
    (`make_agents()`가 실제 에이전트 10개를 만들어 줌).
    """
    missing = [n for n in ALL_NODES if n not in nodes]
    if missing:
        raise ValueError(f"그래프에 필요한 노드가 빠져 있음: {missing}")

    graph = StateGraph(AgentState)
    for name in ALL_NODES:
        graph.add_node(name, _normalize_parallel_update(name, nodes[name]))

    # 순차: 기술 조사가 끝나야 네 관점이 같은 사실 위에서 출발함
    graph.add_edge(START, NODE_SELECT_TECH)
    graph.add_edge(NODE_SELECT_TECH, NODE_TECH_RESEARCH)

    # 병렬 분기: (관점, 기술) 단위 Send fan-out + 합류 (모듈 docstring 참고)
    graph.add_conditional_edges(NODE_TECH_RESEARCH, fan_out_views, list(VIEW_NODES))
    for view in VIEW_NODES:
        graph.add_edge(view, NODE_EVIDENCE_CHECK)

    # 반복 1: 부족한 (관점, 기술)만 재검색, 아니면 종합으로
    graph.add_conditional_edges(
        NODE_EVIDENCE_CHECK,
        route_after_evidence_check,
        [*VIEW_NODES, NODE_EVIDENCE_FINALIZE],
    )

    # 재시도까지 끝난 뒤에만 임시 Evidence key를 최종 정수 ID로 확정한다.
    graph.add_edge(NODE_EVIDENCE_FINALIZE, NODE_SYNTHESIZE)

    # 반복 2: 위반 시 재작성, 아니면 보고서로
    graph.add_edge(NODE_SYNTHESIZE, NODE_JUDGE)
    graph.add_conditional_edges(
        NODE_JUDGE,
        route_after_judge,
        [NODE_SYNTHESIZE, NODE_REPORT],
    )

    graph.add_edge(NODE_REPORT, END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 실제 비즈니스 에이전트 10개 조립 (Doc Pool 색인은 RAG 3종이 공유함, 5장)
# ---------------------------------------------------------------------------


def load_or_build_doc_pool_index(index_dir: Path = config.DOC_POOL_INDEX_DIR) -> FAISS:
    """Doc Pool 공유 색인을 `index_dir/<임베딩 이름>/`에서 로드하거나 없으면 구축함.

    scripts/{agent}/pdf/v1/index 는 청킹·임베딩 비교실험용이라 건드리지 않고,
    그래프 실행용 색인은 별도 경로에 둠. 임베딩 이름별 하위 폴더를 쓰는 이유는
    채택 임베딩을 바꿨을 때(bge-m3 -> Qwen3-Embedding, 둘 다 1024차원) 이전 모델로
    만든 색인이 차원 검사에 걸리지 않고 조용히 로드되는 사고를 막기 위함.
    실제 로직은 tools.get_shared_index와 같음(RAG 3종 독립 실행과 색인을 공유).
    """
    from src.common.tools import _embedding_dir_name, get_shared_index

    return get_shared_index(index_dir=index_dir / _embedding_dir_name(config.EMBEDDING_MODEL))


def make_agents(index: FAISS | None = None) -> dict[str, NodeFn]:
    """설계서 4장 표의 에이전트 10개와 내부 finalizer를 노드로 돌려줌."""
    from src.agents.domain_eval.agent import DomainEvalAgent
    from src.agents.evidence_check.agent import EvidenceCheckAgent
    from src.agents.judge.agent import JudgeAgent
    from src.agents.market_eval.agent import MarketEvalAgent
    from src.agents.report.agent import ReportAgent
    from src.agents.select_tech.agent import SelectTechAgent
    from src.agents.stakeholder_eval.agent import StakeholderEvalAgent
    from src.agents.synthesize.agent import SynthesizeAgent
    from src.agents.tech_research.agent import TechResearchAgent
    from src.agents.trl_eval.agent import TrlEvalAgent
    from src.common.evidence import finalize_evidence

    if index is None:
        index = load_or_build_doc_pool_index()

    agents = [
        SelectTechAgent(),
        TechResearchAgent(index),
        TrlEvalAgent(index),
        MarketEvalAgent(),
        StakeholderEvalAgent(),
        DomainEvalAgent(index),
        EvidenceCheckAgent(),
        SynthesizeAgent(),
        JudgeAgent(),
        ReportAgent(),
    ]
    nodes: dict[str, NodeFn] = {agent.name: agent for agent in agents}
    nodes[NODE_EVIDENCE_FINALIZE] = finalize_evidence
    return nodes


def build_default_graph(index: FAISS | None = None) -> CompiledStateGraph:
    return build_graph(make_agents(index))


# ---------------------------------------------------------------------------
# 통합 실행 (schedule.md 1절 마지막 단계, 8.3 Loop Efficiency 원자료)
# ---------------------------------------------------------------------------


def summarize_run(final_state: AgentState) -> dict[str, Any]:
    """8.3 Agent System 지표 집계에 쓰는 실행 1회 요약."""
    evidence = final_state.get("evidence", [])
    per_perspective: dict[str, dict[str, int]] = {}
    for e in evidence:
        bucket = per_perspective.setdefault(f"{e.perspective}/{e.tech}", {"지지": 0, "반대": 0})
        bucket[e.stance] += 1
    return {
        "retry_count": final_state.get("retry_count", 0),
        "rewrite_count": final_state.get("rewrite_count", 0),
        "evidence_total": len(evidence),
        "evidence_by_perspective_tech": per_perspective,
        "perspective_confidence": final_state.get("perspective_confidence", {}),
        "overall_confidence": (
            final_state["synthesis"].overall_confidence
            if final_state.get("synthesis") is not None
            else 0.0
        ),
        "weakest_perspective": (
            final_state["synthesis"].weakest_perspective
            if final_state.get("synthesis") is not None
            else None
        ),
        "judge_feedback": (
            final_state["judge_feedback"].model_dump()
            if final_state.get("judge_feedback") is not None
            else None
        ),
        "report_path": final_state.get("report_path"),
    }


def main() -> None:
    graph = build_default_graph()
    final_state: AgentState = graph.invoke({}, config={"recursion_limit": 50})
    print(json.dumps(summarize_run(final_state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
