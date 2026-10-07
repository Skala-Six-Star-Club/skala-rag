"""LangGraph 그래프 조립. Orchestrator-Workers 패턴.

    select_tech -> tech_research -> orchestrator --(Send x 계획된 서브 태스크 수)--> [trl | market | stakeholder | domain]_eval
        -> evidence_check --(부족한 (관점, 기술), 재검색 1회)--> orchestrator (re-plan)
                          --(충족 또는 소진)--> evidence_finalize -> synthesize -> report -> quality_eval
        quality_eval --(coverage, bias_control 미달)--> orchestrator
                     --(groundedness, neutrality 미달)--> synthesize
                     --(통과 또는 Loop 상한)--> END

패턴 필수 항목과 코드 위치
- 서브 태스크 목록의 구조화와 State 저장: ``orchestration/orchestrator.py``가 ``Plan``을 ``plan``에 씀
- Dynamic Fan-out: ``orchestration/routing.fan_out_plan``이 계획된 서브 태스크 수만큼 Send를 만듦.
  worker 수는 orchestrator 실행 전에는 정해지지 않음
- 결과 누적과 집계: worker는 ``raw_evidence``(operator.add)와 관점 결과(merge_view_results)에
  누적하고, ``evidence_finalize``가 번호를 확정한 뒤 ``synthesize``(synthesizer)가 집계함
- Fall-back: ``orchestration/worker.with_fallback``이 1회 재시도 후 실패한 서브 태스크를 제외함
- 종료 보장: 재검색 1회(retry_count), 품질 재계획 MAX_QUALITY_REPLANS회와 재작성 MAX_QUALITY_REWRITES회,
  노드 실행 수 MAX_STEPS(step_count), recursion_limit

실행: python app.py (또는 python -m src.graph)
전제: .env(OPENAI_API_KEY, TAVILY_API_KEY), Ollama(qwen3:8b), data/doc_pool/*.pdf
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.common import config
from src.common.observability import log_decision, new_trace_id, read_decisions, run_config
from src.common.state import AgentState
from src.orchestration.routing import (
    NODE_EVIDENCE_CHECK,
    NODE_EVIDENCE_FINALIZE,
    NODE_ORCHESTRATOR,
    NODE_QUALITY_EVAL,
    NODE_REPORT,
    NODE_SELECT_TECH,
    NODE_SYNTHESIZE,
    NODE_TECH_RESEARCH,
    VIEW_NODES,
    fan_out_plan,
    route_after_evidence_check,
    route_after_quality,
)
from src.orchestration.worker import NodeFn, wrap_node, with_fallback

ALL_NODES: tuple[str, ...] = (
    NODE_SELECT_TECH,
    NODE_TECH_RESEARCH,
    NODE_ORCHESTRATOR,
    *VIEW_NODES,
    NODE_EVIDENCE_CHECK,
    NODE_EVIDENCE_FINALIZE,
    NODE_SYNTHESIZE,
    NODE_REPORT,
    NODE_QUALITY_EVAL,
)


def build_graph(nodes: dict[str, NodeFn], checkpointer: Any | None = None) -> CompiledStateGraph:
    """노드 이름 -> 노드 함수 dict를 받아 그래프를 조립함.

    노드 함수는 ``BaseAgent`` 인스턴스 또는 같은 계약의 callable. stub을 넣으면 API 키와
    Doc Pool 없이 흐름만 검증할 수 있음(scripts/graph_flow_check.py).
    """
    missing = [n for n in ALL_NODES if n not in nodes]
    if missing:
        raise ValueError(f"그래프에 필요한 노드가 빠져 있음: {missing}")

    graph = StateGraph(AgentState)
    for name in ALL_NODES:
        node = nodes[name]
        if name in VIEW_NODES:
            node = with_fallback(name, node)
        graph.add_node(name, wrap_node(name, node, finalizer=name == NODE_EVIDENCE_FINALIZE))

    graph.add_edge(START, NODE_SELECT_TECH)
    graph.add_edge(NODE_SELECT_TECH, NODE_TECH_RESEARCH)
    graph.add_edge(NODE_TECH_RESEARCH, NODE_ORCHESTRATOR)

    # Dynamic Fan-out: 계획된 서브 태스크마다 worker 하나
    graph.add_conditional_edges(NODE_ORCHESTRATOR, fan_out_plan, [*VIEW_NODES, NODE_EVIDENCE_FINALIZE])
    # 합류: 같은 superstep의 Send가 모두 끝난 뒤 evidence_check가 1회 실행됨. re-plan에서는
    # 일부 관점만 실행되므로 barrier 합류(add_edge([...], ...))를 쓰지 않음
    for view in VIEW_NODES:
        graph.add_edge(view, NODE_EVIDENCE_CHECK)

    graph.add_conditional_edges(
        NODE_EVIDENCE_CHECK, route_after_evidence_check, [NODE_ORCHESTRATOR, NODE_EVIDENCE_FINALIZE]
    )
    graph.add_edge(NODE_EVIDENCE_FINALIZE, NODE_SYNTHESIZE)
    graph.add_edge(NODE_SYNTHESIZE, NODE_REPORT)
    graph.add_edge(NODE_REPORT, NODE_QUALITY_EVAL)
    graph.add_conditional_edges(
        NODE_QUALITY_EVAL, route_after_quality, [NODE_ORCHESTRATOR, NODE_SYNTHESIZE, END]
    )
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 실제 노드 조립 (Doc Pool 색인은 RAG 3종이 공유함, 5장)
# ---------------------------------------------------------------------------


def load_or_build_doc_pool_index(index_dir: Path = config.DOC_POOL_INDEX_DIR) -> FAISS:
    """Doc Pool 공유 색인을 `index_dir/<임베딩 이름>/`에서 로드하거나 없으면 구축함.

    임베딩 이름별 하위 폴더를 쓰는 이유는 채택 임베딩을 바꿨을 때 이전 모델로 만든
    색인이 차원 검사에 걸리지 않고 조용히 로드되는 사고를 막기 위함.
    """
    from src.common.tools import _embedding_dir_name, get_shared_index

    return get_shared_index(index_dir=index_dir / _embedding_dir_name(config.EMBEDDING_MODEL))


def make_agents(index: FAISS | None = None) -> dict[str, NodeFn]:
    """조정 계층(orchestrator)과 하위 에이전트, 내부 finalizer를 노드로 돌려줌."""
    from src.agents.domain_eval.agent import DomainEvalAgent
    from src.agents.evidence_check.agent import EvidenceCheckAgent
    from src.agents.market_eval.agent import MarketEvalAgent
    from src.agents.quality_eval.agent import QualityEvalAgent
    from src.agents.report.agent import ReportAgent
    from src.agents.select_tech.agent import SelectTechAgent
    from src.agents.stakeholder_eval.agent import StakeholderEvalAgent
    from src.agents.synthesize.agent import SynthesizeAgent
    from src.agents.tech_research.agent import TechResearchAgent
    from src.agents.trl_eval.agent import TrlEvalAgent
    from src.common.evidence import finalize_evidence
    from src.orchestration.orchestrator import OrchestratorAgent

    if index is None:
        index = load_or_build_doc_pool_index()

    agents = [
        SelectTechAgent(),
        TechResearchAgent(index),
        OrchestratorAgent(),
        TrlEvalAgent(index),
        MarketEvalAgent(),
        StakeholderEvalAgent(),
        DomainEvalAgent(index),
        EvidenceCheckAgent(),
        SynthesizeAgent(),
        ReportAgent(),
        QualityEvalAgent(),
    ]
    nodes: dict[str, NodeFn] = {agent.name: agent for agent in agents}
    nodes[NODE_EVIDENCE_FINALIZE] = finalize_evidence
    return nodes


def build_default_graph(index: FAISS | None = None, checkpointer: Any | None = None) -> CompiledStateGraph:
    return build_graph(make_agents(index), checkpointer=checkpointer)


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
