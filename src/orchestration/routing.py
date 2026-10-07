"""조건 분기 함수. 라우팅은 State의 제어 구획만 읽음."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END
from langgraph.types import Send

from src.common import config
from src.common.observability import log_decision
from src.common.state import AgentState, SubTask

NODE_SELECT_TECH = "select_tech"
NODE_TECH_RESEARCH = "tech_research"
NODE_ORCHESTRATOR = "orchestrator"
NODE_EVIDENCE_CHECK = "evidence_check"
NODE_EVIDENCE_FINALIZE = "evidence_finalize"
NODE_SYNTHESIZE = "synthesize"
NODE_REPORT = "report"
NODE_QUALITY_EVAL = "quality_eval"
VIEW_NODES: tuple[str, ...] = ("trl_eval", "market_eval", "stakeholder_eval", "domain_eval")
RESULT_KEYS = {
    "trl": "trl_result",
    "market": "market_result",
    "stakeholder": "stakeholder_result",
    "domain": "domain_result",
}

# evidence_check 재검색 예산의 그래프 쪽 안전장치(실제 예산은 노드가 retry_count로 관리)
MAX_RETRY = 1


def worker_payload(state: AgentState, subtask: SubTask) -> dict[str, Any]:
    """worker가 받는 최소 입력. State 전체를 복사하지 않아 Send 페이로드와 체크포인트를 작게 유지함."""
    payload: dict[str, Any] = {
        "trace_id": state.get("trace_id"),
        "techs": state.get("techs", []),
        "domain": state.get("domain", ""),
        "tech_profiles": state.get("tech_profiles", {}),
        "subtask": subtask,
        "tech_scope": subtask.tech,
    }
    # re-plan round는 앞 round의 미확인 항목을 이어받으므로 해당 관점 결과만 같이 보냄
    result_key = RESULT_KEYS[subtask.perspective]
    if subtask.round > 0 and state.get(result_key) is not None:
        payload[result_key] = state[result_key]
    return payload
