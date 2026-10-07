"""노드 래퍼. 공통 갱신(step_count, 레거시 필드 정리)과 worker Fall-back을 담당함.

Fall-back 정책: worker가 예외를 내면 같은 서브 태스크를 WORKER_MAX_RETRIES회 다시
실행하고, 그래도 실패하면 그 서브 태스크를 제외함. 제외된 서브 태스크는
``excluded_subtasks``에 누적되어 보고서 한계점에 기재되고, 나머지 worker 결과로 진행함.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.common import config
from src.common.observability import log_decision
from src.common.state import AgentState

NodeFn = Callable[[AgentState], dict[str, Any]]

# State에 두지 않는 반환 키. report 본문은 파일로만 남김(체크포인트 크기 관리)
_DROP_KEYS = ("report_md",)


def wrap_node(node_name: str, node: NodeFn, *, finalizer: bool = False) -> NodeFn:
    """모든 노드에 공통 갱신을 붙임.

    - step_count += 1 (종료 가드)
    - 업무 노드가 레거시 계약으로 ``evidence``, ``references``를 반환하면 raw 누적 영역으로 옮김.
      확정 필드는 finalizer만 씀
    - State에 두지 않는 키 제거
    """

    def wrapped(state: AgentState) -> dict[str, Any]:
        update = dict(node(state) or {})
        if not finalizer:
            if "evidence" in update:
                update.setdefault("raw_evidence", update["evidence"])
                update.pop("evidence", None)
            if "references" in update:
                update.setdefault("raw_references", update["references"])
                update.pop("references", None)
        for key in _DROP_KEYS:
            update.pop(key, None)
        update["step_count"] = 1
        return update

    wrapped.__name__ = f"{node_name}_node"
    return wrapped


def with_fallback(node_name: str, node: NodeFn, *, max_retries: int | None = None) -> NodeFn:
    """worker 노드에 재시도 후 제외 Fall-back을 붙임."""

    retries = config.WORKER_MAX_RETRIES if max_retries is None else max_retries

    def guarded(state: AgentState) -> dict[str, Any]:
        subtask = state.get("subtask")
        subtask_id = subtask.subtask_id if subtask is not None else node_name
        trace_id = state.get("trace_id")
        error = ""
        errors: dict[str, str] = {}
        for attempt in range(retries + 1):
            try:
                update = dict(node(state) or {})
            except Exception as exc:  # noqa: BLE001 - worker 하나의 실패가 그래프 전체를 멈추지 않게 함
                error = f"{type(exc).__name__}: {exc}"
                errors[subtask_id] = error
                log_decision(
                    trace_id, node_name, "worker_failed", error, subtask_id=subtask_id, attempt=attempt
                )
                continue
            if attempt:
                log_decision(trace_id, node_name, "retry_succeeded", f"{attempt}회 재시도 후 성공", subtask_id=subtask_id)
            update["node_status"] = {subtask_id: "done"}
            if errors:
                update["task_errors"] = errors
            return update

        log_decision(
            trace_id, node_name, "exclude",
            f"{retries}회 재시도 후에도 실패해 제외하고 나머지 결과로 진행", subtask_id=subtask_id,
        )
        update: dict[str, Any] = {
            "node_status": {subtask_id: "excluded"},
            "task_errors": errors,
            "last_error": f"{subtask_id}: {error}",
        }
        if subtask is not None:
            update["excluded_subtasks"] = [subtask]
        return update

    guarded.__name__ = f"{node_name}_guarded"
    return guarded
