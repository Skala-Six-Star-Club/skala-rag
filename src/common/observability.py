"""관측성 계층. 결정 로그와 LangSmith 실행 설정을 State 밖에서 관리함.

- 결정 로그: 라우팅, 계획, 재작업, 품질 판정의 결정과 사유를
  ``{trace_id, node, decision, reason, ts, ...}`` 한 줄 JSON으로 ``DECISION_LOG_DIR/<trace_id>.jsonl``에 적재함.
- LangSmith: ``LANGSMITH_TRACING=true``와 ``LANGSMITH_API_KEY``가 있으면 LangChain이 자동으로
  추적함. ``run_config``가 같은 ``trace_id``를 metadata와 ``thread_id``에 넣어 트레이스,
  체크포인트, 결정 로그를 하나의 키로 묶음.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import config

logger = logging.getLogger("skala_rag.decision")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def decision_log_path(trace_id: str) -> Path:
    return config.DECISION_LOG_DIR / f"{trace_id}.jsonl"


def log_decision(trace_id: str | None, node: str, decision: str, reason: str, **extra: Any) -> None:
    """결정 한 건을 JSONL로 적재함. trace_id가 없으면(단독 실행) 로거에만 남김."""
    record = {
        "trace_id": trace_id,
        "node": node,
        "decision": decision,
        "reason": reason,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    logger.info("%s %s: %s (%s)", trace_id or "-", node, decision, reason)
    if not trace_id:
        return
    path = decision_log_path(trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_decisions(trace_id: str) -> list[dict[str, Any]]:
    path = decision_log_path(trace_id)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
