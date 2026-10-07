"""체크포인터. 중단된 실행을 같은 trace_id(thread_id)로 이어서 실행할 수 있게 함."""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from src.common import config


def make_checkpointer(kind: str | None = None) -> Any | None:
    """CHECKPOINTER=sqlite(기본) / memory / none.

    sqlite는 langgraph-checkpoint-sqlite가 있어야 하며, 없으면 프로세스 안에서만
    유지되는 memory로 대체함(프로세스를 다시 띄우면 재개 불가).
    """
    kind = (kind or config.CHECKPOINTER).lower()
    if kind == "none":
        return None
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            return InMemorySaver()
        config.CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.CHECKPOINT_DB_PATH, check_same_thread=False)
        return SqliteSaver(conn)
    return InMemorySaver()
