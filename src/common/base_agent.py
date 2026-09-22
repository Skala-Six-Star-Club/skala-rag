"""에이전트 노드 공통 인터페이스.

모든 에이전트는 `def run(self, state: AgentState) -> dict[str, Any]`만 구현하면 됨.
`__call__`이 곧 LangGraph 노드 함수이므로, 그래프에는 `graph.add_node(agent.name, agent)`
형태로 그대로 붙일 수 있음. 이 계약을 지키는 한, 각 에이전트 내부 로직은 서로
독립적으로 병렬 구현 가능함.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.common import config
from src.common.models import get_generation_llm
from src.common.state import AgentState

# 근거 번호 대역. 노드 이름 -> (시작, 끝 미포함). 병렬 관점 노드 간 번호 충돌 방지용.
# 재검색(12장 반복 1)으로 같은 노드가 다시 실행돼도 자기 대역 안에서 이어 붙임.
EVIDENCE_ID_BAND_SIZE = 1000
EVIDENCE_ID_RANGES: dict[str, tuple[int, int]] = {
    name: (i * EVIDENCE_ID_BAND_SIZE, (i + 1) * EVIDENCE_ID_BAND_SIZE)
    for i, name in enumerate(
        ["tech_research", "trl_eval", "market_eval", "stakeholder_eval", "domain_eval"],
        start=1,
    )
}


def rewrite_query(base_query: str, tech: str) -> str:
    """Pre-retrieval Query Rewriting (7.2~7.4 공통). 짧은 텍스트 생성 호출 1회.

    RAG 3개 에이전트가 동일 지시문을 공유해, 검색 대상 문구만 달라지고
    "무엇을 검색할지"는 여전히 코드가 정하는 4장 원칙을 유지함. 모듈 함수로 둬서
    test_runner.py의 3.3절 Query Rewriting 비교실험에서도 에이전트 인스턴스 없이
    그대로 재사용할 수 있게 함.
    """
    llm = get_generation_llm()
    prompt = (
        f"다음 한국어 검색 질의를 '{tech}' 관련 영어 논문에서 실제로 쓰이는 "
        "학술 표현에 가깝게 영어로 다듬어줘. 다른 설명 없이 다듬어진 질의 문구만 출력해.\n\n"
        f"질의: {base_query}"
    )
    return llm.invoke(prompt).content.strip()


class BaseAgent(ABC):
    name: str
    uses_rag: bool = False

    def __call__(self, state: AgentState) -> dict[str, Any]:
        return self.run(state)

    @abstractmethod
    def run(self, state: AgentState) -> dict[str, Any]:
        """state 일부를 읽고, State에 병합할 갱신분만 dict로 반환함(LangGraph 관례)."""
        raise NotImplementedError

    def rewrite_query(self, base_query: str, tech: str) -> str:
        """config.QUERY_REWRITING이 꺼져 있으면 코드가 고정한 질의를 그대로 씀.

        채택 임베딩(Qwen3-Embedding-0.6B)은 한국어 질의를 영어 논문 청크에 직접
        매칭해 리라이팅 이득이 없고, LLM 출력 형식(불리언 검색식 등) 변동이 순위를
        해치는 경우가 확인돼 기본값을 끔(3차 비교실험, 2026-09-22).
        """
        if not config.QUERY_REWRITING:
            return base_query
        return rewrite_query(base_query, tech)

    def next_evidence_id(self, state: AgentState) -> int:
        """evidence 리스트에 새 항목을 append할 때 쓸 다음 번호를 계산함.

        12장대로 관점 노드 4개가 병렬 실행되면 네 노드가 같은 state 스냅샷을 읽어
        전역 max+1이 동일하게 나오고, operator.add reducer로 합쳐질 때 같은 번호의
        Evidence가 중복됨. 이를 막기 위해 노드마다 고정 번호 대역(EVIDENCE_ID_RANGES)을
        두고 그 대역 안에서만 max+1을 계산함. 대역이 없는 노드는 전역 max+1로 동작함.
        """
        existing = state.get("evidence", [])
        band = EVIDENCE_ID_RANGES.get(self.name)
        if band is None:
            return max((e.id for e in existing), default=0) + 1
        start, end = band
        in_band = [e.id for e in existing if start <= e.id < end]
        next_id = max(in_band, default=start - 1) + 1
        if next_id >= end:
            raise ValueError(
                f"{self.name}의 근거 번호 대역 {start}~{end - 1}이 소진됨. "
                "EVIDENCE_ID_RANGES의 EVIDENCE_ID_BAND_SIZE를 늘릴 것."
            )
        return next_id
