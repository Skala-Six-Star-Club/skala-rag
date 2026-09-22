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
from src.common.evidence import make_provisional_key
from src.common.state import AgentState


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

    def scoped_techs(self, state: AgentState) -> list:
        """이 실행이 처리할 기술 목록. graph.py가 Send로 tech_scope를 넣어 주면 그 기술
        하나만, 없으면 techs 전체(독립 실행 스크립트·레거시 호환)."""
        techs = list(state.get("techs", []) or [])
        scope = state.get("tech_scope")
        if scope:
            techs = [t for t in techs if t.name == scope]
        return techs

    def new_evidence(
        self,
        state: AgentState,
        tech: str,
        ordinal: int,
        *,
        perspective: str,
        source_type: str,
        source: str,
        quote: str,
        stance: str = "지지",
        reference_url: str | None = None,
    ):
        """provisional key가 붙은 Evidence를 만듦(id는 evidence_finalize가 부여).

        reference_url은 이 근거가 속한 참고문헌의 URL. 같은 URL의 Reference를
        raw_references에 함께 넣어야 report가 REFERENCE 절에 실을 수 있음.
        """
        from src.common.state import Evidence

        return Evidence(
            key=self.provisional_evidence_key(state, tech, ordinal),
            tech=tech,
            perspective=perspective,
            stance=stance,
            source_type=source_type,
            source=source,
            quote=quote,
            reference_url=reference_url,
        )

    def evidence_attempt(self, state: AgentState) -> int:
        """현재 실행이 초기 수집인지 재시도인지 반환한다."""

        return 1 if self.name in (state.get("retry_targets", []) or []) else 0

    def provisional_evidence_key(
        self,
        state: AgentState,
        tech: str,
        ordinal: int,
    ) -> str:
        """병렬 수집 단계에서 사용할 임시 Evidence key를 만든다."""

        perspective = self.name.removesuffix("_eval")
        return make_provisional_key(
            perspective,
            tech,
            attempt=self.evidence_attempt(state),
            ordinal=ordinal,
        )
