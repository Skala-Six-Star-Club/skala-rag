"""기술 조사 에이전트 (7.2절). 논문에서 기술 개요, 적용 범위, 한계, 같은 진영
다른 방식과의 차이를 추출함.

입력: state["techs"]
출력: {"tech_profiles": ..., "raw_evidence": [...], "raw_references": [...]}
(근거는 provisional key로 발급되고 evidence_finalize가 최종 번호를 부여함)

검색 설정(3차 비교실험 채택안, 2026-09-22): 절 인식 청킹 + Qwen3-Embedding-0.6B
+ Query Rewriting 끔(config.QUERY_REWRITING). 질의는 코드가 고정하고, role=target
필터로 자기 논문에서 개요를, camp 필터로 같은 진영 비교 논문에서 차이를 가져옴.
단일 패스, 재검색 없음(7.2절 "재검색 없음").
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.doc_pool import DOC_POOL_SPECS
from src.common.models import get_generation_llm
from src.common.state import AgentState, Evidence, Reference, TechProfile
from src.common.tools import (
    format_paper_source,
    get_shared_index,
    paper_search,
    strip_citation_tokens,
)


class _ExtractedProfile(BaseModel):
    overview: str = Field(description="기술 개요와 핵심 접근 방식, 2~4문장, 각 문장 끝에 [근거#N]")
    scope: str = Field(description="적용 범위(어떤 모델/워크로드/하드웨어에 적용되는가), [근거#N] 포함")
    limitations: str = Field(description="논문이 스스로 밝힌 한계와 전제 조건, [근거#N] 포함")
    differentiation: str = Field(description="같은 진영 다른 방식과의 차이, 비교 논문 근거 [근거#N] 포함")
    evidence_ids: list[int] = Field(description="위 네 항목에서 실제로 인용한 [근거#N]의 N 전체")


_PROMPT = """\
아래는 '{tech}' 기술({camp} 진영)에 대한 논문 검색 결과임. 각 발췌 앞의 [근거#N]이 근거 번호임.
[개요 발췌]는 {tech} 자기 논문에서, [비교 발췌]는 같은 {camp} 진영의 다른 방식 논문에서 가져옴.

다음 네 항목을 한국어로 채워줘.
- overview: 기술 개요와 핵심 접근 방식
- scope: 적용 범위
- limitations: 논문이 밝힌 한계와 전제
- differentiation: 같은 진영 다른 방식(비교 발췌의 기술)과의 차이

규칙:
1. 모든 문장은 실제로 그 내용이 적힌 발췌의 번호를 [근거#N] 형식으로 문장 끝에 붙임
2. 발췌에 없는 내용은 쓰지 않음. 확인되지 않으면 "발췌에서 확인되지 않음"이라고 적음
3. 다른 기술과의 우열 판정 표현(더 우수함, 뒤처짐 등)은 쓰지 않고 차이만 서술함
4. evidence_ids에는 본문에서 인용한 번호만 넣음

[개요 발췌]
{overview_passages}

[비교 발췌]
{diff_passages}
"""


_INTRO_SECTIONS = ("abstract", "introduction", "1 introduction", "1. introduction")


def _search_overview(index: FAISS, query: str, tech: str, k: int):
    """기술 개요용 검색. 자기 논문(role=target, tech) 안에서 초록·서론 절 청크를 앞세움.

    개요 질의는 표·수치가 밀집한 실험 절 청크와도 유사도가 높게 나와 Top-5가 표로만
    채워지는 경우가 있어(3차 스모크 확인), 후보 3k개 중 section이 Abstract/Introduction
    이거나 1~2쪽인 청크를 먼저 채우고 나머지를 유사도 순으로 채움. 정식 reranker는 아님.
    """
    candidates = index.similarity_search(
        query, k=k * 3, filter={"role": "target", "tech": tech}, fetch_k=index.index.ntotal
    )

    def _is_intro(doc) -> bool:
        section = str(doc.metadata.get("section", "")).strip().lower()
        return section.startswith(_INTRO_SECTIONS) or int(doc.metadata.get("page", 99)) <= 2

    intro = [d for d in candidates if _is_intro(d)]
    rest = [d for d in candidates if not _is_intro(d)]
    return (intro + rest)[:k]


def _reference_for(tech_name: str) -> Reference | None:
    spec = next((s for s in DOC_POOL_SPECS if s["tech"] == tech_name), None)
    if spec is None:
        return None
    return Reference(
        id=0,  # report 단계에서 인용 순서대로 재번호 가능. 여기서는 arXiv 기준 식별자만 채움
        type="paper",
        author_or_org="arXiv",
        year="20" + spec["arxiv"][:2],
        title=spec["file"].removesuffix(".pdf"),
        venue="arXiv",
        url=f"https://arxiv.org/abs/{spec['arxiv']}",
    )


class TechResearchAgent(BaseAgent):
    name = "tech_research"
    uses_rag = True

    def __init__(self, index: FAISS | None = None):
        self.index = index or get_shared_index()

    def run(self, state: AgentState) -> dict[str, Any]:
        techs = state["techs"]
        tech_profiles: dict[str, TechProfile] = {}
        new_evidence: list[Evidence] = []
        new_references: list[Reference] = []
        ordinal = 0
        llm = get_generation_llm().with_structured_output(_ExtractedProfile)

        for tech in techs:
            # role=target + tech: 기술 개요는 자기 논문에서만 근거를 가져옴 (7.2 RAG 및 재검색 전략)
            overview_query = self.rewrite_query(
                f"{tech.name} 논문의 문제 정의, 핵심 아이디어, 접근 방식, 주요 기여", tech.name
            )
            overview_docs = _search_overview(
                self.index, overview_query, tech=tech.name, k=config.DEFAULT_TOP_K
            )
            limit_query = self.rewrite_query(
                f"{tech.name} 논문이 밝힌 한계, 전제 조건, 적용 범위", tech.name
            )
            limit_docs = [
                d for d in paper_search(self.index, limit_query, k=config.DEFAULT_TOP_K, role="target", tech=tech.name)
                if d.page_content not in {o.page_content for o in overview_docs}
            ][:3]
            # camp 전체(role=comparison): 같은 진영 다른 방식과의 차이는 비교 논문에서 가져옴
            diff_query = self.rewrite_query(
                f"{tech.name}과 같은 {tech.camp} 진영의 다른 KV cache 최적화 방식의 접근과 차이", tech.name
            )
            diff_docs = paper_search(
                self.index, diff_query, k=config.DEFAULT_TOP_K, role="comparison", camp=tech.camp
            )

            key_by_num: dict[int, str] = {}

            def _register(docs) -> list[str]:
                nonlocal ordinal
                passages = []
                for doc in docs:
                    ev = self.new_evidence(
                        state, tech.name, ordinal,
                        perspective="tech_research", source_type="논문",
                        source=format_paper_source(doc.metadata.get("tech", tech.name), doc),
                        quote=doc.page_content[:200],
                    )
                    new_evidence.append(ev)
                    num = len(key_by_num) + 1  # 프롬프트용 로컬 번호(기술마다 1부터)
                    key_by_num[num] = ev.key
                    passages.append(f"[근거#{num}] ({doc.metadata.get('tech')}, p.{doc.metadata.get('page')}) {doc.page_content}")
                    ordinal += 1
                return passages

            overview_passages = _register(overview_docs + limit_docs)
            diff_passages = _register(diff_docs)

            extracted: _ExtractedProfile = llm.invoke(  # type: ignore[assignment]
                _PROMPT.format(
                    tech=tech.name,
                    camp=tech.camp,
                    overview_passages="\n\n".join(overview_passages) or "(없음)",
                    diff_passages="\n\n".join(diff_passages) or "(없음)",
                )
            )
            cited_keys = [key_by_num[n] for n in dict.fromkeys(extracted.evidence_ids) if n in key_by_num]
            tech_profiles[tech.name] = TechProfile(
                tech=tech.name,
                # 본문 안의 로컬 번호 토큰은 제거함. 최종 인용 표기는 report(7.10)가
                # evidence_finalize 이후의 evidence_ids로 다시 붙임.
                overview=strip_citation_tokens(extracted.overview),
                scope=strip_citation_tokens(extracted.scope),
                limitations=strip_citation_tokens(extracted.limitations),
                differentiation=strip_citation_tokens(extracted.differentiation),
                evidence_keys=cited_keys,
            )
            ref = _reference_for(tech.name)
            if ref is not None:
                new_references.append(ref)

        return {
            "tech_profiles": tech_profiles,
            "raw_evidence": new_evidence,
            "raw_references": new_references,
            # 독립 실행 스크립트의 기존 계약과 호환. 통합 Graph는
            # _normalize_parallel_update가 raw 영역으로만 병합함.
            "evidence": new_evidence,
            "references": new_references,
        }
