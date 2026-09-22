"""공통 Tool: 논문 검색(paper_search), 웹 검색(web_search), 근거 요약(summarize_sources),
그리고 PDF 로딩/절 인식/청킹 유틸리티 (4장, 5장, 7.2절 구현).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tavily import TavilyClient

from src.common import config
from src.common.models import get_generation_llm

# ---------------------------------------------------------------------------
# PDF 로딩 + 절 구조 인식 + 청킹 (5장 "단계 설계", 7.2절 "Document Loader 및 전처리")
# ---------------------------------------------------------------------------

# 숫자+마침표로 시작하는 제목, 관용적 표제어(Abstract, Introduction, Related Work 등)
_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+[A-Z][\w \-]{2,60}"
    r"|Abstract|Introduction|Related Work|Background|Method(?:ology)?"
    r"|Experiments?|Evaluation|Results?|Discussion|Conclusion|Limitations)\s*$",
    re.MULTILINE,
)
_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(References|Bibliography|참고문헌)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class SectionSegment:
    section: str
    page: int
    text: str


def load_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """PyMuPDF로 쪽 단위 텍스트를 추출함. 반환: [(page_no, text), ...] (1-base)."""
    doc = fitz.open(pdf_path)
    try:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()


def split_into_sections(pages: list[tuple[int, str]]) -> list[SectionSegment]:
    """정규식으로 절 경계를 표시하고 참고문헌 절 이후는 제거함(5장, 7.2절).

    쪽을 넘나드는 절은 이전 절 제목을 이어받음. 절 인식은 문서 6편 규모라
    설계서 5장 지시대로 색인 전 수동 검수를 전제로 함(자동 결과는 근사치).
    """
    segments: list[SectionSegment] = []
    current_section = "Abstract"
    stopped = False

    for page_no, text in pages:
        if stopped:
            break
        if _REFERENCES_HEADING_RE.search(text):
            text = _REFERENCES_HEADING_RE.split(text)[0]
            stopped = True

        pos = 0
        for match in _SECTION_HEADING_RE.finditer(text):
            if match.start() > pos:
                segments.append(
                    SectionSegment(current_section, page_no, text[pos : match.start()])
                )
            current_section = match.group().strip()
            pos = match.end()
        segments.append(SectionSegment(current_section, page_no, text[pos:]))

    return [s for s in segments if s.text.strip()]


def chunk_sections(
    segments: list[SectionSegment],
    tech: str,
    camp: str,
    role: str,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """절 경계 안에서만 분할함(5장 "2단계 방식": 절 우선 분리 후 하위 분할)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs: list[Document] = []
    for seg in segments:
        for chunk in splitter.split_text(seg.text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "tech": tech,
                        "camp": camp,
                        "role": role,
                        "section": seg.section,
                        "page": seg.page,
                    },
                )
            )
    return docs


def build_doc_pool_index_naive(
    doc_pool_dir: Path,
    tech_specs: list[dict[str, str]],
    embedding_model,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> FAISS:
    """비교실험용 baseline(3.2절): 절 구조 인식 없이 쪽 텍스트를 바로 슬라이싱함."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs: list[Document] = []
    for spec in tech_specs:
        pages = load_pdf_pages(doc_pool_dir / spec["file"])
        for page_no, text in pages:
            for chunk in splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "tech": spec["tech"],
                            "camp": spec["camp"],
                            "role": spec["role"],
                            "section": "unknown",
                            "page": page_no,
                        },
                    )
                )
    return FAISS.from_documents(docs, embedding_model)


def build_doc_pool_index(
    doc_pool_dir: Path,
    tech_specs: list[dict[str, str]],
    embedding_model,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> FAISS:
    """Doc Pool 6편을 로딩→절 인식→청킹→임베딩까지 처리해 FAISS 인덱스로 반환함(5장 파이프라인).

    tech_specs: [{"file": "TurboQuant.pdf", "tech": "TurboQuant", "camp": "SW", "role": "target"}, ...]
    """
    all_docs: list[Document] = []
    for spec in tech_specs:
        pages = load_pdf_pages(doc_pool_dir / spec["file"])
        segments = split_into_sections(pages)
        all_docs.extend(
            chunk_sections(
                segments,
                tech=spec["tech"],
                camp=spec["camp"],
                role=spec["role"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return FAISS.from_documents(all_docs, embedding_model)


# ---------------------------------------------------------------------------
# 공유 색인 (5장 "전 에이전트가 하나의 색인을 공유"): 절 인식 청킹 + 채택 임베딩
# ---------------------------------------------------------------------------


def _embedding_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def get_shared_index(embedding_model=None, index_dir: Path | None = None) -> FAISS:
    """Doc Pool 6편의 공유 FAISS 색인을 로드하거나(없으면) 구축해 저장함.

    - 청킹: 절 인식(v1, build_doc_pool_index). 3.2절 비교에서 naive가 수치는 높았으나
      정답 판정이 쪽 번호 기반이라 착시 가능성이 있어 chunk_id 라벨링 전까지 유지함.
    - 임베딩: config.EMBEDDING_MODEL(기본 Qwen3-Embedding-0.6B).
    - 경로: DOC_POOL_INDEX_DIR/<임베딩 이름>/ (임베딩을 바꾸면 자동으로 다른 폴더에 새로 구축).
    """
    from src.common.doc_pool import DOC_POOL_SPECS
    from src.common.models import get_embedding_model

    embedding_model = embedding_model or get_embedding_model()
    index_dir = index_dir or (config.DOC_POOL_INDEX_DIR / _embedding_dir_name(config.EMBEDDING_MODEL))
    if (index_dir / "index.faiss").exists():
        return FAISS.load_local(str(index_dir), embedding_model, allow_dangerous_deserialization=True)
    missing = [s["file"] for s in DOC_POOL_SPECS if not (config.DOC_POOL_DIR / s["file"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"Doc Pool PDF 누락: {missing} — README의 표대로 {config.DOC_POOL_DIR}에 받아 둘 것"
        )
    index = build_doc_pool_index(config.DOC_POOL_DIR, DOC_POOL_SPECS, embedding_model)
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_dir))
    return index


def format_paper_source(tech: str, doc: Document) -> str:
    """Evidence.source 표기: '<기술> p.<쪽> <절>' (13장 REFERENCE, 7.10 인용 안전장치용)."""
    return f"{tech} p.{doc.metadata.get('page')} {doc.metadata.get('section', '')}".strip()


# ---------------------------------------------------------------------------
# paper_search (4장 공통 도구, 5장 검색 단계: 메타데이터 필터 -> Dense Top-K)
# ---------------------------------------------------------------------------


def paper_search(
    index: FAISS,
    query: str,
    k: int = config.DEFAULT_TOP_K,
    role: str | None = None,
    camp: str | None = None,
    tech: str | None = None,
) -> list[Document]:
    """FAISS 색인에서 메타데이터를 먼저 필터링한 뒤 Dense Top-K를 반환함.

    role=target은 대상 기술 2편(TurboQuant, ITME)을 모두 통과시키므로, 한 기술의
    자기 논문에서만 근거를 가져와야 할 때는 tech=기술명을 함께 넘길 것(7.2절
    "기술 개요 추출은 대상 기술 자신의 논문에서만").
    """
    filter_dict: dict[str, Any] = {}
    if role is not None:
        filter_dict["role"] = role
    if camp is not None:
        filter_dict["camp"] = camp
    if tech is not None:
        filter_dict["tech"] = tech
    # LangChain FAISS는 filter를 "fetch_k개(기본 20)를 먼저 뽑고 거르는" 방식으로 적용함.
    # role=target+tech처럼 전체의 1/6만 통과하는 필터면 20개 안에 후보가 부족해 결과가
    # k개 미만이거나 비게 되므로, 전체 벡터를 후보로 잡아 사실상 사전 필터가 되게 함(5장).
    return index.similarity_search(
        query, k=k, filter=filter_dict or None, fetch_k=index.index.ntotal
    )


# ---------------------------------------------------------------------------
# web_search (Tavily, 4장)
# ---------------------------------------------------------------------------


@dataclass
class WebResult:
    title: str
    url: str
    published_date: str | None
    content: str


def _web_search_ddg(query: str, max_results: int) -> list[WebResult]:
    """TAVILY_API_KEY가 없을 때 쓰는 폴백(DuckDuckGo, API 키 불필요).

    질의 템플릿·호출 횟수는 호출자가 그대로 고정하므로 10장 대칭 질의 원칙은
    유지됨. 결과 품질(발췌 길이, 날짜)은 Tavily보다 떨어질 수 있음.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        rows = list(ddgs.text(query, max_results=max_results))
    return [
        WebResult(
            title=r.get("title", ""),
            url=r.get("href", r.get("url", "")),
            published_date=None,
            content=r.get("body", ""),
        )
        for r in rows
    ]


def web_search(query: str, max_results: int = 5) -> list[WebResult]:
    """Tavily로 기사, 발표 자료, 시장 리포트를 검색함. 키가 없으면 DuckDuckGo 폴백."""
    if not config.TAVILY_API_KEY:
        return _web_search_ddg(query, max_results)
    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query=query, max_results=max_results, search_depth="basic")
    return [
        WebResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            published_date=r.get("published_date"),
            content=r.get("content", ""),
        )
        for r in response.get("results", [])
    ]


# ---------------------------------------------------------------------------
# summarize_sources (4장): 검색 결과를 주장 단위로 요약하고 지지/반대 근거로 나눔
# ---------------------------------------------------------------------------


class SummarizedClaim(BaseModel):
    statement: str
    stance: str = Field(description='"지지" 또는 "반대"')
    source: str


class SummarizedClaims(BaseModel):
    claims: list[SummarizedClaim] = Field(default_factory=list)


def summarize_sources(
    results: list[Document] | list[WebResult],
    perspective: str,
    tech: str,
) -> list[SummarizedClaim]:
    """검색 결과를 주장 단위로 요약하고 지지/반대 근거로 나눔(4장 공통 도구 표)."""
    passages = []
    for r in results:
        if isinstance(r, Document):
            passages.append(f"[{r.metadata.get('source', r.metadata)}] {r.page_content}")
        else:
            passages.append(f"[{r.url}] {r.content}")

    prompt = (
        f"다음은 '{tech}' 기술에 대한 '{perspective}' 관점 검색 결과임. "
        "각 근거를 주장 단위로 나누고 지지/반대를 표시해줘.\n\n" + "\n\n".join(passages)
    )
    llm = get_generation_llm().with_structured_output(SummarizedClaims)
    result: SummarizedClaims = llm.invoke(prompt)  # type: ignore[assignment]
    return result.claims


# ---------------------------------------------------------------------------
# 관점 결과 구조화 추출 (7.5·7.6 공통): 근거 발췌 -> TechViewResult
# ---------------------------------------------------------------------------


class _ExtractedClaim(BaseModel):
    statement: str = Field(description="근거에 기반한 한 문장 주장(한국어). 번호 표기는 넣지 않음")
    evidence_nums: list[int] = Field(description="이 주장을 지지하는 [근거#N]의 N 목록, 1개 이상")


class _ExtractedView(BaseModel):
    confirmed_facts: list[_ExtractedClaim] = Field(default_factory=list, description="확인된 사실(지지 근거)")
    counter_facts: list[_ExtractedClaim] = Field(default_factory=list, description="반대 사실·우려·부정적 반응(반대 근거)")
    unconfirmed_items: list[str] = Field(default_factory=list, description="검색 결과로 확인되지 않은 항목")


_CITATION_TOKEN_RE = re.compile(r"\s*\[(?:임시)?근거#[^\]]+\]")


def strip_citation_tokens(text: str) -> str:
    """LLM이 문장 안에 넣은 [근거#N] 토큰을 제거함.

    임시 번호는 evidence_finalize 이후의 최종 번호와 다르므로 본문에 남기면 안 됨.
    최종 인용 표기는 report(7.10절)가 Claim.evidence_ids로 다시 붙임.
    """
    return _CITATION_TOKEN_RE.sub("", text).strip()


def extract_view_result(
    passages: list[str],
    tech: str,
    perspective_label: str,
    required_items: list[str],
    key_by_num: dict[int, str],
    extra_instructions: str = "",
    prior_unconfirmed: list[str] | None = None,
):
    """검색 발췌를 9장 평가 기준의 필수 항목에 맞춰 TechViewResult로 구조화함.

    - passages의 각 발췌는 "[근거#N] ..." 형태이고, key_by_num이 N -> provisional
      Evidence key(src/common/evidence.py)를 이어 줌. 결과 Claim은 evidence_keys를 채우고
      evidence_finalize가 최종 정수 id로 remap함(병렬 노드 간 번호 충돌 회피).
    - 모든 문장은 근거를 1개 이상 참조해야 하고(10장 중립성, 8.2 근거 연결성), 목록에
      없는 번호를 참조한 주장은 버림(7.10 인용 안전장치와 동일 원칙).
    - required_items에 해당하는 내용이 발췌에 없으면 unconfirmed_items에 항목명을 남김.
    """
    from src.common.state import Claim, TechViewResult

    if not passages:
        return TechViewResult(unconfirmed_items=list(required_items))

    prior_block = (
        "\n\n[이전 검색에서 확인되지 않은 항목 — 이번 발췌로 확인되면 confirmed/counter에 넣고, "
        "여전히 없으면 unconfirmed_items에 유지]\n" + "\n".join(f"- {u}" for u in prior_unconfirmed)
        if prior_unconfirmed
        else ""
    )
    prompt = (
        f"아래는 '{tech}' 기술에 대한 '{perspective_label}' 관점 검색 발췌임(논문 청크와 웹 발췌). "
        "각 발췌 앞의 [근거#N]이 근거 번호임.\n\n"
        "다음 필수 항목마다 발췌에서 확인되는 사실을 한국어 한 문장으로 정리해줘:\n"
        + "\n".join(f"- {item}" for item in required_items)
        + "\n\n규칙:\n"
        "1. confirmed_facts에는 긍정적/중립적 사실, counter_facts에는 우려·부정적 반응·한계를 넣음\n"
        "2. 모든 문장은 실제로 그 내용이 적힌 발췌의 근거 번호를 evidence_nums에 1개 이상 넣음. "
        "statement 문장 안에는 [근거#N] 같은 번호 표기를 쓰지 않음\n"
        "3. 발췌에 없는 내용을 지어내지 말고, 확인되지 않는 필수 항목은 unconfirmed_items에 항목명을 그대로 적음\n"
        "4. 다른 기술과의 우열 판정 표현(더 우수함, 뒤처짐 등)은 쓰지 않음\n"
        "5. 각 문장이 위 필수 항목 중 정확히 어떤 항목에 대한 서술인지 나중에 다른 사람이 "
        "한눈에 알아볼 수 있도록, 그 항목의 핵심 용어를 문장 안에 자연스럽게 포함시켜줘 "
        "(예: '투자 업계의 평가'에 대한 문장이면 '투자자'·'애널리스트' 같은 표현을 넣음). "
        "발췌 내용이 필수 항목과 간접적으로만 관련돼도, 그 항목과 관련지어 서술할 수 있으면 "
        "unconfirmed_items로 넘기지 말고 confirmed_facts/counter_facts에 포함시켜줘\n"
        + (extra_instructions + "\n" if extra_instructions else "")
        + prior_block
        + "\n\n[발췌]\n"
        + "\n\n".join(passages)
    )
    llm = get_generation_llm().with_structured_output(_ExtractedView)
    extracted: _ExtractedView = llm.invoke(prompt)  # type: ignore[assignment]

    def _keep(claims: list[_ExtractedClaim]) -> list[Claim]:
        kept = []
        for c in claims:
            keys = [key_by_num[n] for n in dict.fromkeys(c.evidence_nums) if n in key_by_num]
            if keys:
                kept.append(Claim(statement=strip_citation_tokens(c.statement), evidence_keys=keys))
        return kept

    return TechViewResult(
        confirmed_facts=_keep(extracted.confirmed_facts),
        counter_facts=_keep(extracted.counter_facts),
        unconfirmed_items=extracted.unconfirmed_items,
    )
