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
# paper_search (4장 공통 도구, 5장 검색 단계: 메타데이터 필터 -> Dense Top-K)
# ---------------------------------------------------------------------------


def paper_search(
    index: FAISS,
    query: str,
    k: int = config.DEFAULT_TOP_K,
    role: str | None = None,
    camp: str | None = None,
) -> list[Document]:
    """FAISS 색인에서 메타데이터를 먼저 필터링한 뒤 Dense Top-K를 반환함."""
    filter_dict: dict[str, Any] = {}
    if role is not None:
        filter_dict["role"] = role
    if camp is not None:
        filter_dict["camp"] = camp
    return index.similarity_search(query, k=k, filter=filter_dict or None)


# ---------------------------------------------------------------------------
# web_search (Tavily, 4장)
# ---------------------------------------------------------------------------


@dataclass
class WebResult:
    title: str
    url: str
    published_date: str | None
    content: str


def web_search(query: str, max_results: int = 5) -> list[WebResult]:
    """Tavily로 기사, 발표 자료, 시장 리포트를 검색함."""
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
