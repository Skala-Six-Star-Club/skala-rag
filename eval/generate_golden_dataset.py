"""LLM 기반 합성 Golden Dataset 생성 파이프라인 (8.4절 Retrieval 질의셋).

Doc Pool PDF 6편을 읽어 한국어 검색 질의 약 30개(문서당 5개)를 자동 생성하고,
정답 판정에 쓸 쪽 번호(expected_page)와 키워드(expected_keywords)를 함께 만들어
eval/golden/golden_dataset.json에 저장함.

이렇게 한 번 생성한 뒤에는 git에 커밋해 팀원 전원이 동일한 골든셋으로 3.1~3.3
비교실험(임베딩/청킹/Query Rewriting)을 수행하게 함 — 매번 재생성하면 팀원마다
다른 질의로 비교하게 되어 실험이 무의미해짐.

실행: python -m eval.generate_golden_dataset
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from src.common import config
from src.common.doc_pool import DOC_POOL_SPECS
from src.common.models import get_generation_llm
from src.common.tools import load_pdf_pages

QUERIES_PER_TECH = 5


class GeneratedQuery(BaseModel):
    query_ko: str
    perspective: Literal["general", "trl", "domain"]
    expected_page: int
    expected_keywords: list[str]


class GeneratedQueries(BaseModel):
    queries: list[GeneratedQuery]


def _sample_pages_text(pages: list[tuple[int, str]], n: int = 6) -> str:
    step = max(len(pages) // n, 1)
    sampled = pages[::step][:n]
    return "\n\n".join(f"[p.{p}]\n{text[:800]}" for p, text in sampled)


def generate_for_tech(spec: dict[str, str]) -> list[GeneratedQuery]:
    pages = load_pdf_pages(config.DOC_POOL_DIR / spec["file"])
    excerpt = _sample_pages_text(pages)
    prompt = (
        f"다음은 논문 '{spec['tech']}'의 발췌임. 이 발췌만 근거로 삼아, 논문 내용을 "
        f"확인하는 한국어 검색 질의 {QUERIES_PER_TECH}개를 만들어줘. "
        "각 질의는 general(기술 개요/한계/비교), trl(성숙도 근거: 실험·구현·발표 현황), "
        "domain(실험 환경·요구 하드웨어) 중 하나의 perspective를 가져야 하고, "
        "정답이 있는 쪽 번호(expected_page, 위 [p.N] 표기 중 하나)와 정답 판정에 쓸 "
        "영어 키워드 2~3개(expected_keywords)를 함께 달아줘.\n\n" + excerpt
    )
    llm = get_generation_llm().with_structured_output(GeneratedQueries)
    result: GeneratedQueries = llm.invoke(prompt)  # type: ignore[assignment]
    return result.queries


def main() -> None:
    all_queries: list[dict] = []
    qid = 1
    for spec in DOC_POOL_SPECS:
        for gq in generate_for_tech(spec):
            all_queries.append(
                {
                    "id": qid,
                    "query_ko": gq.query_ko,
                    "tech": spec["tech"],
                    "camp": spec["camp"],
                    "role": spec["role"],
                    "perspective": gq.perspective,
                    "expected_page": gq.expected_page,
                    "expected_keywords": gq.expected_keywords,
                }
            )
            qid += 1

    config.GOLDEN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.GOLDEN_DATASET_PATH.write_text(
        json.dumps({"queries": all_queries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(all_queries)}개 질의 생성 완료 -> {config.GOLDEN_DATASET_PATH}")


if __name__ == "__main__":
    main()
