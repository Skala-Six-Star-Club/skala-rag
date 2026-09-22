"""10개 에이전트 test_runner.py가 공유하는 평가 유틸리티.

- RAG 3종(tech_research/trl_eval/domain_eval): Hit Rate@K / MRR (8.1절)
- LLM 생성 에이전트(market_eval/stakeholder_eval/synthesize/report): 8.2절
  LLM-as-a-Judge 루브릭(정확성/완전성/중립성/근거연결성, 1~5)
- 규칙 기반 에이전트(select_tech/evidence_check/judge): 입력→기대 출력 단위 테스트

리포트의 해석·결론 문장은 에이전트별 성격이 달라 각 test_runner.py가 직접 작성
하고, 여기에는 지표 계산과 그래프 생성 같은 중복 로직만 둠.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from matplotlib import font_manager, rcParams
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field

# 한글 라벨이 깨지지 않도록, 설치돼 있는 한글 폰트를 찾아서 씀(없으면 기본값 유지).
for _font_name in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
    if _font_name in {f.name for f in font_manager.fontManager.ttflist}:
        rcParams["font.family"] = _font_name
        break
rcParams["axes.unicode_minus"] = False

# 골든 질의를 어느 test_runner.py가 쓸지 가르는 태그. state.Evidence.perspective
# (tech_research/trl/market/stakeholder/domain)와는 별개 개념 — market/stakeholder는
# RAG를 쓰지 않아 골든 검색 질의가 없고, tech_research는 "general"로 부름.
Perspective = Literal["general", "trl", "domain"]


class GoldenQuery(BaseModel):
    id: int
    query_ko: str
    tech: str
    camp: str
    role: str
    perspective: Perspective
    expected_page: int
    expected_keywords: list[str]


def load_golden_dataset(path: Path, perspective: Perspective) -> list[GoldenQuery]:
    """golden_dataset.json에서 해당 에이전트 관점(perspective)의 질의만 골라 반환함.

    Query Rewriting 비교(3.3절)에 씀 — 리라이팅 효과는 에이전트가 실제로 던지는
    질의 유형에 따라 달라질 수 있어 관점별로 나눠 봄.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenQuery(**q) for q in data["queries"] if q["perspective"] == perspective
    ]


def load_golden_dataset_all(path: Path) -> list[GoldenQuery]:
    """골든 질의 전체(30개)를 관점 구분 없이 반환함.

    청킹·임베딩(3.1~3.2절)은 3개 RAG 에이전트가 공유하는 단일 색인 결정이라
    (5장 "전 에이전트가 하나의 색인을 공유"), schedule.md 3.1~3.2절 원문대로
    전체 골든셋으로 채점함. 세 에이전트 리포트의 청킹/임베딩 수치가 동일하게
    나오는 게 정상임 — 전역 설정을 검증하는 것이지 에이전트별로 다른 값을
    고르는 게 아니기 때문.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenQuery(**q) for q in data["queries"]]


def is_hit(doc: Document, golden: GoldenQuery) -> bool:
    """정답 판정: 같은 기술이면서 쪽 번호가 일치하거나 기대 키워드가 본문에 포함되면 정답.

    ponytail: 청크 ID는 청킹 버전(v1/v2)마다 달라져 그대로 비교할 수 없어 쓰지 않음.
    쪽 번호 + 키워드 근사 매칭으로 대체함 — 골든셋 정답 판정이 더 엄격해야 하면
    사람이 라벨링한 chunk_id 매칭으로 교체.
    """
    if doc.metadata.get("tech") != golden.tech:
        return False
    if doc.metadata.get("page") == golden.expected_page:
        return True
    content = doc.page_content.lower()
    return any(kw.lower() in content for kw in golden.expected_keywords)


def hit_rate_at_k(
    retrieved: list[list[Document]], goldens: list[GoldenQuery], k: int
) -> float:
    if not goldens:
        return 0.0
    hits = sum(
        1
        for docs, g in zip(retrieved, goldens)
        if any(is_hit(d, g) for d in docs[:k])
    )
    return hits / len(goldens)


def mrr(retrieved: list[list[Document]], goldens: list[GoldenQuery], k: int) -> float:
    if not goldens:
        return 0.0
    total = 0.0
    for docs, g in zip(retrieved, goldens):
        for rank, d in enumerate(docs[:k], start=1):
            if is_hit(d, g):
                total += 1.0 / rank
                break
    return total / len(goldens)


def plot_bar_comparison(
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    """버전/후보별 지표 비교 막대 그래프를 PNG로 저장함."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(labels))
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    for i, (name, values) in enumerate(series.items()):
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, values, width=width, label=name)
    ax.set_xticks([xi + width * (n_series - 1) / 2 for xi in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8.2절: LLM-as-a-Judge 루브릭 (market_eval/stakeholder_eval/synthesize/report)
# ---------------------------------------------------------------------------


class RubricScore(BaseModel):
    accuracy: int = Field(ge=1, le=5, description="정확성: 인용한 근거와 실제로 부합하는가")
    completeness: int = Field(ge=1, le=5, description="완전성: 필수 항목이 빠짐없이 포함되는가")
    neutrality: int = Field(ge=1, le=5, description="중립성: 우열 판정 없이 대칭적으로 서술하는가")
    evidence_linkage: int = Field(ge=1, le=5, description="근거 연결성: 문장마다 근거 참조가 있는가")
    notes: str = ""


def score_with_rubric(judge_llm: Any, target_text: str, context: str) -> RubricScore:
    """8.2절 루브릭으로 산출물을 1~5점 채점함. 채점 모델은 생성 모델과 다른 계열을 씀."""
    prompt = (
        "다음 산출물을 아래 기준으로 1~5점 채점해줘.\n"
        "- 정확성: 인용한 근거와 실제로 부합하는가\n"
        "- 완전성: 필수 항목이 빠짐없이 포함되는가\n"
        "- 중립성: 우열 판정·편향된 어조 없이 두 기술을 대칭적으로 서술하는가\n"
        "- 근거 연결성: 문장마다 근거 참조가 있고 그 근거가 실제로 주장을 지지하는가\n\n"
        f"[컨텍스트]\n{context}\n\n[산출물]\n{target_text}"
    )
    scorer = judge_llm.with_structured_output(RubricScore)
    return scorer.invoke(prompt)


def plot_rubric_scores(scores: RubricScore, title: str, save_path: Path) -> None:
    labels = ["정확성", "완전성", "중립성", "근거연결성"]
    values = [scores.accuracy, scores.completeness, scores.neutrality, scores.evidence_linkage]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylim(0, 5)
    ax.set_ylabel("score (1-5)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# select_tech / evidence_check / judge: 입력→기대 출력 단위 테스트
# ---------------------------------------------------------------------------


@dataclass
class UnitCheck:
    name: str
    passed: bool
    detail: str = ""


def plot_unit_checks(checks: list[UnitCheck], title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [c.name for c in checks]
    values = [1 if c.passed else 0 for c in checks]
    colors = ["#55A868" if v else "#C44E52" for v in values]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["FAIL", "PASS"])
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
