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

from src.common.state import ViewResult

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


def best_label(labels: list[str], scores: dict[str, list[float]], key: str) -> str:
    """key 지표(예: "MRR (camp 전체)") 기준으로 가장 점수가 높은 라벨을 반환함.

    "임계값을 넘었는가"만으로는 여러 후보 중 실제로 뭐가 제일 나은지, 그게 채택된
    기본값과 같은지 알 수 없어서 결론에 명시적으로 쓰려고 둠. Hit Rate@5는 표본이
    작으면 1.000으로 자주 천장에 붙어 후보를 못 가르므로, 기본 랭킹 기준은 표본이
    가장 큰(camp 전체, 30개) MRR로 둠 — 더 연속적이고 변별력 있는 지표.
    """
    values = scores[key]
    best_idx = max(range(len(values)), key=lambda i: values[i])
    return labels[best_idx]


# 도표 공통 잉크·상태 색
_STATUS_GOOD = "#0ca30c"
_STATUS_BAD = "#d03b3b"
_INK = "#1f2933"
_INK_MUTED = "#6b7280"

# 같은 지표를 여러 조건에서 비교하는 그래프 두 종류.
# - plot_condition_comparison: 전/후처럼 조건이 2개(또는 순서가 있는 소수)일 때 지표별 점·선
#   (slope chart). 조건 간 변화 방향과 폭이 한눈에 읽힘.
# - plot_bar_comparison: 임베딩 후보처럼 조건이 명목형(순서 없음)이면 선이 "순서"를 암시하므로
#   막대를 유지함. 값 라벨을 붙여 좁은 차이도 읽히게 함.
# 색은 고정 순서(파랑, 주황, 청록, 노랑)로 색각 이상 분리 검증을 통과한 조합이고, 청록·노랑은
# 배경 대비가 낮아 값 라벨과 직접 표기를 항상 함께 둠.
_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _style_axes(ax) -> None:
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")
    ax.tick_params(colors=_INK_MUTED)


def _spread(values: list[float], min_gap: float) -> list[float]:
    """같은 x에서 겹치는 라벨 y좌표를 위아래로 밀어 최소 간격을 확보함(순서 유지)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = []
    for i in order:
        y = values[i]
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((i, y))
    out = [0.0] * len(values)
    for i, y in placed:
        out[i] = y
    return out


def plot_condition_comparison(
    conditions: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    save_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """조건(x) x 지표(선) 비교 그래프. series[지표명] = 조건 순서대로의 값.

    전/후 비교(조건 2개)용 slope chart. 지표마다 고정 순서의 색, 마지막 조건 옆 직접 표기,
    점마다 값 라벨. 같은 값이 겹치면 라벨을 위아래로 벌리고, 같은 x·같은 값의 수치 라벨은
    한 번만 적음.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_cond = len(conditions)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.9 * n_cond + 4.6), 4.4))
    xs = list(range(n_cond))
    all_vals = [v for vals in series.values() for v in vals]
    if ylim is None:
        lo, hi = (min(all_vals), max(all_vals)) if all_vals else (0.0, 1.0)
        pad = max((hi - lo) * 0.25, 0.05)
        ylim = (max(0.0, lo - pad), hi + pad)
    names = list(series)
    for i, name in enumerate(names):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.plot(xs, series[name], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=name, zorder=3)
    # 값 라벨: 같은 x에서 같은 값은 한 번만
    for x in xs:
        seen: set[float] = set()
        for name in names:
            v = round(series[name][x], 3)
            if v in seen:
                continue
            seen.add(v)
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=_INK_MUTED)
    # 직접 표기: 끝점 y가 겹치면 벌림
    ends = [series[n][-1] for n in names]
    label_ys = _spread(ends, (ylim[1] - ylim[0]) * 0.07)
    for i, (name, y) in enumerate(zip(names, label_ys)):
        ax.text(xs[-1] + 0.08, y, name, va="center", fontsize=8,
                color=_SERIES_COLORS[i % len(_SERIES_COLORS)])
    ax.set_xticks(xs)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_xlim(-0.35, n_cond - 1 + 2.1)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK_MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=_INK, pad=26)
    _style_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(names), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_bar_comparison(
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    save_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """명목형 조건(x) x 지표(막대 색) 비교. 막대마다 값 라벨, 막대 사이 여백, 고정 순서 색."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_series = max(len(series), 1)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels) * n_series + 2.5), 4.4))
    x = range(len(labels))
    all_vals = [v for vals in series.values() for v in vals]
    integral = all(float(v).is_integer() for v in all_vals)
    fmt = (lambda v: f"{int(v)}") if integral else (lambda v: f"{v:.3f}")
    for i, (name, values) in enumerate(series.items()):
        offsets = [xi + i * width for xi in x]
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.bar(offsets, values, width=width * 0.9, label=name, color=color, zorder=3)
        for xo, v in zip(offsets, values):
            ax.annotate(fmt(v), (xo, v), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color=_INK_MUTED)
    ax.set_xticks([xi + width * (n_series - 1) / 2 for xi in x])
    ax.set_xticklabels(labels, fontsize=9)
    if ylim is None:
        top = max((v for vals in series.values() for v in vals), default=1.0)
        ylim = (0, top * 1.12)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK_MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=_INK, pad=26)
    _style_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=min(n_series, 4), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8.2절: LLM-as-a-Judge 루브릭 (market_eval/stakeholder_eval/synthesize/report)
# ---------------------------------------------------------------------------


def render_view_result_md(view_result: ViewResult) -> str:
    """ViewResult를 사람이 읽는 마크다운으로 직렬화함.

    test_runner.py 리포트에 str(view_result)(파이썬 repr, 한 줄로 뭉개짐)를 그대로
    박아 넣으면 읽기 어려워서 대신 씀. 채점용 target_text에도 이 형태를 쓰면
    루브릭 채점 모델도 더 안정적으로 읽음.
    """
    def _refs(c) -> str:
        # 최종화 전(id 미확정) 단계에서는 evidence_keys를, 최종화 후에는
        # evidence_ids를 씀 — 최종화 여부와 무관하게 실제로 있는 쪽을 보여줌.
        ids = ", ".join(f"#{i}" for i in c.evidence_ids)
        keys = ", ".join(c.evidence_keys)
        return ids or keys or "근거 없음"

    lines: list[str] = []
    for tech, tv in view_result.by_tech.items():
        lines.append(f"### {tech}\n")
        lines.append("**확인된 사실**")
        for c in tv.confirmed_facts:
            lines.append(f"- {c.statement} ({_refs(c)})")
        lines.append("\n**반대/우려 사실**")
        for c in tv.counter_facts:
            lines.append(f"- {c.statement} ({_refs(c)})")
        lines.append("\n**미확인 항목**")
        for item in tv.unconfirmed_items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


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


# 이진 판정(PASS/FAIL, 포함/미포함)은 크기가 아니라 상태이므로 막대그래프 대신
# 체크리스트·상태 행렬로 그림. 상태 색은 항상 마커 모양과 글자 라벨을 함께 둬서
# 색만으로 의미를 전달하지 않음(색각 이상, 흑백 인쇄 대비).


def _status_marker(ax, x: float, y: float, ok: bool) -> None:
    color = _STATUS_GOOD if ok else _STATUS_BAD
    ax.scatter([x], [y], s=140, marker="o" if ok else "X", color=color, zorder=3)


def plot_unit_checks(checks: list[UnitCheck], title: str, save_path: Path) -> None:
    """단위 테스트 결과를 체크리스트 도표로 저장함: 행마다 상태 마커, PASS/FAIL, 이름, 비고(아랫줄)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(checks), 1)
    passed = sum(c.passed for c in checks)
    has_detail = any(c.detail.strip() for c in checks)
    row_h = 0.62 if has_detail else 0.42
    fig, ax = plt.subplots(figsize=(8, row_h * n + 0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.axis("off")
    for i, c in enumerate(checks):
        y_name = i - 0.12 if (has_detail and c.detail.strip()) else i
        _status_marker(ax, 0.03, i, c.passed)
        ax.text(0.07, i, "PASS" if c.passed else "FAIL", va="center", fontsize=10, fontweight="bold",
                color=_STATUS_GOOD if c.passed else _STATUS_BAD)
        ax.text(0.15, y_name, c.name, va="center", fontsize=10, color=_INK)
        detail = c.detail.replace("\n", " ").strip()
        if detail:
            ax.text(0.15, i + 0.24, detail[:90] + ("…" if len(detail) > 90 else ""), va="center",
                    fontsize=7.5, color=_INK_MUTED)
    ax.set_title(f"{title}  —  {passed}/{len(checks)} 통과", loc="left", fontsize=11, color=_INK)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_status_matrix(
    row_labels: list[str],
    col_labels: list[str],
    values: dict[str, dict[str, bool]],
    title: str,
    save_path: Path,
    true_label: str = "포함",
    false_label: str = "미포함",
) -> None:
    """행 x 열의 이진 상태를 셀 마커와 라벨로 그림(예: 필수 항목 x 기술 커버리지).

    values[col][row] -> bool. 행 이름은 왼쪽, 열 머리글에는 충족 수를 함께 적음.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = max(len(row_labels), 1), max(len(col_labels), 1)
    label_w = 2.6  # 행 이름 영역 폭(데이터 좌표)
    fig, ax = plt.subplots(figsize=(1.9 * n_cols + 4.2, 0.55 * n_rows + 1.5))
    ax.set_xlim(-label_w, n_cols * 1.4 - 0.4)
    ax.set_ylim(-1.3, n_rows - 0.5)
    ax.invert_yaxis()
    ax.axis("off")
    xs = [j * 1.4 for j in range(n_cols)]
    for x, col in zip(xs, col_labels):
        met = sum(1 for r in row_labels if values.get(col, {}).get(r, False))
        ax.text(x + 0.25, -0.9, f"{col}\n{met}/{n_rows} {true_label}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=_INK)
    for i, row in enumerate(row_labels):
        ax.text(-label_w + 0.05, i, row, ha="left", va="center", fontsize=9, color=_INK)
        for x, col in zip(xs, col_labels):
            ok = bool(values.get(col, {}).get(row, False))
            _status_marker(ax, x, i, ok)
            ax.text(x + 0.2, i, true_label if ok else false_label, va="center", fontsize=8,
                    color=_STATUS_GOOD if ok else _STATUS_BAD)
        if i < n_rows - 1:
            ax.axhline(i + 0.5, color="#e5e7eb", linewidth=0.8, zorder=1)
    ax.set_title(title, loc="left", fontsize=11, color=_INK)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9.2·9.3절 필수 항목 커버리지 + 규칙 기반 구조 검사 (market_eval/stakeholder_eval)
# 8.2 루브릭의 "완전성"과 "근거 연결성"을 항목 단위로 분해해 어디가 미달인지 보여줌.
# ---------------------------------------------------------------------------


class ItemCoverage(BaseModel):
    item: str
    covered: bool
    reason: str = ""


class CoverageResult(BaseModel):
    items: list[ItemCoverage] = Field(default_factory=list)


def score_required_items(judge_llm: Any, view_text: str, tech: str, required_items: list[str]) -> CoverageResult:
    """9장 평가 기준표의 필수 항목이 기술별 ViewResult에 실제로 담겼는지 LLM 채점자가 항목별로 판정함."""
    prompt = (
        f"다음은 '{tech}' 기술에 대한 관점 평가 결과임. 아래 필수 항목 각각에 대해, "
        "평가 결과의 확인된 사실/반대 사실 중 그 항목과 주제상 관련된 문장이 하나라도 있으면 "
        "covered=true로 판정해줘 — 항목명이 문장에 그대로 쓰여 있을 필요는 없고, 내용상 그 "
        "항목을 뒷받침하거나 설명하면 충분함(예: '투자 업계의 평가' 항목은 '투자자'·'애널리스트'·"
        "'기관투자가'·'주가' 같은 표현이 들어간 문장이면 포함으로 봄). 관련 문장이 정말 하나도 "
        "없을 때만 covered=false로 판정하고, reason에는 어떤 문장을 근거로 판단했는지(또는 왜 "
        "없다고 판단했는지) 간단히 적어줘. 항목명은 그대로 돌려줘.\n\n"
        "[필수 항목]\n" + "\n".join(f"- {i}" for i in required_items)
        + f"\n\n[평가 결과]\n{view_text}"
    )
    scorer = judge_llm.with_structured_output(CoverageResult)
    result: CoverageResult = scorer.invoke(prompt)
    # 항목명이 바뀌어 돌아오면 순서 기준으로 맞춤
    if len(result.items) == len(required_items):
        for it, name in zip(result.items, required_items):
            it.item = name
    return result


@dataclass
class ViewStructureStats:
    """ViewResult 하나(기술 하나)에 대한 규칙 기반 수치."""

    n_confirmed: int
    n_counter: int
    n_unconfirmed: int
    n_claims_with_evidence: int
    n_claims_valid_evidence: int
    n_evidence: int
    n_evidence_counter: int

    @property
    def n_claims(self) -> int:
        return self.n_confirmed + self.n_counter

    @property
    def evidence_linkage_ratio(self) -> float:
        return self.n_claims_valid_evidence / self.n_claims if self.n_claims else 0.0


def view_structure_stats(view: Any, evidence: list[Any], tech: str) -> ViewStructureStats:
    """8.2 근거 연결성·7.7 반대 근거 유무를 코드로 계산함(LLM 채점과 별개의 확정 수치)."""
    tech_evidence = [e for e in evidence if e.tech == tech]
    valid_ids = {e.id for e in tech_evidence}
    claims = list(view.confirmed_facts) + list(view.counter_facts)
    return ViewStructureStats(
        n_confirmed=len(view.confirmed_facts),
        n_counter=len(view.counter_facts),
        n_unconfirmed=len(view.unconfirmed_items),
        n_claims_with_evidence=sum(1 for c in claims if c.evidence_ids),
        n_claims_valid_evidence=sum(
            1 for c in claims if c.evidence_ids and set(c.evidence_ids) <= valid_ids
        ),
        n_evidence=len(tech_evidence),
        n_evidence_counter=sum(1 for e in tech_evidence if e.stance == "반대"),
    )


def tool_calling_accuracy(
    queries_by_tech: dict[str, list[str]], techs: list[Any], templates: list[str]
) -> tuple[float, list[str]]:
    """8.3절 Tool Calling Accuracy: 실제 던진 질의가 코드에 고정된 템플릿과 일치하는 비율.

    질의는 코드가 생성하므로 정상이면 1.0임. 템플릿 수가 두 기술 간 동일한지
    (10장 대칭 질의)도 함께 확인해 어긋난 항목 목록을 돌려줌.
    """
    issues: list[str] = []
    total = matched = 0
    for tech in techs:
        expected = [t.format(tech=tech.name, anchor=tech.search_anchor) for t in templates]
        actual = queries_by_tech.get(tech.name, [])
        for q in actual:
            total += 1
            if q in expected:
                matched += 1
            else:
                issues.append(f"템플릿 밖 질의: {q}")
        if len(actual) != len(expected):
            issues.append(f"{tech.name}: 질의 수 {len(actual)} != 템플릿 수 {len(expected)}")
    return (matched / total if total else 0.0), issues


def format_view_result(view: Any) -> str:
    """ViewResult(기술 하나)를 채점자가 읽기 좋은 텍스트로 직렬화함."""
    lines = ["확인된 사실:"]
    lines += [f"  - {c.statement} {''.join(f'[근거#{i}]' for i in c.evidence_ids)}" for c in view.confirmed_facts] or ["  (없음)"]
    lines.append("반대 사실:")
    lines += [f"  - {c.statement} {''.join(f'[근거#{i}]' for i in c.evidence_ids)}" for c in view.counter_facts] or ["  (없음)"]
    lines.append("미확인 항목:")
    lines += [f"  - {u}" for u in view.unconfirmed_items] or ["  (없음)"]
    return "\n".join(lines)


def format_evidence(evidence: list[Any], tech: str) -> str:
    return "\n".join(
        f"[근거#{e.id}] ({e.stance}) {e.source} :: {e.quote}" for e in evidence if e.tech == tech
    ) or "(없음)"
