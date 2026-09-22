"""trl_eval 독립 테스트 러너.

Doc Pool을 두 청킹 버전(v1 절 인식 / v2 naive)과 임베딩 후보들로 색인한 뒤,
청킹·임베딩(3.1~3.2절)은 전체 30개 골든셋으로, Query Rewriting(3.3절)은 이
에이전트 관점(perspective=trl)의 질의로 채점함. 청킹/임베딩은 3개 RAG 에이전트가
공유하는 단일 색인 결정이라(5장) schedule.md 원문대로 전체 골든셋을 쓰고, 그래서
세 에이전트 리포트의 청킹/임베딩 수치는 동일하게 나오는 게 정상임. Query
Rewriting만 에이전트마다 실제로 던지는 질의 유형이 달라 역할 차이가 드러남.
tech_research/domain_eval의 색인·결과에는 전혀 손대지 않고 scripts/trl_eval/
아래에만 씀.

실행: python -m scripts.trl_eval.test_runner
전제: data/doc_pool/*.pdf 존재, .env에 OPENAI_API_KEY 설정,
      python -m eval.generate_golden_dataset 로 golden_dataset.json 생성 완료.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS

from scripts.trl_eval.index_config import EMBEDDING_CANDIDATES
from src.common import config
from src.common.base_agent import rewrite_query
from src.common.doc_pool import DOC_POOL_SPECS
from src.common.eval_utils import (
    GoldenQuery,
    best_label,
    hit_rate_at_k,
    load_golden_dataset,
    load_golden_dataset_all,
    mrr,
    plot_bar_comparison,
)
from src.common.models import (
    get_embedding_model,
    get_embedding_model_by_name,
    release_embedding_model,
)
from src.common.tools import build_doc_pool_index, build_doc_pool_index_naive, paper_search

AGENT_NAME = "trl_eval"
PERSPECTIVE = "trl"
HERE = Path(__file__).parent
TOP_K = config.DEFAULT_TOP_K
THRESHOLD_HIT_RATE = 0.8  # 8.1절 제안 임계값
THRESHOLD_MRR = 0.6

# 5장·6.1절이 채택한 기본값. 실측 최고 성능과 다르면 결론에서 재검토 대상으로 표시함.
ADOPTED_CHUNK = "v1_section_aware"
ADOPTED_EMBEDDING = "bge-m3"
ADOPTED_QUERY_REWRITING = "리라이팅 질의"
# 랭킹 기준 지표. Hit Rate@5는 표본이 작으면 1.000에 자주 붙어 후보를 못 가르므로,
# 표본이 가장 큰(30개) camp 전체 필터의 MRR을 씀(best_label 참고).
RANKING_KEY = "MRR (camp 전체)"


def _get_or_build_index(embedding_model, naive: bool, index_dir: Path) -> FAISS:
    if (index_dir / "index.faiss").exists():
        return FAISS.load_local(
            str(index_dir), embedding_model, allow_dangerous_deserialization=True
        )
    builder = build_doc_pool_index_naive if naive else build_doc_pool_index
    index = builder(config.DOC_POOL_DIR, DOC_POOL_SPECS, embedding_model)
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_dir))
    return index


def _retrieve_by_filter(
    index: FAISS, goldens: list[GoldenQuery], filter_mode: str, use_rewrite: bool
) -> tuple[list[GoldenQuery], list[list]]:
    """8.1절: paper_search가 실제로 쓰는 필터 조합을 각각 따로 재현해 검색함.

    - "role_target": tech_research/trl_eval/domain_eval이 대상 기술 개요·판단을
      물을 때 쓰는 필터(role=target). role=target인 골든 질의(10개)에만 적용함.
    - "camp": "같은 진영 다른 방식과의 차이"를 물을 때 쓰는 필터(camp 전체,
      target+comparison 포함). 골든 질의 30개 전체에 적용함.
    반환값은 (그 필터에 실제로 해당하는 골든 질의 부분집합, 검색 결과)쌍 —
    hit_rate_at_k/mrr에 그대로 짝지어 넘길 수 있게 순서를 맞춤.
    """
    if filter_mode == "role_target":
        subset = [g for g in goldens if g.role == "target"]
        results = [
            paper_search(
                index,
                rewrite_query(g.query_ko, g.tech) if use_rewrite else g.query_ko,
                k=TOP_K,
                role="target",
            )
            for g in subset
        ]
        return subset, results
    if filter_mode == "camp":
        results = [
            paper_search(
                index,
                rewrite_query(g.query_ko, g.tech) if use_rewrite else g.query_ko,
                k=TOP_K,
                camp=g.camp,
            )
            for g in goldens
        ]
        return goldens, results
    raise ValueError(f"unknown filter_mode: {filter_mode}")


def _score_both_filters(
    index: FAISS, goldens: list[GoldenQuery], scores: dict[str, list[float]], use_rewrite: bool = False
) -> None:
    subset_t, retrieved_t = _retrieve_by_filter(index, goldens, "role_target", use_rewrite)
    scores["Hit Rate@5 (role=target)"].append(hit_rate_at_k(retrieved_t, subset_t, TOP_K))
    scores["MRR (role=target)"].append(mrr(retrieved_t, subset_t, TOP_K))

    subset_c, retrieved_c = _retrieve_by_filter(index, goldens, "camp", use_rewrite)
    scores["Hit Rate@5 (camp 전체)"].append(hit_rate_at_k(retrieved_c, subset_c, TOP_K))
    scores["MRR (camp 전체)"].append(mrr(retrieved_c, subset_c, TOP_K))


def _empty_filter_scores() -> dict[str, list[float]]:
    return {
        "Hit Rate@5 (role=target)": [],
        "MRR (role=target)": [],
        "Hit Rate@5 (camp 전체)": [],
        "MRR (camp 전체)": [],
    }


def run_chunking_comparison(goldens: list[GoldenQuery]):
    """3.2절: 절 인식(v1) vs naive 슬라이싱(v2) 비교. 전체 골든셋으로, role=target·
    camp 전체 두 필터 조합 각각 채점함(8.1절)."""
    embedding = get_embedding_model()
    labels = ["v1_section_aware", "v2_naive"]
    scores = _empty_filter_scores()
    for label, naive in zip(labels, (False, True)):
        index = _get_or_build_index(embedding, naive, HERE / "pdf" / label.split("_")[0] / "index")
        _score_both_filters(index, goldens, scores)
    plot_bar_comparison(
        labels, scores, f"청킹 전략 비교 ({AGENT_NAME})", "score",
        HERE / "report_assets" / "chunking_comparison.png",
    )
    return labels, scores


def run_embedding_comparison(goldens: list[GoldenQuery]):
    """3.1절: 임베딩 후보 비교(v1 청킹 위에서). 전체 골든셋, 필터 조합별로 채점함(8.1절).

    bge-m3는 청킹 비교에서 이미 v1/index로 색인해 뒀으므로 그 결과를 그대로 재사용함
    (같은 청킹+임베딩 조합을 두 번 embed하지 않기 위함 — 3개 후보 중 CPU에서 가장
    비싼 게 전체 코퍼스 재임베딩이라 여기서 1회분을 아낌).
    """
    labels = list(EMBEDDING_CANDIDATES)
    scores = _empty_filter_scores()
    for name, model_id in EMBEDDING_CANDIDATES.items():
        if name == "bge-m3":
            embedding = get_embedding_model()
            index_dir = HERE / "pdf" / "v1" / "index"
        else:
            embedding = get_embedding_model_by_name(model_id)
            index_dir = HERE / "pdf" / "v1" / f"index_{name}"
        index = _get_or_build_index(embedding, naive=False, index_dir=index_dir)
        _score_both_filters(index, goldens, scores)
        if name != "bge-m3":
            # 후보 모델은 lru_cache 대상이 아니므로 다음 후보 전에 GPU 메모리를 비움
            del index
            release_embedding_model(embedding)
    plot_bar_comparison(
        labels, scores, f"임베딩 모델 비교 ({AGENT_NAME})", "score",
        HERE / "report_assets" / "embedding_comparison.png",
    )
    return labels, scores


def run_query_rewriting_comparison(goldens: list[GoldenQuery]):
    """3.3절: Query Rewriting 적용 전/후 비교(v1 청킹, bge-m3 위에서), 필터 조합별로
    채점함(8.1절).

    이 에이전트 관점(perspective=trl)의 골든 질의만 씀 — 리라이팅 효과는
    질의 유형에 따라 달라질 수 있어, 청킹/임베딩과 달리 에이전트별로 나눠 봄.
    """
    embedding = get_embedding_model()
    index = _get_or_build_index(embedding, naive=False, index_dir=HERE / "pdf" / "v1" / "index")
    labels = ["원본 질의", "리라이팅 질의"]
    scores = _empty_filter_scores()
    for use_rewrite in (False, True):
        _score_both_filters(index, goldens, scores, use_rewrite=use_rewrite)
    plot_bar_comparison(
        labels, scores, f"Query Rewriting 비교 ({AGENT_NAME})", "score",
        HERE / "report_assets" / "query_rewriting_comparison.png",
    )
    return labels, scores


def _fmt_table(labels: list[str], scores: dict[str, list[float]]) -> str:
    header = "| 버전 | " + " | ".join(scores.keys()) + " |\n"
    header += "|---|" + "---|" * len(scores) + "\n"
    rows = "".join(
        f"| {label} | " + " | ".join(f"{scores[k][i]:.3f}" for k in scores) + " |\n"
        for i, label in enumerate(labels)
    )
    return header + rows


def build_report(chunk, emb, qr) -> str:
    chunk_labels, chunk_scores = chunk
    emb_labels, emb_scores = emb
    qr_labels, qr_scores = qr
    best_hit = max(
        chunk_scores["Hit Rate@5 (role=target)"] + chunk_scores["Hit Rate@5 (camp 전체)"]
        + emb_scores["Hit Rate@5 (role=target)"] + emb_scores["Hit Rate@5 (camp 전체)"]
    )
    best_mrr = max(
        chunk_scores["MRR (role=target)"] + chunk_scores["MRR (camp 전체)"]
        + emb_scores["MRR (role=target)"] + emb_scores["MRR (camp 전체)"]
    )
    verdict = (
        f"목표 임계값(Hit Rate@5 ≥ {THRESHOLD_HIT_RATE}, MRR ≥ {THRESHOLD_MRR})을 만족함."
        if best_hit >= THRESHOLD_HIT_RATE and best_mrr >= THRESHOLD_MRR
        else f"목표 임계값(Hit Rate@5 ≥ {THRESHOLD_HIT_RATE}, MRR ≥ {THRESHOLD_MRR})에 미달함 — "
        "6.1절 절차대로 임베딩 후보 교체를 검토해야 함."
    )

    best_chunk = best_label(chunk_labels, chunk_scores, RANKING_KEY)
    best_emb = best_label(emb_labels, emb_scores, RANKING_KEY)
    best_qr = best_label(qr_labels, qr_scores, RANKING_KEY)

    def _match_note(name: str, best: str, adopted: str) -> str:
        if best == adopted:
            return f"- {name}: 채택안({adopted})이 {RANKING_KEY} 기준으로도 최고 성능임."
        return (
            f"- {name}: 채택안은 {adopted}이지만, {RANKING_KEY} 기준 실측 최고 성능은 "
            f"**{best}**임 — 6.1절/5장 절차대로 재검토 대상."
        )

    combination_notes = "\n".join(
        [
            _match_note("청킹", best_chunk, ADOPTED_CHUNK),
            _match_note("임베딩", best_emb, ADOPTED_EMBEDDING),
            _match_note("Query Rewriting", best_qr, ADOPTED_QUERY_REWRITING),
        ]
    )

    return f"""# trl_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.3절 / docs/schedule.md 3.1~3.3절
(TRL 판단 근거인 "실험 수준, 공개 구현 여부"를 얼마나 잘 찾아오는지를 검증함)

모든 표는 두 필터 조합을 각각 채점함(8.1절 "paper_search가 실제로 쓰는 필터
조합 각각에 대해 별도로 측정"): `role=target`은 기술 개요·판단 질의가 쓰는
좁은 필터(대상 기술 2편만), `camp 전체`는 "같은 진영 다른 방식과의 차이" 질의가
쓰는 넓은 필터(해당 진영 3편 모두)임.

## 1. 청킹 전략 비교 (3.2절 — 절 인식 vs naive 슬라이싱, 전체 30개 골든셋)

{_fmt_table(chunk_labels, chunk_scores)}
![청킹 비교](report_assets/chunking_comparison.png)

3개 RAG 에이전트가 공유하는 색인 설정이라, 이 수치는 tech_research·domain_eval
리포트와 동일하게 나오는 게 정상임(전역 결정을 전체 골든셋으로 검증).

## 2. 임베딩 모델 비교 (3.1절, 전체 30개 골든셋)

{_fmt_table(emb_labels, emb_scores)}
![임베딩 비교](report_assets/embedding_comparison.png)

이 수치도 tech_research·domain_eval 리포트와 동일하게 나오는 게 정상임(위와 동일한 이유).

## 3. Query Rewriting 효과 (3.3절, trl_eval 관점 질의만)

{_fmt_table(qr_labels, qr_scores)}
![Query Rewriting 비교](report_assets/query_rewriting_comparison.png)

## 4. 결론

{verdict}

**채택안 vs 실측 최고 성능** ({RANKING_KEY} 기준):

{combination_notes}

trl_eval은 재검색 시 질의 초점을 "구현/공식 발표"에서 "한계·실패 사례·후속
검증"으로 바꾸는 로직(7.3절)이 있으므로, 위 수치는 1차 패스(재검색 전) 기준임.
골든셋이 30개(role=target 필터는 10개)뿐이라 차이가 통계적으로 확고한지는
표본을 늘려 다시 확인해 볼 것.
"""


def main() -> None:
    goldens_all = load_golden_dataset_all(config.GOLDEN_DATASET_PATH)
    goldens_own = load_golden_dataset(config.GOLDEN_DATASET_PATH, PERSPECTIVE)
    if not goldens_all:
        raise SystemExit(
            "골든 질의가 없음. 먼저 `python -m eval.generate_golden_dataset` 를 실행할 것."
        )
    if not goldens_own:
        raise SystemExit(
            f"perspective='{PERSPECTIVE}' 골든 질의가 없음(Query Rewriting 비교용). "
            "먼저 `python -m eval.generate_golden_dataset` 를 실행할 것."
        )

    chunk = run_chunking_comparison(goldens_all)
    emb = run_embedding_comparison(goldens_all)
    qr = run_query_rewriting_comparison(goldens_own)

    report = build_report(chunk, emb, qr)
    report_path = HERE / f"{AGENT_NAME}_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"리포트 저장 완료 -> {report_path}")


if __name__ == "__main__":
    main()
