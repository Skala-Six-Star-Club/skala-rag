"""domain_eval 독립 테스트 러너.

Doc Pool을 두 청킹 버전(v1 절 인식 / v2 naive)과 임베딩 후보들로 색인한 뒤,
청킹·임베딩(3.1~3.2절)은 전체 30개 골든셋으로, Query Rewriting(3.3절)은 이
에이전트 관점(perspective=domain)의 질의로 채점함. 청킹/임베딩은 3개 RAG
에이전트가 공유하는 단일 색인 결정이라(5장) schedule.md 원문대로 전체 골든셋을
쓰고, 그래서 세 에이전트 리포트의 청킹/임베딩 수치는 동일하게 나오는 게 정상임.
Query Rewriting만 에이전트마다 실제로 던지는 질의 유형이 달라 역할 차이가 드러남.
tech_research/trl_eval의 색인·결과에는 전혀 손대지 않고 scripts/domain_eval/
아래에만 씀.

실행: python -m scripts.domain_eval.test_runner
전제: data/doc_pool/*.pdf 존재, .env에 OPENAI_API_KEY 설정,
      python -m eval.generate_golden_dataset 로 golden_dataset.json 생성 완료.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS

from scripts.domain_eval.index_config import EMBEDDING_CANDIDATES
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

AGENT_NAME = "domain_eval"
PERSPECTIVE = "domain"
HERE = Path(__file__).parent
TOP_K = config.DEFAULT_TOP_K
THRESHOLD_HIT_RATE = 0.8  # 8.1절 제안 임계값
THRESHOLD_MRR = 0.6

# 5장·6.1절이 채택한 기본값. 실측 최고 성능과 다르면 결론에서 재검토 대상으로 표시함.
ADOPTED_CHUNK = "v1_section_aware"
# 채택 임베딩은 config.EMBEDDING_MODEL을 후보 목록에서 역조회함(후보 밖이면 모델 id 그대로).
ADOPTED_EMBEDDING = next(
    (name for name, mid in EMBEDDING_CANDIDATES.items() if mid == config.EMBEDDING_MODEL),
    config.EMBEDDING_MODEL,
)


def _index_dir(version: str, embedding_name: str) -> Path:
    """색인 경로를 청킹 버전 x 임베딩 이름으로 고정함. 채택 임베딩을 바꿔도 이전
    모델의 색인을 덮어쓰지 않고, 같은 조합은 재사용함."""
    return HERE / "pdf" / version / f"index_{embedding_name}"
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


# 리라이팅 결과 기록: golden id -> (원본, 리라이팅). 같은 질의를 role=target과 camp 전체
# 두 필터에서 각각 다시 리라이팅하면 필터 간 질의가 달라져 비교가 흐려지므로 1회만
# 호출해 재사용하고, 리포트에도 남겨 실행 간 변동 원인을 추적할 수 있게 함.
REWRITE_LOG: dict[int, tuple[str, str]] = {}


def _query_for(g: GoldenQuery, use_rewrite: bool) -> str:
    if not use_rewrite:
        return g.query_ko
    if g.id not in REWRITE_LOG:
        REWRITE_LOG[g.id] = (g.query_ko, rewrite_query(g.query_ko, g.tech))
    return REWRITE_LOG[g.id][1]


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
                _query_for(g, use_rewrite),
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
                _query_for(g, use_rewrite),
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
    camp 전체 두 필터 조합 각각 채점함(8.1절).

    domain_eval은 실제 운영 시 "실험 환경"/"평가" 절을 우선하는 경량 재랭킹을
    쓰므로(7.4절), 여기서 절 인식(v1)이 v2보다 뚜렷이 나아야 재랭킹의 전제가
    성립함을 확인할 수 있음(schedule.md 3.2절 참고).
    """
    embedding = get_embedding_model()
    labels = ["v1_section_aware", "v2_naive"]
    scores = _empty_filter_scores()
    for label, naive in zip(labels, (False, True)):
        index = _get_or_build_index(embedding, naive, _index_dir(label.split("_")[0], ADOPTED_EMBEDDING))
        _score_both_filters(index, goldens, scores)
    plot_bar_comparison(
        labels, scores, f"청킹 전략 비교 ({AGENT_NAME})", "score",
        HERE / "report_assets" / "chunking_comparison.png",
    )
    return labels, scores


def run_embedding_comparison(goldens: list[GoldenQuery]):
    """3.1절: 임베딩 후보 비교(v1 청킹 위에서). 전체 골든셋, 필터 조합별로 채점함(8.1절).

    채택 임베딩은 청킹 비교에서 이미 색인해 뒀으므로 그대로 재사용함(같은 청킹+임베딩
    조합을 두 번 embed하지 않기 위함).
    """
    labels = list(EMBEDDING_CANDIDATES)
    scores = _empty_filter_scores()
    for name, model_id in EMBEDDING_CANDIDATES.items():
        if name == ADOPTED_EMBEDDING:
            embedding = get_embedding_model()
        else:
            embedding = get_embedding_model_by_name(model_id)
        index = _get_or_build_index(embedding, naive=False, index_dir=_index_dir("v1", name))
        _score_both_filters(index, goldens, scores)
        if name != ADOPTED_EMBEDDING:
            # 후보 모델은 lru_cache 대상이 아니므로 다음 후보 전에 GPU 메모리를 비움
            del index
            release_embedding_model(embedding)
    plot_bar_comparison(
        labels, scores, f"임베딩 모델 비교 ({AGENT_NAME})", "score",
        HERE / "report_assets" / "embedding_comparison.png",
    )
    return labels, scores


def run_query_rewriting_comparison(goldens: list[GoldenQuery]):
    """3.3절: Query Rewriting 적용 전/후 비교(v1 청킹, 채택 임베딩 위에서), 필터 조합별로
    채점함(8.1절).

    이 에이전트 관점(perspective=domain)의 골든 질의만 씀 — 리라이팅 효과는
    질의 유형에 따라 달라질 수 있어, 청킹/임베딩과 달리 에이전트별로 나눠 봄.
    """
    embedding = get_embedding_model()
    index = _get_or_build_index(embedding, naive=False, index_dir=_index_dir("v1", ADOPTED_EMBEDDING))
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


def _fmt_rewrite_table() -> str:
    def _cell(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    rows = "".join(
        f"| {gid} | {_cell(orig)} | {_cell(rew)} |\n"
        for gid, (orig, rew) in sorted(REWRITE_LOG.items())
    )
    return "| id | 원본 질의 | 리라이팅 질의 |\n|---|---|---|\n" + rows


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

    return f"""# domain_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.4절 / docs/schedule.md 3.1~3.3절
(에이전트 코딩 멀티턴 서빙 도입 조건의 근거인 "실험 환경, 요구 하드웨어"를
얼마나 잘 찾아오는지를 검증함)

모든 표는 두 필터 조합을 각각 채점함(8.1절 "paper_search가 실제로 쓰는 필터
조합 각각에 대해 별도로 측정"): `role=target`은 기술 개요·판단 질의가 쓰는
좁은 필터(대상 기술 2편만), `camp 전체`는 "같은 진영 다른 방식과의 차이" 질의가
쓰는 넓은 필터(해당 진영 3편 모두)임.

## 1. 청킹 전략 비교 (3.2절 — 절 인식 vs naive 슬라이싱, 전체 30개 골든셋)

{_fmt_table(chunk_labels, chunk_scores)}
![청킹 비교](report_assets/chunking_comparison.png)

3개 RAG 에이전트가 공유하는 색인 설정이라, 이 수치는 tech_research·trl_eval
리포트와 동일하게 나오는 게 정상임(전역 결정을 전체 골든셋으로 검증).

## 2. 임베딩 모델 비교 (3.1절, 전체 30개 골든셋)

{_fmt_table(emb_labels, emb_scores)}
![임베딩 비교](report_assets/embedding_comparison.png)

이 수치도 tech_research·trl_eval 리포트와 동일하게 나오는 게 정상임(위와 동일한 이유).

## 3. Query Rewriting 효과 (3.3절, domain_eval 관점 질의만)

{_fmt_table(qr_labels, qr_scores)}
![Query Rewriting 비교](report_assets/query_rewriting_comparison.png)

리라이팅 질의 목록(실행마다 LLM 출력이 달라질 수 있어 변동 추적용으로 기록함):

{_fmt_rewrite_table()}

## 4. 결론

{verdict}

**채택안 vs 실측 최고 성능** ({RANKING_KEY} 기준):

{combination_notes}

운영 중인 domain_eval은 여기에 더해 "실험 환경"/"평가" 절 우선 재랭킹을 적용함
(7.4절 2번 항목). 이 재랭킹은 별도 실험 없이 채택된 장치라, 위 수치와 별개로
실제 운영 결과에서 순위 개선 여부를 추가로 관찰할 것(schedule.md 3.2절 참고).
현재 채택 임베딩은 {ADOPTED_EMBEDDING}임(2026-09-22 비교실험 결과로 bge-m3에서 교체).
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
