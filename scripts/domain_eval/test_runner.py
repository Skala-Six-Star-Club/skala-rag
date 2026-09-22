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
    hit_rate_at_k,
    load_golden_dataset,
    load_golden_dataset_all,
    mrr,
    plot_bar_comparison,
)
from src.common.models import get_embedding_model, get_embedding_model_by_name
from src.common.tools import build_doc_pool_index, build_doc_pool_index_naive, paper_search

AGENT_NAME = "domain_eval"
PERSPECTIVE = "domain"
HERE = Path(__file__).parent
TOP_K = config.DEFAULT_TOP_K
THRESHOLD_HIT_RATE = 0.8  # 8.1절 제안 임계값
THRESHOLD_MRR = 0.6


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


def _retrieve_all(index: FAISS, goldens: list[GoldenQuery], use_rewrite: bool) -> list[list]:
    results = []
    for g in goldens:
        query = rewrite_query(g.query_ko, g.tech) if use_rewrite else g.query_ko
        results.append(paper_search(index, query, k=TOP_K, role=g.role))
    return results


def run_chunking_comparison(goldens: list[GoldenQuery]):
    """3.2절: 절 인식(v1) vs naive 슬라이싱(v2) 비교.

    domain_eval은 실제 운영 시 "실험 환경"/"평가" 절을 우선하는 경량 재랭킹을
    쓰므로(7.4절), 여기서 절 인식(v1)이 v2보다 뚜렷이 나아야 재랭킹의 전제가
    성립함을 확인할 수 있음(schedule.md 3.2절 참고). 전체 골든셋으로 채점함.
    """
    embedding = get_embedding_model()
    labels = ["v1_section_aware", "v2_naive"]
    scores = {"Hit Rate@5": [], "MRR": []}
    for label, naive in zip(labels, (False, True)):
        index = _get_or_build_index(embedding, naive, HERE / "pdf" / label.split("_")[0] / "index")
        retrieved = _retrieve_all(index, goldens, use_rewrite=False)
        scores["Hit Rate@5"].append(hit_rate_at_k(retrieved, goldens, TOP_K))
        scores["MRR"].append(mrr(retrieved, goldens, TOP_K))
    plot_bar_comparison(
        labels, scores, f"청킹 전략 비교 ({AGENT_NAME})", "score",
        HERE / "report_assets" / "chunking_comparison.png",
    )
    return labels, scores


def run_embedding_comparison(goldens: list[GoldenQuery]):
    """3.1절: 임베딩 후보 비교(v1 청킹 위에서). 전체 골든셋으로 채점함.

    bge-m3는 청킹 비교에서 이미 v1/index로 색인해 뒀으므로 그 결과를 그대로 재사용함
    (같은 청킹+임베딩 조합을 두 번 embed하지 않기 위함 — 3개 후보 중 CPU에서 가장
    비싼 게 전체 코퍼스 재임베딩이라 여기서 1회분을 아낌).
    """
    labels = list(EMBEDDING_CANDIDATES)
    scores = {"Hit Rate@5": [], "MRR": []}
    for name, model_id in EMBEDDING_CANDIDATES.items():
        if name == "bge-m3":
            embedding = get_embedding_model()
            index_dir = HERE / "pdf" / "v1" / "index"
        else:
            embedding = get_embedding_model_by_name(model_id)
            index_dir = HERE / "pdf" / "v1" / f"index_{name}"
        index = _get_or_build_index(embedding, naive=False, index_dir=index_dir)
        retrieved = _retrieve_all(index, goldens, use_rewrite=False)
        scores["Hit Rate@5"].append(hit_rate_at_k(retrieved, goldens, TOP_K))
        scores["MRR"].append(mrr(retrieved, goldens, TOP_K))
    plot_bar_comparison(
        labels, scores, f"임베딩 모델 비교 ({AGENT_NAME})", "score",
        HERE / "report_assets" / "embedding_comparison.png",
    )
    return labels, scores


def run_query_rewriting_comparison(goldens: list[GoldenQuery]):
    """3.3절: Query Rewriting 적용 전/후 비교(v1 청킹, bge-m3 위에서).

    이 에이전트 관점(perspective=domain)의 골든 질의만 씀 — 리라이팅 효과는
    질의 유형에 따라 달라질 수 있어, 청킹/임베딩과 달리 에이전트별로 나눠 봄.
    """
    embedding = get_embedding_model()
    index = _get_or_build_index(embedding, naive=False, index_dir=HERE / "pdf" / "v1" / "index")
    labels = ["원본 질의", "리라이팅 질의"]
    scores = {"Hit Rate@5": [], "MRR": []}
    for use_rewrite in (False, True):
        retrieved = _retrieve_all(index, goldens, use_rewrite=use_rewrite)
        scores["Hit Rate@5"].append(hit_rate_at_k(retrieved, goldens, TOP_K))
        scores["MRR"].append(mrr(retrieved, goldens, TOP_K))
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
    best_hit = max(chunk_scores["Hit Rate@5"] + emb_scores["Hit Rate@5"])
    best_mrr = max(chunk_scores["MRR"] + emb_scores["MRR"])
    verdict = (
        f"목표 임계값(Hit Rate@5 ≥ {THRESHOLD_HIT_RATE}, MRR ≥ {THRESHOLD_MRR})을 만족함."
        if best_hit >= THRESHOLD_HIT_RATE and best_mrr >= THRESHOLD_MRR
        else f"목표 임계값(Hit Rate@5 ≥ {THRESHOLD_HIT_RATE}, MRR ≥ {THRESHOLD_MRR})에 미달함 — "
        "6.1절 절차대로 임베딩 후보 교체를 검토해야 함."
    )
    return f"""# domain_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.4절 / docs/schedule.md 3.1~3.3절
(에이전트 코딩 멀티턴 서빙 도입 조건의 근거인 "실험 환경, 요구 하드웨어"를
얼마나 잘 찾아오는지를 검증함)

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

## 4. 결론

{verdict}

운영 중인 domain_eval은 여기에 더해 "실험 환경"/"평가" 절 우선 재랭킹을 적용함
(7.4절 2번 항목). 이 재랭킹은 별도 실험 없이 채택된 장치라, 위 수치와 별개로
실제 운영 결과에서 순위 개선 여부를 추가로 관찰할 것(schedule.md 3.2절 참고).
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
