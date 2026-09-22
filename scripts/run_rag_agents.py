"""RAG 3종 에이전트 스모크 실행 (graph.py 이전 단계).

select_tech -> tech_research -> trl_eval, domain_eval 순으로 실제 노드를 돌려
State 갱신분을 output/rag_agents_smoke.md에 사람이 읽을 수 있게 남김.
graph.py가 생기면 이 스크립트는 그래프의 부분 실행으로 대체됨.

실행: python -m scripts.run_rag_agents [--retry]
  --retry: trl_eval/domain_eval을 retry_targets에 넣어 재검색 패스(초점 전환)까지 확인
전제: .env(OPENAI_API_KEY, TAVILY_API_KEY), data/doc_pool PDF 6편.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.agents.domain_eval.agent import DomainEvalAgent
from src.agents.select_tech.agent import SelectTechAgent
from src.agents.tech_research.agent import TechResearchAgent
from src.agents.trl_eval.agent import TrlEvalAgent
from src.common import config
from src.common.eval_utils import format_view_result
from src.common.tools import get_shared_index

OUT = Path("output/rag_agents_smoke.md")


def _merge(state: dict, update: dict) -> None:
    """LangGraph reducer 흉내: evidence/references는 누적, 나머지는 덮어씀(11장)."""
    for k, v in update.items():
        if k in ("evidence", "references"):
            state[k] = state.get(k, []) + v
        else:
            state[k] = v


def main() -> None:
    do_retry = "--retry" in sys.argv
    state: dict = {"evidence": [], "references": []}
    timings: dict[str, float] = {}

    t = time.time(); _merge(state, SelectTechAgent().run(state)); timings["select_tech"] = time.time() - t
    index = get_shared_index()
    for agent in (TechResearchAgent(index), TrlEvalAgent(index), DomainEvalAgent(index)):
        t = time.time(); _merge(state, agent.run(state)); timings[agent.name] = time.time() - t
    if do_retry:
        state["retry_targets"] = ["trl_eval", "domain_eval"]
        for agent in (TrlEvalAgent(index), DomainEvalAgent(index)):
            t = time.time(); _merge(state, agent.run(state)); timings[agent.name + " (retry)"] = time.time() - t

    lines = [
        "# RAG 3종 에이전트 스모크 실행 결과", "",
        f"- 임베딩: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DEVICE})",
        f"- Query Rewriting: {'on' if config.QUERY_REWRITING else 'off'}",
        f"- 도메인: {state['domain']}",
        "- 소요 시간: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()), "",
    ]
    for tech, prof in state["tech_profiles"].items():
        lines += [f"## tech_research: {tech}", "", f"**개요** {prof.overview}", "", f"**적용 범위** {prof.scope}", "",
                  f"**한계** {prof.limitations}", "", f"**같은 진영 차이** {prof.differentiation}", "",
                  f"근거 번호: {prof.evidence_ids}", ""]
    for key, title in (("trl_result", "trl_eval"), ("domain_result", "domain_eval")):
        for tech, view in state[key].by_tech.items():
            lines += [f"## {title}: {tech}", "", "```", format_view_result(view), "```", ""]
    lines += ["## evidence 요약", "", "| 관점 | 기술 | 근거 수 | 반대 근거 수 | 번호 대역 |", "|---|---|---|---|---|"]
    for persp in ("tech_research", "trl", "domain"):
        for tech in state["tech_profiles"]:
            evs = [e for e in state["evidence"] if e.perspective == persp and e.tech == tech]
            if evs:
                lines.append(f"| {persp} | {tech} | {len(evs)} | {sum(e.stance == '반대' for e in evs)} | {min(e.id for e in evs)}~{max(e.id for e in evs)} |")
    ids = [e.id for e in state["evidence"]]
    lines += ["", f"근거 번호 중복: {len(ids) - len(set(ids))}건", ""]
    lines += ["## references", ""] + [f"- {r.title} ({r.year}) {r.url}" for r in state["references"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장 완료 -> {OUT}")


if __name__ == "__main__":
    main()
