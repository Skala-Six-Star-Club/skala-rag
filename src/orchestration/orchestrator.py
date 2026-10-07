"""Orchestrator 노드(orchestrator). 서브 태스크 목록을 구조화해 State에 저장함.

서브 태스크 단위는 (관점, 기술, 초점)이고, 초점은 관점 필수 항목 하나에 대응함(src/common/focus.py).
worker 수와 각 worker가 맡는 일은 이 노드가 계획을 세운 뒤에 정해짐.

round 0 (최초 계획)
    1. 생성 LLM이 tech_research 결과(개요, 한계, 차별점)를 보고 칸(관점, 기술)마다 조사할 초점을 고름
    2. 계획 검증(validate_plan)이 LLM 계획을 보정하고 보정 사유를 남김
       - 카탈로그에 없는 초점 제거
       - 대칭: 한 기술에만 고른 초점은 다른 기술에도 추가(10장 대칭 질의)
       - 필수 초점: 기술 성숙도 칸에 TRL 구간 추정(estimate) 추가
       - 커버리지: 초점이 하나도 없는 칸에 관점 기본 초점 추가(보고서 최소 범위가 관점 4 x 기술 N)
       - 상한: MAX_SUBTASKS를 넘으면 뒤에 고른 초점부터, 칸이 비지 않는 범위에서 제거
    PLANNER_MODE=rule이거나 LLM 호출이 실패하면 빈 계획을 검증에 넘겨 칸마다 기본 초점 하나로 채움

round 1 이상 (re-plan)
    evidence_check(근거 부족)나 quality_eval(커버리지, 편향 미달)이 남긴 pending_gaps의 칸만
    서브 태스크로 만듦. 초점이 지정되지 않은 칸은 그 칸에서 아직 실행하지 않은 초점을, 다 실행했으면
    반대 근거(counter) 초점을 고름. 부족한 기술만 다시 조사해야 기술 간 근거 균형이 회복되므로
    이 단계는 대칭 보정을 하지 않음.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.focus import COUNTER, DEFAULT_FOCUS, FOCI, REQUIRED_FOCUS, format_catalog, is_valid_focus
from src.common.models import get_generation_llm
from src.common.observability import log_decision
from src.common.state import AgentState, Gap, Plan, SubTask, TechProfile, ViewPerspective

VIEW_PERSPECTIVES: tuple[str, ...] = tuple(FOCI)

_PROMPT = """\
당신은 두 기술을 4개 관점(기술 성숙도 trl, 시장성 market, 이해관계자 stakeholder, 도메인 적용 domain)에서
중립적으로 비교 평가하는 조사 계획의 Orchestrator임. 아래 기술 조사 결과를 보고, 각 (관점, 기술) 칸에서
조사할 초점을 골라 서브 태스크 목록을 만들어줘. 서브 태스크 하나는 (관점, 기술, 초점) 하나이고 worker 하나가 실행함.

[평가 대상 기술]
{techs}

[평가 도메인]
{domain}

[기술 조사 결과]
{profiles}

[관점별 초점 목록. 형식은 관점/초점: 설명]
{catalog}

규칙:
1. 서브 태스크는 최대 {max_subtasks}개. 칸마다 1~3개를 고르되, 기술 조사 결과에서 한계가 크거나 정보가 비어 있는
   항목이 걸린 칸에 초점을 더 배정함. 이미 기술 조사 결과로 분명한 항목은 고르지 않아도 됨
2. 기술 성숙도 칸에는 estimate를 포함함
3. 반대 근거가 부족할 것으로 보이는 칸에는 counter를 포함함
4. 특정 기술에 유리하거나 불리한 근거를 찾으려는 목적으로 고르지 않음. 두 기술에 같은 초점을 쓰는 것을 원칙으로 함
5. reason에는 기술 조사 결과의 어떤 공백이나 한계 때문에 그 초점을 고르는지 한 문장으로 적음
"""


class _Pick(BaseModel):
    perspective: ViewPerspective
    tech: str
    focus: str
    reason: str = ""


class _PlanDraft(BaseModel):
    picks: list[_Pick] = Field(default_factory=list)
    rationale: str = ""


def _subtask(round_: int, perspective: str, tech: str, focus: str, reason: str) -> SubTask:
    return SubTask(
        subtask_id=f"r{round_}:{perspective}:{focus}:{tech}",
        perspective=perspective,  # type: ignore[arg-type]
        tech=tech,
        focus=focus,
        round=round_,
        reason=reason,
    )


def validate_plan(picks: list[_Pick], techs: list[str], max_subtasks: int) -> tuple[list[SubTask], list[str]]:
    """LLM 계획을 검증하고 보정함. (서브 태스크 목록, 보정 사유 목록)을 반환."""
    corrections: list[str] = []
    ordered: list[tuple[str, str, str]] = []  # (관점, 초점, 기술). 고른 순서 유지
    reasons: dict[tuple[str, str, str], str] = {}

    def add(perspective: str, focus: str, tech: str, reason: str) -> None:
        key = (perspective, focus, tech)
        if key not in reasons:
            ordered.append(key)
            reasons[key] = reason

    for pick in picks:
        if pick.tech not in techs or not is_valid_focus(pick.perspective, pick.focus):
            corrections.append(f"제거: {pick.perspective}/{pick.focus}/{pick.tech} (카탈로그나 대상 기술에 없음)")
            continue
        add(pick.perspective, pick.focus, pick.tech, pick.reason)

    pairs = list(dict.fromkeys((p, f) for p, f, _ in ordered))
    for perspective, focus in pairs:
        chosen = [t for t in techs if (perspective, focus, t) in reasons]
        for tech in techs:
            if tech not in chosen:
                add(perspective, focus, tech, f"대칭 보정({', '.join(chosen)}와 같은 초점)")
                corrections.append(f"대칭 보정: {perspective}/{focus}를 {tech}에도 추가")

    for perspective, focus in REQUIRED_FOCUS.items():
        for tech in techs:
            if (perspective, focus, tech) not in reasons:
                add(perspective, focus, tech, "필수 초점")
                corrections.append(f"필수 초점 보정: {perspective}/{focus}/{tech}")

    def cell_foci(perspective: str, tech: str, keys: list[tuple[str, str, str]]) -> list[str]:
        return [f for p, f, t in keys if p == perspective and t == tech]

    # 커버리지: 칸마다 반대 근거(counter)가 아닌 초점이 하나 이상 있어야 함. counter만 있는 칸은
    # 부정적 근거만 수집되어 그 칸의 서술이 한쪽으로 기움
    for perspective in VIEW_PERSPECTIVES:
        for tech in techs:
            foci = cell_foci(perspective, tech, ordered)
            if all(f == COUNTER for f in foci):
                kind = "빈 칸 보정" if not foci else "균형 보정"
                add(perspective, DEFAULT_FOCUS[perspective], tech, f"{kind}(관점 커버리지)")
                corrections.append(f"{kind}: {perspective}/{tech}에 {DEFAULT_FOCUS[perspective]} 추가")

    # 상한: 뒤에 고른 (관점, 초점) 쌍부터 두 기술 함께 제거. 필수 초점은 남기고, 제거 뒤에도
    # 모든 칸에 counter가 아닌 초점이 남는 경우만 제거
    if len(ordered) > max_subtasks:
        for perspective, focus in reversed(list(dict.fromkeys((p, f) for p, f, _ in ordered))):
            if len(ordered) <= max_subtasks:
                break
            if REQUIRED_FOCUS.get(perspective) == focus:
                continue
            remaining = [k for k in ordered if (k[0], k[1]) != (perspective, focus)]
            cells_ok = all(
                any(f != COUNTER for f in cell_foci(perspective, tech, remaining)) for tech in techs
            )
            if not cells_ok:
                continue
            ordered = [k for k in ordered if (k[0], k[1]) != (perspective, focus)]
            corrections.append(f"상한 조정: {perspective}/{focus} 제거(서브 태스크 상한 {max_subtasks})")

    subtasks = [_subtask(0, p, t, f, reasons[(p, f, t)]) for p, f, t in ordered]
    return subtasks, corrections


def _executed(state: AgentState) -> set[tuple[str, str, str]]:
    """지금까지 실행을 마친 (관점, 초점, 기술). node_status의 subtask_id에서 읽음."""
    done: set[tuple[str, str, str]] = set()
    for subtask_id, status in (state.get("node_status") or {}).items():
        parts = subtask_id.split(":", 3)
        if len(parts) == 4 and status == "done":
            _, perspective, focus, tech = parts
            done.add((perspective, focus, tech))
    return done


def _format_profiles(profiles: dict[str, TechProfile]) -> str:
    if not profiles:
        return "(기술 조사 결과 없음)"
    blocks = []
    for name, p in profiles.items():
        p = p if isinstance(p, TechProfile) else TechProfile.model_validate(p)
        blocks.append(
            f"- {name}\n  개요: {p.overview}\n  적용 범위: {p.scope}\n  한계: {p.limitations}\n  차별점: {p.differentiation}"
        )
    return "\n".join(blocks)


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    uses_rag = False

    def __init__(self, mode: str | None = None, llm: Any | None = None, max_subtasks: int | None = None):
        self.mode = (mode or config.PLANNER_MODE).lower()
        self._llm = llm
        self.max_subtasks = max_subtasks or config.MAX_SUBTASKS

    def run(self, state: AgentState) -> dict[str, Any]:
        round_ = state.get("plan_round", -1) + 1
        techs = [t.name for t in state.get("techs", []) or []]
        plan = self._initial_plan(state, techs) if round_ == 0 else self._replan(state, round_)

        log_decision(
            state.get("trace_id"),
            self.name,
            f"round {plan.round}: 서브 태스크 {len(plan.subtasks)}개",
            plan.rationale,
            subtasks=[s.subtask_id for s in plan.subtasks],
            corrections=plan.corrections,
        )
        return {
            "plan": plan,
            "plan_round": round_,
            "pending_gaps": [],
            "planned_subtasks": plan.subtasks,
            # 계획 시점에 pending으로 기록. 중단 후 재개하면 pending으로 남은 작업이 미완료 작업임
            "node_status": {s.subtask_id: "pending" for s in plan.subtasks},
        }

    # -- round 0 -----------------------------------------------------------------

    def _initial_plan(self, state: AgentState, techs: list[str]) -> Plan:
        picks: list[_Pick] = []
        if self.mode == "llm":
            picks, rationale = self._draft(state, techs)
        else:
            rationale = "rule 모드: 칸마다 기본 초점 하나"
        subtasks, corrections = validate_plan(picks, techs, self.max_subtasks)
        return Plan(round=0, subtasks=subtasks, rationale=rationale, corrections=corrections)

    def _draft(self, state: AgentState, techs: list[str]) -> tuple[list[_Pick], str]:
        prompt = _PROMPT.format(
            techs=", ".join(techs),
            domain=state.get("domain", ""),
            profiles=_format_profiles(state.get("tech_profiles", {}) or {}),
            catalog=format_catalog(),
            max_subtasks=self.max_subtasks,
        )
        try:
            llm = (self._llm or get_generation_llm()).with_structured_output(_PlanDraft)
            draft: _PlanDraft = llm.invoke(prompt)  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001 - 계획 실패는 검증 단계의 기본 계획으로 대체
            reason = f"계획 LLM 실패({type(exc).__name__}: {exc}). 칸마다 기본 초점 하나로 대체"
            log_decision(state.get("trace_id"), self.name, "fallback_rule_plan", reason)
            return [], reason
        return draft.picks, draft.rationale

    # -- round 1+ ----------------------------------------------------------------

    def _replan(self, state: AgentState, round_: int) -> Plan:
        gaps: list[Gap] = list(state.get("pending_gaps") or [])
        executed = _executed(state)
        subtasks: list[SubTask] = []
        seen: set[tuple[str, str, str]] = set()
        for gap in gaps:
            focus = gap.focus
            if focus is None or not is_valid_focus(gap.perspective, focus):
                unexplored = [
                    f for f in FOCI[gap.perspective]
                    if f != COUNTER and (gap.perspective, f, gap.tech) not in executed
                ]
                focus = unexplored[0] if unexplored else COUNTER
            key = (gap.perspective, focus, gap.tech)
            if key in seen or len(subtasks) >= self.max_subtasks:
                continue
            seen.add(key)
            subtasks.append(_subtask(round_, gap.perspective, gap.tech, focus, f"[{gap.source}] {gap.reason}"))
        sources = sorted({g.source for g in gaps}) or ["없음"]
        rationale = f"re-plan(요청: {', '.join(sources)}). 부족한 칸 {len(subtasks)}건만 재조사"
        return Plan(round=round_, subtasks=subtasks, rationale=rationale)
