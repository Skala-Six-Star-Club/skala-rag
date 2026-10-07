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
