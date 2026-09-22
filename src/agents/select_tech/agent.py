"""기술 선정 에이전트 (7.1절). LLM을 쓰지 않는 순수 로더 노드.

입력: 설정 파일(configs/tech_selection.json)
출력: {"techs": [...], "domain": ...}

기술 선정 자체는 3장에 따라 사람이 이미 내린 결정이므로, 이 노드는 그 값을
State 스키마에 맞춰 검증 후 옮기는 역할만 함(LLM 미사용).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.state import AgentState, TechSpec

_VALID_CAMPS = {"SW", "HW"}
_VALID_ROLES = {"target", "comparison"}


class SelectTechAgent(BaseAgent):
    name = "select_tech"
    uses_rag = False

    def __init__(self, config_path: Path = config.TECH_SELECTION_CONFIG_PATH):
        self.config_path = config_path

    def run(self, state: AgentState) -> dict[str, Any]:
        data = json.loads(self.config_path.read_text(encoding="utf-8"))

        if "domain" not in data or "techs" not in data:
            raise ValueError("설정 파일에 domain/techs 필수 필드가 없음")

        techs: list[TechSpec] = []
        for t in data["techs"]:
            if t.get("camp") not in _VALID_CAMPS:
                raise ValueError(f"invalid camp: {t.get('camp')}")
            if t.get("role") not in _VALID_ROLES:
                raise ValueError(f"invalid role: {t.get('role')}")
            techs.append(TechSpec(**t))

        return {"techs": techs, "domain": data["domain"]}
