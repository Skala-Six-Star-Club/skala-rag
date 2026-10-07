"""보고서 품질 평가 에이전트. 보고서 생성 뒤에 실행하고, 미달이면 Loop를 요청함.

평가 방식은 Hybrid(규칙 + LLM Judge). 판정 대상은 report가 남긴 ``report.json``의
``evidence_token_sections``(인용이 [근거#N] 토큰으로 남아 있는 본문)임.

| 항목 | 방식 | 판정 |
|---|---|---|
| groundedness | 규칙 + 임베딩 | 인용 번호가 모두 실제 근거인지, 인용이 붙은 문장과 인용문의 코사인 유사도 |
| neutrality | 규칙 + LLM | 금지 표현 목록, 검수 LLM(Qwen3-8B)의 우열 판정 표현과 근거 없는 사실 문장 |
| bias_control | 규칙 | (관점, 기술)별 인용 출처 수, 기술별 반대 근거 인용, 기술 간 인용 수 비율 |
| coverage | 규칙 | 관점별 평가 절에 4개 관점 x 기술 모든 칸의 인용이 있는지 |

미달 시 되돌아갈 곳
- coverage, bias_control 미달: orchestrator로 부족한 (관점, 기술)만 re-plan (MAX_QUALITY_REPLANS회)
- groundedness, neutrality 미달: synthesize로 재작성 (MAX_QUALITY_REWRITES회)
- 두 예산은 따로 셈. 둘 다 미달이면 재조사를 먼저 하고, 재조사 뒤 다시 들어온 synthesize가 서술 지적도 반영함
- 예산을 소진했으면 종료하고 미달 내용을 report.json과 보고서 한계점에 남김
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from src.common import config
from src.common.base_agent import BaseAgent
from src.common.models import get_embedding_model, get_judge_llm
from src.common.observability import log_decision
from src.common.state import AgentState, CriterionResult, EvalVerdict, Evidence, Gap

_CITATION_RE = re.compile(r"\[근거#(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# "문장임. [근거#3]"처럼 마침표 뒤에 붙은 인용을 마침표 앞으로 옮겨 다음 문장에 붙지 않게 함
_TRAILING_CITATION_RE = re.compile(r"([.!?])\s*((?:\[근거#\d+\]\s*)+)")
_VIEW_SECTION = "4. 관점별 평가"
_NARRATIVE_SECTIONS = ("SUMMARY", "5. 시사점")
_PERSPECTIVES = ("trl", "market", "stakeholder", "domain")

# 우열 판정, 추천 표현. LLM Judge와 별개로 결정론적으로 걸러냄
_BIASED_PATTERNS = [
    r"더\s*(우수|낫|뛰어나|유리)",
    r"(우월|열등)",
    r"(압도적|명백히)\s*앞",
    r"도입을\s*(추천|권장)",
    r"(추천|권장)한다",
    r"뒤처",
    r"갈 길이 멀",
]

_JUDGE_PROMPT = """\
당신은 두 기술의 다관점 평가 보고서에서 중립성만 검사하는 검수자임.
사실 여부는 검사하지 않고 표현과 형식만 봄. 점수를 매기지 말고 구조화된 필드로만 답함.

[biased_sentences] 한 기술이 전반적으로 더 낫다고 결론짓거나 도입을 추천하는 문장을 그대로 옮기고 kind를 붙임.
kind는 "우열 판정"(두 기술 중 하나가 전반적으로 낫다고 결론), "도입 추천"(특정 기술 도입을 권함),
"해당 없음"(의심했지만 위 두 경우가 아님) 중 하나.
두 기술의 특징, 반응, 제약을 "A는 ~한 반면 B는 ~"처럼 대비해 서술하는 문장은 우열 판정이 아님.
아래 예시의 "기술 A", "기술 B"는 설명용 이름이며 검사 대상 문장이 아님. 예시를 답에 옮기지 말 것.
- 해당 예: "기술 A가 기술 B보다 실용적이므로 도입을 권장함."
- 해당 예: "기술 B는 기술 A에 비해 훨씬 앞선 기술임."
- 해당 아님: "기술 A는 투자자 불확실성을 유발한 반면, 기술 B는 설비투자 흐름과 연결되어 평가됨."
[sentences_without_evidence] [근거#N] 표기가 하나도 없는 사실 주장 문장을 그대로 나열.
접속 문장, 안내 문장, 관점 이름만 적은 문장, 앞 문장의 근거를 요약만 하는 결론 문장은 제외.
[notes] 판정 근거 한두 문장.

===== 검사 대상 =====
{text}
"""
