"""Evidence 수집 중간 상태와 최종 번호 확정을 담당하는 공통 유틸리티.

관점 에이전트는 병렬로 실행되므로 수집 시점에 ``max(id) + 1``을 계산하면
동일 번호가 만들어질 수 있다. 각 에이전트는 이 모듈의 임시 key를 발급하고,
모든 재시도가 끝난 뒤 :func:`finalize_evidence`가 결정적인 순서로 정렬해
연속 정수 ID를 부여한다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import quote, unquote

from src.common.state import (
    AgentState,
    Claim,
    Evidence,
    Reference,
    TechProfile,
    TechViewResult,
    ViewResult,
)

_PERSPECTIVE_ORDER = {
    "tech_research": 0,
    "trl": 1,
    "market": 2,
    "stakeholder": 3,
    "domain": 4,
}


def make_provisional_key(
    perspective: str,
    tech: str,
    *,
    attempt: int = 0,
    ordinal: int,
) -> str:
    """병렬 수집 중 사용할 충돌 없는 Evidence key를 만든다.

    기술명은 URL 인코딩해 key 구분자인 ``:``가 기술명 안에 들어와도 파싱할 수
    있게 한다. ``ordinal``은 해당 에이전트 실행 내에서 증가시키는 값이다.
    """

    encoded_tech = quote(tech, safe="")
    return f"{perspective}:{attempt}:{encoded_tech}:{ordinal:06d}"


def _parse_key(key: str) -> tuple[str, int, str, int]:
    """정렬을 위해 임시 key를 구성 요소로 분해한다."""

    parts = key.split(":", 3)
    if len(parts) != 4:
        return ("legacy", 0, "", 0)
    perspective, attempt_text, encoded_tech, ordinal_text = parts
    try:
        attempt = int(attempt_text)
    except ValueError:
        attempt = 0
    try:
        ordinal = int(ordinal_text)
    except ValueError:
        ordinal = 0
    return perspective, attempt, unquote(encoded_tech), ordinal


def _as_evidence(item: Evidence | dict[str, Any]) -> Evidence:
    if isinstance(item, Evidence):
        return item
    return Evidence.model_validate(item)


def _as_reference(item: Reference | dict[str, Any]) -> Reference:
    if isinstance(item, Reference):
        return item
    return Reference.model_validate(item)


def _remap_claim(
    claim: Claim | dict[str, Any],
    key_to_id: dict[str, int],
    legacy_id_to_id: dict[int, int],
    valid_ids: set[int],
) -> Claim:
    claim = claim if isinstance(claim, Claim) else Claim.model_validate(claim)
    resolved_ids: list[int] = []

    for key in claim.evidence_keys:
        if key not in key_to_id:
            raise ValueError(f"Claim이 존재하지 않는 Evidence key를 참조합니다: {key}")
        if key_to_id[key] not in resolved_ids:
            resolved_ids.append(key_to_id[key])

    for old_id in claim.evidence_ids:
        resolved_id = legacy_id_to_id.get(old_id, old_id)
        if resolved_id not in valid_ids:
            raise ValueError(f"Claim이 존재하지 않는 Evidence ID를 참조합니다: {old_id}")
        if resolved_id not in resolved_ids:
            resolved_ids.append(resolved_id)

    return claim.model_copy(update={"evidence_ids": resolved_ids, "evidence_keys": []})


def _remap_view_result(
    result: ViewResult | dict[str, Any],
    key_to_id: dict[str, int],
    legacy_id_to_id: dict[int, int],
    valid_ids: set[int],
) -> ViewResult:
    result = result if isinstance(result, ViewResult) else ViewResult.model_validate(result)
    by_tech: dict[str, TechViewResult] = {}
    for tech, tech_result in result.by_tech.items():
        tech_result = (
            tech_result
            if isinstance(tech_result, TechViewResult)
            else TechViewResult.model_validate(tech_result)
        )
        by_tech[tech] = tech_result.model_copy(
            update={
                "confirmed_facts": [
                    _remap_claim(c, key_to_id, legacy_id_to_id, valid_ids)
                    for c in tech_result.confirmed_facts
                ],
                "counter_facts": [
                    _remap_claim(c, key_to_id, legacy_id_to_id, valid_ids)
                    for c in tech_result.counter_facts
                ],
            }
        )
    return result.model_copy(update={"by_tech": by_tech})


def _dedupe_references(references: Iterable[Reference | dict[str, Any]]) -> list[Reference]:
    """Reference를 결정적인 순서로 중복 제거하고 1부터 다시 번호를 붙인다."""

    unique: dict[tuple[str, ...], Reference] = {}
    for raw_reference in references:
        reference = _as_reference(raw_reference)
        identity = (
            ("url", reference.url)
            if reference.url
            else (
                "metadata",
                reference.type,
                reference.author_or_org,
                reference.year,
                reference.title,
            )
        )
        unique.setdefault(identity, reference)

    ordered = sorted(
        unique.values(),
        key=lambda ref: (
            ref.url or "",
            ref.title,
            ref.author_or_org,
            ref.year,
            ref.type,
        ),
    )
    return [reference.model_copy(update={"id": index}) for index, reference in enumerate(ordered, 1)]


def finalize_evidence(state: AgentState) -> dict[str, Any]:
    """재시도까지 끝난 Evidence를 확정하고 모든 참조를 ID로 remap한다.

    ``raw_evidence``가 주 입력이며, 레거시 단위 테스트나 수동 호출에서만 쓰는
    ``evidence``도 fallback으로 지원한다. 명시적인 임시 key가 중복되면 조용히
    덮어쓰지 않고 실패시켜 팀 계약 위반을 즉시 발견하게 한다.
    """

    raw_items = state.get("raw_evidence")
    if raw_items is None:
        raw_items = state.get("evidence", [])

    entries: list[tuple[str, Evidence]] = []
    seen_keys: set[str] = set()
    for index, raw_item in enumerate(raw_items or []):
        evidence = _as_evidence(raw_item)
        if evidence.key:
            key = evidence.key
            if key in seen_keys:
                raise ValueError(f"중복된 provisional Evidence key: {key}")
        else:
            # 레거시 입력은 각 원소의 위치까지 포함해 고유하게 만든다.
            old_id = "none" if evidence.id is None else str(evidence.id)
            key = f"legacy:{old_id}:{index:06d}"
            while key in seen_keys:
                key = f"{key}:dup"
        seen_keys.add(key)
        entries.append((key, evidence))

    tech_order = {
        tech.name: index for index, tech in enumerate(state.get("techs", []) or [])
    }

    def sort_key(item: tuple[str, Evidence]) -> tuple[Any, ...]:
        key, evidence = item
        perspective, attempt, key_tech, ordinal = _parse_key(key)
        return (
            _PERSPECTIVE_ORDER.get(perspective, 99),
            tech_order.get(key_tech or evidence.tech, 99),
            attempt,
            ordinal,
            key,
            evidence.tech,
            evidence.source,
        )

    entries.sort(key=sort_key)

    key_to_id: dict[str, int] = {}
    legacy_id_to_id: dict[int, int] = {}
    final_evidence: list[Evidence] = []
    for final_id, (key, evidence) in enumerate(entries, 1):
        key_to_id[key] = final_id
        if evidence.id is not None:
            legacy_id_to_id.setdefault(evidence.id, final_id)
        final_evidence.append(evidence.model_copy(update={"id": final_id, "key": key}))

    valid_ids = set(key_to_id.values())
    updates: dict[str, Any] = {
        "evidence": final_evidence,
        "evidence_finalized": True,
    }

    profiles: dict[str, TechProfile] = {}
    for tech, raw_profile in (state.get("tech_profiles", {}) or {}).items():
        profile = (
            raw_profile
            if isinstance(raw_profile, TechProfile)
            else TechProfile.model_validate(raw_profile)
        )
        profile_ids: list[int] = []
        for key in profile.evidence_keys:
            if key not in key_to_id:
                raise ValueError(f"TechProfile이 존재하지 않는 Evidence key를 참조합니다: {key}")
            if key_to_id[key] not in profile_ids:
                profile_ids.append(key_to_id[key])
        for old_id in profile.evidence_ids:
            resolved_id = legacy_id_to_id.get(old_id, old_id)
            if resolved_id not in valid_ids:
                raise ValueError(
                    f"TechProfile이 존재하지 않는 Evidence ID를 참조합니다: {old_id}"
                )
            if resolved_id not in profile_ids:
                profile_ids.append(resolved_id)
        profiles[tech] = profile.model_copy(
            update={"evidence_ids": profile_ids, "evidence_keys": []}
        )
    if state.get("tech_profiles") is not None:
        updates["tech_profiles"] = profiles

    for field in ("trl_result", "market_result", "stakeholder_result", "domain_result"):
        result = state.get(field)
        if result is not None:
            updates[field] = _remap_view_result(
                result, key_to_id, legacy_id_to_id, valid_ids
            )

    raw_references = list(state.get("raw_references") or [])
    raw_references.extend(state.get("references") or [])
    updates["references"] = _dedupe_references(raw_references)
    return updates


__all__ = ["finalize_evidence", "make_provisional_key"]
