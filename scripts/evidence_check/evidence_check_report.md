# evidence_check 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.7절 / docs/schedule.md 2절
(LLM 미사용, 규칙 기반 노드라 단위 테스트로 검증함)

| 테스트 | 결과 | 비고 |
|---|---|---|
| 충분한 근거 -> 재검색 없음 | PASS |  |
| 근거 부족(<3건, 규칙 1) -> 재검색 대상 | PASS |  |
| 반대 근거 0건(규칙 2) -> 재검색 대상 | PASS |  |
| 근거 수 비율 2배 초과(규칙 3) -> 재검색 대상 | PASS |  |
| 재시도 소진 -> 더 이상 재검색 안 함 | PASS |  |
| 출처 없는 주장(규칙 4) -> confirmed_facts에서 제거 | PASS |  |
| 존재하지 않는 근거 번호(규칙 4) -> 제거 | PASS |  |
| 근거와 무관한 주장(규칙 4) -> 제거 | PASS |  |
| 근거와 부합하는 주장(규칙 4) -> 유지됨 | PASS |  |
| perspective_confidence: 목표치 이상 -> 1.0, 목표치 미만 -> 비례 | PASS | A=1.0, B=0.4 |

![단위 테스트 결과](report_assets/unit_checks.png)

## 결론

전부 통과함.
