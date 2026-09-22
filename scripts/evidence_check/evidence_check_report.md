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

![단위 테스트 결과](report_assets/unit_checks.png)

## 결론

전부 통과함.
