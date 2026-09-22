# select_tech 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.1절 / docs/schedule.md 2절
(LLM 미사용, 규칙 기반 노드라 단위 테스트로 검증함)

| 테스트 | 결과 | 비고 |
|---|---|---|
| 정상 설정 로드 | PASS |  |
| 잘못된 camp 값 거부 | PASS |  |
| 잘못된 role 값 거부 | PASS |  |
| domain 누락 거부 | PASS |  |

![단위 테스트 결과](report_assets/unit_checks.png)

## 결론

전부 통과함.
