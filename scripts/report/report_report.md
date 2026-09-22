# report 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.10절, 9.5절, 13장 / docs/schedule.md 2절

## 1부: 단위 테스트 (인용 안전장치 · 챕터 직렬화 · REFERENCE 표기 · JSON/PDF)

| 테스트 | 결과 | 비고 |
|---|---|---|
| 환각 근거 번호 -> 원문으로 되돌림 | PASS | TurboQuant는 채널당 3.5비트에서 품질 저하가 없다고 보고됨[근거#1]. |
| 유효 근거 번호 -> 다듬은 문장 유지 | PASS | TurboQuant는 채널당 3.5비트 수준에서 품질 저하가 거의 없다고 보고됨[근거#1]. |
| 기술 선정: 두 기술명 + Human 기반 명시 | PASS | | 기술 | 진영 | 역할 | 검색 앵커 |
|---|---|---|---|
| TurboQuant | SW | target  |
| 관점별 평가: 4관점 + TRL 추정 라벨 + 관점마다 두 기술 | PASS | 블록 4개 |
| 관점별 평가: 모든 주장에 근거 번호 | PASS | ### 4.1 기술 성숙도 (TRL, 공개 정보 기반 추정)

**TurboQuant**
- (확인) 공개 구현이 vLLM에  |
| 시사점: 일치점·상충점 각 1건 + 상충 이유 | PASS | ### 관점 간 일치점

- 두 기술 모두 실사용 검증 부족[근거#1][근거#3]

### 관점 간 상충점

- **비용**: |
| 한계점: 공개 정보 한계 + 미확인 항목 + 편향 조치 + 검수 반영 | PASS | ### 공개 정보 기반 추정의 한계

본 보고서의 모든 평가는 논문, 기사, 제조사 발표 자료 등 공개 정보에 한정한 추정임. |
| 논문: 저자(YYYY). 제목. *학술지*. | PASS | Zandieh, A. et al.(2025). TurboQuant. *arXiv, 2504.19874*. |
| 특허: 출원인(YYYY-MM). *특허명*, 번호, URL | PASS | SK hynix(2026-01). *계층 메모리*, KR10-2026-0000001, https://x |
| 웹: 작성자(YYYY-MM-DD). *제목*. 사이트, URL | PASS | vLLM Team(2026-05-11). *TurboQuant Study*. vLLM Blog, https://vllm.ai |
| JSON: 왕복 + 필수 키 + judge_passed | PASS | cited_evidence_ids, domain, generated_at, judge_feedback, judge_passed |
| PDF: 변환 성공 + 한글 임베딩 | PASS | ok=True, size=22910, 한글 추출=True |

![단위 테스트](report_assets/citation_checks.png)

단위 테스트 전부 통과함.

## 2부: LLM-as-a-Judge 루브릭 (8.2절)

(2부 미실행 — API 키/Ollama 설정 후 재실행할 것)

## 검사 대상 보고서 초안

```

```
