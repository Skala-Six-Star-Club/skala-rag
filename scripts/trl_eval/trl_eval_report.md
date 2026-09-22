# trl_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.3절 / docs/schedule.md 3.1~3.3절
(TRL 판단 근거인 "실험 수준, 공개 구현 여부"를 얼마나 잘 찾아오는지를 검증함)

모든 표는 두 필터 조합을 각각 채점함(8.1절 "paper_search가 실제로 쓰는 필터
조합 각각에 대해 별도로 측정"): `role=target`은 기술 개요·판단 질의가 쓰는
좁은 필터(대상 기술 2편만), `camp 전체`는 "같은 진영 다른 방식과의 차이" 질의가
쓰는 넓은 필터(해당 진영 3편 모두)임.

## 1. 청킹 전략 비교 (3.2절 — 절 인식 vs naive 슬라이싱, 전체 30개 골든셋)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| v1_section_aware | 1.000 | 0.883 | 0.900 | 0.844 |
| v2_naive | 1.000 | 0.950 | 0.967 | 0.933 |

![청킹 비교](report_assets/chunking_comparison.png)

3개 RAG 에이전트가 공유하는 색인 설정이라, 이 수치는 tech_research·domain_eval
리포트와 동일하게 나오는 게 정상임(전역 결정을 전체 골든셋으로 검증).

## 2. 임베딩 모델 비교 (3.1절, 전체 30개 골든셋)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| bge-m3 | 1.000 | 0.883 | 0.900 | 0.844 |
| multilingual-e5-large | 1.000 | 0.950 | 0.933 | 0.867 |
| qwen3-embedding-0.6b | 1.000 | 1.000 | 0.933 | 0.892 |

![임베딩 비교](report_assets/embedding_comparison.png)

이 수치도 tech_research·domain_eval 리포트와 동일하게 나오는 게 정상임(위와 동일한 이유).

## 3. Query Rewriting 효과 (3.3절, trl_eval 관점 질의만)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| 원본 질의 | 1.000 | 0.833 | 0.889 | 0.833 |
| 리라이팅 질의 | 1.000 | 0.833 | 0.889 | 0.833 |

![Query Rewriting 비교](report_assets/query_rewriting_comparison.png)

## 4. 결론

목표 임계값(Hit Rate@5 ≥ 0.8, MRR ≥ 0.6)을 만족함.

trl_eval은 재검색 시 질의 초점을 "구현/공식 발표"에서 "한계·실패 사례·후속
검증"으로 바꾸는 로직(7.3절)이 있으므로, 위 수치는 1차 패스(재검색 전) 기준임.
