# domain_eval 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.4절 / docs/schedule.md 3.1~3.3절
(에이전트 코딩 멀티턴 서빙 도입 조건의 근거인 "실험 환경, 요구 하드웨어"를
얼마나 잘 찾아오는지를 검증함)

모든 표는 두 필터 조합을 각각 채점함(8.1절 "paper_search가 실제로 쓰는 필터
조합 각각에 대해 별도로 측정"): `role=target`은 기술 개요·판단 질의가 쓰는
좁은 필터(대상 기술 2편만), `camp 전체`는 "같은 진영 다른 방식과의 차이" 질의가
쓰는 넓은 필터(해당 진영 3편 모두)임.

## 1. 청킹 전략 비교 (3.2절 — 절 인식 vs naive 슬라이싱, 전체 30개 골든셋)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| v1_section_aware | 1.000 | 1.000 | 0.933 | 0.892 |
| v2_naive | 1.000 | 1.000 | 1.000 | 0.967 |

![청킹 비교](report_assets/chunking_comparison.png)

3개 RAG 에이전트가 공유하는 색인 설정이라, 이 수치는 tech_research·trl_eval
리포트와 동일하게 나오는 게 정상임(전역 결정을 전체 골든셋으로 검증).

## 2. 임베딩 모델 비교 (3.1절, 전체 30개 골든셋)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| bge-m3 | 1.000 | 0.883 | 0.900 | 0.844 |
| multilingual-e5-large | 1.000 | 0.950 | 0.933 | 0.867 |
| qwen3-embedding-0.6b | 1.000 | 1.000 | 0.933 | 0.892 |

![임베딩 비교](report_assets/embedding_comparison.png)

이 수치도 tech_research·trl_eval 리포트와 동일하게 나오는 게 정상임(위와 동일한 이유).

## 3. Query Rewriting 효과 (3.3절, domain_eval 관점 질의만)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| 원본 질의 | 1.000 | 1.000 | 0.857 | 0.857 |
| 리라이팅 질의 | 1.000 | 0.833 | 0.857 | 0.786 |

![Query Rewriting 비교](report_assets/query_rewriting_comparison.png)

리라이팅 질의 목록(실행마다 LLM 출력이 달라질 수 있어 변동 추적용으로 기록함):

| id | 원본 질의 | 리라이팅 질의 |
|---|---|---|
| 3 | 식(3)과 그 전후 설명에서 zi(및 평균화된 샘플)의 분산이 d에 대해 어떻게 스케일하는지(즉 분산이 1/d 또는 다른 형태로 감소하는지)를 보여주는 수식을 찾아라. | Locate an equation in or around Eq. (3) that characterizes how Var(z_i) (and the sample mean) scale with d (i.e., whether the variance decays as 1/d or follows a different dependence). |
| 5 | Figure 5의 실험 비교 설정을 확인하라: 어떤 데이터셋(예: GloVe, OpenAI3)과 embedding 차원(d 값들), 비교 대상 방법들(PQ, RabitQ 등), 그리고 사용된 비트수(예: 2 bits, 4 bits)는 무엇인가? | In "TurboQuant" (Figure 5), what are the experimental comparison settings: which datasets (e.g., GloVe, OpenAI3), embedding dimensionalities (values of d), compared methods (PQ, RabitQ, etc.), and quantization bit‑widths (e.g., 2‑bit, 4‑bit)? |
| 8 | 논문은 DRAM 캐시를 64 GB로 확장할 때 SRAM 요구량이 얼마나 증가한다고 보고하며, 그 이유로 32 GB를 기준으로 선택했다고 명시하는가? | Does the paper quantify the increase in SRAM requirements when scaling the DRAM cache to 64 GB and explicitly state that 32 GB was chosen as the baseline for that comparison? |
| 14 | 참고문헌에서 'trillion parameter' 모델 훈련을 위한 메모리 최적화 논문으로 언급된 작업의 이름은 무엇인가? | Which referenced work is cited as a memory-efficient training method for trillion-parameter models? |
| 18 | 논문에서 제시한 KIVI 알고리즘의 전체 파이프라인은 어떻게 구성되나요? (예: Figure 3의 Q_MatMul, Prefill/Decoding 단계와 채널/토큰별 양자화 흐름) | How is the end-to-end pipeline of the KIVI algorithm structured in the paper (e.g., Figure 3: Q_MatMul, prefill/decoding stages, and per-channel and per-token quantization flows)? |
| 22 | 논문에서 비교한 Transformer 블록 실행 스타일들(Full GPU, KV cache on CPU, Prefetch KV cache, Prefetch critical KV) 간의 차이와 각 실행 흐름(Load Cache / Attention / FFN)의 타이밍 비교는 어디에 나오는가? | Where in the InfiniGen paper are the differences between Transformer block execution modes (Full GPU, KV cache on CPU, Prefetch KV cache, Prefetch critical KV) and the per-stage timing breakdowns for Load Cache, Attention, and FFN reported? |
| 28 | 하드웨어/구성 확인: VPU(벡터 처리 유닛)의 아키텍처와 설정(예: Configurable Array/Tree)이 설명되어 있으며 CXL-PNM 컨트롤러가 LPDDR5X와 AXI4 기반 중재로 메모리 접근을 조율하는가? | Hardware/configuration verification: Are the VPU (vector processing unit) microarchitecture and configuration (e.g., configurable array/tree topology) described, and does a CXL‑PNM controller arbitrate memory access to LPDDR5X over an AXI4‑based interconnect? |


## 4. 결론

목표 임계값(Hit Rate@5 ≥ 0.8, MRR ≥ 0.6)을 만족함.

**채택안 vs 실측 최고 성능** (MRR (camp 전체) 기준):

- 청킹: 채택안은 v1_section_aware이지만, MRR (camp 전체) 기준 실측 최고 성능은 **v2_naive**임 — 6.1절/5장 절차대로 재검토 대상.
- 임베딩: 채택안(qwen3-embedding-0.6b)이 MRR (camp 전체) 기준으로도 최고 성능임.
- Query Rewriting: 채택안(원본 질의)이 MRR (camp 전체) 기준으로도 최고 성능임.

운영 중인 domain_eval은 여기에 더해 "실험 환경"/"평가" 절 우선 재랭킹을 적용함
(7.4절 2번 항목). 이 재랭킹은 별도 실험 없이 채택된 장치라, 위 수치와 별개로
실제 운영 결과에서 순위 개선 여부를 추가로 관찰할 것(schedule.md 3.2절 참고).
현재 채택 임베딩은 qwen3-embedding-0.6b임(2026-09-22 비교실험 결과로 bge-m3에서 교체).
골든셋이 30개(role=target 필터는 10개)뿐이라 차이가 통계적으로 확고한지는
표본을 늘려 다시 확인해 볼 것.
