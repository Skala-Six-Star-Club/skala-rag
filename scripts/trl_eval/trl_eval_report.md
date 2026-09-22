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
| v1_section_aware | 1.000 | 1.000 | 0.933 | 0.892 |
| v2_naive | 1.000 | 1.000 | 1.000 | 0.967 |

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
| 원본 질의 | 1.000 | 1.000 | 0.889 | 0.750 |
| 리라이팅 질의 | 1.000 | 1.000 | 0.889 | 0.833 |

![Query Rewriting 비교](report_assets/query_rewriting_comparison.png)

리라이팅 질의 목록(실행마다 LLM 출력이 달라질 수 있어 변동 추적용으로 기록함):

| id | 원본 질의 | 리라이팅 질의 |
|---|---|---|
| 2 | 논문에서 GPU 같은 하드웨어 가속기에서 성능 저하의 원인으로 지적한 알고리즘적 특징은 무엇인가? 'lack of vectorization' 관련 진술을 찾아라. | "TurboQuant" AND ("GPU" OR "hardware accelerator" OR "accelerator") AND ("performance degradation" OR "performance loss" OR "performance slowdown") AND ("algorithmic characteristics" OR "algorithmic factors" OR "algorithmic limitations") AND ("lack of vectorization" OR "non-vectorizable" OR "not vectorized" OR "insufficient vectorization") |
| 9 | ITME는 어떤 소프트웨어/실험 환경에서 구현·평가되었는가(예: vLLM 기반) 및 Recomp./All-caching 설정과 메모리 사용 비교는 어떻게 보고되는가? | In which software/experimental environments is ITME implemented and evaluated (e.g., vLLM‑based), and how are recomputation versus all‑caching configurations and memory usage reported and compared? |
| 10 | 논문은 FPGA 기반 ITME 프로토타입을 제시하는가? FPGA 프로토타입과 CMM 기반 평가 간의 읽기/쓰기 대역폭 비교 및 읽기 우선 스케줄링 관련 결과를 보고하는가? | Does the paper present an FPGA-based ITME prototype and provide a comparative evaluation of read/write bandwidth against a CMM-based evaluation, including results on read-priority scheduling? |
| 13 | 사람 선호에 맞추기 위한 강화학습에 논문에서 사용한 알고리즘 이름은 무엇이며, 선택 이유(비용 절감 관련)는 무엇인가? | Which reinforcement learning algorithm was used for human-preference alignment, and what rationale—particularly with respect to cost reduction—motivated its selection? |
| 15 | MMLU Humanity-Moral 서브셋에 대한 세 명의 주석가 간 일치도는 어떠했으며, 논문이 이로부터 내린 결론은 무엇인가? | Inter-annotator agreement (three annotators) on the MMLU Humanity–Moral subset and the conclusions drawn in DeepSeek-V2 |
| 17 | KIVI의 성능 검증 근거는 무엇인가요? LongBench나 논문 내 표(예: Table 4)를 통해 실험 결과가 어떻게 제시되었나요? | Empirical validation of KIVI: evidence for performance and presentation of experimental results (e.g., LongBench evaluations and in-paper tables such as Table 4) |
| 24 | 실험적 근거(TRL)를 확인하려면 OPT-13B, 시퀀스 길이 2048 배치 8에서 Transformer 블록의 지연 시간 분해(latency breakdown)와 데이터 전송이 차지하는 비중은 어디에 제시되어 있는가? | To validate the paper's TRL claims, where does InfiniGen report the latency breakdown of Transformer blocks and the fraction of runtime attributable to data transfer for OPT‑13B (sequence length = 2048, batch size = 8)? |
| 27 | 서버 수준 평가 포함 여부: In-server PNM과 KV-cache 오프로드의 처리량 및 에너지 비교 실험을 제시하는가? | Does the paper include a server-level evaluation comparing throughput and energy of in-server PNM versus KV-cache offloading? |
| 30 | Steady Selection 실험 결과 확인: PNM 장치 수 증가 시 recall 수가 급감하고 GPU 이용률이 개선된다는 결과를 제시하는가? | Does the "Steady Selection" experiment report that increasing the number of PNM devices causes a sharp decline in recall while improving GPU utilization? |


## 4. 결론

목표 임계값(Hit Rate@5 ≥ 0.8, MRR ≥ 0.6)을 만족함.

**채택안 vs 실측 최고 성능** (MRR (camp 전체) 기준):

- 청킹: 채택안은 v1_section_aware이지만, MRR (camp 전체) 기준 실측 최고 성능은 **v2_naive**임 — 6.1절/5장 절차대로 재검토 대상.
- 임베딩: 채택안(qwen3-embedding-0.6b)이 MRR (camp 전체) 기준으로도 최고 성능임.
- Query Rewriting: 채택안은 원본 질의이지만, MRR (camp 전체) 기준 실측 최고 성능은 **리라이팅 질의**임 — 6.1절/5장 절차대로 재검토 대상.

trl_eval은 재검색 시 질의 초점을 "구현/공식 발표"에서 "한계·실패 사례·후속
검증"으로 바꾸는 로직(7.3절)이 있으므로, 위 수치는 1차 패스(재검색 전) 기준임.
현재 채택 임베딩은 qwen3-embedding-0.6b임(2026-09-22 비교실험 결과로 bge-m3에서 교체).
골든셋이 30개(role=target 필터는 10개)뿐이라 차이가 통계적으로 확고한지는
표본을 늘려 다시 확인해 볼 것.
