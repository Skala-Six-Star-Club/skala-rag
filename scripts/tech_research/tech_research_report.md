# tech_research 테스트 리포트

설계 근거: docs/agentic-rag-design.md 5장·7.2절 / docs/schedule.md 3.1~3.3절

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

3개 RAG 에이전트가 공유하는 색인 설정이라, 이 수치는 trl_eval·domain_eval
리포트와 동일하게 나오는 게 정상임(전역 결정을 전체 골든셋으로 검증).

## 2. 임베딩 모델 비교 (3.1절, 전체 30개 골든셋)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| bge-m3 | 1.000 | 0.883 | 0.900 | 0.844 |
| multilingual-e5-large | 1.000 | 0.950 | 0.933 | 0.867 |
| qwen3-embedding-0.6b | 1.000 | 1.000 | 0.933 | 0.892 |

![임베딩 비교](report_assets/embedding_comparison.png)

이 수치도 trl_eval·domain_eval 리포트와 동일하게 나오는 게 정상임(위와 동일한 이유).

## 3. Query Rewriting 효과 (3.3절, tech_research 관점 질의만)

| 버전 | Hit Rate@5 (role=target) | MRR (role=target) | Hit Rate@5 (camp 전체) | MRR (camp 전체) |
|---|---|---|---|---|
| 원본 질의 | 1.000 | 1.000 | 1.000 | 1.000 |
| 리라이팅 질의 | 1.000 | 1.000 | 1.000 | 0.964 |

![Query Rewriting 비교](report_assets/query_rewriting_comparison.png)

리라이팅 질의 목록(실행마다 LLM 출력이 달라질 수 있어 변동 추적용으로 기록함):

| id | 원본 질의 | 리라이팅 질의 |
|---|---|---|
| 1 | TurboQuant의 목적과 초록에서 주장하는 핵심 기여는 무엇인가? 특히 'near-optimal distortion rates'와 온라인(data-oblivious) 특성에 대한 설명을 찾아라. | TurboQuant: objective and primary contributions stated in the abstract; explanation of its "near‑optimal distortion rates" and its online (data‑oblivious) properties |
| 4 | Theorem 2(Quantprod)의 보장을 요약하라: 기대 inner-product, inner-product distortion Dprod에 대한 상한식(특히 d와 b에 대한 스케일링)과 b=1,2,3,4일 때 근사 Dprod 값들을 찾아라. | Summarize the guarantees of Theorem 2 (Quantprod): bounds on the expected inner product and on the inner-product distortion D_prod—particularly the scaling with d and b—and give approximate values of D_prod for b = 1, 2, 3, 4. |
| 6 | 논문은 ITME가 SSD 기반 용량을 'direct-access memory expansion'으로 변환해 공유 컨텍스트 계층(T3.5)을 실현한다고 주장하는가? | Does the paper claim that ITME converts SSD‑backed capacity into a direct‑access memory expansion to realize a shared‑context layer (T3.5)? |
| 7 | Figure 2(및 관련 설명)는 모델 가중치와 장기 컨텍스트(KV) 캐시의 접근 동작과 어떤 레이어별 프리페칭(prefetching) 기회를 보여주는가? | Does Figure 2 (and its accompanying description) illustrate the access patterns of model weights and the long-term key–value (KV) context cache, and indicate layer-wise prefetching opportunities? |
| 11 | DeepSeek-V2의 전체 파라미터 수, 토큰당 활성화되는 파라미터 수와 지원되는 컨텍스트 길이는 각각 얼마인가? | What are DeepSeek‑V2's total parameter count, parameters activated per token, and maximum supported context length? |
| 12 | Table 1에서 능력이 'Weak'로 분류된 어텐션 메커니즘은 무엇인가? | Which attention mechanism is classified as 'Weak' in Table 1 of the DeepSeek-V2 paper? |
| 16 | KIVI가 해결하려는 주요 문제와 제안한 방법(특히 'tuning-free asymmetric 2bit quantization'의 핵심 아이디어)은 무엇인가요? | KIVI: problem addressed and proposed methodology, with emphasis on the core idea of "tuning-free asymmetric 2-bit quantization" |
| 19 | 논문에 제시된 메모리·속도 분석 예시(예: OPT-175B, batch size 512, prompt 512 등)에서 KV 캐시가 차지하는 용량 수치는 얼마이며 어떻게 비교되나요? | In the paper's memory and performance analyses (e.g., OPT-175B, batch size 512, context length 512), what is the KV-cache memory footprint and how does it compare to other model components and its impact on throughput/latency? |
| 20 | 저자들이 언급한 'attention sparsity'가 어떻게 토큰별 양자화가 채널별 양자화보다 더 작게 만드는지(OB2 현상에 대한 직관적 설명)는 무엇인가요? | What is the intuitive explanation (Observation 2 / OB2) for how attention sparsity, as described by the authors, causes per-token quantization to incur lower quantization error than per-channel quantization? |
| 21 | InfiniGen의 핵심 아이디어는 무엇인가? 특히 긴 텍스트 생성에서 KV 캐시 관리를 어떻게 하며 오프로딩 기반 추론 시스템과는 어떻게 연계되는지 확인 | InfiniGen: core concept, KV-cache management for long-context generation, and integration with offloading-based inference systems |
| 23 | InfiniGen을 다른 기법들(예: Full Cache, Quantization, H2O 등)과 성능 비교한 결과는 어떤 모델·벤치마크에서 제시되며 어떤 비교 대상들이 포함되어 있는지 확인 | Which models and benchmarks are used to present InfiniGen’s performance comparisons with other techniques (e.g., Full Cache, Quantization, H2O), and which baseline methods are included? |
| 25 | 논문에서 Transformer 블록 간 입력 유사성 시각화와 쿼리 행렬(예: OPT-13B의 Layer 18)에서 관찰된 채널/열 패턴은 어디에 설명되어 있는가? | Where do InfiniGen and related papers describe visualizations of input similarity across Transformer blocks and the channel/columnar patterns observed in query matrices (e.g., Layer 18 of OPT-13B)? |
| 26 | 논문 요약: 이 논문이 CXL 기반 KV-cache 관리로 1M-토큰 LLM 추론 확장 문제를 다루는가? | Abstract: Does this paper address scaling LLM inference to a 1M-token context via CXL-based KV-cache management? |
| 29 | KV-cache 관리 방법 분류: 논문이 eviction(영구 삭제) 방식과 dynamic selection 방식을 구분하고 StreamingLLM, H2O, InfiniPot 등 관련 작업을 언급하는가? | Classification of KV-cache management schemes: eviction (permanent deletion) vs. dynamic selection; mentions StreamingLLM, H2O, InfiniPot; PIM/CXL |


## 4. 결론

목표 임계값(Hit Rate@5 ≥ 0.8, MRR ≥ 0.6)을 만족함.

**채택안 vs 실측 최고 성능** (MRR (camp 전체) 기준):

- 청킹: 채택안은 v1_section_aware이지만, MRR (camp 전체) 기준 실측 최고 성능은 **v2_naive**임 — 6.1절/5장 절차대로 재검토 대상.
- 임베딩: 채택안(qwen3-embedding-0.6b)이 MRR (camp 전체) 기준으로도 최고 성능임.
- Query Rewriting: 채택안은 리라이팅 질의이지만, MRR (camp 전체) 기준 실측 최고 성능은 **원본 질의**임 — 6.1절/5장 절차대로 재검토 대상.

현재 설정은 v1_section_aware + qwen3-embedding-0.6b + Query Rewriting 적용을 기본값으로
채택했음(임베딩은 2026-09-22 비교실험 결과로 bge-m3에서 교체). 위에서 "재검토 대상"으로 표시된 항목이 있다면 6장·7.2절 설계를
재검토할 것 — 다만 골든셋이 30개(role=target 필터는 10개)뿐이라 차이가 통계적으로
확고한지는 표본을 늘려 다시 확인해 볼 것.
