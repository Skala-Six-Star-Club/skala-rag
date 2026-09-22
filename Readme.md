# KV cache 최적화 기술 평가 보고서 생성 Agent

SW(TurboQuant)와 HW(ITME) 두 KV cache 최적화 기술을, 기술 성숙도·시장성·이해관계자·
도메인 적용 네 관점에서 비교하는 LangGraph 기반 Multi-Agent 평가 보고서 생성기.
설계 근거는 [docs/agentic-rag-design.md](docs/agentic-rag-design.md), 테스트 계획은
[docs/schedule.md](docs/schedule.md) 참고.

이 저장소는 **설계서의 10개 비즈니스 에이전트와 통합 Graph**를 갖추고 있음. RAG를
실제로 쓰는 3개(`tech_research`, `trl_eval`, `domain_eval`, 4·5장)와 RAG를 쓰지
않는 7개(`select_tech`, `market_eval`, `stakeholder_eval`, `evidence_check`,
`synthesize`, `judge`, `report`)가 각각 `src/agents/{agent}/`, `scripts/{agent}/`에
대응함. `src/graph.py`는 병렬 관점 실행, 최대 1회 재검색, Evidence ID 최종화,
종합·검수·보고서 흐름을 연결함.

> **검색 설정 채택안 (2026-09-22 비교실험, `scripts/{rag 3종}/*_report.md`)**
> 절 인식 청킹(800자/overlap 120) + `Qwen/Qwen3-Embedding-0.6B` + Query Rewriting 끔.
> 설계서 6.1절의 bge-m3와 7.2~7.4절의 리라이팅은 실측(Hit Rate@5, MRR)에서 각각
> Qwen3-Embedding에 뒤지고 원본 질의와 같거나 낮아 교체·비활성화함. 리라이팅은
> `.env`의 `QUERY_REWRITING=1`로 다시 켤 수 있음. RAG 3종은 이 설정으로 구현이 끝나
> 있으며 `python -m scripts.run_rag_agents`로 그래프 없이 실제 순서대로 돌려 볼 수 있음.

---

## 디렉토리 구조

```
skala-rag/
├── .env.example            # 환경 변수 템플릿
├── requirements.txt         # 팀 공통 패키지 버전
├── configs/
│   └── tech_selection.json  # select_tech가 읽는 기술 선정 결과 (3장, Human 기반)
├── docs/                    # 설계서, 테스트 계획 (원본)
├── assets/
│   └── fonts/NanumGothic-Regular.ttf  # report PDF 한글 폰트 (OFL, xhtml2pdf는 시스템 폰트로 폴백하지 않음)
├── data/
│   └── doc_pool/            # Doc Pool PDF 6편 (gitignore, 직접 내려받아 채움)
├── output/                  # report 산출물 report.md / report.pdf / report.json (gitignore)
├── src/
│   ├── graph.py               # 10개 노드를 잇는 LangGraph 그래프 (12장) — 통합 실행 진입점
│   ├── common/               # 10개 에이전트 + 테스트 스크립트가 공유하는 공통 모듈
│   │   ├── config.py          # .env 로더
│   │   ├── state.py           # LangGraph State (11장)
│   │   ├── evidence.py        # 임시 Evidence key 발급·최종 ID 확정
│   │   ├── models.py          # 생성/검수 LLM, 임베딩 로더 (6장)
│   │   ├── tools.py           # paper_search, web_search, summarize_sources, PDF 로딩/청킹
│   │   ├── base_agent.py      # 에이전트 노드 공통 인터페이스
│   │   ├── doc_pool.py         # Doc Pool 6편 스펙 (5장 표)
│   │   └── eval_utils.py       # Hit Rate@K/MRR·루브릭·단위테스트용 그래프 (테스트 스크립트 공용)
│   └── agents/                # 10개 에이전트 노드 (총 10개 폴더)
│       ├── select_tech/agent.py        # 7.1절 — RAG 미사용, 규칙 기반
│       ├── tech_research/agent.py      # 7.2절 — RAG 사용
│       ├── trl_eval/agent.py           # 7.3절 — RAG 사용
│       ├── market_eval/agent.py        # 7.5절 — RAG 미사용, 웹 검색만
│       ├── stakeholder_eval/agent.py   # 7.6절 — RAG 미사용, 웹 검색만
│       ├── domain_eval/agent.py        # 7.4절 — RAG 사용
│       ├── evidence_check/agent.py     # 7.7절 — RAG 미사용, 규칙 기반
│       ├── synthesize/agent.py         # 7.8절 — RAG 미사용, 생성만
│       ├── judge/agent.py              # 7.9절 — RAG 미사용, Qwen3-8B 이진 판정 + judge_passed()
│       └── report/agent.py             # 7.10절 — RAG 미사용, 챕터 직렬화 + 인용 안전장치 + MD/PDF/JSON
├── scripts/                  # 에이전트별 독립 테스트 환경 (총 10개, 서로 영향 없음)
│   ├── tech_research/
│   │   ├── pdf/v1/index/      # 절 인식 청킹 버전 FAISS 색인 (gitignore)
│   │   ├── pdf/v2/index/      # naive 슬라이싱 비교 버전 FAISS 색인 (gitignore)
│   │   ├── index_config.py    # 청킹/임베딩 비교실험 설정
│   │   ├── test_runner.py     # Hit Rate@5·MRR 측정 + 그래프 + 리포트 생성
│   │   └── tech_research_report.md  # 실행 결과물 (팀 공유용, git 추적함)
│   ├── trl_eval/, domain_eval/   # 위와 동일 구조(RAG 3종)
│   ├── select_tech/, evidence_check/, judge/   # 단위 테스트 / 스팟체크 test_runner.py
│   └── market_eval/, stakeholder_eval/, synthesize/, report/  # 8.2 루브릭 test_runner.py
└── eval/
    ├── generate_golden_dataset.py  # LLM 합성 Golden Dataset 생성 파이프라인 (8.4절)
    └── golden/golden_dataset.json  # 생성 결과 (팀 공용, git 추적함)
```

## 에이전트별 테스트 방식 (schedule.md 2절)

RAG 여부에 따라 `test_runner.py`가 검증하는 방식이 다르고, **"지금 바로 실행"과
"담당자가 TODO를 채운 뒤 실행"이 나뉨** — schedule.md 1절의 "만든다 → 돌린다 →
채점한다 → 기준 미달이면 프롬프트를 고쳐 재실행한다" 피드백 루프가 각 에이전트
개발 단계에서 도는 것이지, 지금 전부 일괄 실행하라는 뜻이 아님.

| 에이전트 | RAG | 검증 방식 | 지금 바로 실행 가능? |
|---|---|---|---|
| `select_tech`, `evidence_check` | X | 입력→기대 출력 단위 테스트 | **가능** — 규칙 기반이라 TODO 자체가 없음(이미 완성) |
| `report` 1부(인용 안전장치·챕터 직렬화·REFERENCE 표기·JSON/PDF) | X | 단위 테스트 | **가능** — 순수 함수라 API 키 불필요 |
| `judge` 1부(`judge_passed`) | X | 단위 테스트 | **가능** — graph.py 조건부 엣지용 순수 함수 |
| `tech_research`, `trl_eval`, `domain_eval` | O | Hit Rate@5·MRR, 청킹/임베딩/Query Rewriting 비교 (3.1~3.3절) | 색인/청킹 로직은 완성돼 있어 Doc Pool·API 키만 있으면 지금도 가능. TODO(구조화 추출)는 검증 대상 밖 |
| `judge`, `synthesize`, `report` 2부 | X | 이진 판정 스팟체크 / 8.2 루브릭 | 세 에이전트 모두 구조화 출력 호출까지 이미 구현돼 있어(자리표시자 아님) API 키만 있으면 지금도 실행 가능. 다만 프롬프트가 다듬어지기 전 초안이라 낮은 점수가 정상 — 담당자가 프롬프트를 고쳐 재실행하는 용도임 |
| `market_eval`, `stakeholder_eval` | X | 8.2절 LLM-as-a-Judge 루브릭(1~5점) | **담당자가 `agent.py`의 `TODO`(구조화 추출)를 채운 뒤** — 지금 돌리면 `by_tech`가 빈 자리표시자라 채점 자체가 무의미함 |

(실행 명령은 아래 "에이전트별 테스트 실행" 참고)

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `src/graph.py` | 12장 그래프 조립. `build_graph(nodes)`에 노드 이름→callable dict를 넣으면 `select_tech → tech_research → 관점 4종 × 기술 N(Send로 병렬) → evidence_check → (부족한 관점·기술 조합만 재검색 1회) → evidence_finalize → synthesize → judge → (위반 시 재작성 1회) → report` 순서로 잇고, `make_agents()`가 실제 에이전트 10개를 만들어 줌. `python -m src.graph`로 통합 실행 |
| `scripts/graph_flow_check.py` | API 키·PDF 없이 stub 노드로 그래프 흐름 검증(fan-out 8회, by_tech 병합, 기술 단위 재검색, key 충돌 없음, 재작성 1회). `python -m scripts.graph_flow_check` |
| `src/common/state.py` | `AgentState`(TypedDict) + `TechSpec`/`TechProfile`/`Evidence`/`Reference`/`ViewResult`/`Synthesis`/`JudgeFeedback`. 11장 표의 필드명·타입·갱신 방식(덮어쓰기/누적)을 그대로 구현함 |
| `src/common/evidence.py` | 병렬 수집용 provisional key 발급, 재시도 후 결정적 정렬·ID 부여, Claim/TechProfile/Reference remap |
| `src/common/models.py` | `get_generation_llm()`(GPT-5 mini), `get_judge_llm()`(Qwen3-8B, Ollama), `get_embedding_model()`(bge-m3). 3.1절 비교실험용 `get_embedding_model_by_name()` 포함 |
| `src/common/tools.py` | PDF 로딩(PyMuPDF) → 절 구조 인식(정규식) → 절 경계 내 청킹 → FAISS 색인(`build_doc_pool_index`), 공유 색인 로더(`get_shared_index`, `DOC_POOL_INDEX_DIR/<임베딩>/`), `paper_search`(role/camp/tech 사전 필터, 전체 벡터 대상), `web_search`(Tavily, 키 없으면 DuckDuckGo), `summarize_sources`, 관점 결과 구조화 추출(`extract_view_result`) |
| `scripts/run_rag_agents.py` | RAG 3종 스모크 실행(select_tech → tech_research → trl_eval, domain_eval → evidence_finalize). `--retry`로 재검색 패스까지 확인. 결과는 `output/rag_agents_smoke.md` |
| `src/common/base_agent.py` | `BaseAgent.run(state) -> dict` 하나만 구현하면 되는 노드 인터페이스. 모듈 함수 `rewrite_query()`(Pre-retrieval Query Rewriting, 7.2~7.4 공통)도 여기 있음 |
| `src/common/doc_pool.py` | Doc Pool 6편의 파일명·기술명·진영·역할·arXiv ID (5장 표) |
| `src/common/eval_utils.py` | `hit_rate_at_k`/`mrr`/`plot_bar_comparison`(RAG 3종), `score_with_rubric`/`plot_rubric_scores`(8.2 루브릭), `UnitCheck`/`plot_unit_checks`(단위 테스트) — 10개 `test_runner.py`가 공유하는 지표·그래프 유틸 |
| `src/graph.py` | 10개 비즈니스 에이전트와 내부 `evidence_finalize`를 연결하는 통합 Graph |
| `src/agents/{agent}/agent.py` | 실제 노드 구현. State 입출력 키는 고정돼 있고, `TODO` 표시된 LLM 구조화 추출/프롬프트 로직만 담당자가 채우면 됨(`select_tech`/`evidence_check`는 이미 완성돼 있음 — 규칙 기반이라 판단할 여지가 없음) |
| `configs/tech_selection.json` | `select_tech`가 읽는 기술 선정 결과(3장: TurboQuant/ITME, Human 기반 결정) |
| `eval/generate_golden_dataset.py` | Doc Pool PDF를 읽어 LLM으로 한국어 검색 질의 약 30개(문서당 5개)를 합성하고 정답 쪽 번호·키워드를 붙여 `golden_dataset.json`에 저장 |
| `scripts/{rag 3종}/index_config.py` | 그 에이전트의 임베딩 후보 목록(3.1절) |
| `scripts/{agent}/test_runner.py` | 위 "에이전트별 테스트 방식" 표에 따른 검증을 실행하고, PNG 그래프와 `{agent}_report.md`를 생성 |

## State 설계 요약

`src/common/state.py`가 11장 표를 그대로 구현함. 핵심만 짚으면:

- 병렬 노드는 `raw_evidence`, `raw_references`를 `Annotated[list[...], operator.add]` reducer로 누적함. `evidence_finalize`가 재시도까지 끝난 뒤 결정적인 순서로 정렬해 `evidence`와 `references`에 연속 ID를 부여함. 보고서와 인용 검증은 확정 영역을 사용함.
- `evidence_check`는 근거량·균형뿐 아니라 Claim의 provisional key/최종 ID와 인용문 임베딩을 확인하고, `perspective_confidence`를 계산해 `synthesize`에 전달함.
- `Synthesis`에는 관점별 신뢰도, 전체 평균(`overall_confidence`), 가장 낮은 관점(`weakest_perspective`)이 코드로 집계되어 저장됨.
- `Evidence.perspective`는 설계서 4개 관점(`trl`/`market`/`stakeholder`/`domain`)에 조사 단계인 `tech_research`를 더해 5가지 값을 가짐 — "조사와 관점 에이전트는 공통으로 evidence에도 기록함"(4장)을 반영
- `ViewResult`는 `by_tech: dict[기술명, TechViewResult]` 형태로 두 기술을 나란히 담음(9.5절 "두 기술을 나란히 서술")
- 관점 노드 4개는 `graph.py`가 `Send`로 (관점, 기술) 단위로 호출함. 각 호출은 state에 `tech_scope`(기술명 하나)를 받아 `BaseAgent.scoped_techs`로 그 기술만 처리하고, 같은 관점 필드(`trl_result` 등)에 동시에 쓰는 결과는 `merge_view_results` reducer가 기술 키로 합침. 독립 실행 스크립트처럼 `tech_scope` 없이 부르면 techs 전체를 처리함
- `retry_targets`에는 관점 코드(`trl`)가 아니라 실제 노드 이름(`trl_eval`)이 들어가고, `retry_scopes`에는 노드별로 부족한 기술 목록이 들어감 — `evidence_check`가 세 규칙을 기술별로 평가해 채우고, `graph.py`가 부족한 (노드, 기술)만 `Send`로 다시 실행하며, 각 관점 노드는 `self.name in state["retry_targets"]`로 재검색 초점을 바꿈(12장 "반복 1")
- `rewrite_count`는 11장 표에 없지만 `retry_count`의 짝으로 추가함 — `synthesize`가 `judge_feedback`을 받아 다시 쓴 횟수를 기록하고, `graph.py`의 `judge` 뒤 조건 분기가 이 값으로 재작성 예산(1회)을 확인함(12장 "반복 2")

---

## 개발 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env에 OPENAI_API_KEY, TAVILY_API_KEY 채우기
```

임베딩 장치는 `EMBEDDING_DEVICE=auto`(기본)면 cuda → mps(Apple Silicon) → cpu 순으로
자동 선택함. fp16 로드는 CUDA에서만 켜지고 MPS/CPU는 fp32로 동작함.

### Ollama 설치 및 실행 (검수 모델 Qwen3-8B, 6.3절)

`judge`, `synthesize`, `market_eval`, `stakeholder_eval`, `report` 2부를 돌리려면
로컬에 Ollama가 떠 있어야 함.

```bash
# 1. 설치
brew install ollama          # macOS
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download 에서 설치 프로그램 받기

# 2. 서버 실행 (macOS는 앱으로 설치하면 자동 실행됨. 터미널에서 직접 띄우려면:)
ollama serve &

# 3. 검수 모델 다운로드 (4비트 양자화, 약 5GB, 6.3절)
ollama pull qwen3:8b

# 4. 확인
ollama list                       # qwen3:8b가 보여야 함
curl http://localhost:11434       # "Ollama is running" 응답 확인
```

`.env`의 `OLLAMA_BASE_URL`(기본 `http://localhost:11434`)과 `OLLAMA_JUDGE_MODEL`
(기본 `qwen3:8b`)이 위 설정과 일치해야 함. 포트를 바꿨거나 원격 Ollama를 쓰면
`OLLAMA_BASE_URL`을 그에 맞게 고칠 것.

### Doc Pool 준비 (RAG 3종에만 필요)

`data/doc_pool/`에 아래 파일명으로 PDF 6편을 받아 둠(`src/common/doc_pool.py`의
arXiv ID 참고, 예: `https://arxiv.org/pdf/2504.19874` → `TurboQuant.pdf`).

| 파일명 | 기술 | arXiv |
|---|---|---|
| `TurboQuant.pdf` | TurboQuant | 2504.19874 |
| `CXL-based.pdf` | ITME | 2606.12556 |
| `DeepSeek-V2.pdf` | DeepSeek-V2 | 2405.04434 |
| `KIVI.pdf` | KIVI | 2402.02750 |
| `Dynamic KV Cache Mgmt.pdf` | InfiniGen | 2406.19707 |
| `PIM-CXL.pdf` | PIM/CXL | 2511.00321 |

파일명은 `src/common/doc_pool.py`의 `DOC_POOL_SPECS`에 고정돼 있음(코드가 그 이름을
그대로 찾음) — 다른 이름으로 받았다면 이 표대로 리네임하거나 `doc_pool.py`를 맞춰 고칠 것.

### Golden Dataset 생성 (최초 1회, 팀 공용, RAG 3종에만 필요)

```bash
python -m eval.generate_golden_dataset
```

`eval/golden/golden_dataset.json`을 생성해 git에 커밋함. 팀원 전원이 같은 질의셋으로
비교실험을 해야 결과가 비교 가능하므로, 이미 파일이 있으면 임의로 재생성하지 말 것.

### 에이전트별 테스트 실행 (팀원이 각자, 독립적으로)

**규칙 기반, TODO 없이 이미 완성됨 — 지금 바로 실행**:

```bash
python -m scripts.select_tech.test_runner
python -m scripts.evidence_check.test_runner
```

**LLM 호출까지 이미 구현됨 — API 키만 있으면 지금 실행 가능** (프롬프트 초안
단계라 점수가 낮게 나올 수 있음. 그건 실패가 아니라 "고쳐서 재실행"의 시작점):

```bash
python -m scripts.report.test_runner       # 1부(직렬화·REFERENCE·JSON·PDF)는 키 없이 실행, 2부는 OpenAI+Ollama 필요
python -m scripts.synthesize.test_runner   # OpenAI+Ollama 필요
python -m scripts.judge.test_runner        # 1부(judge_passed)는 키 없이 실행, 2부 스팟체크는 Ollama 필요
python -m scripts.tech_research.test_runner   # OpenAI API 키 + Doc Pool PDF + golden_dataset.json 필요
python -m scripts.trl_eval.test_runner
python -m scripts.domain_eval.test_runner
```

**담당자가 `agent.py`의 `TODO`(구조화 추출)를 채운 뒤에 실행** — 지금 돌리면
`by_tech`가 빈 `TechViewResult()`라 채점 자체가 무의미함:

```bash
python -m scripts.market_eval.test_runner
python -m scripts.stakeholder_eval.test_runner
```

RAG 3종의 `test_runner.py`는:

1. `scripts/{agent}/pdf/v1/index`(절 인식 청킹)와 `v2/index`(naive 슬라이싱)에
   FAISS 색인을 만들거나(최초 1회) 재사용함
2. `v1` 청킹 위에서 임베딩 후보(bge-m3/multilingual-e5-large/Qwen3-Embedding-0.6B)를
   비교함
3. Query Rewriting 적용 전/후를 비교함
4. `report_assets/*.png` 막대그래프와 `{agent}_report.md`를 저장함(둘 다 git 추적함 —
   팀원끼리 결과를 비교·공유하려고 커밋 대상으로 둠. 실행할 때마다 값이 바뀔 수
   있으니, 재실행 후에는 바뀐 리포트를 다시 커밋해서 최신 상태로 유지할 것)

모든 `test_runner.py`는 자신의 `scripts/{agent}/` 아래에만 쓰기 때문에 10개를 동시에
실행해도 서로 영향 없음.

### 통합 실행

```bash
python -m src.graph
```

`src/graph.py`가 Doc Pool 색인을 `data/doc_pool_index/`에 한 번 만들어 RAG 3종이
공유하게 하고(`.env`의 `DOC_POOL_INDEX_DIR`로 변경 가능, gitignore), 10개 노드를 12장
순서로 실행한 뒤 `output/report.md`를 쓰고 8.3절 Loop Efficiency 원자료(`retry_count`,
`rewrite_count`, 관점·기술별 지지/반대 근거 수, `judge_feedback`)를 JSON으로 출력함.
OpenAI·Tavily 키, Ollama(qwen3:8b), Doc Pool PDF 6편이 모두 필요함.

그래프 흐름만 확인하고 싶으면 `build_graph(nodes)`에 실제 에이전트 대신 같은 계약
(`state -> dict`)의 stub을 넣으면 됨 — API 키·PDF 없이 병렬 분기, 부분 재검색, 재작성
루프가 12장대로 도는지 검증할 수 있음.

### 다음 단계

- 각 `agent.py`의 `TODO` 채우기(담당자, 14장 역할 분담 참고) — LLM 구조화 추출과
  프롬프트 로직이 대상이며, `select_tech`/`evidence_check`는 규칙 기반이라 이미
  완성돼 있고 Graph와 Evidence ID 정책도 구현되어 있음
