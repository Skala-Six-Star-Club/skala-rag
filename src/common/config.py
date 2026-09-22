"""환경 변수 로더. .env.example의 키와 1:1로 대응함."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# override=True: .env 값이 셸에 이미 깔려 있는(예: 다른 프로젝트에서 export한 낡은
# 키) 동일 이름 환경변수보다 항상 우선하게 함. 기본값(override=False)이면
# load_dotenv()가 기존 환경변수를 덮어쓰지 않아, .env를 고쳐도 조용히 무시되는
# 문제가 생길 수 있음.
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "qwen3:8b")
# OPENAI_API_KEY가 비어 있을 때 생성 LLM을 대신할 로컬 Ollama 모델(선택).
# 비워 두면 OpenAI 키가 없을 때 그대로 에러가 남.
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "")


def _detect_device() -> str:
    """EMBEDDING_DEVICE 미설정/auto면 cuda -> mps -> cpu 순으로 가용한 장치를 고름.

    - cuda: Linux/Windows NVIDIA GPU
    - mps: Apple Silicon Mac (Metal). torch.backends.mps.is_available()로 확인
    - cpu: 그 외
    명시적으로 cpu / cuda / cuda:0 / mps를 넣으면 그대로 씀.
    """
    requested = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"


# 6.1절 채택 임베딩. 2026-09-22 비교실험(3.1절)에서 MRR 기준 최상위였던
# Qwen3-Embedding-0.6B로 교체함(bge-m3 대비 camp 전체 MRR 0.844 -> 0.892).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DEVICE = _detect_device()
EMBEDDING_IS_CUDA = EMBEDDING_DEVICE.startswith("cuda")
EMBEDDING_IS_MPS = EMBEDDING_DEVICE == "mps"
# CUDA에서는 배치를 키워 임베딩 처리량을 올림. MPS는 통합 메모리를 CPU와 나눠 쓰므로
# CPU 기본값(32)을 유지함.
EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE") or ("64" if EMBEDDING_IS_CUDA else "32")
)
# fp16은 CUDA에서만 켬. MPS는 fp16 연산 정밀도 문제(일부 모델에서 NaN)가 보고돼
# 기본 dtype(fp32)로 두고, EMBEDDING_FP16=1을 명시해도 CUDA가 아니면 무시함.
EMBEDDING_FP16 = os.getenv("EMBEDDING_FP16", "1") == "1" and EMBEDDING_IS_CUDA

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

DOC_POOL_DIR = Path(os.getenv("DOC_POOL_DIR", "./data/doc_pool"))
# 그래프·에이전트 공유 색인 루트. 실제 색인은 <루트>/<임베딩 이름>/ 아래에 두어
# 채택 임베딩을 바꿔도 이전 모델 색인이 섞여 로드되지 않게 함(tools.get_shared_index).
DOC_POOL_INDEX_DIR = Path(
    os.getenv("DOC_POOL_INDEX_DIR", "./data/doc_pool_index")
)
GOLDEN_DATASET_PATH = Path(
    os.getenv("GOLDEN_DATASET_PATH", "./eval/golden/golden_dataset.json")
)
TECH_SELECTION_CONFIG_PATH = Path("./configs/tech_selection.json")
# 7.2~7.4 Pre-retrieval Query Rewriting. 2026-09-22 3차 비교실험에서 Qwen3-Embedding
# 위에서는 리라이팅이 원본 질의보다 같거나 낮아(출력 형식 불안정 포함) 기본값을 끔.
# 켜려면 QUERY_REWRITING=1. 테스트 러너의 3.3절 비교실험은 이 값과 무관하게 전/후를 모두 잼.
QUERY_REWRITING = os.getenv("QUERY_REWRITING", "0") == "1"

# 5장: 청킹 기본값 (RecursiveCharacterTextSplitter, 절 경계 내부 분할)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# 5장: 검색 기본값
DEFAULT_TOP_K = 5

# 7.7절 확장(evidence_check 환각 검증): 주장(Claim)과 인용 근거(Evidence.quote) 간
# bge-m3 임베딩 코사인 유사도 임계값. 초기값은 경험적 휴리스틱이며(8.1절 Hit Rate@K
# 임계값과 같은 성격), Golden Dataset(8.4절)으로 실측 후 조정 대상임.
GROUNDING_MIN_SIMILARITY = float(os.getenv("GROUNDING_MIN_SIMILARITY", "0.35"))

# 관점당 목표 근거 수 5~8건의 하한을 신뢰도 만점 기준으로 사용한다.
# 재검색 규칙의 최소 3건과는 별도의 값이다.
TARGET_EVIDENCE_COUNT = int(os.getenv("TARGET_EVIDENCE_COUNT", "5"))

# 7.10 report 본문의 인용 표기 방식. 내부 검증(인용 안전장치, REFERENCE 집계)은 항상
# [근거#N] 토큰으로 하고, 출력 직전에만 바꿈.
#   numeric: REFERENCE 번호로 표기 — [3], 여러 자료면 [1, 3], 논문은 쪽 번호 포함 [1, p.3]
#   none:    본문에서 인용 표기를 모두 제거(REFERENCE 절과 report.json의 근거 번호는 유지)
#   raw:     [근거#N] 토큰을 그대로 둠(디버깅용)
REPORT_CITATION_STYLE = os.getenv("REPORT_CITATION_STYLE", "numeric").strip().lower()
