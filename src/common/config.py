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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

DOC_POOL_DIR = Path(os.getenv("DOC_POOL_DIR", "./data/doc_pool"))
GOLDEN_DATASET_PATH = Path(
    os.getenv("GOLDEN_DATASET_PATH", "./eval/golden/golden_dataset.json")
)
TECH_SELECTION_CONFIG_PATH = Path("./configs/tech_selection.json")

# 5장: 청킹 기본값 (RecursiveCharacterTextSplitter, 절 경계 내부 분할)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# 5장: 검색 기본값
DEFAULT_TOP_K = 5

# 7.7절 확장(evidence_check 환각 검증): 주장(Claim)과 인용 근거(Evidence.quote) 간
# bge-m3 임베딩 코사인 유사도 임계값. 초기값은 경험적 휴리스틱이며(8.1절 Hit Rate@K
# 임계값과 같은 성격), Golden Dataset(8.4절)으로 실측 후 조정 대상임.
GROUNDING_MIN_SIMILARITY = float(os.getenv("GROUNDING_MIN_SIMILARITY", "0.35"))

# 관점당 목표 근거 수 5~8건(7.7절)의 하한 쪽을 관점별 신뢰도 만점 기준으로 씀.
TARGET_EVIDENCE_COUNT = 5
