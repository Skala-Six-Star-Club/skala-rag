"""환경 변수 로더. .env.example의 키와 1:1로 대응함."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
GOLDEN_DATASET_PATH = Path(
    os.getenv("GOLDEN_DATASET_PATH", "./eval/golden/golden_dataset.json")
)
TECH_SELECTION_CONFIG_PATH = Path("./configs/tech_selection.json")

# 5장: 청킹 기본값 (RecursiveCharacterTextSplitter, 절 경계 내부 분할)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# 5장: 검색 기본값
DEFAULT_TOP_K = 5
