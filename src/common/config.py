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
    """EMBEDDING_DEVICE 미설정/auto면 CUDA 가용 여부로 자동 결정함."""
    requested = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = _detect_device()
# GPU에서는 배치를 키워 임베딩 처리량을 올림(CPU 기본값 32는 그대로 둠).
EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "64" if EMBEDDING_DEVICE.startswith("cuda") else "32")
)
# CUDA에서는 fp16으로 로드해 메모리와 시간을 절반 가까이 줄임.
EMBEDDING_FP16 = os.getenv("EMBEDDING_FP16", "1") == "1" and EMBEDDING_DEVICE.startswith("cuda")

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
