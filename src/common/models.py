"""LLM 및 Embedding 로더. 6장 모델 선정 결과를 그대로 코드화함.

- 생성 LLM: OpenAI GPT-5 mini (6.2절)
- 검수 LLM: Qwen3-8B, Ollama 로컬 (6.3절)
- 임베딩: bge-m3, 로컬 (6.1절)
"""

from __future__ import annotations

import gc
from functools import lru_cache
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.common import config


@lru_cache(maxsize=1)
def get_generation_llm():
    """모든 에이전트가 공유하는 생성 LLM. Structured Outputs로 스키마를 강제함(6.2절).

    gpt-5-mini류 추론 모델은 temperature를 기본값(1)에서 바꾸는 걸 지원하지 않아
    (요청 시 400 에러) temperature를 아예 넘기지 않음.

    OPENAI_API_KEY가 없고 OLLAMA_GENERATION_MODEL이 설정돼 있으면 로컬 Ollama
    모델로 대체함(오프라인/GPU 서버 실행용 폴백). 이 경우 검수 모델과 계열이
    겹칠 수 있으니 리포트에 명시할 것.
    """
    if not config.OPENAI_API_KEY and config.OLLAMA_GENERATION_MODEL:
        return ChatOllama(
            model=config.OLLAMA_GENERATION_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            reasoning=False,  # qwen3 계열 thinking 비활성화(짧은 구조화 출력 용도)
        )
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


@lru_cache(maxsize=1)
def get_judge_llm(temperature: float = 0.0) -> ChatOllama:
    """judge 노드 전용 검수 LLM. 생성 모델과 계열을 분리해 자기 선호 편향을 피함(6.3절)."""
    return ChatOllama(
        model=config.OLLAMA_JUDGE_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        reasoning=False,  # qwen3 thinking 비활성화: 구조화 출력만 필요함
    )


def _embedding_model_kwargs() -> dict[str, Any]:
    """CUDA면 fp16으로 로드하고, 아니면 기본 dtype을 씀."""
    kwargs: dict[str, Any] = {"device": config.EMBEDDING_DEVICE}
    if config.EMBEDDING_FP16:
        import torch

        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return kwargs


def _build_embedding(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=_embedding_model_kwargs(),
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": config.EMBEDDING_BATCH_SIZE,
        },
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """bge-m3 임베딩. 벡터는 정규화 후 내적 유사도로 사용함(6.1절).

    EMBEDDING_DEVICE=auto(기본)면 CUDA 가용 시 GPU에 fp16으로 올림.
    """
    return _build_embedding(config.EMBEDDING_MODEL)


def get_embedding_model_by_name(model_name: str) -> HuggingFaceEmbeddings:
    """비교실험(3.1절)용: 후보 임베딩 모델을 이름으로 직접 로드함."""
    return _build_embedding(model_name)


def release_embedding_model(embedding: HuggingFaceEmbeddings | None) -> None:
    """비교실험에서 후보 모델을 바꿔 탈 때 GPU 메모리를 되돌려줌.

    24GB급 단일 GPU에 bge-m3(약 2.2GB fp16)·e5-large·Qwen3-Embedding을 차례로
    올릴 때 이전 모델이 남아 있으면 Ollama 검수 모델과 함께 OOM이 날 수 있음.
    """
    if embedding is None:
        return
    try:
        del embedding._client  # SentenceTransformer 인스턴스
    except AttributeError:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
