"""LLM 및 Embedding 로더. 6장 모델 선정 결과를 그대로 코드화함.

- 생성 LLM: OpenAI GPT-5 mini (6.2절)
- 검수 LLM: Qwen3-8B, Ollama 로컬 (6.3절)
- 임베딩: bge-m3, 로컬 (6.1절)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.common import config


@lru_cache(maxsize=1)
def get_generation_llm() -> ChatOpenAI:
    """모든 에이전트가 공유하는 생성 LLM. Structured Outputs로 스키마를 강제함(6.2절).

    gpt-5-mini류 추론 모델은 temperature를 기본값(1)에서 바꾸는 걸 지원하지 않아
    (요청 시 400 에러) temperature를 아예 넘기지 않음.
    """
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
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """bge-m3 임베딩. 벡터는 정규화 후 내적 유사도로 사용함(6.1절)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embedding_model_by_name(model_name: str) -> HuggingFaceEmbeddings:
    """비교실험(3.1절)용: 후보 임베딩 모델을 이름으로 직접 로드함."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
