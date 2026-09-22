"""domain_eval 비교실험용 색인 버전 설정 (schedule.md 3.1~3.2절).

v1: 5장에서 채택한 방식(절 구조 인식 후 절 경계 안에서만 800자/overlap 120 분할)
v2: 비교 대상(절 구조 인식 없이 쪽 텍스트를 바로 800자 슬라이싱)
두 버전 모두 scripts/domain_eval/pdf/{v1,v2}/index/ 에 FAISS로 저장됨.
"""

EMBEDDING_CANDIDATES = {
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}
