"""Doc Pool 6편 스펙 (5장 표). 색인 구축(tools.build_doc_pool_index)과 Golden
Dataset 생성(eval/generate_golden_dataset.py)이 공유함.

PDF 실물은 저작권 때문에 리포에 커밋하지 않음(.gitignore). 아래 arXiv ID로 받아
data/doc_pool/ 아래 파일명 그대로 저장할 것(README 참고).
파일명에 콜론(:)이나 슬래시(/)를 쓰지 말 것 — macOS Finder와 Windows에서 만들 수 없음.
"""

DOC_POOL_SPECS: list[dict[str, str]] = [
    {"file": "TurboQuant.pdf", "tech": "TurboQuant", "camp": "SW", "role": "target", "arxiv": "2504.19874"},
    {"file": "CXL-based.pdf", "tech": "ITME", "camp": "HW", "role": "target", "arxiv": "2606.12556"},
    {"file": "DeepSeek-V2.pdf", "tech": "DeepSeek-V2", "camp": "SW", "role": "comparison", "arxiv": "2405.04434"},
    {"file": "KIVI.pdf", "tech": "KIVI", "camp": "SW", "role": "comparison", "arxiv": "2402.02750"},
    {"file": "Dynamic KV Cache Mgmt.pdf", "tech": "InfiniGen", "camp": "HW", "role": "comparison", "arxiv": "2406.19707"},
    {"file": "PIM-CXL.pdf", "tech": "PIM/CXL", "camp": "HW", "role": "comparison", "arxiv": "2511.00321"},
]
