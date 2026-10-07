"""Doc Pool 6편 스펙 (5장 표). 색인 구축(tools.build_doc_pool_index)과 Golden
Dataset 생성(eval/generate_golden_dataset.py)이 공유함.

PDF 실물은 data/doc_pool/에 파일명 그대로 둠. authors, title은 REFERENCE 표기용.
파일명에 콜론(:)이나 슬래시(/)를 쓰지 말 것 — macOS Finder와 Windows에서 만들 수 없음.
"""

DOC_POOL_SPECS: list[dict[str, str]] = [
    {"file": "TurboQuant.pdf", "tech": "TurboQuant", "camp": "SW", "role": "target", "arxiv": "2504.19874",
     "authors": "Zandieh, A. et al.", "title": "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"},
    {"file": "CXL-based.pdf", "tech": "ITME", "camp": "HW", "role": "target", "arxiv": "2606.12556",
     "authors": "Jang, H. et al.", "title": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories"},
    {"file": "DeepSeek-V2.pdf", "tech": "DeepSeek-V2", "camp": "SW", "role": "comparison", "arxiv": "2405.04434",
     "authors": "DeepSeek-AI", "title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"},
    {"file": "KIVI.pdf", "tech": "KIVI", "camp": "SW", "role": "comparison", "arxiv": "2402.02750",
     "authors": "Liu, Z. et al.", "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"},
    {"file": "Dynamic KV Cache Mgmt.pdf", "tech": "InfiniGen", "camp": "HW", "role": "comparison", "arxiv": "2406.19707",
     "authors": "Lee, W. et al.", "title": "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management"},
    {"file": "PIM-CXL.pdf", "tech": "PIM/CXL", "camp": "HW", "role": "comparison", "arxiv": "2511.00321",
     "authors": "Kim, D. et al.", "title": "Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits"},
]
