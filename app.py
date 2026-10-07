"""통합 실행 진입점.

    python app.py                       # 새 실행 (trace_id 자동 생성)
    python app.py --trace-id <id>       # trace_id 지정
    python app.py --resume <trace_id>   # 중단된 실행을 마지막 체크포인트에서 재개
"""

from __future__ import annotations

import argparse
import json
import logging

from src.graph import run


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache 다관점 평가 Orchestrator-Workers 그래프 실행")
    parser.add_argument("--trace-id", help="실행 식별자. LangSmith metadata, 결정 로그, 체크포인트 thread_id에 함께 쓰임")
    parser.add_argument("--resume", metavar="TRACE_ID", help="같은 trace_id의 마지막 체크포인트에서 재개")
    parser.add_argument("-v", "--verbose", action="store_true", help="결정 로그를 콘솔에도 출력")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run(trace_id=args.resume or args.trace_id, resume=bool(args.resume))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
