"""관점별 조사 초점 카탈로그. orchestrator가 고르는 서브 태스크의 초점과 worker의 질의를 함께 정의함.

초점 하나는 관점의 필수 항목(9장 평가 기준) 하나에 대응하고, 그 항목을 묻는 질의 템플릿을 가짐.
orchestrator는 초점 id만 고르고 질의 문구는 이 카탈로그가 정하므로, 두 기술에 같은 초점이면 같은
질의 구조가 적용됨(10장 대칭 질의). 템플릿 자리표시자: {tech}, {anchor}, {domain}.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FocusSpec:
    id: str
    label: str  # 보고서와 결정 로그에 쓰는 짧은 이름
    items: tuple[str, ...]  # 추출 단계에서 확인할 필수 항목
    desc: str  # orchestrator 프롬프트에 보이는 설명
    paper: str | None = None  # 논문 검색 질의(RAG 관점만)
    web: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (한국어, 영어) 웹 질의 사다리


COUNTER = "counter"

FOCI: dict[str, dict[str, FocusSpec]] = {
    "trl": {
        "estimate": FocusSpec(
            "estimate", "TRL 구간 추정",
            ("TRL 구간 추정(1~3, 4~6, 7~9)과 추정임을 명시한 판단 사유", "공개 정보로 확인되지 않는 부분"),
            "구현과 공식 발표 현황을 종합해 TRL 구간을 추정. 기술 성숙도 칸마다 반드시 포함",
            paper="{tech} 구현과 공식 발표 현황, 실험 수준, 공개 구현",
            web=(("{tech} {anchor} 제품 발표 오픈소스 구현 TRL", '"{tech}" {anchor} product announcement open source TRL'),),
        ),
        "experiment": FocusSpec(
            "experiment", "실험 수준",
            ("논문의 실험 수준(시뮬레이션, 프로토타입, 실제 시스템)",),
            "논문 실험이 시뮬레이션, 프로토타입, 실제 시스템 중 어디까지인지",
            paper="{tech} 실험 환경과 평가 규모, 프로토타입 구현",
            web=(("{tech} {anchor} 벤치마크 실험 결과 재현", '"{tech}" {anchor} benchmark results reproduction'),),
        ),
        "open_impl": FocusSpec(
            "open_impl", "공개 구현",
            ("공개 구현 유무(오픈소스, 코드 공개)",),
            "오픈소스 공개, 서빙 프레임워크 통합 여부",
            paper="{tech} 공개 구현과 코드 공개, 서빙 프레임워크 통합",
            web=(("{tech} {anchor} 오픈소스 GitHub vLLM 통합", '"{tech}" {anchor} open source GitHub vLLM integration'),),
        ),
        "productization": FocusSpec(
            "productization", "제품화",
            ("시제품과 제품 발표, 샘플 공급과 양산 보도",),
            "시제품, 제품 발표, 샘플 공급, 양산 보도",
            paper="{tech} 하드웨어 시제품과 실제 시스템 배포",
            web=(("{tech} {anchor} 제품 발표 샘플 공급 양산", '"{tech}" {anchor} product launch sampling mass production'),),
        ),
        COUNTER: FocusSpec(
            COUNTER, "한계와 후속 검증",
            ("한계와 실패 사례, 독립 재현과 후속 검증",),
            "한계, 실패 사례, 독립 재현과 후속 검증(반대 근거)",
            paper="{tech} 한계와 실패 사례, 후속 검증",
            web=(("{tech} {anchor} 한계 실패 사례 후속 검증", '"{tech}" {anchor} limitations failure follow-up evaluation'),),
        ),
    },
    "market": {
        "size": FocusSpec(
            "size", "시장 규모", ("시장 규모와 성장성",), "시장 규모와 성장성",
            web=(("{tech} {anchor} 시장 규모", '"{tech}" {anchor} market size'),),
        ),
        "adoption": FocusSpec(
            "adoption", "채택 현황", ("상용화와 채택 현황",), "상용화와 채택 사례",
            web=(("{tech} {anchor} 채택 사례", '"{tech}" {anchor} adoption deployment'),),
        ),
        "ecosystem": FocusSpec(
            "ecosystem", "생태계", ("프레임워크 지원과 표준화 같은 생태계 지지",), "프레임워크 지원, 표준화, 생태계 지지",
            web=(("{tech} {anchor} 프레임워크 지원", '"{tech}" {anchor} framework support integration'),),
        ),
        COUNTER: FocusSpec(
            COUNTER, "시장 비판", ("시장의 비판, 도입 실패 우려, 회의론",), "한계 비판, 도입 실패 우려, 시장 회의론(반대 근거)",
            web=(
                ("{tech} {anchor} 한계 비판", '"{tech}" {anchor} limitations criticism'),
                ("{tech} {anchor} 도입 실패 우려", '"{tech}" {anchor} adoption risk concerns'),
                ("{tech} {anchor} 시장 회의론", '"{tech}" {anchor} skepticism'),
            ),
        ),
    },
    "stakeholder": {
        "competitor": FocusSpec(
            "competitor", "경쟁 진영", ("경쟁 진영의 반응과 대응 기술",), "경쟁 진영의 반응과 대응 기술",
            web=(("{tech} {anchor} 경쟁 기술 반응", '"{tech}" {anchor} competitors response'),),
        ),
        "developer": FocusSpec(
            "developer", "개발자와 도입 기업", ("도입 기업과 개발자의 의견과 도입 장벽",), "도입 기업과 개발자 커뮤니티의 의견, 도입 장벽",
            web=(("{tech} {anchor} 개발자 커뮤니티 반응", '"{tech}" {anchor} developer community reaction'),),
        ),
        "investor": FocusSpec(
            "investor", "투자 업계", ("투자 업계의 평가",), "투자자, 애널리스트의 평가",
            web=(("{tech} {anchor} 투자 업계 평가", '"{tech}" {anchor} investors analysts'),),
        ),
        COUNTER: FocusSpec(
            COUNTER, "이해관계자 반론", ("경쟁사 반박, 개발자 불만, 투자 리스크",), "경쟁사 반박, 개발자 불만, 투자 리스크(반대 근거)",
            web=(
                ("{tech} {anchor} 경쟁사 반박", '"{tech}" {anchor} competitor rebuttal'),
                ("{tech} {anchor} 개발자 불만 한계", '"{tech}" {anchor} developer complaints limitations'),
                ("{tech} {anchor} 투자 리스크", '"{tech}" {anchor} investment risk'),
            ),
        ),
    },
    "domain": {
        "cache_reuse": FocusSpec(
            "cache_reuse", "캐시 재사용", ("세션 간 캐시 재사용에 미치는 영향",), "멀티턴 세션 간 prefix cache 재사용에 미치는 영향",
            paper="{tech} {domain} 멀티턴 세션 간 prefix cache 재사용",
            web=(("{tech} {anchor} prefix cache 재사용 멀티턴", '"{tech}" {anchor} prefix cache reuse multi-turn'),),
        ),
        "latency": FocusSpec(
            "latency", "응답 지연", ("턴당 응답 지연",), "턴당 응답 지연, TTFT, 처리량",
            paper="{tech} {domain} 턴당 지연 시간, TTFT, 처리량",
            web=(("{tech} {anchor} 지연 시간 처리량 서빙", '"{tech}" {anchor} latency throughput serving'),),
        ),
        "storage_cost": FocusSpec(
            "storage_cost", "캐시 보관 비용", ("캐시 보관 비용(메모리, 저장 장치)",), "KV cache를 보관하는 메모리, 저장 장치 비용",
            paper="{tech} {domain} KV cache 메모리 사용량과 저장 비용",
            web=(("{tech} {anchor} KV cache 메모리 비용 저장 장치", '"{tech}" {anchor} KV cache memory cost storage'),),
        ),
        "accuracy": FocusSpec(
            "accuracy", "정확도 유지", ("누적 턴에서의 정확도 유지",), "긴 문맥과 누적 턴에서의 정확도 유지",
            paper="{tech} {domain} 긴 문맥과 누적 턴에서의 정확도 유지",
            web=(("{tech} {anchor} 장문맥 정확도 저하", '"{tech}" {anchor} long context accuracy degradation'),),
        ),
        "integration": FocusSpec(
            "integration", "도입 변경 범위", ("도입에 필요한 하드웨어와 소프트웨어 변경 범위",), "도입에 필요한 하드웨어, 소프트웨어 변경 범위",
            paper="{tech} {domain} 도입에 필요한 하드웨어 요구사항과 소프트웨어 변경",
            web=(("{tech} {anchor} 서빙 프레임워크 도입 요구사항", '"{tech}" {anchor} serving framework requirements'),),
        ),
        COUNTER: FocusSpec(
            COUNTER, "도메인 적용 한계", ("도메인 적용의 한계와 실패 사례",), "에이전트 코딩 서빙에 적용할 때의 한계와 실패 사례(반대 근거)",
            paper="{tech} {domain} 한계와 실패 사례, 후속 검증",
            web=(("{tech} {anchor} 한계 실패 사례", '"{tech}" {anchor} limitations failure cases'),),
        ),
    },
}
