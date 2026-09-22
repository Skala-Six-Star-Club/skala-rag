# judge 테스트 리포트

설계 근거: docs/agentic-rag-design.md 7.9절, 12장 "반복 2" / docs/schedule.md 2절
(이진 판정 노드라 판정 정확도 스팟체크로 검증함)

## 1부: judge_passed() 순수 함수 (graph.py 조건부 엣지용)

| 테스트 | 결과 | 비고 |
|---|---|---|
| judge_passed: 모두 통과 -> True | PASS | has_biased_expression=False sentences_without_evidence=[] is_balanced=True notes='' |
| judge_passed: 우열 표현 -> False | PASS | has_biased_expression=True sentences_without_evidence=[] is_balanced=True notes='' |
| judge_passed: 근거 없는 문장 -> False | PASS | has_biased_expression=False sentences_without_evidence=['근거 없는 문장'] is_balanced=True notes='' |
| judge_passed: 불균형 -> False | PASS | has_biased_expression=False sentences_without_evidence=[] is_balanced=False notes='' |

순수 함수 검사 전부 통과함.

## 2부: Qwen3-8B 판정 스팟체크

| 테스트 | 결과 | 비고 |
|---|---|---|
| 우열 판정 표현 탐지 | PASS | has_biased_expression=True sentences_without_evidence=['TurboQuant가 ITME보다 훨씬 우수하고 실용적인 기술임. ITME는 아직 갈 길이 멀다.'] is_balanced=False notes="요약문은 '훨씬 우수하다', '갈 길이 멀다' 등의 우열 판정 표현을 사용해 편향적이고, 근거 표기가 없으며, 근거 분포는 균형이지만 어조와 분량에서 TurboQuant에 대한 부정적 평가가 압도적이라 균형이 깨진다." |
| 근거 없는 문장 탐지 | PASS | has_biased_expression=True sentences_without_evidence=['TurboQuant가 ITME보다 훨씬 우수하고 실용적인 기술임. ITME는 아직 갈 길이 멀다.'] is_balanced=False notes="요약문은 '훨씬 우수하다', '갈 길이 멀다' 등의 우열 판정 표현을 사용해 편향적이고, 근거 표기가 없으며, 근거 분포는 균형이지만 어조와 분량에서 TurboQuant에 대한 부정적 평가가 압도적이라 균형이 깨진다." |
| 위반 입력 -> judge_passed False | PASS | has_biased_expression=True sentences_without_evidence=['TurboQuant가 ITME보다 훨씬 우수하고 실용적인 기술임. ITME는 아직 갈 길이 멀다.'] is_balanced=False notes="요약문은 '훨씬 우수하다', '갈 길이 멀다' 등의 우열 판정 표현을 사용해 편향적이고, 근거 표기가 없으며, 근거 분포는 균형이지만 어조와 분량에서 TurboQuant에 대한 부정적 평가가 압도적이라 균형이 깨진다." |
| 깨끗한 입력 -> 우열 표현 없음 | PASS | has_biased_expression=False sentences_without_evidence=[] is_balanced=True notes='우열 판정 표현이 없고 모든 문장에 근거 표기가 포함되어 있으며, 두 기술에 대한 지지/반대 근거가 균형 있게 분포해 있습니다.' |
| 깨끗한 입력 -> 근거 없는 문장 없음 | PASS | has_biased_expression=False sentences_without_evidence=[] is_balanced=True notes='우열 판정 표현이 없고 모든 문장에 근거 표기가 포함되어 있으며, 두 기술에 대한 지지/반대 근거가 균형 있게 분포해 있습니다.' |
| 근거 분포 균형 -> is_balanced True | PASS | has_biased_expression=False sentences_without_evidence=[] is_balanced=True notes='우열 판정 표현이 없고 모든 문장에 근거 표기가 포함되어 있으며, 두 기술에 대한 지지/반대 근거가 균형 있게 분포해 있습니다.' |
| 깨끗한 입력 -> judge_passed True | PASS | has_biased_expression=False sentences_without_evidence=[] is_balanced=True notes='우열 판정 표현이 없고 모든 문장에 근거 표기가 포함되어 있으며, 두 기술에 대한 지지/반대 근거가 균형 있게 분포해 있습니다.' |
| 한쪽 반대 근거 0건 -> is_balanced False | PASS | has_biased_expression=False sentences_without_evidence=[] is_balanced=False notes='SUMMARY 문장은 우열 판정 표현이 없고 근거 표기가 포함되어 있어 문제 없음. 근거 분포에서 ITME에 반대 근거가 0건으로 균형이 깨짐. 반대 근거가 없는 기술은 서술이 불균형한 것으로 판단됨.' |

스팟체크 전부 통과함.

![스팟체크 결과](report_assets/unit_checks.png)
