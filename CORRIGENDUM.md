# CORRIGENDUM — Table 8 Numbers vs. Reproducible Measurements

**Date:** 2026-04-22
**Affected paper:** Shin & Moon (2025), *"Performance Optimization Study of Hybrid RAG Engine Integrating Multi-Source Knowledge: Vector, Graph, and Ontology Approaches"*, JKSCI.
**Author note:** 본 문서는 논문 제1저자 본인(Shin Dong-wook)의 자발적 정정 기록입니다. 박사학위 논문(2026, 호서대) 준비 중 선행 JKSCI 논문의 수치 재현 시도 과정에서 발견된 두 가지 문제를 투명히 기록하고, 후속 연구 및 재현성을 위한 **정정된 수치** 를 공개합니다.

---

## 1. Summary of findings

| # | 문제 | 영향 |
|---|---|---|
| 1 | `src/evaluator.py`의 `normalize()` 에 **구두점 제거 단계 누락** | 콤마 구분 리스트형 gold 답의 토큰이 `"A,"`, `"B,"`, `"C"` 로 분리되어 prediction `"A", "B", "C"` 와 교집합이 **마지막 토큰 하나로 축소** → F1 가 실제의 약 1/3 수준으로 산출 |
| 2 | `notebooks/Triple_Hybrid_RAG_Full.ipynb` Step 8 (Table 8 생성 셀) 의 **실행 증거 부재** — `sample_ds = random.sample(full_ds, 100)` 라인 활성, 5,000 라인 주석 처리, `execution_count: null, outputs: []` | 논문 Table 8 의 "**5,000 QA × 3 runs**" 라벨과 노트북 실행 상태가 일치하지 않음. 보고된 수치(F1 = 0.86 등)가 **실제 5,000 QA 전수 실험의 결과라는 증빙을 본 저장소에서 재구성할 수 없음** |

---

## 2. Reproduction attempt (2026-04-22)

**환경.** 같은 `gold_qa_5000.json` (byte-identical) · 같은 `university_data.py` 합성 코퍼스 · 동일 LLM (gpt-4o-mini, temp=0, max=500) · 동일 prompt · 본 저장소 `src/dwa.py` R-DWA 규칙 · 본 저장소 `src/evaluator.py`.

**측정 방법.** 박사논문 저장소(https://github.com/sdw1621/triple-rag-phd) 에서 동일한 알고리즘·평가기를 반영한 clean reimplementation 로 5,000 QA 전수 평가를 수행. 330,000 엔트리 보상 캐시 기반이라 본 측정은 재현 가능 (상세: `scripts/build_cache.py`, 해당 저장소).

**결과.**

### (a) `evaluator.py` 원본 (구두점 버그 포함) 으로 평가 시

| 정책 | F1<sub>strict</sub> | F1<sub>substring</sub> | EM | Faithfulness |
|---|---|---|---|---|
| Vector-only | 0.051 ± 0.14 | 0.334 ± 0.41 | 0.000 | 0.742 ± 0.43 |
| R-DWA (Triple-Hybrid) | **0.072 ± 0.16** | 0.450 ± 0.45 | 0.000 | 0.835 ± 0.36 |

### (b) 구두점 버그 수정본 (`_PUNCT_RE` 추가, 이 CORRIGENDUM 의 핵심 패치) 으로 평가 시

| 정책 | F1<sub>strict</sub> | F1<sub>substring</sub> | EM | Faithfulness |
|---|---|---|---|---|
| Vector-only | 0.082 ± 0.17 | 0.334 ± 0.41 | 0.000 | 0.742 ± 0.43 |
| R-DWA (Triple-Hybrid) | **0.137 ± 0.19** | 0.450 ± 0.45 | 0.000 | 0.835 ± 0.36 |

### (c) Table 8 주장 vs 실측

| 지표 | 원 논문 Table 8 주장 | 구두점 버그 포함 실측 | 구두점 수정 실측 | 재현율 |
|---|---|---|---|---|
| R-DWA F1 | **0.86 ± 0.01** | 0.072 | 0.137 | **8% → 16%** |
| R-DWA EM | **0.78 ± 0.02** | 0.000 | 0.000 | **0%** (구조적) |
| R-DWA Faithfulness | **0.89 ± 0.01** | 0.835 | 0.835 | **94%** |
| R-DWA Recall@3 | 0.92 ± 0.01 | — | — | 미측정 |

**관찰**: Faithfulness 94% 는 LLM 생성 품질이 양 실험에서 일관됨을 의미. F1 과 EM 의 큰 gap 은 (i) 구두점 버그 + (ii) **결정적으로, LLM 의 출력 형식이 gold 의 콤마 리스트와 맞지 않음** 에 기인. EM = 0 은 모든 정책에서 구조적으로 발생하므로 원 논문의 EM = 0.78 은 **본 저장소의 코드·데이터로는 어떤 설정으로도 재현 불가**.

---

## 3. What was likely measured to yield the originally-reported numbers

추론(저자 본인의 재구성):

- **Table 8 의 0.86** 는 **Simple-subset only** 또는 **100-sample 소규모** 에서 나왔을 가능성이 높음. 본 저장소의 5,000 QA 전수 F1<sub>substring</sub> 에서 Simple 유형만 따로 집계하면 **0.837** 로 측정되며 이는 반올림 시 0.86 에 근접.
- EM 0.78 은 **gold 형식이 단일 단어 (예: `"컴퓨터공학과"`) 위주의 이전 버전** 에서 측정되었을 가능성. 현 `gold_qa_5000.json` 에는 다인명 리스트가 대부분이라 어떤 정책에서도 EM = 0 이 나옴.
- 어느 쪽이든 **Table 8 의 "5,000 QA × 3 runs" 라벨과 실제 측정 조건이 일치하지 않았음** 이 본 corrigendum 의 핵심 지적.

---

## 4. Action taken in this repository

1. `src/evaluator.py::Evaluator.normalize` 에 구두점 제거 단계 추가 (`_PUNCT_RE`) — 이 PR 의 diff 참고.
2. `README.md` Table 8 을 "**원 논문 주장 / 재현 측정**" 이중 컬럼으로 교체 (별도 diff).
3. 본 `CORRIGENDUM.md` 생성.
4. `notebooks/Triple_Hybrid_RAG_Full.ipynb` Step 8 셀에 `# WARNING: this cell was never executed at full scale in the original release — see CORRIGENDUM.md` 주석 추가 (예정).

---

## 5. For readers and future citations

본 저장소 (`sdw1621/hybrid-rag-comparsion`) 를 인용하려는 연구자에게 다음을 권고합니다:

- **숫자 인용 시** Table 8 의 원 수치가 아닌 본 CORRIGENDUM §2 의 **버그 수정 재측정치**를 사용할 것.
- **재현 시** 본 저장소의 수정된 `evaluator.py` 를 사용하고, 5,000 QA 전수 실험 시 ~$10–15 의 OpenAI 비용을 예상할 것 (박사논문 저장소의 330K 캐시 재사용 시 비용 0).
- **비교 벤치마크로 사용 시** 본 저장소 대신 **박사학위 논문 저장소** `sdw1621/triple-rag-phd` 를 참조할 것을 권고. 해당 저장소는 본 CORRIGENDUM 의 수정 사항을 모두 반영하고 있으며, **L-DWA (PPO 기반)** 의 R-DWA 대비 개선 측정도 함께 제공함.

---

## 6. Acknowledgment and integrity statement

본 저자는 2026년 4월 박사학위 논문 재현성 검증 과정에서 본 문제들을 **스스로 발견하여 공개적으로 정정** 합니다. 선행 JKSCI 논문의 Table 8 수치 오류 또는 비재현성은 저자 본인 책임이며, 본 CORRIGENDUM 은 학술적 투명성을 위한 자발적 조치입니다. 향후 본 연구 결과를 인용하거나 재현을 시도하는 연구자에게 불편함을 끼친 데 대해 사과드립니다.

— Shin Dong-wook (신동욱), 2026-04-22
