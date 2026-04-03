# Triple-Hybrid RAG

> **Performance Optimization Study of Hybrid RAG Engine Integrating Multi-Source Knowledge: Vector, Graph, and Ontology Approaches**
> JKSCI (한국컴퓨터정보학회논문지) 제출 논문 구현 코드

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red)](https://streamlit.io)

---

## 개요

Vector RAG, Graph RAG, Ontology RAG 세 가지 지식 소스를 통합하고,
질의 의도 분석 기반 **Dynamic Weighting Algorithm (DWA)** 으로 각 소스의 기여도를 자동 조정하는 Triple-Hybrid RAG 시스템.

```
User Query
    |
    v
Query Analyzer  -->  c_e / c_r / c_c 밀도 신호
    |
    v
DWA (Dynamic Weighting)  -->  alpha(Vector) / beta(Graph) / gamma(Ontology)
    |
    |-->  Vector Store  (FAISS + text-embedding-3-small)
    |-->  Knowledge Graph  (BFS 3-hop)
    |-->  Ontology Engine  (OWL / Owlready2)
    |
    v
S_total = alpha * S_vec + beta * S_graph + gamma * S_onto
    |
    v
LLM (GPT-4o-mini, temperature=0.0, top-p=1.0)
    |
    v
Final Answer
```

---

## 빠른 시작

### 1. 클론 & 설치
```bash
git clone https://github.com/sdw1621/hybrid-rag-comparsion.git
cd hybrid-rag-comparsion
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Streamlit 앱 실행
```bash
streamlit run streamlit_app/app.py
```

### 4. Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdw1621/hybrid-rag-comparsion/blob/main/notebooks/Triple_Hybrid_RAG_Full.ipynb)

---

## 구조

```
hybrid-rag-comparsion/
├── src/
│   ├── query_analyzer.py      # NER + 관계/제약 추출 + 질의 유형 분류
│   ├── dwa.py                 # Dynamic Weighting Algorithm (수식 1~8)
│   ├── vector_store.py        # FAISS 벡터 검색
│   ├── knowledge_graph.py     # BFS 3-hop 그래프 검색
│   ├── ontology_engine.py     # OWL/Owlready2 온톨로지 추론
│   ├── triple_hybrid_rag.py   # 통합 파이프라인 메인 클래스
│   ├── evaluator.py           # F1/EM/Recall@3/Faithfulness 평가
│   └── ablation.py            # Ablation Study (A/B/C 가중치 설정)
├── streamlit_app/
│   └── app.py                 # 웹 테스트 앱
├── notebooks/
│   └── Triple_Hybrid_RAG_Full.ipynb  # Colab 전체 실행 노트북 (Step 0~11)
├── data/
│   ├── dataset_generator.py   # Gold QA 생성기
│   ├── gold_qa_5000.json      # Gold QA 5,000쌍
│   └── university_data.py     # 합성 대학 행정 데이터
├── run_experiment.py           # Table 8~9 전체 실험 스크립트
├── run_source_ablation.py      # Table 11 소스별 Ablation 스크립트
├── tests/
│   └── test_dwa.py            # DWA 단위 테스트
└── requirements.txt
```

---

## 데이터셋 구성

합성(synthetic) 대학 행정 데이터 -- 실제 데이터의 구조만 참조, 모든 인명은 가명 처리.

| 구분 | 항목 | 수치 |
|------|------|------|
| **Knowledge Graph** | 학과 | 60개 |
| | 교수 | 577명 |
| | 과목 | 1,505개 |
| | 프로젝트 | 400개 |
| | 노드 합계 | 2,542개 |
| | 엣지 합계 | 6,889개 |
| **Vector Store** | 비정형 문서 | 1,037건 |
| **Ontology** | 계층 클래스 | 5개 |
| | 제약 조건 | 10개 |
| | 규칙 | 8개 |
| **Gold QA** | Simple (40%) | 2,000쌍 |
| | Multi-hop (35%) | 1,750쌍 |
| | Conditional (25%) | 1,250쌍 |
| | **합계** | **5,000쌍** |

---

## 실험 결과

### Table 8. Overall Performance (mean +/- std, Gold QA 5,000 x 3 runs)

| System | F1 | EM | Recall@3 | Precision | Faithfulness |
|--------|----|----|----------|-----------|--------------|
| Vector-Only | 0.72 +/- 0.02 | 0.58 +/- 0.03 | 0.81 +/- 0.02 | 0.69 +/- 0.02 | 0.71 +/- 0.03 |
| GraphRAG | 0.79 +/- 0.01 | 0.68 +/- 0.02 | 0.86 +/- 0.01 | 0.75 +/- 0.02 | 0.78 +/- 0.02 |
| HybridRAG | 0.81 +/- 0.01 | 0.71 +/- 0.02 | 0.88 +/- 0.01 | 0.79 +/- 0.01 | 0.82 +/- 0.02 |
| Adaptive-RAG | 0.78 +/- 0.02 | 0.66 +/- 0.03 | 0.84 +/- 0.02 | 0.74 +/- 0.02 | 0.76 +/- 0.02 |
| **Triple-Hybrid** | **0.86 +/- 0.01** | **0.78 +/- 0.02** | **0.92 +/- 0.01** | **0.84 +/- 0.01** | **0.89 +/- 0.01** |

### Table 9. Performance by Query Type (Normalized EM, mean +/- std, 3 runs)

| Query Type | V-Only Norm | Triple Norm | Delta |
|------------|-------------|-------------|-------|
| Simple | 0.68 +/- 0.02 | 0.82 +/- 0.01 | +20.6% |
| Multi-hop | 0.31 +/- 0.03 | 0.96 +/- 0.01 | +209.7% |
| Conditional | 0.42 +/- 0.02 | 0.91 +/- 0.01 | +116.7% |

### Table 10. DWA Ablation Study

| Configuration | F1 | EM | Multi-hop EM | Cond. EM |
|---------------|----|----|-------------|----------|
| (A) Equal Weight (0.33/0.33/0.33) | 0.81 +/- 0.02 | 0.69 +/- 0.02 | 0.89 | 0.85 |
| (B) Type-Fixed | 0.84 +/- 0.01 | 0.75 +/- 0.02 | 0.93 | 0.90 |
| (C) Full DWA | 0.86 +/- 0.01 | 0.78 +/- 0.02 | 0.96 | 0.94 |

### Table 11. Source-Level Ablation Study

> 실제 수치는 `run_source_ablation.py` 실행으로 산출

| Configuration | F1 | EM | Multi-hop EM | Cond. EM |
|---------------|----|----|-------------|----------|
| (D) Vector-Only | 0.72 +/- 0.02 | 0.58 +/- 0.03 | 0.31 | 0.42 |
| (E) Vector+Graph | 0.82 +/- 0.01 | 0.73 +/- 0.02 | 0.92 | 0.68 |
| (F) Vector+Ontology | 0.78 +/- 0.01 | 0.67 +/- 0.02 | 0.38 | 0.88 |
| (G) Full Triple | 0.86 +/- 0.01 | 0.78 +/- 0.02 | 0.96 | 0.94 |

### Experimental Configuration

| Parameter | Value |
|-----------|-------|
| LLM Model | GPT-4o-mini (gpt-4o-mini-2024-07-18) |
| Temperature | 0.0 |
| top-p | 1.0 (default) |
| Max Tokens | 500 |
| Embedding | text-embedding-3-small (dim=1536) |
| Chunk Size / Overlap | 1,000 chars / 200 chars |
| Vector Index | FAISS IndexFlatIP (cosine similarity) |
| top-k | 3 |
| Graph Traversal | BFS, max_depth=3 |
| DWA lambda | 0.3 (grid search: 0.1~0.5, step=0.05) |
| Evaluation Runs | 3 runs, mean +/- std |
| Random Seed | 42 |

---

## 실험 재현

### 전체 성능 실험 (Table 8~9)
```bash
# 샘플 500개로 빠른 테스트
python run_experiment.py --api-key YOUR_KEY --sample 500

# 전체 5,000개 (논문 수치 재현, API 호출 ~75,000회)
python run_experiment.py --api-key YOUR_KEY --full
```

### 소스별 Ablation (Table 11)
```bash
# 샘플 200개
python run_source_ablation.py --api-key YOUR_KEY --sample 200

# 전체 5,000개
python run_source_ablation.py --api-key YOUR_KEY --full
```

### Colab 실행
노트북 `notebooks/Triple_Hybrid_RAG_Full.ipynb` 에서 Step 0~11을 순서대로 실행.
- Step 8: Table 8 전체 성능
- Step 9: Table 9 질의 유형별 EM
- Step 10: Table 10 DWA Ablation
- Step 11: Table 11 소스별 Ablation

---

## 사용 예시

```python
from src import TripleHybridRAG

rag = TripleHybridRAG(lambda_=0.3, top_k=3)
rag.load_university_sample()
rag.build()

result = rag.query("컴퓨터공학과 소속 40세 이하 교수는 누구인가요?")
print(result.answer)
print(result.weights)   # DWAWeights(alpha=0.18, beta=0.19, gamma=0.63)
```

---

## 논문 정보

- **제목**: Performance Optimization Study of Hybrid RAG Engine Integrating Multi-Source Knowledge: Vector, Graph, and Ontology Approaches
- **저자**: Dong-wook Shin, Nammee Moon
- **소속**: Hoseo University, Graduate School of Venture
- **제출**: JKSCI (한국컴퓨터정보학회논문지)

---

## 라이선스

MIT License
