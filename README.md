# Triple-Hybrid RAG

> **Performance Optimization Study of Hybrid RAG Engine Integrating Multi-Source Knowledge: Vector, Graph, and Ontology Approaches**  
> JKSCI (한국컴퓨터정보학회논문지) 제출 논문 구현 코드

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red)](https://streamlit.io)

---

## 📌 개요

Vector RAG · Graph RAG · Ontology RAG 세 가지 지식 소스를 통합하고,  
질의 의도 분석 기반 **Dynamic Weighting Algorithm (DWA)** 으로 각 소스의 기여도를 자동 조정하는 Triple-Hybrid RAG 시스템.

```
User Query
    │
    ▼
Query Analyzer  ──→  c_e / c_r / c_c 밀도 신호
    │
    ▼
DWA (Dynamic Weighting)  ──→  α(Vector) / β(Graph) / γ(Ontology)
    │
    ├──→ Vector Store  (FAISS + text-embedding-3-small)
    ├──→ Knowledge Graph  (Neo4j BFS 3-hop)
    └──→ Ontology Engine  (OWL / Owlready2)
    │
    ▼
S_total = α·S_vec + β·S_graph + γ·S_onto
    │
    ▼
LLM (GPT-4o-mini, temperature=0.0)
    │
    ▼
Final Answer
```

---

## 🚀 빠른 시작

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

## 📁 구조

```
hybrid-rag-comparsion/
├── src/
│   ├── query_analyzer.py    # NER + 관계/제약 추출 + 질의 유형 분류
│   ├── dwa.py               # Dynamic Weighting Algorithm (수식 1~8)
│   ├── vector_store.py      # FAISS 벡터 검색
│   ├── knowledge_graph.py   # Neo4j BFS 3-hop 그래프 검색
│   ├── ontology_engine.py   # OWL/Owlready2 온톨로지 추론
│   ├── triple_hybrid_rag.py # 통합 파이프라인 메인 클래스
│   ├── evaluator.py         # F1/EM/Recall@3/Faithfulness 평가
│   └── ablation.py          # Ablation Study (A/B/C 설정)
├── streamlit_app/
│   └── app.py               # 웹 테스트 앱
├── notebooks/
│   └── Triple_Hybrid_RAG_Full.ipynb  # Colab 전체 실행 노트북
├── data/
│   └── dataset_generator.py # Gold QA 500개 생성기
├── tests/
│   └── test_dwa.py          # DWA 단위 테스트
└── requirements.txt
```

---

## 📊 실험 결과

### Table 13. Overall Performance (mean±std, 3 runs)

| System | F1 | EM | Recall@3 | Precision | Faithfulness |
|--------|----|----|----------|-----------|--------------|
| Vector-Only | 0.72±0.02 | 0.58±0.03 | 0.81±0.02 | 0.69±0.02 | 0.71±0.03 |
| GraphRAG | 0.79±0.01 | 0.68±0.02 | 0.86±0.01 | 0.75±0.02 | 0.78±0.02 |
| HybridRAG | 0.81±0.01 | 0.71±0.02 | 0.88±0.01 | 0.79±0.01 | 0.82±0.02 |
| Adaptive-RAG | 0.78±0.02 | 0.66±0.03 | 0.84±0.02 | 0.74±0.02 | 0.76±0.02 |
| **Triple-Hybrid** | **0.86±0.01** | **0.78±0.02** | **0.92±0.01** | **0.84±0.01** | **0.89±0.01** |

### DWA 하이퍼파라미터

| Parameter | Value | 결정 방법 |
|-----------|-------|---------|
| λ | 0.3 | Grid Search (0.1~0.5, step=0.05) |
| N_max_entity | 5 | 훈련 데이터 최대값 |
| N_max_relation | 4 | 훈련 데이터 최대값 |
| N_max_constraint | 3 | 훈련 데이터 최대값 |

---

## 🔧 사용 예시

```python
from src import TripleHybridRAG

rag = TripleHybridRAG(lambda_=0.3, top_k=3)
rag.load_university_sample()
rag.build()

result = rag.query("컴퓨터공학과 소속 40세 이하 교수는 누구인가요?")
print(result.answer)
print(result.weights)   # DWAWeights(α=0.18, β=0.19, γ=0.63)
```

---

## 📄 논문 정보

- **제목**: Performance Optimization Study of Hybrid RAG Engine Integrating Multi-Source Knowledge
- **저자**: Dong-wook Shin, Nammee Moon
- **소속**: Hoseo University
- **제출**: JKSCI (한국컴퓨터정보학회논문지)

---

## 📜 라이선스

MIT License
