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
│   └── dataset_generator.py # Gold QA 5,000개 생성기 (60개 학과 / 577명 교수 / 1,505개 과목 / 400개 프로젝트)
├── tests/
│   └── test_dwa.py          # DWA 단위 테스트
└── requirements.txt
```

---

## 🗄️ 데이터셋 구성

| 구분 | 항목 | 수치 |
|------|------|------|
| **Knowledge Graph** | 학과 | 60개 |
| | 교수 | 577명 |
| | 과목 | 1,505개 |
| | 프로젝트 | 400개 |
| | 노드 합계 | 2,542개 |
| | 엣지 합계 | 6,889개 |
| **Vector Store** | 비정형 문서 (교수·학과·프로젝트 소개) | 1,037건 |
| **Gold QA** | Simple (40%) | 2,000쌍 |
| | Multi-hop (35%) | 1,750쌍 |
| | Conditional (25%) | 1,250쌍 |
| | **합계** | **5,000쌍** |

---

## 📊 실험 결과

### Table 13. Overall Performance (mean±std, Gold QA 5,000쌍 · 3 runs)

| System | F1 | EM | Recall@3 | Precision | Faithfulness |
|--------|----|----|----------|-----------|--------------|
| Vector-Only | 0.71±0.01 | 0.57±0.02 | 0.80±0.01 | 0.68±0.01 | 0.70±0.01 |
| GraphRAG | 0.78±0.01 | 0.67±0.01 | 0.85±0.01 | 0.74±0.01 | 0.77±0.01 |
| HybridRAG | 0.80±0.01 | 0.70±0.01 | 0.87±0.01 | 0.78±0.01 | 0.81±0.01 |
| Adaptive-RAG | 0.77±0.01 | 0.65±0.01 | 0.83±0.01 | 0.73±0.01 | 0.75±0.01 |
| **Triple-Hybrid** | **0.85±0.01** | **0.77±0.01** | **0.91±0.01** | **0.83±0.01** | **0.88±0.01** |

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
