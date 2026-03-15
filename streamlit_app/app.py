"""
Triple-Hybrid RAG — Streamlit 테스트 앱
논문 Section IV.4 구현

실행: streamlit run streamlit_app/app.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Triple-Hybrid RAG", page_icon="🔬", layout="wide")

# ── 사이드바 설정 ──
st.sidebar.markdown("### ⚙️ 설정")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
st.sidebar.markdown("---")
lambda_ = st.sidebar.slider("lambda (DWA)", 0.0, 0.5, 0.3, 0.05)
top_k = st.sidebar.slider("top-k", 1, 5, 3)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📖 페이지",
    ["🔍 질의 테스트", "📊 성능 비교", "⚖️ Ablation Study", "ℹ️ 시스템 정보"],
    label_visibility="visible",
)


@st.cache_resource
def load_rag():
    from src.triple_hybrid_rag import TripleHybridRAG

    rag = TripleHybridRAG(lambda_=0.3, top_k=3)
    rag.load_university_sample(extended=True)
    rag.build()
    return rag


# ── 헤더 ──
st.markdown("## 🔬 Triple-Hybrid RAG")
st.markdown("Vector · Graph · Ontology 통합")

# ════════════════════════════════════════════
# 1. 질의 테스트
# ════════════════════════════════════════════
if page == "🔍 질의 테스트":
    st.subheader("질의 입력")
    examples = [
        "김철수 교수가 담당하는 과목은?",
        "컴퓨터공학과 소속 40세 이하 교수는?",
        "이영희 교수와 같은 학과 교수는?",
        "인공지능학과 소속 조교수는 누구인가?",
        "딥러닝 과목 담당 교수의 연구 분야는?",
        "AI융합연구소 참여 교수 목록은?",
    ]
    c1, c2 = st.columns([1, 1])
    with c1:
        query = st.text_area(
            "질문을 입력하세요",
            placeholder="예: 김철수 교수의 연구 분야는?",
            height=100,
        )
    with c2:
        ex = st.selectbox("예시 질의", ["직접 입력"] + examples)
    if ex != "직접 입력":
        query = ex

    if st.button("🔎 검색", use_container_width=True) and query:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("⚠️ 사이드바에서 OpenAI API Key를 입력해 주세요.")
            st.stop()
        with st.spinner("처리 중..."):
            try:
                rag = load_rag()
                rag.dwa.lambda_ = lambda_
                rag.top_k = top_k
                _t0 = time.time()
                result = rag.query(query)
                _wall = time.time() - _t0
            except Exception as e:
                st.error(f"오류 발생: {e}")
                st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("응답 시간", f"{_wall:.2f}s")
        m2.metric("질의 유형", result.intent.query_type)
        m3.metric("복잡도", f"{result.intent.complexity_score:.2f}")
        m4.metric("엔티티 수", len(result.intent.entities))
        st.markdown("---")

        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### 💬 답변")
            st.markdown(
                f'<div style="background:rgba(216,90,48,0.15);border-radius:8px;padding:16px;'
                f'border-left:4px solid #D85A30;color:#fff">{result.answer}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### 📄 검색 컨텍스트")
            t1, t2, t3 = st.tabs(["Vector", "Graph", "Ontology"])
            for tab, ctxs in zip(
                [t1, t2, t3],
                [result.vector_contexts, result.graph_contexts, result.onto_contexts],
            ):
                with tab:
                    for i, c in enumerate(ctxs, 1):
                        st.markdown(f"**[{i}]** {c}")
                    if not ctxs:
                        st.info("결과 없음")
        with right:
            st.markdown("#### ⚖️ DWA 가중치")
            w = result.weights
            fig = go.Figure(
                go.Bar(
                    x=["Vector(α)", "Graph(β)", "Ontology(γ)"],
                    y=[w.alpha, w.beta, w.gamma],
                    marker_color=["#378ADD", "#1D9E75", "#D85A30"],
                    text=[f"{v:.3f}" for v in [w.alpha, w.beta, w.gamma]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                yaxis=dict(range=[0, 1.15]), height=260, showlegend=False,
                margin=dict(l=20, r=20, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# 2. 성능 비교
# ════════════════════════════════════════════
elif page == "📊 성능 비교":
    st.subheader("Table 13 — 전체 성능 비교 (0~1)")
    perf = {
        "시스템": ["Vector-Only", "GraphRAG", "HybridRAG", "Adaptive-RAG", "Triple-Hybrid"],
        "F1": [0.72, 0.79, 0.81, 0.78, 0.86],
        "EM": [0.58, 0.68, 0.71, 0.66, 0.78],
        "Recall@3": [0.81, 0.86, 0.88, 0.84, 0.92],
        "Precision": [0.69, 0.75, 0.79, 0.74, 0.84],
        "Faithfulness": [0.71, 0.78, 0.82, 0.76, 0.89],
    }
    df = pd.DataFrame(perf)
    st.dataframe(
        df.style.highlight_max(subset=df.columns[1:], color="#D85A30"),
        hide_index=True,
        use_container_width=True,
    )

    # 성능 비교 차트
    fig_perf = go.Figure()
    colors = {"Vector-Only": "#378ADD", "GraphRAG": "#D85A30", "HybridRAG": "#D4537E",
              "Adaptive-RAG": "#BA7517", "Triple-Hybrid": "#1D9E75"}
    for _, row in df.iterrows():
        fig_perf.add_trace(go.Bar(
            name=row["시스템"],
            x=["F1", "EM", "Recall@3", "Precision", "Faithfulness"],
            y=[row["F1"], row["EM"], row["Recall@3"], row["Precision"], row["Faithfulness"]],
            marker_color=colors.get(row["시스템"], "#888"),
        ))
    fig_perf.update_layout(
        barmode="group", yaxis=dict(range=[0, 1]),
        height=350, margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")
    st.subheader("Table 14 — 질의 유형별 EM 점수")
    st.dataframe(
        pd.DataFrame({
            "질의 유형": ["Simple", "Multi-hop", "Conditional"],
            "V-Only Raw": [0.62, 0.25, 0.36],
            "V-Only Norm": [0.68, 0.31, 0.42],
            "Triple Raw": [0.76, 0.91, 0.85],
            "Triple Norm": [0.82, 0.96, 0.91],
            "개선율": ["+20.6%", "+310%", "+116.7%"],
        }),
        hide_index=True,
        use_container_width=True,
    )

# ════════════════════════════════════════════
# 3. Ablation Study
# ════════════════════════════════════════════
elif page == "⚖️ Ablation Study":
    st.subheader("Table 15 — Ablation Study")
    abl = {
        "구성": ["(A) 균등 가중치", "(B) 유형별 고정", "(C) Full DWA"],
        "F1": [0.81, 0.84, 0.86],
        "EM": [0.69, 0.75, 0.78],
        "Multi-hop EM": [0.89, 0.93, 0.96],
        "Conditional EM": [0.85, 0.90, 0.94],
    }
    df2 = pd.DataFrame(abl)
    st.dataframe(
        df2.style.highlight_max(
            subset=["F1", "EM", "Multi-hop EM", "Conditional EM"], color="#D85A30"
        ),
        hide_index=True,
        use_container_width=True,
    )

    # Ablation 차트
    fig_abl = go.Figure()
    fig_abl.add_trace(go.Bar(name="F1", x=df2["구성"], y=df2["F1"], marker_color="#1D9E75"))
    fig_abl.add_trace(go.Bar(name="EM", x=df2["구성"], y=df2["EM"], marker_color="#378ADD"))
    fig_abl.add_trace(go.Bar(name="Multi-hop EM", x=df2["구성"], y=df2["Multi-hop EM"], marker_color="#D85A30"))
    fig_abl.add_trace(go.Bar(name="Conditional EM", x=df2["구성"], y=df2["Conditional EM"], marker_color="#D4537E"))
    fig_abl.update_layout(
        barmode="group", yaxis=dict(range=[0, 1]),
        height=350, margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_abl, use_container_width=True)

    st.info(
        "💡 **(A)→(B)**: 유형별 고정 가중치 적용 시 +3.7%  |  "
        "**(B)→(C)**: Full DWA 적용 시 Multi-hop EM +3.2%p 추가 향상\n\n"
        "DWA(Dynamic Weighting Algorithm)를 제거하면 F1 점수가 가장 크게 하락하며, "
        "이는 질의 유형에 따른 동적 가중치 조절이 시스템 성능에 핵심적 역할을 합니다."
    )

# ════════════════════════════════════════════
# 4. 시스템 정보
# ════════════════════════════════════════════
elif page == "ℹ️ 시스템 정보":
    st.subheader("시스템 구성")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🧠 Vector RAG")
        st.markdown(
            "FAISS + OpenAI `text-embedding-3-small` 기반 의미 유사도 검색.  \n"
            "비정형 텍스트 질의에 강점."
        )
        st.markdown("##### 🔗 Graph RAG")
        st.markdown(
            "Neo4j + Cypher 기반 지식 그래프 탐색.  \n"
            "관계 기반 질의에 강점. BFS 3-hop 탐색 지원."
        )
    with col2:
        st.markdown("##### 📐 Ontology RAG")
        st.markdown(
            "OWL / Owlready2 기반 온톨로지 추론 엔진.  \n"
            "조건부 질의 및 규칙 기반 추론에 강점."
        )
        st.markdown("##### ⚖️ DWA 알고리즘")
        st.markdown(
            "Dynamic Weighting Algorithm.  \n"
            "질의 분석 결과(c_e, c_r, c_c)를 기반으로 세 RAG의 가중치를 동적 조절."
        )

    st.markdown("---")
    st.markdown("##### 📋 상세 설정")
    info_data = {
        "항목": ["LLM", "임베딩", "Vector Store", "Graph DB", "Ontology",
                 "DWA λ", "지식 베이스", "평가 데이터셋", "GitHub"],
        "값": [
            "GPT-4o-mini (temperature=0.0)",
            "text-embedding-3-small (dim=1536)",
            "FAISS IndexFlatIP / chunk 1000 / overlap 200 / top-k=3",
            "BFS max_depth=3 / Neo4j / nodes 93 / edges 192",
            "OWL / Owlready2",
            "0.3 (grid search 0.1~0.5)",
            "30 교수 / 8 학과 / 40 과목 / 15 프로젝트 / 186 문서",
            "1,000 쌍 (Simple 40% / Multi-hop 35% / Conditional 25%)",
            "https://github.com/sdw1621/hybrid-rag-comparsion",
        ],
    }
    st.dataframe(pd.DataFrame(info_data), hide_index=True, use_container_width=True)
