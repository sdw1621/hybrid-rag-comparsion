"""
Triple-Hybrid RAG — Streamlit 테스트 앱
논문 Section IV.4 구현

실행: streamlit run streamlit_app/app.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="Triple-Hybrid RAG",
    page_icon="🔬",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {font-size:2rem; font-weight:700; color:#1F3864; margin-bottom:0.2rem;}
.sub-title  {font-size:1rem; color:#666; margin-bottom:1.5rem;}
.weight-box {background:#f0f4ff; border-radius:8px; padding:12px; margin:4px 0;}
.answer-box {background:#e8f5e9; border-radius:8px; padding:16px; border-left:4px solid #43a047;}
.badge-simple     {background:#EBF5FB; color:#1A5276; padding:2px 10px; border-radius:12px; font-size:0.85rem;}
.badge-multi_hop  {background:#E9F7EF; color:#145A32; padding:2px 10px; border-radius:12px; font-size:0.85rem;}
.badge-conditional{background:#FDEDEC; color:#922B21; padding:2px 10px; border-radius:12px; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


# ── 세션 상태 ────────────────────────────────────────────
@st.cache_resource
def load_rag():
    from src.triple_hybrid_rag import TripleHybridRAG
    api_key = os.environ.get("OPENAI_API_KEY", "")
    rag = TripleHybridRAG(openai_api_key=api_key if api_key else None)
    rag.load_university_sample()
    rag.build()
    return rag


# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
    st.markdown("### ⚙️ 설정")

    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.environ.get("OPENAI_API_KEY",""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("**λ (DWA 조정 강도)**")
    lambda_ = st.slider("lambda", 0.0, 0.5, 0.3, 0.05)

    st.markdown("**top-k 검색 수**")
    top_k = st.slider("top_k", 1, 5, 3)

    st.markdown("---")
    st.markdown("**📖 페이지 이동**")
    page = st.radio("", ["🔍 질의 테스트", "📊 성능 비교", "⚖️ Ablation Study", "ℹ️ 시스템 정보"])


# ── 메인 ─────────────────────────────────────────────────
st.markdown('<p class="main-title">🔬 Triple-Hybrid RAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Vector · Graph · Ontology 통합 검색 증강 생성 시스템</p>',
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# Page 1: 질의 테스트
# ════════════════════════════════════════════════════════
if page == "🔍 질의 테스트":
    st.subheader("질의 입력")

    example_queries = [
        "컴퓨터공학과 소속 교수는 몇 명인가요?",
        "인공지능학과 소속 45세 이하 교수는 누구인가요?",
        "소프트웨어공학과 교수가 담당하는 과목 목록은?",
        "AI기술연구 프로젝트 참여 교수들의 소속 학과는?",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("질문을 입력하세요", placeholder="예: 컴퓨터공학과 소속 교수 중 45세 이하는 누구인가요?")
    with col2:
        example = st.selectbox("예시 질의", ["직접 입력"] + example_queries)
        if example != "직접 입력":
            query = example

    if st.button("🚀 검색", use_container_width=True) and query:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OpenAI API Key를 사이드바에 입력해주세요.")
        else:
            with st.spinner("분석 중..."):
                rag = load_rag()
                rag.dwa.lambda_ = lambda_
                rag.top_k       = top_k
                result = rag.query(query)

            # 결과 표시
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⏱ 응답시간", f"{result.elapsed:.2f}s")
            c2.metric("🎯 질의 유형", result.intent.query_type)
            c3.metric("📊 복잡도", f"{result.intent.complexity_score:.2f}")
            c4.metric("🔢 엔티티 수", len(result.intent.entities))

            st.markdown("---")
            col_l, col_r = st.columns([2, 1])

            with col_l:
                st.markdown("#### 💬 답변")
                st.markdown(f'<div class="answer-box">{result.answer}</div>',
                            unsafe_allow_html=True)

                st.markdown("#### 🔍 검색 컨텍스트")
                tabs = st.tabs(["📄 Vector", "🔗 Graph", "🏛 Ontology"])
                for tab, ctxs, label in zip(
                    tabs,
                    [result.vector_contexts, result.graph_contexts, result.onto_contexts],
                    ["Vector", "Graph", "Ontology"]
                ):
                    with tab:
                        if ctxs:
                            for i, c in enumerate(ctxs, 1):
                                st.markdown(f"**[{i}]** {c}")
                        else:
                            st.info(f"{label} 컨텍스트 없음")

            with col_r:
                st.markdown("#### ⚖️ DWA 가중치")
                w = result.weights
                fig = go.Figure(go.Bar(
                    x=["Vector (α)", "Graph (β)", "Ontology (γ)"],
                    y=[w.alpha, w.beta, w.gamma],
                    marker_color=["#5B9BD5", "#70AD47", "#ED7D31"],
                    text=[f"{v:.3f}" for v in [w.alpha, w.beta, w.gamma]],
                    textposition='outside',
                ))
                fig.update_layout(
                    yaxis=dict(range=[0, 1]),
                    height=280,
                    margin=dict(t=20, b=20, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 🔬 밀도 신호")
                df_sig = pd.DataFrame({
                    "신호": ["c_e (개체)", "c_r (관계)", "c_c (제약)"],
                    "값":   [result.intent.c_e, result.intent.c_r, result.intent.c_c],
                })
                st.dataframe(df_sig, hide_index=True, use_container_width=True)

                st.markdown("#### 🧩 추출 정보")
                if result.intent.entities:
                    st.write("**개체명:**", ", ".join(result.intent.entities))
                if result.intent.relations:
                    st.write("**관계:**",   ", ".join(result.intent.relations))
                if result.intent.constraints:
                    st.write("**제약:**",   ", ".join(result.intent.constraints))


# ════════════════════════════════════════════════════════
# Page 2: 성능 비교
# ════════════════════════════════════════════════════════
elif page == "📊 성능 비교":
    st.subheader("📊 논문 Table 13 — 시스템별 성능 비교")

    # 논문 실험 결과 데이터 (Gold QA 5,000쌍 · 3회 반복 기준)
    perf_data = {
        "System":      ["Vector-Only","GraphRAG","HybridRAG","Adaptive-RAG","Triple-Hybrid"],
        "F1":          [0.71, 0.78, 0.80, 0.77, 0.85],
        "EM":          [0.57, 0.67, 0.70, 0.65, 0.77],
        "Recall@3":    [0.80, 0.85, 0.87, 0.83, 0.91],
        "Precision":   [0.68, 0.74, 0.78, 0.73, 0.83],
        "Faithfulness":[0.70, 0.77, 0.81, 0.75, 0.88],
    }
    df_perf = pd.DataFrame(perf_data)

    st.dataframe(
        df_perf.style.highlight_max(subset=df_perf.columns[1:], color='#c8e6c9'),
        hide_index=True, use_container_width=True
    )

    # 그룹 바 차트
    metrics = ["F1","EM","Recall@3","Precision","Faithfulness"]
    colors  = ["#5B9BD5","#ED7D31","#70AD47","#FFC000","#FF4444"]
    fig = go.Figure()
    for i, row in df_perf.iterrows():
        fig.add_trace(go.Bar(
            name=row['System'],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors[i],
        ))
    fig.update_layout(
        barmode='group', height=400,
        yaxis=dict(range=[0,1.1], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Query Type별 EM
    st.markdown("---")
    st.subheader("질의 유형별 EM (Table 14)")
    qt_data = {
        "Query Type":  ["Simple","Multi-hop","Conditional"],
        "V-Only Raw":  [0.61, 0.24, 0.35],
        "V-Only Norm": [0.67, 0.30, 0.41],
        "Triple Raw":  [0.75, 0.90, 0.84],
        "Triple Norm": [0.81, 0.95, 0.90],
        "Improvement": ["+20.9%", "+216.7%", "+119.5%"],
    }
    df_qt = pd.DataFrame(qt_data)
    st.dataframe(df_qt, hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════
# Page 3: Ablation Study
# ════════════════════════════════════════════════════════
elif page == "⚖️ Ablation Study":
    st.subheader("⚖️ Ablation Study — Table 15")

    abl_data = {
        "Config":          ["(A) Equal Weight","(B) Type-Fixed","(C) Full DWA"],
        "F1":              [0.79, 0.83, 0.85],
        "F1_std":          [0.02, 0.01, 0.01],
        "EM":              [0.67, 0.74, 0.77],
        "Multi-hop EM":    [0.87, 0.92, 0.95],
        "Conditional EM":  [0.83, 0.88, 0.93],
        "ΔF1 vs (C)":      ["-7.1%","-2.4%","baseline"],
    }
    df_abl = pd.DataFrame(abl_data)
    st.dataframe(
        df_abl.style.highlight_max(
            subset=["F1","EM","Multi-hop EM","Conditional EM"], color='#c8e6c9'),
        hide_index=True, use_container_width=True
    )

    fig2 = go.Figure()
    for metric, color in [("F1","#5B9BD5"),("Multi-hop EM","#FF4444"),("Conditional EM","#ED7D31")]:
        fig2.add_trace(go.Bar(
            name=metric, x=df_abl["Config"], y=df_abl[metric], marker_color=color,
        ))
    fig2.update_layout(
        barmode='group', height=350,
        yaxis=dict(range=[0.7, 1.0], title="Score"),
        legend=dict(orientation="h"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 (A)→(B): 기본 가중치 분화로 F1 +5.1% 향상 | (B)→(C): 연속 조정으로 Multi-hop EM +3.0%p 추가 향상")


# ════════════════════════════════════════════════════════
# Page 4: 시스템 정보
# ════════════════════════════════════════════════════════
elif page == "ℹ️ 시스템 정보":
    st.subheader("ℹ️ 시스템 구성")
    info = {
        "LLM": "GPT-4o-mini (gpt-4o-mini-2024-07-18)",
        "Temperature": "0.0",
        "Embedding": "text-embedding-3-small (dim=1536)",
        "Vector Index": "FAISS IndexFlatIP (cosine)",
        "Chunk Size": "1,000자 / overlap 200자",
        "top-k": "3",
        "Graph": "BFS max_depth=3 (Neo4j 호환) | 노드 2,542개 · 엣지 6,889개",
        "Ontology": "OWL / Owlready2",
        "DWA λ": "0.3 (grid search 0.1~0.5)",
        "Dataset": "Gold QA 5,000쌍 (Simple 40% · Multi-hop 35% · Conditional 25%) | Vector 1,037건 · KG 노드 2,542 · 엣지 6,889 | 60개 학과 · 577명 교수 · 1,505개 과목 · 400개 프로젝트",
        "GitHub": "https://github.com/sdw1621/hybrid-rag-comparsion",
    }
    for k, v in info.items():
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"**{k}**")
        c2.markdown(v)
