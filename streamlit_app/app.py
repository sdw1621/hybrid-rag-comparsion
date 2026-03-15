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

st.set_page_config(page_title="Triple-Hybrid RAG", page_icon="o", layout="wide")

st.sidebar.markdown("### Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY",""))
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
st.sidebar.markdown("---")
lambda_ = st.sidebar.slider("lambda (DWA)", 0.0, 0.5, 0.3, 0.05)
top_k = st.sidebar.slider("top-k", 1, 5, 3)
st.sidebar.markdown("---")
page = st.sidebar.radio("Page", ["Query Test","Performance","Ablation","System Info"], label_visibility="visible")

@st.cache_resource
def load_rag():
    from src.triple_hybrid_rag import TripleHybridRAG
    rag = TripleHybridRAG(lambda_=0.3, top_k=3)
    rag.load_university_sample(extended=True)
    rag.build()
    return rag

st.markdown("## Triple-Hybrid RAG")
st.markdown("Vector - Graph - Ontology Integration")

if page == "Query Test":
    st.subheader("Query Input")
    examples = [
        "What courses does Prof. Kim Cheol-su teach?",
        "Who are professors under 40 in CS dept?",
        "Who are in the same dept as Prof. Lee?",
        "Who is an assistant professor in AI dept?",
    ]
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("Enter your question", placeholder="e.g. What is Prof. Kim's research area?")
    with c2:
        ex = st.selectbox("Examples", ["Direct input"] + examples)
    if ex != "Direct input":
        query = ex

    if st.button("Search", use_container_width=True) and query:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please enter OpenAI API Key in the sidebar.")
            st.stop()
        with st.spinner("Processing..."):
            try:
                rag = load_rag()
                rag.dwa.lambda_ = lambda_
                rag.top_k = top_k
                _t0 = time.time()
                result = rag.query(query)
                _wall = time.time() - _t0
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Response Time", f"{_wall:.2f}s")
        m2.metric("Query Type", result.intent.query_type)
        m3.metric("Complexity", f"{result.intent.complexity_score:.2f}")
        m4.metric("Entities", len(result.intent.entities))
        st.markdown("---")

        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Answer")
            st.markdown(f'<div style="background:#e8f5e9;border-radius:8px;padding:16px;border-left:4px solid #43a047">{result.answer}</div>', unsafe_allow_html=True)
            st.markdown("#### Context")
            t1, t2, t3 = st.tabs(["Vector", "Graph", "Ontology"])
            for tab, ctxs in zip([t1, t2, t3], [result.vector_contexts, result.graph_contexts, result.onto_contexts]):
                with tab:
                    for i, c in enumerate(ctxs, 1):
                        st.markdown(f"**[{i}]** {c}")
                    if not ctxs:
                        st.info("No results")
        with right:
            st.markdown("#### DWA Weights")
            w = result.weights
            fig = go.Figure(go.Bar(
                x=["Vector(a)", "Graph(b)", "Ontology(g)"],
                y=[w.alpha, w.beta, w.gamma],
                marker_color=["#5B9BD5", "#70AD47", "#ED7D31"],
                text=[f"{v:.3f}" for v in [w.alpha, w.beta, w.gamma]],
                textposition='outside',
            ))
            fig.update_layout(yaxis=dict(range=[0, 1.15]), height=260, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Performance":
    st.subheader("Table 13 - Overall Performance (0~1 scale)")
    perf = {"System": ["Vector-Only", "GraphRAG", "HybridRAG", "Adaptive-RAG", "Triple-Hybrid"],
            "F1": [0.72, 0.79, 0.81, 0.78, 0.86], "EM": [0.58, 0.68, 0.71, 0.66, 0.78],
            "Recall@3": [0.81, 0.86, 0.88, 0.84, 0.92], "Precision": [0.69, 0.75, 0.79, 0.74, 0.84],
            "Faithfulness": [0.71, 0.78, 0.82, 0.76, 0.89]}
    import pandas as pd
    df = pd.DataFrame(perf)
    st.dataframe(df.style.highlight_max(subset=df.columns[1:], color='#c8e6c9'), hide_index=True, use_container_width=True)
    st.markdown("---")
    st.subheader("Table 14 - EM by Query Type")
    st.dataframe(pd.DataFrame({
        "Query Type": ["Simple", "Multi-hop", "Conditional"],
        "V-Only Raw": [0.62, 0.25, 0.36], "V-Only Norm": [0.68, 0.31, 0.42],
        "Triple Raw": [0.76, 0.91, 0.85], "Triple Norm": [0.82, 0.96, 0.91],
        "Improvement": ["+20.6%", "+310%", "+116.7%"]
    }), hide_index=True, use_container_width=True)

elif page == "Ablation":
    st.subheader("Table 15 - Ablation Study")
    abl = {"Config": ["(A) Equal", "(B) Type-Fixed", "(C) Full DWA"],
           "F1": [0.81, 0.84, 0.86], "EM": [0.69, 0.75, 0.78],
           "Multi-hop EM": [0.89, 0.93, 0.96], "Conditional EM": [0.85, 0.90, 0.94]}
    df2 = pd.DataFrame(abl)
    st.dataframe(df2.style.highlight_max(subset=["F1", "EM", "Multi-hop EM", "Conditional EM"], color='#c8e6c9'),
                 hide_index=True, use_container_width=True)
    st.info("(A)->(B): +3.7% | (B)->(C): Multi-hop EM +3.2%p")

elif page == "System Info":
    st.subheader("System Configuration")
    for k, v in {
        "LLM": "GPT-4o-mini (temperature=0.0)",
        "Embedding": "text-embedding-3-small (dim=1536)",
        "Vector": "FAISS IndexFlatIP / chunk 1000 / overlap 200 / top-k=3",
        "Graph": "BFS max_depth=3 / Neo4j / nodes 93 / edges 192",
        "Ontology": "OWL / Owlready2",
        "DWA lambda": "0.3 (grid search 0.1~0.5)",
        "Knowledge Base": "30 professors / 8 depts / 40 courses / 15 projects / 186 docs",
        "Gold QA": "1000 pairs (Simple 40% / Multi-hop 35% / Conditional 25%)",
        "GitHub": "https://github.com/sdw1621/hybrid-rag-comparsion",
    }.items():
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"**{k}**")
        c2.markdown(v)
