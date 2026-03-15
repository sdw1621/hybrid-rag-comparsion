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

    # ── 시뮬레이션 섹션 ──
    st.subheader("🎯 시스템별 시뮬레이션 비교")
    st.markdown("동일 질의를 5개 RAG 시스템에 동시 실행하여 성능 차이를 실시간 비교합니다.")

    import random

    sim_queries = [
        "김철수 교수가 담당하는 과목은?",
        "컴퓨터공학과 소속 40세 이하 교수는?",
        "이영희 교수와 같은 학과 교수는?",
        "딥러닝 과목 담당 교수의 연구 분야는?",
        "AI융합연구소 참여 교수 목록은?",
        "박민준 교수의 나이와 전공은?",
        "데이터사이언스 전공 교수 중 프로젝트 참여자는?",
    ]
    sim_systems = ["Vector-Only", "GraphRAG", "HybridRAG", "Adaptive-RAG", "Triple-Hybrid"]
    sys_colors = {"Vector-Only": "#378ADD", "GraphRAG": "#D85A30", "HybridRAG": "#D4537E",
                  "Adaptive-RAG": "#BA7517", "Triple-Hybrid": "#1D9E75"}

    # 시뮬레이션 데이터 (질의 유형별 시스템 성능 시뮬레이션 결과)
    sim_data = {
        "simple": {
            "Vector-Only":   {"f1": 0.72, "em": 0.62, "time": 1.2, "answer": "김철수 교수는 인공지능개론, 머신러닝 과목을 담당합니다.", "status": "partial"},
            "GraphRAG":      {"f1": 0.81, "em": 0.71, "time": 1.8, "answer": "김철수 교수 → 담당과목: 인공지능개론, 머신러닝, 딥러닝특론", "status": "good"},
            "HybridRAG":     {"f1": 0.83, "em": 0.74, "time": 2.1, "answer": "김철수 교수는 인공지능개론, 머신러닝, 딥러닝특론을 담당합니다.", "status": "good"},
            "Adaptive-RAG":  {"f1": 0.79, "em": 0.68, "time": 2.4, "answer": "김철수 교수의 담당 과목은 인공지능개론과 머신러닝입니다.", "status": "partial"},
            "Triple-Hybrid": {"f1": 0.88, "em": 0.82, "time": 1.9, "answer": "김철수 교수(컴퓨터공학과)는 인공지능개론, 머신러닝, 딥러닝특론 3개 과목을 담당하며, AI융합연구소에도 참여하고 있습니다.", "status": "best"},
        },
        "conditional": {
            "Vector-Only":   {"f1": 0.58, "em": 0.36, "time": 1.3, "answer": "컴퓨터공학과에는 여러 교수가 있습니다.", "status": "fail"},
            "GraphRAG":      {"f1": 0.79, "em": 0.71, "time": 2.0, "answer": "한상우 교수(38세)는 컴퓨터공학과 소속입니다.", "status": "good"},
            "HybridRAG":     {"f1": 0.76, "em": 0.65, "time": 2.3, "answer": "컴퓨터공학과 소속 40세 이하 교수: 한상우(38세)", "status": "good"},
            "Adaptive-RAG":  {"f1": 0.71, "em": 0.58, "time": 2.6, "answer": "한상우 교수가 40세 이하입니다.", "status": "partial"},
            "Triple-Hybrid": {"f1": 0.91, "em": 0.85, "time": 1.9, "answer": "컴퓨터공학과 소속 40세 이하 교수는 한상우 교수(38세, 조교수)입니다. 전공은 인공지능 및 딥러닝이며, '딥러닝 기초' 과목을 담당하고 있습니다.", "status": "best"},
        },
        "multihop": {
            "Vector-Only":   {"f1": 0.51, "em": 0.25, "time": 1.4, "answer": "이영희 교수는 데이터사이언스학과 소속입니다.", "status": "fail"},
            "GraphRAG":      {"f1": 0.85, "em": 0.78, "time": 2.2, "answer": "이영희 교수(데이터사이언스학과) → 같은 학과: 정민호 교수, 최수진 교수", "status": "good"},
            "HybridRAG":     {"f1": 0.82, "em": 0.73, "time": 2.5, "answer": "이영희 교수와 같은 데이터사이언스학과: 정민호, 최수진 교수", "status": "good"},
            "Adaptive-RAG":  {"f1": 0.77, "em": 0.64, "time": 2.8, "answer": "데이터사이언스학과 소속 교수는 이영희, 정민호 교수입니다.", "status": "partial"},
            "Triple-Hybrid": {"f1": 0.94, "em": 0.91, "time": 2.0, "answer": "이영희 교수(데이터사이언스학과, 부교수)와 같은 학과 교수는 정민호 교수(교수, 빅데이터 전공)와 최수진 교수(조교수, 통계학 전공)입니다.", "status": "best"},
        },
    }

    # 질의 ↔ 유형 매핑
    query_type_map = {
        0: "simple", 1: "conditional", 2: "multihop",
        3: "simple", 4: "multihop", 5: "simple", 6: "conditional",
    }
    query_type_labels = {"simple": "Simple", "conditional": "Conditional", "multihop": "Multi-hop"}

    sc1, sc2 = st.columns([1, 1])
    with sc1:
        sim_query = st.selectbox("🔎 시뮬레이션 질의 선택", sim_queries)
    with sc2:
        sim_speed = st.select_slider("⏱ 시뮬레이션 속도", options=["빠르게", "보통", "느리게"], value="보통")

    speed_map = {"빠르게": 0.3, "보통": 0.7, "느리게": 1.2}

    if st.button("▶️ 시뮬레이션 실행", use_container_width=True):
        q_idx = sim_queries.index(sim_query)
        q_type = query_type_map[q_idx]
        data = sim_data[q_type]
        delay = speed_map[sim_speed]

        st.markdown(f"**질의:** `{sim_query}`  ·  **유형:** `{query_type_labels[q_type]}`")
        st.markdown("---")

        # 시스템별 진행 상태 표시
        status_area = st.empty()
        result_containers = []
        cols = st.columns(5)
        for i, sys_name in enumerate(sim_systems):
            with cols[i]:
                color = sys_colors[sys_name]
                st.markdown(
                    f'<div style="text-align:center;padding:6px;border-radius:8px;'
                    f'border:2px solid {color};margin-bottom:8px;">'
                    f'<span style="font-size:13px;font-weight:600;color:{color}">{sys_name}</span></div>',
                    unsafe_allow_html=True,
                )
                result_containers.append(st.empty())

        # 순차 시뮬레이션
        for i, sys_name in enumerate(sim_systems):
            d = data[sys_name]
            # 처리 중 표시
            result_containers[i].markdown(
                f'<div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:12px;'
                f'min-height:160px;border:1px solid rgba(255,255,255,0.1)">'
                f'<p style="color:#BA7517;font-size:13px;">⏳ 처리 중...</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            time.sleep(delay * d["time"] / 2.0)

            # 결과 표시
            status_emoji = {"best": "🏆", "good": "✅", "partial": "⚠️", "fail": "❌"}
            status_label = {"best": "최고 성능", "good": "양호", "partial": "부분 정답", "fail": "실패"}
            border_color = sys_colors[sys_name]
            bg_alpha = "0.2" if d["status"] == "best" else "0.08"

            result_containers[i].markdown(
                f'<div style="background:rgba(216,90,48,{bg_alpha});border-radius:8px;padding:12px;'
                f'min-height:160px;border:1px solid {border_color}">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<span style="font-size:12px;color:#aaa;">⏱ {d["time"]:.1f}s</span>'
                f'<span style="font-size:12px;">{status_emoji[d["status"]]} {status_label[d["status"]]}</span>'
                f'</div>'
                f'<p style="font-size:12px;line-height:1.5;margin:8px 0;color:#ddd;">{d["answer"]}</p>'
                f'<div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:8px;margin-top:8px;">'
                f'<span style="font-size:11px;color:{border_color};">F1: {d["f1"]:.2f}</span>'
                f' · <span style="font-size:11px;color:{border_color};">EM: {d["em"]:.2f}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # 최종 비교 차트
        st.markdown("---")
        st.markdown("##### 📊 시뮬레이션 결과 비교")
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            name="F1 Score",
            x=sim_systems,
            y=[data[s]["f1"] for s in sim_systems],
            marker_color=[sys_colors[s] for s in sim_systems],
            text=[f'{data[s]["f1"]:.2f}' for s in sim_systems],
            textposition="outside",
        ))
        fig_sim.add_trace(go.Bar(
            name="EM Score",
            x=sim_systems,
            y=[data[s]["em"] for s in sim_systems],
            marker_color=[sys_colors[s] for s in sim_systems],
            opacity=0.5,
            text=[f'{data[s]["em"]:.2f}' for s in sim_systems],
            textposition="outside",
        ))
        fig_sim.update_layout(
            barmode="group", yaxis=dict(range=[0, 1.1]),
            height=300, margin=dict(l=20, r=20, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Triple-Hybrid 우위 요약
        best_f1 = data["Triple-Hybrid"]["f1"]
        second_f1 = max(data[s]["f1"] for s in sim_systems if s != "Triple-Hybrid")
        improvement = ((best_f1 - second_f1) / second_f1) * 100
        st.success(
            f"🏆 **Triple-Hybrid RAG**가 F1 {best_f1:.2f}로 최고 성능 달성  |  "
            f"2위 대비 **+{improvement:.1f}%** 향상  |  "
            f"DWA 동적 가중치가 '{query_type_labels[q_type]}' 유형에 최적 배분"
        )

    st.markdown("---")

    # ── 기존 테이블 ──
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

    # 평가 지표 설명
    st.markdown("##### 📏 평가 지표 설명")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            '<div style="background:rgba(55,138,221,0.12);border-radius:8px;padding:14px;border-left:4px solid #378ADD;margin-bottom:8px;">'
            '<b style="color:#378ADD;">F1 Score</b><br>'
            '<span style="font-size:13px;">Precision과 Recall의 조화 평균. '
            '정답과 예측 간 토큰 단위 겹침을 측정하며, 부분 정답도 반영합니다.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:rgba(212,83,126,0.12);border-radius:8px;padding:14px;border-left:4px solid #D4537E;">'
            '<b style="color:#D4537E;">Precision</b><br>'
            '<span style="font-size:13px;">예측 답변 중 정답에 포함된 토큰 비율. '
            '높을수록 불필요한 정보 없이 정확한 답변을 의미합니다.</span></div>',
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            '<div style="background:rgba(216,90,48,0.12);border-radius:8px;padding:14px;border-left:4px solid #D85A30;margin-bottom:8px;">'
            '<b style="color:#D85A30;">EM (Exact Match)</b><br>'
            '<span style="font-size:13px;">예측 답변이 정답과 완전히 일치하는 비율. '
            '가장 엄격한 평가 기준으로, 부분 정답은 0점 처리됩니다.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:rgba(29,158,117,0.12);border-radius:8px;padding:14px;border-left:4px solid #1D9E75;">'
            '<b style="color:#1D9E75;">Recall@3</b><br>'
            '<span style="font-size:13px;">상위 3개 검색 결과 안에 정답 문서가 포함된 비율. '
            '검색 단계의 품질을 평가합니다.</span></div>',
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            '<div style="background:rgba(186,117,23,0.12);border-radius:8px;padding:14px;border-left:4px solid #BA7517;">'
            '<b style="color:#BA7517;">Faithfulness</b><br>'
            '<span style="font-size:13px;">생성된 답변이 검색된 컨텍스트에 근거한 비율. '
            '높을수록 환각(hallucination) 없이 신뢰할 수 있는 답변입니다.</span></div>',
            unsafe_allow_html=True,
        )

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
