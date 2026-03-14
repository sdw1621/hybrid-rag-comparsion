"""
Triple-Hybrid RAG — 통합 테스트 대시보드
각 모듈을 인터랙티브하게 검증 + pytest 결과 시각화

실행: streamlit run streamlit_app/test_app.py
"""
import os, sys, subprocess, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Triple-Hybrid RAG — 테스트 대시보드",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
.test-title {font-size:1.8rem; font-weight:700; color:#1a237e;}
.pass-badge {background:#e8f5e9; color:#2e7d32; border-radius:6px; padding:2px 10px; font-size:0.85rem;}
.fail-badge {background:#ffebee; color:#c62828; border-radius:6px; padding:2px 10px; font-size:0.85rem;}
.section-header {font-size:1.1rem; font-weight:600; color:#37474f; border-bottom:2px solid #e0e0e0; padding-bottom:4px; margin:16px 0 8px;}
</style>
""", unsafe_allow_html=True)


# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 테스트 대시보드")
    st.markdown("---")
    page = st.radio("페이지 선택", [
        "🔬 컴포넌트 단위 테스트",
        "📐 DWA 파라미터 분석",
        "📊 Evaluator 배치 테스트",
        "⚡ pytest 자동 실행",
    ])
    st.markdown("---")
    st.markdown("**프로젝트**")
    st.markdown("Triple-Hybrid RAG")
    st.markdown("[GitHub](https://github.com/sdw1621/hybrid-rag-comparsion)")


st.markdown('<p class="test-title">🧪 Triple-Hybrid RAG 테스트 대시보드</p>', unsafe_allow_html=True)
st.markdown("각 모듈의 동작을 인터랙티브하게 검증합니다.")


# ════════════════════════════════════════════════════════
# Page 1: 컴포넌트 단위 테스트
# ════════════════════════════════════════════════════════
if page == "🔬 컴포넌트 단위 테스트":

    tabs = st.tabs(["🧩 QueryAnalyzer", "⚖️ DWA", "📏 Evaluator", "🔗 KnowledgeGraph", "🏛 OntologyEngine"])

    # ── QueryAnalyzer ────────────────────────────────────
    with tabs[0]:
        st.markdown('<p class="section-header">QueryAnalyzer — 질의 분석기 테스트</p>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            query_input = st.text_area(
                "테스트 질의 입력",
                value="김철수 교수가 소속된 학과의 40세 이하 교수는?",
                height=80,
            )
            preset = st.selectbox("예시 질의", [
                "직접 입력",
                "김철수 교수는 누구인가요?",                          # simple
                "김철수 교수와 이영희 교수가 협력하는 연구는?",        # multi_hop
                "컴퓨터공학과 교수 중 40세 이하는 누구인가요?",        # conditional
                "이영희 교수가 소속되고 담당하는 것은?",               # multi_hop (relation×2)
                "이영희 교수를 제외한 딥러닝 담당 교수 목록",          # conditional (제외)
            ])
            if preset != "직접 입력":
                query_input = preset

        with col2:
            st.markdown("**예상 유형 가이드**")
            st.markdown("- **simple**: 개체 1개, 관계 1개 이하")
            st.markdown("- **multi_hop**: 개체 2+개 or 관계 2+개")
            st.markdown("- **conditional**: 이하/이상/제외 등 제약")

        if st.button("분석 실행", key="qa_run"):
            from src.query_analyzer import QueryAnalyzer
            qa = QueryAnalyzer()
            intent = qa.analyze(query_input)

            c1, c2, c3, c4 = st.columns(4)
            badge = {"simple": "🔵 simple", "multi_hop": "🟢 multi_hop", "conditional": "🔴 conditional"}
            c1.metric("질의 유형", badge.get(intent.query_type, intent.query_type))
            c2.metric("복잡도", f"{intent.complexity_score:.3f}")
            c3.metric("개체 수", len(intent.entities))
            c4.metric("관계 수", len(intent.relations))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**추출 정보**")
                st.json({
                    "entities":    intent.entities,
                    "relations":   intent.relations,
                    "constraints": intent.constraints,
                })
            with col_b:
                st.markdown("**밀도 신호**")
                fig = go.Figure(go.Bar(
                    x=["c_e (개체)", "c_r (관계)", "c_c (제약)"],
                    y=[intent.c_e, intent.c_r, intent.c_c],
                    marker_color=["#5B9BD5", "#70AD47", "#ED7D31"],
                    text=[f"{v:.3f}" for v in [intent.c_e, intent.c_r, intent.c_c]],
                    textposition='outside',
                ))
                fig.update_layout(yaxis=dict(range=[0, 1.1]), height=220,
                                  margin=dict(t=10, b=20, l=10, r=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # 단위 검증
            st.markdown("**자동 검증**")
            checks = [
                ("query_type 유효", intent.query_type in ('simple','multi_hop','conditional')),
                ("c_e ∈ [0,1]",    0.0 <= intent.c_e <= 1.0),
                ("c_r ∈ [0,1]",    0.0 <= intent.c_r <= 1.0),
                ("c_c ∈ [0,1]",    0.0 <= intent.c_c <= 1.0),
                ("complexity ∈ [0,1]", 0.0 <= intent.complexity_score <= 1.0),
            ]
            for name, ok in checks:
                badge_html = f'<span class="{"pass-badge" if ok else "fail-badge"}">{"✅ PASS" if ok else "❌ FAIL"}</span>'
                st.markdown(f"{badge_html} &nbsp; {name}", unsafe_allow_html=True)

    # ── DWA ─────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<p class="section-header">DWA — 동적 가중치 알고리즘 테스트</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            q_type   = st.selectbox("질의 유형", ["simple", "multi_hop", "conditional"])
            lambda_v = st.slider("λ (조정 강도)", 0.0, 0.5, 0.3, 0.05)
            c_e      = st.slider("c_e (개체 밀도)", 0.0, 1.0, 0.2, 0.05)
            c_r      = st.slider("c_r (관계 밀도)", 0.0, 1.0, 0.0, 0.05)
            c_c      = st.slider("c_c (제약 밀도)", 0.0, 1.0, 0.0, 0.05)
        with col2:
            from src.dwa import DWA
            from src.query_analyzer import QueryIntent
            intent  = QueryIntent(q_type, [], [], [], 0.0, c_e=c_e, c_r=c_r, c_c=c_c)
            dwa     = DWA(lambda_=lambda_v)
            weights = dwa.compute(intent)

            fig = go.Figure(go.Bar(
                x=["Vector (α)", "Graph (β)", "Ontology (γ)"],
                y=[weights.alpha, weights.beta, weights.gamma],
                marker_color=["#5B9BD5","#70AD47","#ED7D31"],
                text=[f"{v:.4f}" for v in [weights.alpha, weights.beta, weights.gamma]],
                textposition='outside',
            ))
            fig.update_layout(yaxis=dict(range=[0,1.1]), height=300,
                              margin=dict(t=20,b=20,l=10,r=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            total = weights.alpha + weights.beta + weights.gamma
            st.markdown(f"**합계:** `{total:.8f}` {'✅' if abs(total-1.0)<1e-6 else '❌'}")
            st.code(dwa.explain(intent))

    # ── Evaluator ────────────────────────────────────────
    with tabs[2]:
        st.markdown('<p class="section-header">Evaluator — 평가 지표 테스트</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            pred_text = st.text_area("예측 답변 (pred)", value="김철수 교수는 컴퓨터공학과 소속이다", height=80)
            gold_text = st.text_area("정답 (gold)",      value="김철수 교수는 컴퓨터공학과 소속입니다", height=80)
        with col2:
            docs_text = st.text_area("검색 문서 (줄바꿈 구분)", height=80,
                value="김철수 교수는 45세이며 컴퓨터공학과 소속이다\n이영희 교수는 38세이며 인공지능학과 소속이다")
            ctx_text  = st.text_area("컨텍스트 (줄바꿈 구분)", height=80,
                value="김철수 교수는 인공지능개론, 딥러닝을 담당한다")

        if st.button("평가 실행", key="eval_run"):
            from src.evaluator import Evaluator
            ev   = Evaluator()
            docs = [d for d in docs_text.strip().split('\n') if d]
            ctx  = [c for c in ctx_text.strip().split('\n') if c]
            result = ev.evaluate_single(pred_text, gold_text, docs, ctx)

            st.markdown("**평가 결과**")
            d = result.as_dict()
            cols = st.columns(len(d))
            for col, (metric, score) in zip(cols, d.items()):
                col.metric(metric, f"{score:.4f}")

            st.markdown("**정규화 비교**")
            c1, c2 = st.columns(2)
            c1.code(f"pred (norm): {ev.normalize(pred_text)}")
            c2.code(f"gold (norm): {ev.normalize(gold_text)}")

    # ── KnowledgeGraph ───────────────────────────────────
    with tabs[3]:
        st.markdown('<p class="section-header">KnowledgeGraph — BFS 검색 테스트</p>', unsafe_allow_html=True)

        @st.cache_resource
        def load_kg():
            from src.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            kg.load_university_data()
            return kg

        kg = load_kg()
        st.info(f"그래프: 노드 {len(kg.nodes)}개 / 엣지 {len(kg.edges)}개 로드됨")

        col1, col2 = st.columns([2, 1])
        with col1:
            kg_query = st.text_input("검색 키워드", value="김철수")
            kg_topk  = st.slider("top_k", 1, 10, 3)
        with col2:
            kg_hops  = st.slider("max_hops", 1, 3, 3)

        if st.button("그래프 검색", key="kg_run"):
            results = kg._bfs_search(kg_query, kg_topk, kg_hops)
            if results:
                st.markdown("**검색 결과 (경로)**")
                for i, r in enumerate(results, 1):
                    st.markdown(f"`[{i}]` {r}")
            else:
                st.warning("결과 없음")

            # 그래프 노드 시각화
            st.markdown("**그래프 노드 현황**")
            node_df = pd.DataFrame([
                {"ID": nid, "이름": info["name"], "유형": info["type"]}
                for nid, info in kg.nodes.items()
            ])
            st.dataframe(node_df, hide_index=True, use_container_width=True)

    # ── OntologyEngine ───────────────────────────────────
    with tabs[4]:
        st.markdown('<p class="section-header">OntologyEngine — 온톨로지 제약 테스트</p>', unsafe_allow_html=True)

        @st.cache_resource
        def load_onto():
            from src.ontology_engine import OntologyEngine
            return OntologyEngine()

        onto = load_onto()

        col1, col2 = st.columns(2)
        with col1:
            onto_query = st.text_input("검색 질의", value="이영희")
            onto_topk  = st.slider("top_k ", 1, 5, 3)
            if st.button("온톨로지 검색", key="onto_search"):
                results = onto.search(onto_query, onto_topk)
                for r in results:
                    st.markdown(f"- {r}")

        with col2:
            st.markdown("**제약 조건 검증**")
            entity_name  = st.selectbox("교수", ["김철수","이영희","박민수","정수진"])
            constraint   = st.text_input("제약 표현식", value="40세 이하")
            if st.button("제약 검증", key="onto_check"):
                ok = onto.check_constraint(entity_name, constraint)
                if ok:
                    st.success(f"✅ {entity_name} → '{constraint}' 조건 충족")
                else:
                    st.error(f"❌ {entity_name} → '{constraint}' 조건 불충족")

        # 전체 교수 × 제약 매트릭스
        st.markdown("---")
        st.markdown("**제약 조건 매트릭스 (전체 교수)**")
        constraints_list = ["36세 미만", "40세 이하", "45세 이하", "50세 이상", "52세 이상"]
        professors_list  = ["김철수","이영희","박민수","정수진"]
        matrix = {
            c: [onto.check_constraint(p, c) for p in professors_list]
            for c in constraints_list
        }
        df_matrix = pd.DataFrame(matrix, index=professors_list)
        df_display = df_matrix.map(lambda x: "✅" if x else "❌")
        st.dataframe(df_display, use_container_width=True)


# ════════════════════════════════════════════════════════
# Page 2: DWA 파라미터 분석
# ════════════════════════════════════════════════════════
elif page == "📐 DWA 파라미터 분석":
    st.subheader("📐 DWA 파라미터 민감도 분석")

    from src.dwa import DWA
    from src.query_analyzer import QueryIntent

    # λ 민감도 분석
    st.markdown('<p class="section-header">λ 변화에 따른 가중치 변화 (c_r=0.5, c_c=0.3 고정)</p>',
                unsafe_allow_html=True)

    q_type_sel = st.selectbox("질의 유형 선택", ["simple", "multi_hop", "conditional"])
    lambda_vals = np.arange(0.0, 0.55, 0.05)
    alphas, betas, gammas = [], [], []
    for lam in lambda_vals:
        intent = QueryIntent(q_type_sel, [], [], [], 0.0, c_e=0.3, c_r=0.5, c_c=0.3)
        w = DWA(lambda_=float(lam)).compute(intent)
        alphas.append(w.alpha); betas.append(w.beta); gammas.append(w.gamma)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambda_vals, y=alphas, name="α (Vector)", line=dict(color="#5B9BD5", width=2)))
    fig.add_trace(go.Scatter(x=lambda_vals, y=betas,  name="β (Graph)",  line=dict(color="#70AD47", width=2)))
    fig.add_trace(go.Scatter(x=lambda_vals, y=gammas, name="γ (Ontology)", line=dict(color="#ED7D31", width=2)))
    fig.update_layout(
        xaxis_title="λ", yaxis_title="가중치", yaxis=dict(range=[0,1]),
        height=350, margin=dict(t=20, b=40), legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # c_r × c_c 히트맵
    st.markdown('<p class="section-header">c_r × c_c 히트맵 (λ=0.3)</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    cr_vals = np.arange(0.0, 1.1, 0.1)
    cc_vals = np.arange(0.0, 1.1, 0.1)

    for col, (weight_name, weight_attr) in zip(
        [col1, col2, col3],
        [("α (Vector)", "alpha"), ("β (Graph)", "beta"), ("γ (Ontology)", "gamma")]
    ):
        grid = np.zeros((len(cr_vals), len(cc_vals)))
        for i, cr in enumerate(cr_vals):
            for j, cc in enumerate(cc_vals):
                intent = QueryIntent(q_type_sel, [], [], [], 0.0, c_e=0.2, c_r=float(cr), c_c=float(cc))
                w = DWA(lambda_=0.3).compute(intent)
                grid[i, j] = getattr(w, weight_attr)

        fig_hm = go.Figure(go.Heatmap(
            z=grid,
            x=[f"{v:.1f}" for v in cc_vals],
            y=[f"{v:.1f}" for v in cr_vals],
            colorscale="Blues",
            zmin=0, zmax=1,
        ))
        fig_hm.update_layout(
            title=weight_name,
            xaxis_title="c_c", yaxis_title="c_r",
            height=280, margin=dict(t=40, b=30, l=40, r=10),
        )
        col.plotly_chart(fig_hm, use_container_width=True)

    # 3가지 질의 유형 비교
    st.markdown('<p class="section-header">질의 유형별 기본 가중치 비교 (c_r=0, c_c=0)</p>',
                unsafe_allow_html=True)
    types = ["simple", "multi_hop", "conditional"]
    base_weights = []
    for qt in types:
        intent = QueryIntent(qt, [], [], [], 0.0, c_e=0.0, c_r=0.0, c_c=0.0)
        w = DWA(lambda_=0.3).compute(intent)
        base_weights.append({"유형": qt, "α Vector": w.alpha, "β Graph": w.beta, "γ Ontology": w.gamma})
    df_bw = pd.DataFrame(base_weights)

    fig_type = go.Figure()
    for metric, color in [("α Vector","#5B9BD5"),("β Graph","#70AD47"),("γ Ontology","#ED7D31")]:
        fig_type.add_trace(go.Bar(name=metric, x=df_bw["유형"], y=df_bw[metric], marker_color=color))
    fig_type.update_layout(
        barmode='group', yaxis=dict(range=[0,1], title="가중치"),
        height=300, margin=dict(t=20,b=20), legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_type, use_container_width=True)


# ════════════════════════════════════════════════════════
# Page 3: Evaluator 배치 테스트
# ════════════════════════════════════════════════════════
elif page == "📊 Evaluator 배치 테스트":
    st.subheader("📊 Evaluator 배치 평가 테스트")

    from src.evaluator import Evaluator
    ev = Evaluator()

    # 기본 테스트 케이스
    default_cases = [
        {"pred": "김철수 교수는 컴퓨터공학과 소속이다",       "gold": "김철수 교수는 컴퓨터공학과 소속입니다",    "type": "simple"},
        {"pred": "이영희 교수는 딥러닝과 컴퓨터비전을 담당한다", "gold": "이영희 교수가 담당하는 과목은 딥러닝, 컴퓨터비전입니다", "type": "simple"},
        {"pred": "인공지능학과",                            "gold": "컴퓨터공학과",                      "type": "simple"},
        {"pred": "AI융합프로젝트에 김철수, 이영희 교수가 참여한다", "gold": "AI융합프로젝트 참여 교수: 김철수, 이영희",  "type": "multi_hop"},
        {"pred": "이영희(38세)와 정수진(36세)",              "gold": "이영희 교수와 정수진 교수가 40세 이하",   "type": "conditional"},
    ]

    st.markdown("**테스트 케이스 (편집 가능)**")
    edited_cases = []
    for i, case in enumerate(default_cases):
        with st.expander(f"케이스 {i+1}: [{case['type']}]", expanded=(i==0)):
            col1, col2 = st.columns(2)
            pred = col1.text_input(f"pred_{i}", value=case["pred"], label_visibility="collapsed")
            gold = col2.text_input(f"gold_{i}", value=case["gold"], label_visibility="collapsed")
            edited_cases.append({"pred": pred, "gold": gold, "type": case["type"]})

    if st.button("배치 평가 실행", use_container_width=True):
        results_data = []
        for case in edited_cases:
            r = ev.evaluate_single(case["pred"], case["gold"], [], [case["pred"]])
            d = r.as_dict()
            d["유형"] = case["type"]
            d["pred (norm)"] = ev.normalize(case["pred"])[:40]
            d["gold (norm)"] = ev.normalize(case["gold"])[:40]
            results_data.append(d)

        df_results = pd.DataFrame(results_data)
        metric_cols = ["F1", "EM_norm", "Recall@3", "Precision", "Faithfulness"]

        st.markdown("**결과 테이블**")
        st.dataframe(
            df_results[["유형","pred (norm)","gold (norm)"] + metric_cols]
            .style.background_gradient(subset=metric_cols, cmap="RdYlGn", vmin=0, vmax=1),
            hide_index=True, use_container_width=True,
        )

        st.markdown("**메트릭별 평균**")
        means = df_results[metric_cols].mean()
        cols  = st.columns(len(metric_cols))
        for col, (metric, score) in zip(cols, means.items()):
            col.metric(metric, f"{score:.4f}")

        # 레이더 차트
        fig = go.Figure(go.Scatterpolar(
            r=[float(means[m]) for m in metric_cols],
            theta=metric_cols,
            fill='toself',
            line_color='#5B9BD5',
            fillcolor='rgba(91,155,213,0.3)',
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            height=350, margin=dict(t=20,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 유형별 F1
        if len(df_results) > 1:
            st.markdown("**질의 유형별 F1**")
            type_f1 = df_results.groupby("유형")["F1"].mean().reset_index()
            fig2 = px.bar(type_f1, x="유형", y="F1",
                          color="유형", range_y=[0,1],
                          color_discrete_map={"simple":"#5B9BD5","multi_hop":"#70AD47","conditional":"#ED7D31"})
            fig2.update_layout(height=280, margin=dict(t=20,b=20), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════
# Page 4: pytest 자동 실행
# ════════════════════════════════════════════════════════
elif page == "⚡ pytest 자동 실행":
    st.subheader("⚡ pytest 자동 실행 & 결과 시각화")

    root_dir = os.path.join(os.path.dirname(__file__), '..')

    test_files = {
        "test_dwa.py (기존)":        "tests/test_dwa.py",
        "test_integration.py (신규)": "tests/test_integration.py",
        "전체 테스트":               "tests/",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_test = st.selectbox("실행할 테스트", list(test_files.keys()))
    with col2:
        verbose = st.checkbox("상세 출력 (-v)", value=True)
        tb_mode = st.selectbox("Traceback", ["short", "line", "no"])

    if st.button("🚀 pytest 실행", use_container_width=True):
        test_path = os.path.join(root_dir, test_files[selected_test])
        cmd = [sys.executable, "-m", "pytest", test_path, "--tb=" + tb_mode, "--no-header"]
        if verbose:
            cmd.append("-v")

        with st.spinner("테스트 실행 중..."):
            start_t = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=root_dir,
                    timeout=120,
                )
                elapsed_t = time.time() - start_t
                stdout = proc.stdout
                stderr = proc.stderr
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                st.error("테스트 시간 초과 (120초)")
                st.stop()

        # 결과 파싱
        lines = stdout.split('\n')
        passed, failed, errors, total = 0, 0, 0, 0
        summary_line = ""
        for line in reversed(lines):
            if "passed" in line or "failed" in line or "error" in line:
                summary_line = line.strip()
                import re
                p = re.search(r'(\d+) passed', line)
                f = re.search(r'(\d+) failed', line)
                e = re.search(r'(\d+) error', line)
                if p: passed = int(p.group(1))
                if f: failed = int(f.group(1))
                if e: errors = int(e.group(1))
                total = passed + failed + errors
                break

        # 요약 메트릭
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("전체", total)
        col2.metric("통과", passed, delta=None)
        col3.metric("실패", failed, delta=None)
        col4.metric("소요시간", f"{elapsed_t:.1f}s")

        if returncode == 0:
            st.success(f"✅ 모든 테스트 통과! {summary_line}")
        else:
            st.error(f"❌ 테스트 실패 — {summary_line}")

        # 파이 차트
        if total > 0:
            fig = go.Figure(go.Pie(
                labels=["통과", "실패", "오류"],
                values=[passed, failed, errors],
                marker_colors=["#43a047", "#e53935", "#fb8c00"],
                hole=0.4,
            ))
            fig.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        # 테스트별 결과 (verbose 모드)
        if verbose:
            test_lines = [l for l in lines if " PASSED" in l or " FAILED" in l or " ERROR" in l]
            if test_lines:
                st.markdown("**테스트별 결과**")
                for tl in test_lines:
                    icon = "✅" if "PASSED" in tl else "❌"
                    st.markdown(f"{icon} `{tl.strip()}`")

        # 전체 출력
        with st.expander("전체 pytest 출력 보기"):
            st.code(stdout, language="text")
        if stderr:
            with st.expander("stderr"):
                st.code(stderr, language="text")

    st.markdown("---")
    st.markdown("**빠른 테스트 (컴포넌트 mock)**")
    if st.button("기본 동작 검증 (API 불필요)", key="quick_test"):
        results = []
        try:
            from src.query_analyzer import QueryAnalyzer
            qa = QueryAnalyzer()
            intent = qa.analyze("김철수 교수가 소속된 학과의 40세 이하 교수는?")
            results.append(("QueryAnalyzer — conditional 분류", intent.query_type == 'conditional'))
            results.append(("QueryAnalyzer — entities 추출", len(intent.entities) > 0))
            results.append(("QueryAnalyzer — constraints 추출", len(intent.constraints) > 0))
        except Exception as e:
            results.append((f"QueryAnalyzer 오류: {e}", False))

        try:
            from src.dwa import DWA
            from src.query_analyzer import QueryIntent
            dwa = DWA(0.3)
            w = dwa.compute(QueryIntent('conditional',[],[],[],0.0,c_e=0.2,c_r=0.3,c_c=0.5))
            results.append(("DWA — 가중치 합=1.0", abs(w.alpha+w.beta+w.gamma-1.0)<1e-6))
            results.append(("DWA — conditional: gamma 우세", w.gamma > w.alpha))
        except Exception as e:
            results.append((f"DWA 오류: {e}", False))

        try:
            from src.evaluator import Evaluator
            ev = Evaluator()
            r = ev.evaluate_single("김철수 교수", "김철수 교수", [], [])
            results.append(("Evaluator — F1 perfect match", r.f1 == 1.0))
            results.append(("Evaluator — EM_norm perfect match", r.em_norm == 1.0))
        except Exception as e:
            results.append((f"Evaluator 오류: {e}", False))

        try:
            from src.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            kg.load_university_data()
            results.append(("KnowledgeGraph — 노드 13개", len(kg.nodes) == 13))
            results.append(("KnowledgeGraph — 엣지 15개", len(kg.edges) == 15))
            r = kg.search("김철수", top_k=3)
            results.append(("KnowledgeGraph — BFS 결과 있음", len(r) > 0))
        except Exception as e:
            results.append((f"KnowledgeGraph 오류: {e}", False))

        try:
            from src.ontology_engine import OntologyEngine
            onto = OntologyEngine()
            results.append(("OntologyEngine — 이영희 40세이하 True", onto.check_constraint("이영희","40세 이하") is True))
            results.append(("OntologyEngine — 박민수 40세이하 False", onto.check_constraint("박민수","40세 이하") is False))
        except Exception as e:
            results.append((f"OntologyEngine 오류: {e}", False))

        passed_q = sum(1 for _, ok in results if ok)
        total_q  = len(results)

        st.markdown(f"**결과: {passed_q}/{total_q} 통과**")
        for name, ok in results:
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} {name}")
