"""
실험 실행 스크립트 — 논문 Table 8~10 수치 산출
5,000 Gold QA × 3 runs → 실제 OpenAI API 호출

실행 방법:
  python run_experiment.py --api-key YOUR_KEY --sample 500
  python run_experiment.py --api-key YOUR_KEY --full   (전체 5,000개)
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_experiment(api_key: str, sample_size: int, runs: int = 3, seed: int = 42):
    os.environ["OPENAI_API_KEY"] = api_key

    print("=" * 60)
    print("Triple-Hybrid RAG — 실험 실행")
    print(f"샘플 크기: {sample_size}개  |  반복: {runs}회  |  총 질의: {sample_size * runs}회")
    print("=" * 60)

    # ── 1. RAG 빌드 ──────────────────────────────────────────
    print("\n[1/4] RAG 시스템 빌드 중...")
    from src.triple_hybrid_rag import TripleHybridRAG
    rag = TripleHybridRAG(lambda_=0.3, top_k=3)
    rag.load_university_sample(extended=True)
    rag.build()
    print("  ✅ 빌드 완료")

    # ── 2. 베이스라인 시스템 빌드 ─────────────────────────────
    print("\n[2/4] 베이스라인 시스템 빌드 중...")

    # Vector-Only
    from src.vector_store import VectorStore
    from src.query_analyzer import QueryAnalyzer
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    vector_only_store = VectorStore()
    vector_only_store.build(rag._documents)

    PROMPT = (
        "다음 컨텍스트를 기반으로 질문에 정확하게 답변하세요. "
        "컨텍스트에서 답을 찾을 수 없으면 '정보를 찾을 수 없습니다'라고 답하세요.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    def vector_only_query(q):
        docs = [d for d, _ in vector_only_store.search(q, 3)]
        ctx = "\n".join(docs)
        resp = llm.invoke(PROMPT.format(context=ctx, query=q))
        return resp.content, docs, []

    def graphrag_query(q):
        g_ctxs = rag.graph.search(q, 3)
        v_ctxs = [d for d, _ in vector_only_store.search(q, 1)]
        ctx = "\n".join(v_ctxs + g_ctxs)
        resp = llm.invoke(PROMPT.format(context=ctx, query=q))
        return resp.content, v_ctxs, g_ctxs

    def hybridrag_query(q):
        v = [d for d, _ in vector_only_store.search(q, 2)]
        g = rag.graph.search(q, 1)
        ctx = "\n".join(v + g)
        resp = llm.invoke(PROMPT.format(context=ctx, query=q))
        return resp.content, v, g

    def adaptive_query(q):
        from src.query_analyzer import QueryAnalyzer
        intent = QueryAnalyzer().analyze(q)
        if intent.query_type == "simple":
            docs = [d for d, _ in vector_only_store.search(q, 3)]
            ctx = "\n".join(docs)
            resp = llm.invoke(PROMPT.format(context=ctx, query=q))
            return resp.content, docs, []
        elif intent.query_type == "multi_hop":
            g = rag.graph.search(q, 3)
            ctx = "\n".join(g)
            resp = llm.invoke(PROMPT.format(context=ctx, query=q))
            return resp.content, [], g
        else:
            o = rag.ontology.search(q, 3)
            ctx = "\n".join(o)
            resp = llm.invoke(PROMPT.format(context=ctx, query=q))
            return resp.content, [], o

    print("  ✅ 베이스라인 준비 완료")

    # ── 3. 데이터셋 로드 ─────────────────────────────────────
    print("\n[3/4] 데이터셋 로드 중...")
    with open("data/gold_qa_5000.json", encoding="utf-8") as f:
        full_ds = json.load(f)

    random.seed(seed)
    if sample_size >= len(full_ds):
        sample_ds = full_ds
    else:
        # 유형별 비율 유지하며 샘플링
        simple_pool      = [d for d in full_ds if d["type"] == "simple"]
        multihop_pool    = [d for d in full_ds if d["type"] == "multi_hop"]
        conditional_pool = [d for d in full_ds if d["type"] == "conditional"]
        n_s = int(sample_size * 0.40)
        n_m = int(sample_size * 0.35)
        n_c = sample_size - n_s - n_m
        sample_ds = (
            random.sample(simple_pool, min(n_s, len(simple_pool))) +
            random.sample(multihop_pool, min(n_m, len(multihop_pool))) +
            random.sample(conditional_pool, min(n_c, len(conditional_pool)))
        )
    random.shuffle(sample_ds)
    print(f"  ✅ 샘플 {len(sample_ds)}개 준비 (Simple {sum(1 for d in sample_ds if d['type']=='simple')} / "
          f"Multi-hop {sum(1 for d in sample_ds if d['type']=='multi_hop')} / "
          f"Conditional {sum(1 for d in sample_ds if d['type']=='conditional')})")

    # ── 4. 평가 실행 ─────────────────────────────────────────
    print("\n[4/4] 평가 실행 중...")
    from src.evaluator import Evaluator
    ev = Evaluator()

    systems = {
        "Vector-Only":  vector_only_query,
        "GraphRAG":     graphrag_query,
        "HybridRAG":    hybridrag_query,
        "Adaptive-RAG": adaptive_query,
        "Triple-Hybrid": None,  # rag.query() 사용
    }

    results = {}
    total_calls = len(sample_ds) * runs * len(systems)
    call_count = 0
    start_all = time.time()

    for sys_name, query_fn in systems.items():
        print(f"\n  ▶ {sys_name} 평가 중...")
        f1s, ems, r3s, precs, faiths = [], [], [], [], []
        by_type = {"simple": [], "multi_hop": [], "conditional": []}

        for i, item in enumerate(sample_ds):
            q, gold, qtype = item["query"], item["answer"], item["type"]
            run_ems = []

            for r in range(runs):
                try:
                    if sys_name == "Triple-Hybrid":
                        res = rag.query(q)
                        answer = res.answer
                        retrieved = res.vector_contexts + res.graph_contexts + res.onto_contexts
                        contexts = res.vector_contexts
                    else:
                        answer, v_docs, g_docs = query_fn(q)
                        retrieved = v_docs + g_docs
                        contexts = v_docs

                    single = ev.evaluate_single(answer, gold, retrieved, contexts)
                    run_ems.append(single.em_norm)
                    f1s.append(single.f1)
                    r3s.append(single.recall_at_k)
                    precs.append(single.precision)
                    faiths.append(single.faithfulness)

                except Exception as e:
                    print(f"    ⚠️  오류 ({sys_name} / {q[:30]}): {e}")
                    run_ems.append(0.0)
                    f1s.append(0.0)
                    r3s.append(0.0)
                    precs.append(0.0)
                    faiths.append(0.0)

                call_count += 1

            em_mean = sum(run_ems) / len(run_ems)
            ems.append(em_mean)
            by_type[qtype].append(em_mean)

            # 진행률 표시
            elapsed = time.time() - start_all
            pct = call_count / total_calls * 100
            eta = (elapsed / call_count * (total_calls - call_count)) if call_count else 0
            print(f"    [{i+1}/{len(sample_ds)}] {pct:.1f}%  경과={elapsed:.0f}s  남은={eta:.0f}s  "
                  f"현재EM={em_mean:.3f}", end="\r")

        import numpy as np
        results[sys_name] = {
            "F1":          {"mean": float(np.mean(f1s)),   "std": float(np.std(f1s))},
            "EM":          {"mean": float(np.mean(ems)),   "std": float(np.std(ems))},
            "Recall@3":    {"mean": float(np.mean(r3s)),   "std": float(np.std(r3s))},
            "Precision":   {"mean": float(np.mean(precs)), "std": float(np.std(precs))},
            "Faithfulness":{"mean": float(np.mean(faiths)),"std": float(np.std(faiths))},
            "by_type": {
                t: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for t, v in by_type.items() if v
            },
        }
        print(f"\n    ✅ F1={results[sys_name]['F1']['mean']:.4f}  "
              f"EM={results[sys_name]['EM']['mean']:.4f}")

    # ── 5. 결과 저장 ─────────────────────────────────────────
    out = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(sample_ds),
            "runs": runs,
            "total_api_calls": call_count,
            "elapsed_sec": round(time.time() - start_all, 1),
        },
        "results": results,
    }

    out_path = f"data/experiment_results_{len(sample_ds)}qa_{runs}runs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 6. 결과 출력 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 최종 결과 (Table 8 형식)")
    print("=" * 60)
    print(f"{'시스템':<16} {'F1':>8} {'EM':>8} {'Recall@3':>10} {'Precision':>10} {'Faithfulness':>13}")
    print("-" * 65)
    for sname, r in results.items():
        print(f"{sname:<16} "
              f"{r['F1']['mean']:.2f}±{r['F1']['std']:.2f}  "
              f"{r['EM']['mean']:.2f}±{r['EM']['std']:.2f}  "
              f"{r['Recall@3']['mean']:.2f}±{r['Recall@3']['std']:.2f}    "
              f"{r['Precision']['mean']:.2f}±{r['Precision']['std']:.2f}    "
              f"{r['Faithfulness']['mean']:.2f}±{r['Faithfulness']['std']:.2f}")

    print(f"\n질의 유형별 EM (Table 9):")
    th = results.get("Triple-Hybrid", {}).get("by_type", {})
    vo = results.get("Vector-Only",   {}).get("by_type", {})
    for t in ["simple", "multi_hop", "conditional"]:
        if t in th and t in vo:
            delta = (th[t]["mean"] - vo[t]["mean"]) / max(vo[t]["mean"], 1e-9) * 100
            print(f"  {t:<14} V-Only={vo[t]['mean']:.4f}  Triple={th[t]['mean']:.4f}  Δ={delta:+.1f}%")

    print(f"\n✅ 결과 저장: {out_path}")
    print(f"⏱  총 소요: {round(time.time() - start_all)}초  |  API 호출: {call_count}회")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="OpenAI API Key")
    parser.add_argument("--sample",  type=int, default=500,
                        help="샘플 크기 (기본 500 / 전체 5000)")
    parser.add_argument("--full",    action="store_true",
                        help="전체 5,000개 실험 (주의: 비용 큼)")
    parser.add_argument("--runs",    type=int, default=3, help="반복 횟수")
    args = parser.parse_args()

    sample = len(json.load(open("data/gold_qa_5000.json", encoding="utf-8"))) \
             if args.full else args.sample

    run_experiment(api_key=args.api_key, sample_size=sample, runs=args.runs)
