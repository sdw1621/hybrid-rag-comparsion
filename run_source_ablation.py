"""
소스별 Ablation Study — 논문 Table 15-b 수치 산출
(D) Vector-Only
(E) Vector+Graph       (Ontology 비활성)
(F) Vector+Ontology    (Graph 비활성)
(G) Vector+Graph+Onto  (Full Triple-Hybrid)

실행:
  python run_source_ablation.py --api-key YOUR_KEY --sample 200
  python run_source_ablation.py --api-key YOUR_KEY --full
"""
import argparse
import json
import os
import random
import sys
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def query_with_disabled_sources(rag, question, disable_sources=None):
    """특정 소스를 비활성화한 상태로 질의 수행"""
    from src.dwa import DWAWeights
    import time as _time

    start = _time.time()
    intent = rag.analyzer.analyze(question)
    weights = rag.dwa.compute(intent)

    # 소스별 검색
    disable = disable_sources or []
    v_ctxs = [] if 'vector' in disable else (
        [doc for doc, _ in rag.vector.search(question, rag.top_k)] if rag._built else []
    )
    g_ctxs = [] if 'graph' in disable else rag.graph.search(question, rag.top_k)
    o_ctxs = [] if 'ontology' in disable else rag.ontology.search(question, rag.top_k)

    # 가중치 재정규화
    a = 0.0 if 'vector' in disable else weights.alpha
    b = 0.0 if 'graph' in disable else weights.beta
    g = 0.0 if 'ontology' in disable else weights.gamma
    total = a + b + g
    if total > 0:
        a, b, g = a / total, b / total, g / total

    # 컨텍스트 통합
    context = rag._merge_contexts(v_ctxs, g_ctxs, o_ctxs, a, b, g)

    # LLM 호출
    from src.triple_hybrid_rag import PROMPT_TEMPLATE
    prompt = PROMPT_TEMPLATE.format(context=context, query=question)
    response = rag.llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)

    elapsed = _time.time() - start
    return answer, v_ctxs, g_ctxs, o_ctxs, elapsed


def run_source_ablation(api_key: str, sample_size: int, runs: int = 3, seed: int = 42):
    os.environ["OPENAI_API_KEY"] = api_key

    print("=" * 60)
    print("Source-Level Ablation Study — Table 15-b")
    print(f"샘플: {sample_size}개  |  반복: {runs}회")
    print("=" * 60)

    # 1. RAG 빌드
    print("\n[1/3] RAG 시스템 빌드 중...")
    from src.triple_hybrid_rag import TripleHybridRAG
    rag = TripleHybridRAG(lambda_=0.3, top_k=3)
    rag.load_university_sample(extended=True)
    rag.build()
    print("  ✅ 빌드 완료")

    # 2. 데이터셋 로드
    print("\n[2/3] 데이터셋 로드 중...")
    with open("data/gold_qa_5000.json", encoding="utf-8") as f:
        full_ds = json.load(f)

    random.seed(seed)
    if sample_size >= len(full_ds):
        sample_ds = full_ds
    else:
        simple_pool = [d for d in full_ds if d["type"] == "simple"]
        multihop_pool = [d for d in full_ds if d["type"] == "multi_hop"]
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
    print(f"  ✅ 샘플 {len(sample_ds)}개")

    # 3. 평가 실행
    print("\n[3/3] 소스별 Ablation 평가 중...")
    from src.evaluator import Evaluator
    ev = Evaluator()

    configs = {
        "D_VectorOnly":   {"disable": ["graph", "ontology"], "desc": "Vector-Only"},
        "E_VectorGraph":  {"disable": ["ontology"],          "desc": "Vector+Graph"},
        "F_VectorOnto":   {"disable": ["graph"],             "desc": "Vector+Ontology"},
        "G_FullTriple":   {"disable": [],                    "desc": "Vector+Graph+Ontology (Full)"},
    }

    results = {}
    start_all = time.time()

    for cfg_name, cfg in configs.items():
        print(f"\n  ▶ [{cfg_name}] {cfg['desc']}...")
        f1s, ems = [], []
        by_type = {"simple": [], "multi_hop": [], "conditional": []}

        for i, item in enumerate(sample_ds):
            q, gold, qtype = item["query"], item["answer"], item["type"]
            run_ems = []

            for r in range(runs):
                try:
                    answer, v_ctxs, g_ctxs, o_ctxs, elapsed = query_with_disabled_sources(
                        rag, q, disable_sources=cfg["disable"]
                    )
                    retrieved = v_ctxs + g_ctxs + o_ctxs
                    single = ev.evaluate_single(answer, gold, retrieved, v_ctxs)
                    run_ems.append(single.em_norm)
                    f1s.append(single.f1)
                except Exception as e:
                    print(f"    ⚠️ 오류: {e}")
                    run_ems.append(0.0)
                    f1s.append(0.0)

            em_mean = sum(run_ems) / len(run_ems)
            ems.append(em_mean)
            by_type[qtype].append(em_mean)

            if (i + 1) % 20 == 0 or i == len(sample_ds) - 1:
                elapsed_total = time.time() - start_all
                print(f"    [{i+1}/{len(sample_ds)}]  경과={elapsed_total:.0f}s  현재EM={em_mean:.3f}")

        results[cfg_name] = {
            "desc": cfg["desc"],
            "F1_mean": round(float(np.mean(f1s)), 4),
            "F1_std": round(float(np.std(f1s)), 4),
            "EM_mean": round(float(np.mean(ems)), 4),
            "EM_std": round(float(np.std(ems)), 4),
            "by_type": {
                t: {
                    "mean": round(float(np.mean(v)), 4) if v else 0.0,
                    "std": round(float(np.std(v)), 4) if v else 0.0,
                }
                for t, v in by_type.items()
            },
        }
        print(f"    ✅ F1={results[cfg_name]['F1_mean']:.4f}±{results[cfg_name]['F1_std']:.4f}  "
              f"EM={results[cfg_name]['EM_mean']:.4f}±{results[cfg_name]['EM_std']:.4f}")

    # 결과 출력
    print("\n" + "=" * 80)
    print("Table 15-b. Source-Level Ablation Study (DWA applied, mean±std, 3 runs)")
    print("=" * 80)
    print(f"{'Configuration':<25} {'F1':>12} {'EM':>12} {'Multi-hop EM':>14} {'Cond. EM':>12}")
    print("-" * 80)
    for cfg_name, r in results.items():
        mh = r["by_type"].get("multi_hop", {})
        cd = r["by_type"].get("conditional", {})
        print(f"({cfg_name[0]}) {r['desc']:<20} "
              f"{r['F1_mean']:.2f}±{r['F1_std']:.2f}   "
              f"{r['EM_mean']:.2f}±{r['EM_std']:.2f}   "
              f"{mh.get('mean', 0):.2f}±{mh.get('std', 0):.2f}       "
              f"{cd.get('mean', 0):.2f}±{cd.get('std', 0):.2f}")
    print("=" * 80)

    # 저장
    out = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(sample_ds),
            "runs": runs,
            "elapsed_sec": round(time.time() - start_all, 1),
        },
        "results": results,
    }
    out_path = f"data/source_ablation_{len(sample_ds)}qa_{runs}runs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 결과 저장: {out_path}")
    print(f"⏱  총 소요: {round(time.time() - start_all)}초")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="OpenAI API Key")
    parser.add_argument("--sample", type=int, default=200, help="샘플 크기")
    parser.add_argument("--full", action="store_true", help="전체 5,000개")
    parser.add_argument("--runs", type=int, default=3, help="반복 횟수")
    args = parser.parse_args()

    sample = 5000 if args.full else args.sample
    run_source_ablation(api_key=args.api_key, sample_size=sample, runs=args.runs)
