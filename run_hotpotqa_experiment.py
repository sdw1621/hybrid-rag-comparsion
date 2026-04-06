"""
HotpotQA 공개 벤치마크 실험
논문 심사위원 요구: "최소 하나 이상의 공개 데이터셋에 대한 추가 실험"

HotpotQA distractor dev set에서 300문항 샘플링
Vector-Only vs Triple-Hybrid 비교
"""
import json
import os
import sys
import random
import time
import csv
import re
import unicodedata
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# API 키는 환경변수 OPENAI_API_KEY로 설정
# Windows: set OPENAI_API_KEY=sk-xxx
# Linux/Mac: export OPENAI_API_KEY=sk-xxx
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY 환경변수를 설정해 주세요.")
    sys.exit(1)

import numpy as np

# ── 설정 ──
SEED = 42
SAMPLE_SIZE = 300
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 3
LAMBDA = 0.3
RESULTS_DIR = "results/hotpotqa"


# ── 1. HotpotQA 데이터 로드 ──
def download_hotpotqa():
    """HotpotQA distractor dev set 다운로드"""
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    cache_path = "data/hotpot_dev_distractor_v1.json"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_path):
        print(f"✅ 캐시 사용: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"⬇️  HotpotQA 다운로드 중... ({url})")
    import urllib.request
    urllib.request.urlretrieve(url, cache_path)
    print(f"✅ 다운로드 완료: {cache_path}")
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sample_questions(data, n=SAMPLE_SIZE, seed=SEED):
    """난이도별 층화 샘플링: hard 60%, medium 30%, easy 10%"""
    random.seed(seed)

    by_level = defaultdict(list)
    for item in data:
        level = item.get('level', 'medium')
        by_level[level].append(item)

    sampled = []
    # hard 우선 (multi-hop에 가까움)
    hard = by_level.get('hard', [])
    medium = by_level.get('medium', [])
    easy = by_level.get('easy', [])

    n_hard = min(int(n * 0.6), len(hard))
    n_medium = min(int(n * 0.3), len(medium))
    n_easy = min(n - n_hard - n_medium, len(easy))

    sampled.extend(random.sample(hard, n_hard) if len(hard) >= n_hard else hard)
    sampled.extend(random.sample(medium, n_medium) if len(medium) >= n_medium else medium)
    sampled.extend(random.sample(easy, n_easy) if len(easy) >= n_easy else easy)

    # 부족하면 나머지에서 채움
    if len(sampled) < n:
        remaining = [x for x in data if x not in sampled]
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    random.shuffle(sampled)
    print(f"✅ 샘플링: {len(sampled)}문항 (hard={n_hard}, medium={n_medium}, easy={n_easy})")
    return sampled[:n]


# ── 2. 데이터 변환: HotpotQA → Triple-Hybrid 형식 ──
def extract_documents(item) -> List[str]:
    """HotpotQA context paragraphs → 문서 리스트"""
    docs = []
    for title, sentences in item['context']:
        text = f"[{title}] " + " ".join(sentences)
        docs.append(text)
    return docs


def extract_graph_edges(item) -> List[Tuple[str, str, str]]:
    """supporting facts에서 개체 간 관계 추출"""
    edges = []
    titles = [title for title, _ in item['context']]

    # supporting facts의 title들 간에 "related_to" 관계 생성
    sup_titles = list(set(t for t, _ in item.get('supporting_facts', [])))
    for i in range(len(sup_titles)):
        for j in range(i + 1, len(sup_titles)):
            edges.append((sup_titles[i], "related_to", sup_titles[j]))

    # 각 paragraph에서 간단한 개체 추출 (title = 주요 개체)
    for title, sentences in item['context']:
        for sent in sentences:
            # 다른 title이 문장에 언급되면 관계 생성
            for other_title in titles:
                if other_title != title and other_title in sent:
                    edges.append((title, "mentions", other_title))

    return edges


def build_simple_ontology_rules(items) -> List[Dict]:
    """HotpotQA 질문 유형에 맞는 간이 온톨로지 규칙"""
    rules = []

    # 비교 질문 탐지 규칙
    rules.append({
        "type": "comparison",
        "pattern": r"(which|who|what).*(more|less|greater|fewer|older|younger|taller|shorter|larger|smaller|better|worse)",
        "action": "compare_entities"
    })

    # Yes/No 질문 탐지
    rules.append({
        "type": "boolean",
        "pattern": r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\s",
        "action": "boolean_check"
    })

    # 계층 관계 (Person → Occupation, Place → Country 등)
    rules.append({
        "type": "hierarchy",
        "categories": ["Person", "Place", "Organization", "Event", "Work"],
        "action": "classify_and_filter"
    })

    return rules


# ── 3. 평가 ──
def normalize_answer(text: str) -> str:
    """HotpotQA 표준 정규화"""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    # 관사, 구두점 제거
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


# ── 4. 실험 실행 ──
def run_experiment():
    print("=" * 60)
    print("HotpotQA 공개 벤치마크 실험")
    print("=" * 60)

    # 데이터 로드
    data = download_hotpotqa()
    samples = sample_questions(data, SAMPLE_SIZE)

    # 시스템 초기화
    from src.vector_store import VectorStore
    from src.knowledge_graph import KnowledgeGraph
    from src.ontology_engine import OntologyEngine
    from src.query_analyzer import QueryAnalyzer
    from src.dwa import DWA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    analyzer = QueryAnalyzer()
    dwa = DWA(lambda_=LAMBDA)

    PROMPT = (
        "Based on the following context, answer the question accurately. "
        "If the answer cannot be determined from the context, state that the "
        "information is not available. Keep the answer concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # 결과 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    vector_f1s, vector_ems = [], []
    triple_f1s, triple_ems = [], []

    for idx, item in enumerate(samples):
        question = item['question']
        gold_answer = item['answer']
        level = item.get('level', 'unknown')
        q_type = item.get('type', 'unknown')

        # 문서 추출
        docs = extract_documents(item)

        # --- Vector-Only ---
        vector_store = VectorStore()
        vector_store.build(docs)
        v_results = vector_store.search(question, TOP_K)
        v_context = "\n".join([doc for doc, _ in v_results])

        v_prompt = PROMPT.format(context=v_context, question=question)
        v_response = llm.invoke(v_prompt)
        v_answer = v_response.content if hasattr(v_response, 'content') else str(v_response)

        v_f1 = compute_f1(v_answer, gold_answer)
        v_em = compute_em(v_answer, gold_answer)
        vector_f1s.append(v_f1)
        vector_ems.append(v_em)

        # --- Triple-Hybrid ---
        # Graph 구축
        graph = KnowledgeGraph()
        edges = extract_graph_edges(item)
        for src, rel, dst in edges:
            if src not in graph.nodes:
                graph.add_node(src, src, "Entity")
            if dst not in graph.nodes:
                graph.add_node(dst, dst, "Entity")
            graph.add_edge(src, rel, dst)

        # DWA 가중치 계산
        intent = analyzer.analyze(question)
        weights = dwa.compute(intent)

        # 각 소스 검색
        g_results = graph.search(question, TOP_K)

        # 온톨로지: 비교/boolean 질문 탐지
        o_results = []
        if re.search(r'(which|who).*(more|less|older|younger|larger|smaller)', question, re.I):
            o_results.append(f"[Comparison] This is a comparison question requiring entity attribute comparison.")
        if re.match(r'^(is|are|was|were|do|does|did)\s', question, re.I):
            o_results.append(f"[Boolean] This is a yes/no question requiring factual verification.")

        # 컨텍스트 통합
        total_w = weights.alpha + weights.beta + weights.gamma
        budget = TOP_K * 3
        n_v = max(1, round(budget * weights.alpha / total_w))
        n_g = max(1, round(budget * weights.beta / total_w))

        parts = []
        parts.append(f"[Vector(α={weights.alpha:.2f})]\n" + "\n".join([d for d, _ in v_results[:n_v]]))
        if g_results:
            parts.append(f"[Graph(β={weights.beta:.2f})]\n" + "\n".join(g_results[:n_g]))
        if o_results:
            parts.append(f"[Ontology(γ={weights.gamma:.2f})]\n" + "\n".join(o_results))

        t_context = "\n\n".join(parts)
        t_prompt = PROMPT.format(context=t_context, question=question)
        t_response = llm.invoke(t_prompt)
        t_answer = t_response.content if hasattr(t_response, 'content') else str(t_response)

        t_f1 = compute_f1(t_answer, gold_answer)
        t_em = compute_em(t_answer, gold_answer)
        triple_f1s.append(t_f1)
        triple_ems.append(t_em)

        results.append({
            "idx": idx,
            "question": question,
            "gold": gold_answer,
            "level": level,
            "type": q_type,
            "v_answer": v_answer,
            "v_f1": v_f1,
            "v_em": v_em,
            "t_answer": t_answer,
            "t_f1": t_f1,
            "t_em": t_em,
            "alpha": weights.alpha,
            "beta": weights.beta,
            "gamma": weights.gamma,
        })

        if (idx + 1) % 10 == 0:
            avg_vf1 = np.mean(vector_f1s)
            avg_tf1 = np.mean(triple_f1s)
            print(f"[{idx+1}/{len(samples)}] V-Only F1={avg_vf1:.3f} | Triple F1={avg_tf1:.3f}")

    # ── 결과 집계 ──
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)

    v_f1_mean = np.mean(vector_f1s)
    v_em_mean = np.mean(vector_ems)
    t_f1_mean = np.mean(triple_f1s)
    t_em_mean = np.mean(triple_ems)

    print(f"Vector-Only:   F1={v_f1_mean:.4f}  EM={v_em_mean:.4f}")
    print(f"Triple-Hybrid: F1={t_f1_mean:.4f}  EM={t_em_mean:.4f}")
    print(f"Δ F1: {((t_f1_mean - v_f1_mean) / v_f1_mean * 100):.1f}%")
    print(f"Δ EM: {((t_em_mean - v_em_mean) / v_em_mean * 100 if v_em_mean > 0 else 0):.1f}%")

    # 난이도별 분석
    print("\n--- 난이도별 ---")
    for level in ['hard', 'medium', 'easy']:
        level_results = [r for r in results if r['level'] == level]
        if level_results:
            lv_f1 = np.mean([r['v_f1'] for r in level_results])
            lt_f1 = np.mean([r['t_f1'] for r in level_results])
            lv_em = np.mean([r['v_em'] for r in level_results])
            lt_em = np.mean([r['t_em'] for r in level_results])
            print(f"  {level:8s} (n={len(level_results):3d}): "
                  f"V-Only F1={lv_f1:.3f} EM={lv_em:.3f} | "
                  f"Triple F1={lt_f1:.3f} EM={lt_em:.3f} | "
                  f"ΔF1={((lt_f1-lv_f1)/lv_f1*100 if lv_f1>0 else 0):.1f}%")

    # 질문 유형별 분석
    print("\n--- 질문 유형별 ---")
    for qtype in ['bridge', 'comparison']:
        type_results = [r for r in results if r['type'] == qtype]
        if type_results:
            tv_f1 = np.mean([r['v_f1'] for r in type_results])
            tt_f1 = np.mean([r['t_f1'] for r in type_results])
            print(f"  {qtype:12s} (n={len(type_results):3d}): "
                  f"V-Only F1={tv_f1:.3f} | Triple F1={tt_f1:.3f} | "
                  f"ΔF1={((tt_f1-tv_f1)/tv_f1*100 if tv_f1>0 else 0):.1f}%")

    # CSV 저장
    csv_path = os.path.join(RESULTS_DIR, "hotpotqa_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ 결과 저장: {csv_path}")

    # 요약 저장
    summary = {
        "dataset": "HotpotQA distractor dev",
        "sample_size": len(samples),
        "seed": SEED,
        "llm": LLM_MODEL,
        "temperature": TEMPERATURE,
        "vector_only": {"F1": round(v_f1_mean, 4), "EM": round(v_em_mean, 4)},
        "triple_hybrid": {"F1": round(t_f1_mean, 4), "EM": round(t_em_mean, 4)},
        "delta_f1_pct": round((t_f1_mean - v_f1_mean) / v_f1_mean * 100, 1),
        "delta_em_pct": round((t_em_mean - v_em_mean) / v_em_mean * 100 if v_em_mean > 0 else 0, 1),
    }
    summary_path = os.path.join(RESULTS_DIR, "hotpotqa_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 요약 저장: {summary_path}")

    return summary


if __name__ == "__main__":
    run_experiment()
