"""
Ablation Study
논문 Table 15 / Section VI.3 구현
(A) Equal Weight  α=β=γ=0.33
(B) Type-Fixed    Table 4 기본 가중치만 (Stage 2 없음)
(C) Full DWA      연속 조정 포함 (λ=0.3)
"""
import json
import numpy as np
from typing import List, Dict
from .dwa import DWA, DWAWeights
from .query_analyzer import QueryIntent
from .evaluator import Evaluator


class AblationStudy:
    """
    3가지 가중치 설정 비교
    결과: Table 15 형식 DataFrame
    """

    CONFIGS = {
        "A_Equal":     {"desc": "Equal Weight (α=β=γ=0.33)",    "lambda": 0.0, "force_equal": True},
        "B_TypeFixed": {"desc": "Type-Fixed Base Weight",        "lambda": 0.0, "force_equal": False},
        "C_FullDWA":   {"desc": "Full DWA (λ=0.3)",             "lambda": 0.3, "force_equal": False},
    }

    def __init__(self, rag_factory, dataset: List[Dict]):
        """
        rag_factory: callable → TripleHybridRAG 인스턴스 반환
        dataset    : Gold QA 리스트
        """
        self.rag_factory = rag_factory
        self.dataset     = dataset
        self.evaluator   = Evaluator()

    def _make_equal_dwa(self):
        """항상 α=β=γ=1/3 반환하는 DWA"""
        class EqualDWA(DWA):
            def compute(self, intent):
                return DWAWeights(alpha=1/3, beta=1/3, gamma=1/3)
        return EqualDWA(lambda_=0.0)

    def _make_type_fixed_dwa(self):
        """Stage 1만 적용 (λ=0) DWA"""
        return DWA(lambda_=0.0)

    def run(self, sample_size: int = 50, runs: int = 3) -> Dict:
        """
        sample_size: 전체 데이터셋 중 평가할 개수 (시간 절약)
        """
        import random
        random.seed(42)
        sample = random.sample(self.dataset, min(sample_size, len(self.dataset)))
        # 질의 유형별 비율 유지
        by_type = {'simple':[], 'multi_hop':[], 'conditional':[]}
        for d in sample:
            by_type[d['type']].append(d)

        results = {}
        for cfg_name, cfg in self.CONFIGS.items():
            print(f"\n{'='*50}")
            print(f"[{cfg_name}] {cfg['desc']}")
            print('='*50)

            rag = self.rag_factory()
            # DWA 교체
            if cfg['force_equal']:
                rag.dwa = self._make_equal_dwa()
            else:
                rag.dwa = DWA(lambda_=cfg['lambda'])

            f1s, ems, mh_ems, cond_ems = [], [], [], []
            for item in sample:
                for _ in range(runs):
                    res  = rag.query(item['query'])
                    ret  = res.vector_contexts + res.graph_contexts + res.onto_contexts
                    ev   = self.evaluator.evaluate_single(
                        res.answer, item['answer'], ret, res.vector_contexts)
                    f1s.append(ev.f1)
                    ems.append(ev.em_norm)
                    if item['type'] == 'multi_hop':
                        mh_ems.append(ev.em_norm)
                    if item['type'] == 'conditional':
                        cond_ems.append(ev.em_norm)

            results[cfg_name] = {
                "F1":            round(float(np.mean(f1s)), 4),
                "F1_std":        round(float(np.std(f1s)),  4),
                "EM":            round(float(np.mean(ems)), 4),
                "EM_std":        round(float(np.std(ems)),  4),
                "MultiHop_EM":   round(float(np.mean(mh_ems))   if mh_ems   else 0.0, 4),
                "Conditional_EM":round(float(np.mean(cond_ems)) if cond_ems else 0.0, 4),
            }
            print(f"  F1={results[cfg_name]['F1']:.4f}  EM={results[cfg_name]['EM']:.4f}"
                  f"  MH_EM={results[cfg_name]['MultiHop_EM']:.4f}")

        self._print_table(results)
        return results

    def _print_table(self, results: Dict):
        print("\n" + "="*70)
        print("Table 15. Ablation Study Results")
        print("="*70)
        print(f"{'Config':<15} {'F1':>8} {'EM':>8} {'MH_EM':>10} {'Cond_EM':>10}")
        print("-"*70)
        for cfg, r in results.items():
            delta = ""
            if cfg != "C_FullDWA":
                base_f1 = results.get("C_FullDWA", {}).get("F1", r["F1"])
                diff = r["F1"] - base_f1
                delta = f"  ({diff:+.1%})"
            print(f"{cfg:<15} {r['F1']:>8.4f} {r['EM']:>8.4f}"
                  f" {r['MultiHop_EM']:>10.4f} {r['Conditional_EM']:>10.4f}{delta}")
        print("="*70)
