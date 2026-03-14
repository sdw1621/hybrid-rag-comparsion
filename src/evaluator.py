"""
Evaluator — 논문 Table 11 평가지표 완전 구현
F1 Score / Exact Match (EM) / Recall@3 / Precision / Faithfulness
EM 정규화: Unicode NFC + 소문자 + 공백통일 + 한국어 조사 제거
"""
import re
import unicodedata
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    f1: float
    em_raw: float
    em_norm: float
    recall_at_k: float
    precision: float
    faithfulness: float

    def as_dict(self):
        return {
            "F1": round(self.f1, 4),
            "EM_raw": round(self.em_raw, 4),
            "EM_norm": round(self.em_norm, 4),
            "Recall@3": round(self.recall_at_k, 4),
            "Precision": round(self.precision, 4),
            "Faithfulness": round(self.faithfulness, 4),
        }


class Evaluator:
    """
    논문 Section V.2 / Table 11 전체 지표 계산
    """

    # 한국어 조사/어미 목록 (EM 정규화)
    KO_PARTICLES = [
        '은','는','이','가','을','를','의','에','에서','로','으로',
        '와','과','도','만','까지','부터','에게','한테','께',
    ]

    def normalize(self, text: str) -> str:
        """EM 정규화 (논문 Section V.2)"""
        # 1. Unicode NFC
        text = unicodedata.normalize('NFC', text)
        # 2. 소문자
        text = text.lower()
        # 3. 공백 통일
        text = re.sub(r'\s+', ' ', text).strip()
        # 4. 한국어 조사 제거
        for p in sorted(self.KO_PARTICLES, key=len, reverse=True):
            text = re.sub(rf'(?<=[가-힣]){p}(?=\s|$)', '', text)
        # 5. 숫자 표현 통일 (한글 숫자 → 아라비아)
        ko_nums = {'일':'1','이':'2','삼':'3','사':'4','오':'5',
                   '육':'6','칠':'7','팔':'8','구':'9','십':'10'}
        for k, v in ko_nums.items():
            text = text.replace(k, v)
        return text.strip()

    def exact_match(self, pred: str, gold: str, normalize: bool = True) -> float:
        if normalize:
            return 1.0 if self.normalize(pred) == self.normalize(gold) else 0.0
        return 1.0 if pred.strip() == gold.strip() else 0.0

    def f1_score(self, pred: str, gold: str) -> float:
        """토큰 레벨 F1"""
        pred_toks = set(self.normalize(pred).split())
        gold_toks = set(self.normalize(gold).split())
        if not pred_toks or not gold_toks:
            return 0.0
        common = pred_toks & gold_toks
        if not common:
            return 0.0
        prec = len(common) / len(pred_toks)
        rec  = len(common) / len(gold_toks)
        return 2 * prec * rec / (prec + rec)

    def recall_at_k(self, retrieved_docs: List[str], gold_answer: str, k: int = 3) -> float:
        """Recall@k: top-k 검색 결과 안에 정답 포함 여부"""
        gold_norm = self.normalize(gold_answer)
        for doc in retrieved_docs[:k]:
            if gold_norm in self.normalize(doc):
                return 1.0
        return 0.0

    def precision(self, retrieved_docs: List[str], gold_answer: str) -> float:
        """Precision: 검색된 문서 중 정답 관련 비율"""
        if not retrieved_docs:
            return 0.0
        gold_norm = self.normalize(gold_answer)
        relevant = sum(1 for d in retrieved_docs if gold_norm in self.normalize(d))
        return relevant / len(retrieved_docs)

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        RAGAS Faithfulness 근사:
        답변의 각 문장이 컨텍스트에서 지지되는 비율
        """
        if not contexts or not answer:
            return 0.0
        sentences = [s.strip() for s in re.split(r'[.!?。]', answer) if s.strip()]
        if not sentences:
            return 0.0
        ctx_combined = self.normalize(" ".join(contexts))
        supported = sum(
            1 for s in sentences
            if any(tok in ctx_combined for tok in self.normalize(s).split() if len(tok) > 1)
        )
        return supported / len(sentences)

    def evaluate_single(self, pred: str, gold: str,
                        retrieved_docs: List[str],
                        all_contexts: List[str]) -> EvalResult:
        return EvalResult(
            f1            = self.f1_score(pred, gold),
            em_raw        = self.exact_match(pred, gold, normalize=False),
            em_norm       = self.exact_match(pred, gold, normalize=True),
            recall_at_k   = self.recall_at_k(retrieved_docs, gold),
            precision     = self.precision(retrieved_docs, gold),
            faithfulness  = self.faithfulness(pred, all_contexts),
        )

    def evaluate_dataset(self, rag, dataset: List[Dict],
                         runs: int = 3, verbose: bool = True) -> Dict:
        """
        전체 데이터셋 평가 (3회 반복 → 평균±표준편차)
        논문 Table 13 형식으로 결과 반환
        """
        import numpy as np

        all_f1, all_em, all_r3, all_prec, all_faith = [], [], [], [], []
        by_type: Dict[str, list] = {'simple':[], 'multi_hop':[], 'conditional':[]}

        for item in dataset:
            q, gold, qtype = item['query'], item['answer'], item['type']
            run_em = []
            for _ in range(runs):
                result = rag.query(q)
                retrieved = result.vector_contexts + result.graph_contexts + result.onto_contexts
                ev = self.evaluate_single(result.answer, gold, retrieved,
                                          result.vector_contexts)
                run_em.append(ev.em_norm)
                all_f1.append(ev.f1)
                all_r3.append(ev.recall_at_k)
                all_prec.append(ev.precision)
                all_faith.append(ev.faithfulness)
            all_em.append(np.mean(run_em))
            by_type[qtype].append(np.mean(run_em))

            if verbose:
                print(f"[{qtype}] Q: {q[:40]}... | EM={np.mean(run_em):.3f}")

        def stat(arr):
            a = np.array(arr)
            return float(np.mean(a)), float(np.std(a))

        summary = {
            "F1":          {"mean": stat(all_f1)[0],  "std": stat(all_f1)[1]},
            "EM":          {"mean": stat(all_em)[0],  "std": stat(all_em)[1]},
            "Recall@3":    {"mean": stat(all_r3)[0],  "std": stat(all_r3)[1]},
            "Precision":   {"mean": stat(all_prec)[0],"std": stat(all_prec)[1]},
            "Faithfulness":{"mean": stat(all_faith)[0],"std": stat(all_faith)[1]},
            "by_type": {
                t: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for t, v in by_type.items() if v
            }
        }
        return summary
