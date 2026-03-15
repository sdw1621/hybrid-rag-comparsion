"""
Dynamic Weighting Algorithm (DWA)
논문 Section III.4 / 수식 (1)~(8) 완전 구현
"""
from dataclasses import dataclass
from .query_analyzer import QueryIntent


@dataclass
class DWAWeights:
    alpha: float   # Vector 가중치
    beta:  float   # Graph  가중치
    gamma: float   # Ontology 가중치

    def as_dict(self):
        return {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}

    def __repr__(self):
        return f"DWA(α={self.alpha:.3f}, β={self.beta:.3f}, γ={self.gamma:.3f})"


class DWA:
    """
    Stage 1 : 질의 유형 → 기본 가중치 (Table 4)
    Stage 2 : 밀도 신호 c_e/c_r/c_c → 연속 조정 (수식 5~7)
    Normalize: 합 = 1.0 (수식 8)
    λ = 0.3 (grid search 결과)
    """

    # 논문 Table 4
    BASE_WEIGHTS = {
        'simple':      {'alpha': 0.6, 'beta': 0.2, 'gamma': 0.2},
        'multi_hop':   {'alpha': 0.2, 'beta': 0.6, 'gamma': 0.2},
        'conditional': {'alpha': 0.2, 'beta': 0.2, 'gamma': 0.6},
    }

    def __init__(self, lambda_: float = 0.3):
        self.lambda_ = lambda_

    def compute(self, intent: QueryIntent) -> DWAWeights:
        b = self.BASE_WEIGHTS[intent.query_type]
        a0, b0, g0 = b['alpha'], b['beta'], b['gamma']
        lam = self.lambda_
        c_r, c_c = intent.c_r, intent.c_c

        # 수식 (5)~(7)
        a_ = a0 * (1 - lam * (c_r + c_c) / 2)
        b_ = b0 + lam * c_r * (1 - b0)
        g_ = g0 + lam * c_c * (1 - g0)

        # 수식 (8) 정규화
        total = a_ + b_ + g_
        return DWAWeights(alpha=a_/total, beta=b_/total, gamma=g_/total)

    def explain(self, intent: QueryIntent) -> str:
        b = self.BASE_WEIGHTS[intent.query_type]
        w = self.compute(intent)
        return (
            f"[Stage 1] type={intent.query_type} "
            f"→ α={b['alpha']}, β={b['beta']}, γ={b['gamma']}\n"
            f"[Stage 2] c_r={intent.c_r:.2f} c_c={intent.c_c:.2f} λ={self.lambda_}\n"
            f"[Final  ] α={w.alpha:.3f} β={w.beta:.3f} γ={w.gamma:.3f}"
        )
