"""
DWA 단위 테스트 — 논문 Table 6 수치 검증
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.query_analyzer import QueryAnalyzer, QueryIntent
from src.dwa import DWA, DWAWeights


def test_table6_case1():
    """c_r=0, c_c=0 → 가중치 변화 없음"""
    intent = QueryIntent('simple', [], [], [], 0.0, c_e=0.2, c_r=0.0, c_c=0.0)
    w = DWA(0.3).compute(intent)
    assert abs(w.alpha - 0.60) < 0.01, f"α={w.alpha}"
    assert abs(w.beta  - 0.20) < 0.01, f"β={w.beta}"
    assert abs(w.gamma - 0.20) < 0.01, f"γ={w.gamma}"
    print("✅ Table6 Case1 통과")


def test_table6_case2():
    """c_r=0.25 → Graph 가중치 상승"""
    intent = QueryIntent('simple', [], [], [], 0.0, c_e=0.4, c_r=0.25, c_c=0.0)
    w = DWA(0.3).compute(intent)
    assert w.beta > 0.20, f"β should increase: {w.beta}"
    assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6, "합이 1.0 아님"
    print(f"✅ Table6 Case2 통과: α={w.alpha:.2f} β={w.beta:.2f} γ={w.gamma:.2f}")


def test_table6_case3():
    """c_r=0.25, c_c=0.33 → Graph+Ontology 동시 상승, Vector 감소"""
    intent = QueryIntent('simple', [], [], [], 0.0, c_e=0.4, c_r=0.25, c_c=0.33)
    w = DWA(0.3).compute(intent)
    assert w.alpha < 0.60, f"α should decrease: {w.alpha}"
    assert w.beta  > 0.20, f"β should increase: {w.beta}"
    assert w.gamma > 0.20, f"γ should increase: {w.gamma}"
    assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6
    print(f"✅ Table6 Case3 통과: α={w.alpha:.2f} β={w.beta:.2f} γ={w.gamma:.2f}")


def test_normalization():
    """정규화: 가중치 합 항상 1.0"""
    dwa = DWA(0.3)
    for qtype in ['simple','multi_hop','conditional']:
        for c_r in [0.0, 0.5, 1.0]:
            for c_c in [0.0, 0.5, 1.0]:
                intent = QueryIntent(qtype,[],[],[],0.0,c_e=0.2,c_r=c_r,c_c=c_c)
                w = dwa.compute(intent)
                total = w.alpha + w.beta + w.gamma
                assert abs(total - 1.0) < 1e-6, f"합={total} (type={qtype})"
    print("✅ 정규화 테스트 통과 (27 케이스)")


def test_query_analyzer():
    """QueryAnalyzer 기본 동작"""
    qa = QueryAnalyzer()
    intent = qa.analyze("김철수 교수가 소속된 학과의 40세 이하 교수 목록은?")
    assert intent.query_type == 'conditional', f"type={intent.query_type}"
    assert len(intent.entities) > 0, "개체명 없음"
    assert len(intent.constraints) > 0, "제약조건 없음"
    print(f"✅ QueryAnalyzer 테스트 통과: {intent.query_type}, 개체={intent.entities}")


if __name__ == "__main__":
    test_table6_case1()
    test_table6_case2()
    test_table6_case3()
    test_normalization()
    test_query_analyzer()
    print("\n🎉 모든 테스트 통과!")
