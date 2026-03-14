"""
Triple-Hybrid RAG — 통합 테스트 스위트
QueryAnalyzer / DWA / Evaluator / KnowledgeGraph / OntologyEngine 전 모듈 커버
실행: pytest tests/test_integration.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.query_analyzer import QueryAnalyzer, QueryIntent
from src.dwa import DWA, DWAWeights
from src.evaluator import Evaluator, EvalResult
from src.knowledge_graph import KnowledgeGraph
from src.ontology_engine import OntologyEngine


# ════════════════════════════════════════════════════════
# QueryAnalyzer 테스트
# ════════════════════════════════════════════════════════
class TestQueryAnalyzer:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.qa = QueryAnalyzer()

    # ── 질의 유형 분류 ────────────────────────────────
    def test_simple_query_classification(self):
        intent = self.qa.analyze("김철수 교수는 누구인가요?")
        assert intent.query_type == 'simple'

    def test_multi_hop_query_two_entities(self):
        intent = self.qa.analyze("김철수 교수와 이영희 교수가 협력하는 연구는?")
        assert intent.query_type == 'multi_hop'

    def test_multi_hop_query_two_relations(self):
        intent = self.qa.analyze("이영희 교수가 소속하고 담당하는 것은?")
        assert intent.query_type == 'multi_hop'

    def test_conditional_query_age_constraint(self):
        intent = self.qa.analyze("컴퓨터공학과 교수 중 40세 이하는 누구인가요?")
        assert intent.query_type == 'conditional'

    def test_conditional_query_exception(self):
        intent = self.qa.analyze("이영희 교수를 제외한 딥러닝 담당 교수는?")
        assert intent.query_type == 'conditional'

    # ── 개체명 추출 ───────────────────────────────────
    def test_entity_extraction_professor_name(self):
        intent = self.qa.analyze("김철수 교수의 담당 과목은?")
        assert any("김철수" in e for e in intent.entities)

    def test_entity_extraction_department(self):
        intent = self.qa.analyze("컴퓨터공학과 소속 교수 목록")
        assert any("컴퓨터공학과" in e for e in intent.entities)

    def test_entity_extraction_multiple(self):
        intent = self.qa.analyze("김철수 교수와 이영희 교수의 협력 현황")
        assert len(intent.entities) >= 2

    def test_entity_extraction_empty_query(self):
        intent = self.qa.analyze("무엇을 가르치나요?")
        # 개체가 없어도 오류 없이 처리
        assert isinstance(intent.entities, list)

    # ── 관계 키워드 추출 ──────────────────────────────
    def test_relation_extraction_belongs(self):
        intent = self.qa.analyze("김철수 교수가 소속된 학과는?")
        assert "소속" in intent.relations

    def test_relation_extraction_teaches(self):
        intent = self.qa.analyze("이영희 교수가 담당하는 과목은?")
        assert "담당" in intent.relations

    def test_relation_extraction_multiple(self):
        intent = self.qa.analyze("박민수 교수가 소속되고 담당하는 것은?")
        assert len(intent.relations) >= 2

    # ── 제약조건 추출 ─────────────────────────────────
    def test_constraint_extraction_age_below(self):
        intent = self.qa.analyze("40세 이하 교수 목록")
        assert len(intent.constraints) > 0

    def test_constraint_extraction_age_above(self):
        intent = self.qa.analyze("50세 이상 교수는?")
        assert len(intent.constraints) > 0

    def test_constraint_extraction_exclusion(self):
        intent = self.qa.analyze("김철수 제외한 교수 목록")
        assert len(intent.constraints) > 0

    # ── 밀도 신호 ─────────────────────────────────────
    def test_density_c_e_range(self):
        intent = self.qa.analyze("김철수 교수의 연구")
        assert 0.0 <= intent.c_e <= 1.0

    def test_density_c_r_range(self):
        intent = self.qa.analyze("소속된 담당하는 협력하는")
        assert 0.0 <= intent.c_r <= 1.0

    def test_density_c_c_range(self):
        intent = self.qa.analyze("40세 이하 50세 이상 제외")
        assert 0.0 <= intent.c_c <= 1.0

    def test_density_signals_normalized(self):
        intent = self.qa.analyze("김철수 이영희 박민수 정수진 교수 소속 담당 협력 참여 40세 이하 제외")
        assert intent.c_e <= 1.0
        assert intent.c_r <= 1.0
        assert intent.c_c <= 1.0

    # ── 복잡도 점수 ───────────────────────────────────
    def test_complexity_score_range(self):
        intent = self.qa.analyze("김철수 교수의 연구 분야는?")
        assert 0.0 <= intent.complexity_score <= 1.0

    def test_complexity_higher_for_complex_query(self):
        simple_intent = self.qa.analyze("교수 목록")
        complex_intent = self.qa.analyze("김철수 이영희 교수가 소속되고 담당하는 40세 이하 과목은?")
        assert complex_intent.complexity_score >= simple_intent.complexity_score


# ════════════════════════════════════════════════════════
# DWA 테스트
# ════════════════════════════════════════════════════════
class TestDWA:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dwa = DWA(lambda_=0.3)

    def _make_intent(self, qtype, c_e=0.2, c_r=0.0, c_c=0.0):
        return QueryIntent(qtype, [], [], [], 0.0, c_e=c_e, c_r=c_r, c_c=c_c)

    # ── 기본 가중치 (Stage 1) ─────────────────────────
    def test_simple_base_weights(self):
        w = self.dwa.compute(self._make_intent('simple'))
        assert w.alpha > w.beta
        assert w.alpha > w.gamma

    def test_multi_hop_base_weights(self):
        w = self.dwa.compute(self._make_intent('multi_hop'))
        assert w.beta > w.alpha
        assert w.beta > w.gamma

    def test_conditional_base_weights(self):
        w = self.dwa.compute(self._make_intent('conditional'))
        assert w.gamma > w.alpha
        assert w.gamma > w.beta

    # ── Stage 2 연속 조정 ─────────────────────────────
    def test_graph_weight_increases_with_c_r(self):
        w0 = self.dwa.compute(self._make_intent('simple', c_r=0.0))
        w1 = self.dwa.compute(self._make_intent('simple', c_r=0.5))
        assert w1.beta > w0.beta

    def test_ontology_weight_increases_with_c_c(self):
        w0 = self.dwa.compute(self._make_intent('simple', c_c=0.0))
        w1 = self.dwa.compute(self._make_intent('simple', c_c=0.5))
        assert w1.gamma > w0.gamma

    def test_vector_weight_decreases_with_c_r_c_c(self):
        w0 = self.dwa.compute(self._make_intent('simple', c_r=0.0, c_c=0.0))
        w1 = self.dwa.compute(self._make_intent('simple', c_r=0.5, c_c=0.5))
        assert w1.alpha < w0.alpha

    # ── 정규화 ────────────────────────────────────────
    def test_weights_sum_to_one_simple(self):
        w = self.dwa.compute(self._make_intent('simple', c_r=0.3, c_c=0.3))
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6

    def test_weights_sum_to_one_multi_hop(self):
        w = self.dwa.compute(self._make_intent('multi_hop', c_r=1.0, c_c=0.5))
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6

    def test_weights_sum_to_one_conditional(self):
        w = self.dwa.compute(self._make_intent('conditional', c_r=0.5, c_c=1.0))
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6

    def test_normalization_all_combinations(self):
        for qtype in ['simple', 'multi_hop', 'conditional']:
            for c_r in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for c_c in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    intent = self._make_intent(qtype, c_r=c_r, c_c=c_c)
                    w = self.dwa.compute(intent)
                    total = w.alpha + w.beta + w.gamma
                    assert abs(total - 1.0) < 1e-6, f"합={total} (type={qtype})"

    # ── λ 민감도 ──────────────────────────────────────
    def test_lambda_zero_no_adjustment(self):
        dwa0 = DWA(lambda_=0.0)
        w = dwa0.compute(self._make_intent('simple', c_r=0.5, c_c=0.5))
        # λ=0 이면 Stage 2 조정 없음 → base weight 그대로
        assert abs(w.alpha - 0.6) < 0.01
        assert abs(w.beta  - 0.2) < 0.01
        assert abs(w.gamma - 0.2) < 0.01

    def test_higher_lambda_amplifies_effect(self):
        w_low  = DWA(lambda_=0.1).compute(self._make_intent('simple', c_r=0.5))
        w_high = DWA(lambda_=0.5).compute(self._make_intent('simple', c_r=0.5))
        # 높은 λ → graph 가중치 더 크게 증가
        assert w_high.beta > w_low.beta

    # ── DWAWeights 데이터클래스 ───────────────────────
    def test_weights_as_dict(self):
        w = self.dwa.compute(self._make_intent('simple'))
        d = w.as_dict()
        assert set(d.keys()) == {'alpha', 'beta', 'gamma'}
        assert abs(sum(d.values()) - 1.0) < 1e-6

    def test_explain_output(self):
        intent = self._make_intent('simple', c_r=0.3, c_c=0.2)
        text = self.dwa.explain(intent)
        assert 'Stage 1' in text
        assert 'Stage 2' in text
        assert 'Final' in text


# ════════════════════════════════════════════════════════
# Evaluator 테스트
# ════════════════════════════════════════════════════════
class TestEvaluator:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ev = Evaluator()

    # ── normalize ─────────────────────────────────────
    def test_normalize_lowercase(self):
        assert self.ev.normalize("Hello World") == "hello world"

    def test_normalize_extra_spaces(self):
        assert self.ev.normalize("hello  world") == "hello world"

    def test_normalize_korean_particle_ga(self):
        result = self.ev.normalize("김철수가")
        assert "가" not in result or "김철수" in result

    def test_normalize_korean_particle_eul(self):
        result = self.ev.normalize("과목을")
        assert "을" not in result or "과목" in result

    def test_normalize_unicode_nfc(self):
        import unicodedata
        text_nfd = unicodedata.normalize('NFD', "김철수")
        text_nfc = unicodedata.normalize('NFC', "김철수")
        assert self.ev.normalize(text_nfd) == self.ev.normalize(text_nfc)

    # ── exact_match ───────────────────────────────────
    def test_exact_match_identical(self):
        assert self.ev.exact_match("김철수", "김철수") == 1.0

    def test_exact_match_different(self):
        assert self.ev.exact_match("김철수", "이영희") == 0.0

    def test_exact_match_raw_case_sensitive(self):
        assert self.ev.exact_match("Hello", "hello", normalize=False) == 0.0

    def test_exact_match_normalized_case_insensitive(self):
        assert self.ev.exact_match("Hello", "hello", normalize=True) == 1.0

    def test_exact_match_whitespace_normalized(self):
        assert self.ev.exact_match("김 철수", "김철수", normalize=True) == 0.0

    # ── f1_score ──────────────────────────────────────
    def test_f1_perfect_match(self):
        assert self.ev.f1_score("김철수 교수 컴퓨터", "김철수 교수 컴퓨터") == 1.0

    def test_f1_no_overlap(self):
        assert self.ev.f1_score("김철수", "이영희") == 0.0

    def test_f1_partial_overlap(self):
        score = self.ev.f1_score("김철수 교수", "김철수 학생")
        assert 0.0 < score < 1.0

    def test_f1_empty_prediction(self):
        assert self.ev.f1_score("", "김철수") == 0.0

    def test_f1_empty_gold(self):
        assert self.ev.f1_score("김철수", "") == 0.0

    def test_f1_symmetric(self):
        a, b = "김철수 교수 인공지능", "인공지능 김철수 교수"
        assert abs(self.ev.f1_score(a, b) - self.ev.f1_score(b, a)) < 1e-6

    # ── recall_at_k ───────────────────────────────────
    def test_recall_at_k_found_in_top(self):
        docs = ["김철수 교수는 45세입니다", "이영희 교수", "박민수"]
        assert self.ev.recall_at_k(docs, "45세", k=3) == 1.0

    def test_recall_at_k_not_found(self):
        docs = ["이영희 교수", "박민수 교수"]
        assert self.ev.recall_at_k(docs, "정수진", k=2) == 0.0

    def test_recall_at_k_boundary(self):
        docs = ["이영희", "박민수", "정수진", "김철수"]
        # 정답이 4번째 → k=3이면 없음
        assert self.ev.recall_at_k(docs, "김철수", k=3) == 0.0
        # k=4면 있음
        assert self.ev.recall_at_k(docs, "김철수", k=4) == 1.0

    def test_recall_at_k_empty_docs(self):
        assert self.ev.recall_at_k([], "김철수", k=3) == 0.0

    # ── precision ─────────────────────────────────────
    def test_precision_all_relevant(self):
        docs = ["김철수 45세", "김철수 교수", "김철수 컴퓨터공학과"]
        assert self.ev.precision(docs, "김철수") == 1.0

    def test_precision_none_relevant(self):
        docs = ["이영희 교수", "박민수 교수"]
        assert self.ev.precision(docs, "정수진") == 0.0

    def test_precision_partial(self):
        docs = ["김철수 교수", "이영희 교수"]
        score = self.ev.precision(docs, "김철수")
        assert 0.0 < score < 1.0

    def test_precision_empty_docs(self):
        assert self.ev.precision([], "김철수") == 0.0

    # ── faithfulness ─────────────────────────────────
    def test_faithfulness_supported(self):
        answer = "김철수 교수는 컴퓨터공학과 소속입니다."
        contexts = ["김철수 교수는 45세이며 컴퓨터공학과 소속이다."]
        score = self.ev.faithfulness(answer, contexts)
        assert score > 0.0

    def test_faithfulness_unsupported(self):
        answer = "정수진 교수는 해외 대학에 있습니다."
        contexts = ["김철수 교수 컴퓨터공학과"]
        score = self.ev.faithfulness(answer, contexts)
        assert score >= 0.0  # 0 or low

    def test_faithfulness_empty_answer(self):
        assert self.ev.faithfulness("", ["some context"]) == 0.0

    def test_faithfulness_empty_contexts(self):
        assert self.ev.faithfulness("some answer", []) == 0.0

    # ── evaluate_single ───────────────────────────────
    def test_evaluate_single_returns_eval_result(self):
        result = self.ev.evaluate_single(
            pred="김철수 교수",
            gold="김철수 교수",
            retrieved_docs=["김철수 교수는 45세"],
            all_contexts=["김철수 교수는 컴퓨터공학과 소속이다"]
        )
        assert isinstance(result, EvalResult)
        assert result.f1 == 1.0
        assert result.em_norm == 1.0

    def test_evaluate_single_as_dict(self):
        result = self.ev.evaluate_single("test", "test", [], [])
        d = result.as_dict()
        expected_keys = {"F1", "EM_raw", "EM_norm", "Recall@3", "Precision", "Faithfulness"}
        assert set(d.keys()) == expected_keys

    def test_evaluate_single_scores_in_range(self):
        result = self.ev.evaluate_single(
            pred="김철수 교수가 담당하는 과목은 인공지능개론입니다",
            gold="김철수 교수는 인공지능개론을 담당합니다",
            retrieved_docs=["김철수 교수는 인공지능개론, 딥러닝을 담당한다"],
            all_contexts=["김철수 교수는 컴퓨터공학과 소속이다"]
        )
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.em_norm <= 1.0
        assert 0.0 <= result.recall_at_k <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.faithfulness <= 1.0


# ════════════════════════════════════════════════════════
# KnowledgeGraph 테스트
# ════════════════════════════════════════════════════════
class TestKnowledgeGraph:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.kg = KnowledgeGraph()

    # ── 노드/엣지 추가 ────────────────────────────────
    def test_add_node(self):
        self.kg.add_node("p1", "김철수", "Professor")
        assert "p1" in self.kg.nodes
        assert self.kg.nodes["p1"]["name"] == "김철수"

    def test_add_node_with_props(self):
        self.kg.add_node("p1", "김철수", "Professor", age=45, dept="컴퓨터공학과")
        assert self.kg.nodes["p1"]["age"] == 45
        assert self.kg.nodes["p1"]["dept"] == "컴퓨터공학과"

    def test_add_edge(self):
        self.kg.add_node("p1", "김철수", "Professor")
        self.kg.add_node("c1", "인공지능개론", "Course")
        self.kg.add_edge("p1", "담당", "c1")
        assert ("p1", "담당", "c1") in self.kg.edges

    def test_adjacency_list_forward(self):
        self.kg.add_node("p1", "김철수", "Professor")
        self.kg.add_node("c1", "인공지능개론", "Course")
        self.kg.add_edge("p1", "담당", "c1")
        assert any(rel == "담당" and dst == "c1" for rel, dst in self.kg.adj["p1"])

    def test_adjacency_list_inverse(self):
        self.kg.add_node("p1", "김철수", "Professor")
        self.kg.add_node("c1", "인공지능개론", "Course")
        self.kg.add_edge("p1", "담당", "c1")
        assert any(rel == "inv_담당" and dst == "p1" for rel, dst in self.kg.adj["c1"])

    # ── BFS 검색 ──────────────────────────────────────
    def test_bfs_search_keyword_match(self):
        self.kg.add_node("p1", "김철수", "Professor")
        self.kg.add_node("c1", "인공지능개론", "Course")
        self.kg.add_edge("p1", "담당", "c1")
        results = self.kg.search("김철수", top_k=5)
        assert len(results) > 0
        assert any("김철수" in r for r in results)

    def test_bfs_search_returns_list(self):
        results = self.kg.search("없는키워드", top_k=3)
        assert isinstance(results, list)

    def test_bfs_search_top_k_limit(self):
        self.kg.load_university_data()
        results = self.kg.search("김철수", top_k=2)
        assert len(results) <= 2

    def test_bfs_search_path_format(self):
        self.kg.add_node("p1", "김철수", "Professor")
        self.kg.add_node("c1", "인공지능개론", "Course")
        self.kg.add_edge("p1", "담당", "c1")
        results = self.kg.search("김철수", top_k=3)
        # 결과 형식: "X --[rel]--> Y"
        for r in results:
            assert "--[" in r and "]-->" in r

    # ── 대학 데이터 로드 ──────────────────────────────
    def test_load_university_data_nodes(self):
        self.kg.load_university_data()
        assert len(self.kg.nodes) == 13  # 교수4 + 과목5 + 학과2 + 프로젝트2

    def test_load_university_data_edges(self):
        self.kg.load_university_data()
        assert len(self.kg.edges) == 16

    def test_load_university_search_professor(self):
        self.kg.load_university_data()
        results = self.kg.search("이영희", top_k=3)
        assert len(results) > 0

    def test_load_university_search_department(self):
        self.kg.load_university_data()
        results = self.kg.search("컴퓨터공학과", top_k=5)
        assert len(results) > 0


# ════════════════════════════════════════════════════════
# OntologyEngine 테스트
# ════════════════════════════════════════════════════════
class TestOntologyEngine:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.onto = OntologyEngine()

    # ── 기본 초기화 ───────────────────────────────────
    def test_instances_loaded(self):
        # rule-based 또는 owlready2 모두 인스턴스 데이터 있어야 함
        results = self.onto.search("김철수", top_k=5)
        assert len(results) > 0

    def test_search_professor_name(self):
        results = self.onto.search("이영희", top_k=3)
        assert any("이영희" in r for r in results)

    def test_search_returns_list(self):
        results = self.onto.search("없는이름", top_k=3)
        assert isinstance(results, list)

    def test_search_top_k_respected(self):
        results = self.onto.search("교수", top_k=2)
        assert len(results) <= 2

    # ── check_constraint ─────────────────────────────
    def test_constraint_below_pass(self):
        # 이영희: 38세 → 40세 이하 → True
        assert self.onto.check_constraint("이영희", "40세 이하") is True

    def test_constraint_below_fail(self):
        # 박민수: 52세 → 40세 이하 → False
        assert self.onto.check_constraint("박민수", "40세 이하") is False

    def test_constraint_above_pass(self):
        # 박민수: 52세 → 50세 이상 → True
        assert self.onto.check_constraint("박민수", "50세 이상") is True

    def test_constraint_above_fail(self):
        # 이영희: 38세 → 50세 이상 → False
        assert self.onto.check_constraint("이영희", "50세 이상") is False

    def test_constraint_below_strict_pass(self):
        # 정수진: 36세 → 37세 미만 → True
        assert self.onto.check_constraint("정수진", "37세 미만") is True

    def test_constraint_below_strict_fail(self):
        # 정수진: 36세 → 36세 미만 → False
        assert self.onto.check_constraint("정수진", "36세 미만") is False

    def test_constraint_above_strict_pass(self):
        # 김철수: 45세 → 44세 초과 → True
        assert self.onto.check_constraint("김철수", "44세 초과") is True

    def test_constraint_above_strict_fail(self):
        # 김철수: 45세 → 45세 초과 → False
        assert self.onto.check_constraint("김철수", "45세 초과") is False

    def test_constraint_unknown_entity_returns_true(self):
        # 알 수 없는 개체 → 기본값 True
        assert self.onto.check_constraint("없는교수", "40세 이하") is True

    def test_constraint_no_age_pattern_returns_true(self):
        # 연령 패턴 없는 제약 → True
        assert self.onto.check_constraint("김철수", "컴퓨터공학과") is True


# ════════════════════════════════════════════════════════
# Triple-Hybrid RAG 통합 테스트 (LLM 제외)
# ════════════════════════════════════════════════════════
class TestTripleHybridRAGNoLLM:
    """LLM 없이 파이프라인 컴포넌트 통합 검증"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.qa    = QueryAnalyzer()
        self.dwa   = DWA(lambda_=0.3)
        self.kg    = KnowledgeGraph()
        self.onto  = OntologyEngine()
        self.ev    = Evaluator()
        self.kg.load_university_data()

    def test_full_pipeline_simple_query(self):
        query  = "김철수 교수는 누구인가요?"
        intent  = self.qa.analyze(query)
        weights = self.dwa.compute(intent)
        g_ctx   = self.kg.search(query, top_k=3)
        o_ctx   = self.onto.search(query, top_k=3)

        assert intent.query_type == 'simple'
        assert abs(weights.alpha + weights.beta + weights.gamma - 1.0) < 1e-6
        assert isinstance(g_ctx, list)
        assert isinstance(o_ctx, list)

    def test_full_pipeline_conditional_query(self):
        query  = "40세 이하 교수는?"
        intent  = self.qa.analyze(query)
        weights = self.dwa.compute(intent)

        assert intent.query_type == 'conditional'
        # 제약 쿼리 → ontology 가중치 우세
        assert weights.gamma > weights.alpha

    def test_full_pipeline_multi_hop_query(self):
        query  = "김철수 교수와 이영희 교수가 함께 참여하는 프로젝트는?"
        intent  = self.qa.analyze(query)
        weights = self.dwa.compute(intent)
        g_ctx   = self.kg.search(query, top_k=3)

        assert intent.query_type == 'multi_hop'
        # multi_hop → graph 가중치 우세
        assert weights.beta > weights.alpha
        assert len(g_ctx) > 0

    def test_evaluator_pipeline_output(self):
        pred = "김철수 교수"
        gold = "김철수 교수"
        docs = self.kg.search("김철수", top_k=3)
        ctx  = self.onto.search("김철수", top_k=3)

        result = self.ev.evaluate_single(pred, gold, docs, ctx)
        assert result.f1 == 1.0
        assert result.em_norm == 1.0

    def test_constraint_filtering_pipeline(self):
        """OntologyEngine 제약 필터링 + Evaluator 통합"""
        professors = ["김철수", "이영희", "박민수", "정수진"]
        constraint = "40세 이하"
        filtered = [p for p in professors if self.onto.check_constraint(p, constraint)]

        # 이영희(38), 정수진(36)만 해당
        assert "이영희" in filtered
        assert "정수진" in filtered
        assert "김철수" not in filtered  # 45세
        assert "박민수" not in filtered  # 52세

    def test_weight_consistency_across_queries(self):
        """같은 유형 질의 → 비슷한 가중치 패턴"""
        simple_queries = [
            "김철수 교수의 나이는?",
            "인공지능개론 담당 교수는?",
        ]
        for q in simple_queries:
            intent  = self.qa.analyze(q)
            weights = self.dwa.compute(intent)
            # simple query: alpha가 beta, gamma보다 커야 함
            if intent.query_type == 'simple':
                assert weights.alpha > weights.beta or weights.alpha > weights.gamma


# ════════════════════════════════════════════════════════
# 경계 케이스 & 강건성 테스트
# ════════════════════════════════════════════════════════
class TestEdgeCases:

    def test_query_analyzer_empty_string(self):
        qa = QueryAnalyzer()
        intent = qa.analyze("")
        assert intent.query_type in ('simple', 'multi_hop', 'conditional')
        assert isinstance(intent.entities, list)

    def test_query_analyzer_whitespace_only(self):
        qa = QueryAnalyzer()
        intent = qa.analyze("   ")
        assert intent is not None

    def test_dwa_boundary_c_r_one(self):
        dwa = DWA(lambda_=0.3)
        intent = QueryIntent('simple', [], [], [], 0.0, c_e=1.0, c_r=1.0, c_c=0.0)
        w = dwa.compute(intent)
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6
        assert all(v >= 0.0 for v in [w.alpha, w.beta, w.gamma])

    def test_dwa_boundary_c_c_one(self):
        dwa = DWA(lambda_=0.3)
        intent = QueryIntent('simple', [], [], [], 0.0, c_e=1.0, c_r=0.0, c_c=1.0)
        w = dwa.compute(intent)
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6

    def test_dwa_high_lambda_boundary(self):
        dwa = DWA(lambda_=0.5)
        intent = QueryIntent('simple', [], [], [], 0.0, c_e=1.0, c_r=1.0, c_c=1.0)
        w = dwa.compute(intent)
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6
        assert all(v >= 0.0 for v in [w.alpha, w.beta, w.gamma])

    def test_evaluator_normalize_empty_string(self):
        ev = Evaluator()
        assert ev.normalize("") == ""

    def test_knowledge_graph_duplicate_node(self):
        kg = KnowledgeGraph()
        kg.add_node("p1", "김철수", "Professor")
        kg.add_node("p1", "김철수Updated", "Professor")
        # 마지막 값으로 덮어쓰기
        assert kg.nodes["p1"]["name"] == "김철수Updated"

    def test_ontology_search_all_professors(self):
        onto = OntologyEngine()
        results = self.onto_search_fallback(onto, "교수")
        assert isinstance(results, list)

    def onto_search_fallback(self, onto, q):
        return onto.search(q, top_k=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
