"""
TripleHybridRAG — Vector + Graph + Ontology 통합 파이프라인
논문 전체 아키텍처 구현 (Fig. 1)

S_total = α_final·S_vector + β_final·S_graph + γ_final·S_ontology
"""
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .query_analyzer import QueryAnalyzer, QueryIntent
from .dwa import DWA, DWAWeights
from .vector_store import VectorStore
from .knowledge_graph import KnowledgeGraph
from .ontology_engine import OntologyEngine


@dataclass
class RAGResult:
    """질의 처리 결과"""
    answer: str
    elapsed: float
    weights: DWAWeights
    intent: QueryIntent
    vector_contexts: List[str]
    graph_contexts: List[str]
    onto_contexts: List[str]
    prompt_used: str = ""


PROMPT_TEMPLATE = (
    "다음 컨텍스트를 기반으로 질문에 정확하게 답변하세요. "
    "컨텍스트에서 답을 찾을 수 없는 경우에만 '정보를 찾을 수 없습니다'라고 답하세요. "
    "Graph 경로 정보(A --[관계]--> B)도 참고하여 관계를 추론하세요. "
    "가능한 한 구체적이고 상세하게 답변하세요.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


class TripleHybridRAG:
    """
    메인 클래스
    - add_knowledge() : 지식 추가
    - query()         : 질의 처리 (DWA 가중치 자동 적용)
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 llm_model: str = "gpt-4o-mini",
                 temperature: float = 0.0,
                 top_k: int = 3,
                 lambda_: float = 0.3):

        import os
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self.top_k       = top_k
        self.llm_model   = llm_model
        self.temperature = temperature

        # 컴포넌트 초기화
        self.analyzer = QueryAnalyzer()
        self.dwa       = DWA(lambda_=lambda_)
        self.vector    = VectorStore()
        self.graph     = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        self.ontology  = OntologyEngine()

        # LLM
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        self._documents: List[str] = []
        self._built = False

    # ── 지식 추가 ──────────────────────────────────────────
    def add_documents(self, docs: List[str]):
        self._documents.extend(docs)

    def add_graph_edge(self, src: str, relation: str, dst: str,
                       src_type: str = "Node", dst_type: str = "Node"):
        if src not in self.graph.nodes:
            self.graph.add_node(src, src, src_type)
        if dst not in self.graph.nodes:
            self.graph.add_node(dst, dst, dst_type)
        self.graph.add_edge(src, relation, dst)

    def load_university_sample(self, extended: bool = True):
        """
        논문 실험 데이터 일괄 로드
        extended=True  → 30명/8학과/40과목/15프로젝트/문서 186건 (권장)
        extended=False → 원래 소규모 데이터 (하위 호환)
        """
        if extended:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from data.extended_loader import load_extended_graph, generate_documents
            load_extended_graph(self.graph)
            docs = generate_documents()
            self.add_documents(docs)
        else:
            self.graph.load_university_data()
            self.add_documents([
                "김철수 교수는 45세이며 컴퓨터공학과 소속이다. 인공지능개론, 딥러닝, 강화학습을 담당한다.",
                "이영희 교수는 38세이며 인공지능학과 소속이다. 딥러닝, 컴퓨터비전을 담당한다.",
                "박민수 교수는 52세이며 컴퓨터공학과 소속이다. 자연어처리를 담당한다.",
                "정수진 교수는 36세이며 인공지능학과 소속이다. 컴퓨터비전을 담당한다.",
            ])

    def build(self):
        """벡터 인덱스 구축"""
        if self._documents:
            self.vector.build(self._documents)
        self._built = True
        print("✅ TripleHybridRAG 빌드 완료")

    # ── 질의 처리 ──────────────────────────────────────────
    def query(self, question: str) -> RAGResult:
        start = time.time()

        # 1. 질의 분석
        intent  = self.analyzer.analyze(question)
        weights = self.dwa.compute(intent)

        # 2. 병렬 검색 (각 소스)
        v_ctxs = [doc for doc, _ in self.vector.search(question, self.top_k)] if self._built else []
        g_ctxs = self.graph.search(question, self.top_k)
        o_ctxs = self.ontology.search(question, self.top_k)

        # 3. 가중치 기반 컨텍스트 통합
        context = self._merge_contexts(
            v_ctxs, g_ctxs, o_ctxs,
            weights.alpha, weights.beta, weights.gamma
        )

        # 4. LLM 답변 생성
        prompt = PROMPT_TEMPLATE.format(context=context, query=question)
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        elapsed = time.time() - start
        return RAGResult(
            answer=answer, elapsed=elapsed,
            weights=weights, intent=intent,
            vector_contexts=v_ctxs,
            graph_contexts=g_ctxs,
            onto_contexts=o_ctxs,
            prompt_used=prompt,
        )

    def _merge_contexts(self, v_ctxs, g_ctxs, o_ctxs,
                        alpha, beta, gamma) -> str:
        """가중치 비례 컨텍스트 조합 — 지배적 소스에 더 많은 슬롯 배분"""
        total = alpha + beta + gamma
        budget = self.top_k * 3  # 총 컨텍스트 슬롯 확대
        n_v = max(1, round(budget * alpha / total))
        n_g = max(1, round(budget * beta  / total))
        n_o = max(1, round(budget * gamma / total))

        parts = []
        if v_ctxs:
            parts.append(f"[Vector(α={alpha:.2f})]\n" + "\n".join(v_ctxs[:n_v]))
        if g_ctxs:
            parts.append(f"[Graph(β={beta:.2f})]\n"  + "\n".join(g_ctxs[:n_g]))
        if o_ctxs:
            parts.append(f"[Ontology(γ={gamma:.2f})]\n" + "\n".join(o_ctxs[:n_o]))
        return "\n\n".join(parts)
