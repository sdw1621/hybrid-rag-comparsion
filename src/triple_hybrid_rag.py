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
    "Based on the following context, answer the question accurately. "
    "If the answer cannot be determined from the context, state that the information is not available.\n\n"
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

    def load_university_sample(self):
        """논문 실험 데이터 일괄 로드"""
        self.graph.load_university_data()
        docs = [
            "김철수 교수는 45세이며 컴퓨터공학과 소속이다. 인공지능개론, 딥러닝, 강화학습을 담당한다. 머신러닝과 자연어처리를 연구하며 이영희 교수와 협력한다.",
            "이영희 교수는 38세이며 인공지능학과 소속이다. 딥러닝, 컴퓨터비전을 담당한다. 딥러닝과 컴퓨터비전을 연구하며 김철수, 박민수 교수와 협력한다.",
            "박민수 교수는 52세이며 컴퓨터공학과 소속이다. 자연어처리를 담당한다. 자연어처리와 정보검색을 연구한다.",
            "정수진 교수는 36세이며 인공지능학과 소속이다. 컴퓨터비전을 담당한다. 딥러닝과 컴퓨터비전을 연구한다.",
            "컴퓨터공학과에는 김철수 교수(45세)와 박민수 교수(52세)가 소속되어 있다.",
            "인공지능학과에는 이영희 교수(38세)와 정수진 교수(36세)가 소속되어 있다.",
            "인공지능개론은 3학점 과목으로 김철수 교수가 담당한다. AI 기초 개념과 알고리즘을 다룬다.",
            "딥러닝은 3학점 과목으로 김철수, 이영희 교수가 공동 담당한다. CNN, RNN, Transformer를 다룬다.",
            "자연어처리는 3학점 과목으로 박민수 교수가 담당한다. 텍스트 분석과 언어 모델을 다룬다.",
            "컴퓨터비전은 3학점 과목으로 이영희, 정수진 교수가 공동 담당한다. 이미지 인식과 객체 탐지를 다룬다.",
            "강화학습은 3학점 과목으로 김철수 교수가 담당한다. MDP, Q-Learning, PPO를 다룬다.",
            "AI융합프로젝트에는 김철수, 이영희 교수가 참여한다. NLP연구프로젝트에는 박민수 교수가 참여한다.",
        ]
        self.add_documents(docs)

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
        """가중치 비례 컨텍스트 조합"""
        total = alpha + beta + gamma
        n_v = max(1, round(self.top_k * alpha / total))
        n_g = max(1, round(self.top_k * beta  / total))
        n_o = max(1, round(self.top_k * gamma / total))

        parts = []
        if v_ctxs:
            parts.append(f"[Vector(α={alpha:.2f})]\n" + "\n".join(v_ctxs[:n_v]))
        if g_ctxs:
            parts.append(f"[Graph(β={beta:.2f})]\n"  + "\n".join(g_ctxs[:n_g]))
        if o_ctxs:
            parts.append(f"[Ontology(γ={gamma:.2f})]\n" + "\n".join(o_ctxs[:n_o]))
        return "\n\n".join(parts)
